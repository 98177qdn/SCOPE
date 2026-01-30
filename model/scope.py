import os
import torch as T
import torch.nn.functional as F
from agents.scope_agent import SCOPE_Agent

from envs.CBF import solve_joint_cbf, check_future_cbf_feasibility, compute_cbf_violation_per_agent
import numpy as np

class SCOPE:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions, env,
                 scenario='simple', alpha=0.01, beta=0.02, fc1=128,
                 fc2=128, gamma=0.99, tau=0.01, chkpt_dir='tmp/scope/'):
        self.agents = []
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.env = env
        chkpt_dir += scenario
        self.chkpt_dir = chkpt_dir

        self.lambda_max = 1.0
        self.lambda_decay = 0.999
        self.lambda_update_every = 10
        self.v_ema_alpha = 0.1

        self.learn_step = 0
        self.lambdas = [0.0 for _ in range(len(self.agents))]
        self.violation_ema = [0.0 for _ in range(len(self.agents))]

        self.lambdas = [0.0 for _ in range(self.n_agents)]
        self.eta_lambda = 5e-4
        self.epsilon = 0
        self.tau = 0.2

        for agent_idx in range(self.n_agents):
            self.agents.append(SCOPE_Agent(actor_dims[agent_idx], critic_dims,
                                           n_actions, n_agents, agent_idx, alpha=alpha, beta=beta, fc1=fc1, fc2=fc2, gamma=gamma,
                                           chkpt_dir=chkpt_dir))

    def save_checkpoint(self, path=None):
        if not path:
            print('... saving checkpoint ...')
            path = self.chkpt_dir
        for agent in self.agents:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            agent.save_models(path)

    def save_checkpoint_foreach(self, num):
        print('... saving agent', num, 'checkpoint ...')
        os.makedirs(os.path.dirname(self.agents[num].actor.chkpt_file), exist_ok=True)
        self.agents[num].save_models()

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent in self.agents:
            agent.load_models()

    def load_checkpoint_foreach(self, num):
        print('... loading agent', num, 'checkpoint ...')
        self.agents[num].load_models()

    def agents_choose_action(self, raw_obs, time_step, test):  # timestep for exploration
        raw_actions = []
        for agent_idx, agent in enumerate(self.agents):
            action = agent.scope_choose_action(raw_obs[agent_idx], time_step, self.env, test)
            raw_actions.append(action.detach().cpu().numpy())

        positions = [agent.pos for agent in self.env.hunters]
        velocities = [agent.vel for agent in self.env.hunters]
        u_raw_list = [action for action in raw_actions]

        obs_positions = [obs.pos for obs in self.env.obstacles]
        obs_velocities = [obs.vel for obs in self.env.obstacles]
        obs_radii = [obs.radius for obs in self.env.obstacles]

        target_positions = [t.pos for t in self.env.targets]
        target_velocities = [t.vel for t in self.env.targets]

        boundary_low = np.array([0, 0])
        boundary_high = np.array([self.env.length, self.env.width])
        violations = compute_cbf_violation_per_agent( positions, velocities, u_raw_list, r_hunter=self.env.d_safe, r_target=self.env.d_safe_e,
            obs_positions=obs_positions, obs_velocities=obs_velocities, obs_radii=obs_radii, target_positions=target_positions, target_velocities=target_velocities,
            boundary_low=boundary_low, boundary_high=boundary_high, lambda1=3.0, lambda2=2.0, buffer=0.03
        )

        _, safe_actions = solve_joint_cbf( positions, velocities, u_raw_list, r_hunter=self.env.d_safe, r_target=self.env.d_safe_e,
            obs_positions=obs_positions, obs_velocities=obs_velocities, obs_radii=obs_radii, target_positions=target_positions, target_velocities=target_velocities,
            boundary_low=boundary_low, boundary_high=boundary_high, u_max=2*self.env.a_max,
        )

        if not check_future_cbf_feasibility( positions, velocities, safe_actions, r_hunter=self.env.d_safe, r_target=self.env.d_safe_e,
                obs_positions=obs_positions, obs_velocities=obs_velocities, obs_radii=obs_radii, target_positions=target_positions, target_velocities=target_velocities,
                boundary_low=boundary_low, boundary_high=boundary_high, u_max=self.env.a_max, dt=self.env.v_max/self.env.a_max, delta=0
            ):

            _, safe_actions = solve_joint_cbf( positions, velocities, safe_actions, r_hunter=self.env.d_safe, r_target=self.env.d_safe_e,
                obs_positions=obs_positions, obs_velocities=obs_velocities, obs_radii=obs_radii, target_positions=target_positions, target_velocities=target_velocities,
                boundary_low=boundary_low, boundary_high=boundary_high, u_max=self.env.a_max, buffer=0.03, delta=0.06
            )

        modified = violations.astype(np.float32)

        return safe_actions, modified

    def learn(self, memory):
        if not memory.ready():
            return

        actor_states, states, actions, rewards, \
            actor_new_states, states_, dones, safe_value = memory.sample_buffer()

        device = self.agents[0].actor.device

        # === to tensors ===
        states = T.tensor(states, dtype=T.float).to(device)
        actions = T.tensor(actions, dtype=T.float).to(device)
        rewards = T.tensor(rewards, dtype=T.float).to(device)
        states_ = T.tensor(states_, dtype=T.float).to(device)
        dones = T.tensor(dones).to(device)

        # === build new actions from target actors (for critic target) ===
        all_agents_new_actions = []
        old_agents_actions = []

        for agent_idx, agent in enumerate(self.agents):
            new_states = T.tensor(actor_new_states[agent_idx], dtype=T.float).to(device)
            new_pi = agent.target_actor.forward(new_states)
            all_agents_new_actions.append(new_pi)
            old_agents_actions.append(actions[agent_idx])  # actions is originally a list-like per-agent

        # === concatenate along action-dimension ===
        new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)
        old_actions = T.cat([acts for acts in old_agents_actions], dim=1)

        # === per-agent updates ===
        for agent_idx, agent in enumerate(self.agents):

            with T.no_grad():
                critic_value_ = agent.target_critic.forward(states_, new_actions, agent_idx).flatten()
                target = rewards[:, agent_idx] + (1 - dones[:, 0].int()) * agent.gamma * critic_value_

            critic_value = agent.critic.forward(states, old_actions, agent_idx).flatten()
            critic_loss = F.mse_loss(target, critic_value)
            agent.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            agent.critic.optimizer.step()
            agent.critic.scheduler.step()

            mu_states = T.tensor(actor_states[agent_idx], dtype=T.float).to(device)
            oa = old_actions.clone()
            oa[:, agent_idx * self.n_actions:agent_idx * self.n_actions + self.n_actions] = agent.actor.forward(
                mu_states)

            actor_loss_rl = -T.mean(agent.critic.forward(states, oa, agent_idx).flatten())

            try:
                g_val = -safe_value[:, agent_idx]
                phi = F.softplus((-g_val) / self.tau) * self.tau
                phi = T.clamp(phi, max=1.0)
                mean_violation = phi.mean()
            except Exception:
                mean_violation = T.tensor(0.0, device=device)

            lambda_i = self.lambdas[agent_idx] if agent_idx < len(self.lambdas) else 0.0
            if isinstance(lambda_i, T.Tensor):
                lambda_val = lambda_i.item()
            else:
                lambda_val = float(lambda_i)

            actor_loss = actor_loss_rl + lambda_val * (mean_violation - self.epsilon)

            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            agent.actor.optimizer.step()
            agent.actor.scheduler.step()

            with T.no_grad():
                v_hat = (1.0 - self.v_ema_alpha) * self.violation_ema[agent_idx] + self.v_ema_alpha * float(
                    mean_violation.item())
                self.violation_ema[agent_idx] = v_hat

                if (self.learn_step % self.lambda_update_every) == 0:
                    new_lambda = lambda_val + self.eta_lambda * (v_hat - self.epsilon)
                    new_lambda = max(0.0, min(self.lambda_max, new_lambda))
                    new_lambda *= self.lambda_decay

                    if agent_idx < len(self.lambdas):
                        if isinstance(self.lambdas[agent_idx], T.Tensor):
                            self.lambdas[agent_idx].fill_(new_lambda)
                        else:
                            self.lambdas[agent_idx] = new_lambda
                    else:
                        self.lambdas.append(new_lambda)

        # update target networks as before
        for agent in self.agents:
            agent.update_network_parameters()

