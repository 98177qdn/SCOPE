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


        # Adapt each agent's safety weight using actual CBF violations.
        self.lambda_init = 0.1
        self.lambda_max = 1.0
        self.lambda_decay = 0.999
        self.lambda_update_every = 10
        self.v_ema_alpha = 0.1
        self.eta_lambda = 5e-4
        self.epsilon = 0.0
        self.tau = 0.2

        self.learn_step = 0
        self.lambdas = [self.lambda_init for _ in range(self.n_agents)]
        self.violation_ema = [0.0 for _ in range(self.n_agents)]
        self.last_safety_stats = [None for _ in range(self.n_agents)]
        self._online_cbf_cache = None

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

        if not test:
            self._cache_online_cbf(raw_obs, raw_actions)
        else:
            self._online_cbf_cache = None

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

    def _cache_online_cbf(self, raw_obs, raw_actions):
        """Freeze the latest pre-step CBF coefficients; keep replay unchanged."""
        # Match the existing compute_cbf_violation_per_agent formulas.
        lambda1, lambda2, buffer = 3.0, 2.0, 0.03
        upper = np.array(
            [self.env.length, self.env.width, self.env.height][:self.n_actions],
            dtype=np.float64,
        )
        cached_agents = []
        for i, hunter in enumerate(self.env.hunters):
            pi = np.asarray(hunter.pos, dtype=np.float64)
            vi = np.asarray(hunter.vel, dtype=np.float64)
            rows, offsets = [], []

            def add_distance_constraint(entity, radius, other_action=None):
                d = pi - np.asarray(entity.pos, dtype=np.float64)
                v = vi - np.asarray(entity.vel, dtype=np.float64)
                dist2 = float(np.dot(d, d))
                if dist2 < 1e-12:
                    return
                h = dist2 - radius ** 2
                dh = 2.0 * np.dot(d, v)
                offset = -2.0 * np.dot(v, v) + 2.0 * lambda1 * dh + lambda2 * h
                if other_action is not None:
                    offset -= 2.0 * np.dot(d, other_action)
                rows.append(2.0 * d)
                offsets.append(float(offset))

            for j, other in enumerate(self.env.hunters):
                if j != i:
                    add_distance_constraint(
                        other, 2.0 * self.env.d_safe + buffer, raw_actions[j]
                    )
            for target in self.env.targets:
                add_distance_constraint(
                    target, self.env.d_safe + self.env.d_safe_e + buffer
                )
            for obstacle in self.env.obstacles:
                add_distance_constraint(
                    obstacle, self.env.d_safe + obstacle.radius + buffer
                )

            for d in range(self.n_actions):
                direction = np.eye(self.n_actions)[d]
                rows.extend([-direction, direction])
                offsets.extend([
                    -2.0 * lambda1 * vi[d] + lambda2 * (upper[d] - pi[d] - buffer),
                    2.0 * lambda1 * vi[d] + lambda2 * (pi[d] - buffer),
                ])
            cached_agents.append({
                "obs": np.array(raw_obs[i], dtype=np.float32, copy=True),
                "A": np.asarray(rows, dtype=np.float32),
                "b": np.asarray(offsets, dtype=np.float32),
            })
        self._online_cbf_cache = {
            "agents": cached_agents,
            "a_max": float(self.env.a_max),
        }

    def _online_cbf_loss(self, agent_idx):
        """Return a differentiable penalty and a detached violation magnitude."""
        if self._online_cbf_cache is None:
            raise RuntimeError("Training actions must be chosen before learn().")
        agent = self.agents[agent_idx]
        cached = self._online_cbf_cache["agents"][agent_idx]
        device = agent.actor.device
        obs = T.as_tensor(cached["obs"], dtype=T.float32, device=device).unsqueeze(0)
        action = agent.actor(obs).squeeze(0)
        norm = T.linalg.vector_norm(action).clamp_min(1e-8)
        scale = (self._online_cbf_cache["a_max"] / norm).clamp(max=1.0)
        action = action * scale
        A = T.as_tensor(cached["A"], dtype=action.dtype, device=device)
        b = T.as_tensor(cached["b"], dtype=action.dtype, device=device)
        ell = (A @ action + b).amin()
        safety_loss = F.softplus(-ell)
        violation = F.relu(-ell.detach())
        return safety_loss, violation


    def learn(self, memory):
        if not memory.ready():
            return

        if self._online_cbf_cache is None:
            raise RuntimeError("Training actions must be chosen before learn().")

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

        # Count successful replay sampling once, not once per agent.
        self.learn_step += 1

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

            actor_loss_rl = -T.mean(
                agent.critic.forward(states, oa, agent_idx).flatten()
            )

            # Online safety uses its own matching observation and CBF snapshot.
            safety_loss, violation = self._online_cbf_loss(agent_idx)
            lambda_val = float(self.lambdas[agent_idx])
            actor_loss = actor_loss_rl + lambda_val * safety_loss

            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            agent.actor.optimizer.step()
            agent.actor.scheduler.step()

            with T.no_grad():
                violation_value = float(violation.item())
                v_hat = (
                    (1.0 - self.v_ema_alpha) * self.violation_ema[agent_idx]
                    + self.v_ema_alpha * violation_value
                )
                self.violation_ema[agent_idx] = v_hat

                if self.learn_step % self.lambda_update_every == 0:
                    new_lambda = (
                        lambda_val + self.eta_lambda * (v_hat - self.epsilon)
                    )
                    new_lambda = max(0.0, min(self.lambda_max, new_lambda))
                    self.lambdas[agent_idx] = new_lambda * self.lambda_decay

                safety_value = float(safety_loss.detach().item())
                self.last_safety_stats[agent_idx] = {
                    "lambda_used": lambda_val,
                    "lambda_next": float(self.lambdas[agent_idx]),
                    "violation": violation_value,
                    "violation_ema": v_hat,
                    "safety_loss": safety_value,
                    "weighted_safety_loss": lambda_val * safety_value,
                }

        # update target networks as before
        for agent in self.agents:
            agent.update_network_parameters()

