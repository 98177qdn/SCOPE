import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims,
                 n_agents, n_actions, name, chkpt_dir, attention_heads=4):
        super(CriticNetwork, self).__init__()
        self.name = name
        self.chkpt_file = chkpt_dir
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.state_dim = int(input_dims / n_agents)
        # === Layers ===
        self.fc1 = nn.Linear(input_dims + n_agents * n_actions, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)
        self.sa_embed = nn.Linear(self.state_dim + n_actions + n_agents, fc1_dims)

        self.attn = nn.MultiheadAttention(embed_dim=fc1_dims, num_heads=attention_heads, batch_first=True)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5000, gamma=0.33)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action, agent_index):

        B = state.size(0)

        # === reshape ===
        states = state.view(B, self.n_agents, self.state_dim)  # [B, n_agents, state_dim]
        actions = action.view(B, self.n_agents, self.n_actions)  # [B, n_agents, action_dim]

        # === agent ID embedding ===
        id_emb = F.one_hot(T.arange(self.n_agents), num_classes=self.n_agents).float().to(
            state.device)  # [n_agents, n_agents]
        id_emb = id_emb.unsqueeze(0).repeat(B, 1, 1)  # [B, n_agents, n_agents]

        # === concatenate state, action, id  ===
        sa = T.cat([states, actions, id_emb], dim=-1)  # [B, n_agents, state_dim + action_dim + n_agents]
        sa_embed = F.relu(self.sa_embed(sa))  # [B, n_agents, embed_dim]

        # === Attention ===
        q_embed = sa_embed[:, agent_index:agent_index + 1, :]  # [B, 1, embed_dim]
        attn_out, _ = self.attn(query=q_embed, key=sa_embed, value=sa_embed)  # [B, 1, embed_dim]
        attn_out = attn_out.squeeze(1)  # [B, embed_dim]

        # === MLP ===
        x_mlp = T.cat([state, action], dim=-1)  # [B, total_dim]
        x_mlp = F.relu(self.fc1(x_mlp))  # [B, fc1_dim]

        # === Feature Fusion ===
        x = F.relu(self.fc2(x_mlp + attn_out))  # [B, fc2_dim]
        q = self.q(x)  # [B, 1]

        return q

    def save_checkpoint(self, path=None):
        if not path:
            path = self.chkpt_file
        path = str(os.path.join(path, self.name))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        T.save(self.state_dict(), path)

    def load_checkpoint(self):
        path = str(os.path.join(self.chkpt_file, self.name))
        self.load_state_dict(T.load(path))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims,
                 n_actions, name, chkpt_dir):
        super(ActorNetwork, self).__init__()
        self.name = name
        self.chkpt_file = str(os.path.join(chkpt_dir, name))

        # === feature extract ===
        self.feature = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU()
        )

        # === Gate ===
        self.gate = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.Sigmoid()
        )
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.8)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = self.feature(state)
        gate = self.gate(state)
        x = x * gate

        x = F.relu(self.fc2(x))
        pi = T.tanh(self.pi(x))
        return pi

    def save_checkpoint(self, path=None):
        if not path:
            path = self.chkpt_file
        path = str(os.path.join(path, self.name))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        T.save(self.state_dict(), path)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))


class SCOPE_Agent:
    def __init__(self, actor_dims, critic_dims, n_actions, n_agents, agent_idx, chkpt_dir,
                    alpha=0.0001, beta=0.002, fc1=128,
                    fc2=128, gamma=0.99, tau=0.01):
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.agent_name = 'agent_%s' % agent_idx
        self.actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions,
                                  chkpt_dir=chkpt_dir, name=self.agent_name+'_actor.pt')
        self.critic = CriticNetwork(beta, critic_dims,
                            fc1, fc2, n_agents, n_actions,
                            chkpt_dir=chkpt_dir, name=self.agent_name+'_critic.pt')
        self.target_actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions,
                                        chkpt_dir=chkpt_dir,
                                        name=self.agent_name+'_target_actor.pt')
        self.target_critic = CriticNetwork(beta, critic_dims,
                                            fc1, fc2, n_agents, n_actions,
                                            chkpt_dir=chkpt_dir,
                                            name=self.agent_name+'_target_critic.pt')

        self.update_network_parameters(tau=1)

    def scope_choose_action(self, observation, time_step, env, test=False):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        actions = self.actor.forward(state)
        actions = actions.squeeze(0)

        if T.isnan(actions).any():
            print("NaN detected in actions!")
            print("Actions:", actions)
            print("Input state:", state)
            raise ValueError("NaN found in action output!")

        max_noise = 0.9
        min_noise = 0.01
        decay_rate = 0.999995
        noise_scale = max(min_noise, max_noise * (decay_rate ** time_step))
        noise = 2 * T.rand(self.n_actions).to(self.actor.device) - 1

        if not test:
            noise = noise_scale * noise
        else:
            noise = 0 * noise

        action = actions + noise

        magnitude = T.norm(action)
        if magnitude > env.a_max:
            action = action / magnitude * env.a_max

        return action

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_actor_params = self.target_actor.named_parameters()
        actor_params = self.actor.named_parameters()

        target_actor_state_dict = dict(target_actor_params)
        actor_state_dict = dict(actor_params)
        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                    (1-tau)*target_actor_state_dict[name].clone()

        self.target_actor.load_state_dict(actor_state_dict)

        target_critic_params = self.target_critic.named_parameters()
        critic_params = self.critic.named_parameters()

        target_critic_state_dict = dict(target_critic_params)
        critic_state_dict = dict(critic_params)
        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                    (1-tau)*target_critic_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

    def save_models(self, path=None):
        self.actor.save_checkpoint(path)
        self.target_actor.save_checkpoint(path)
        self.critic.save_checkpoint(path)
        self.target_critic.save_checkpoint(path)

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()
