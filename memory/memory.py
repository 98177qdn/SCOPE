import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, max_size, critic_dims, actor_dims,
                 n_actions, n_agents, batch_size):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.n_agents = n_agents
        self.actor_dims = actor_dims
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.state_memory = np.zeros((self.mem_size, critic_dims))
        self.new_state_memory = np.zeros((self.mem_size, critic_dims))
        self.reward_memory = np.zeros((self.mem_size, n_agents))
        self.safe_memory = np.zeros((self.mem_size, n_agents))
        self.terminal_memory = np.zeros((self.mem_size, n_agents), dtype=bool)
        self.episode_memory = np.zeros(max_size)

        self.actor_state_memory = np.zeros((self.mem_size, n_agents, max(actor_dims)))
        self.actor_new_state_memory = np.zeros((self.mem_size, n_agents, max(actor_dims)))
        self.actor_action_memory = np.zeros((self.mem_size, n_agents, n_actions))

    def store_transition(self, raw_obs, state, action, reward, raw_obs_, state_, done, episode, safe_values):
        index = self.mem_cntr % self.mem_size

        for agent_idx in range(self.n_agents):
            # 只存储每个agent实际维度的数据，避免零填充
            self.actor_state_memory[index, agent_idx, :len(raw_obs[agent_idx])] = raw_obs[agent_idx]
            self.actor_new_state_memory[index, agent_idx, :len(raw_obs_[agent_idx])] = raw_obs_[agent_idx]
            self.actor_action_memory[index, agent_idx] = action[agent_idx]

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.safe_memory[index] = safe_values
        self.terminal_memory[index] = done
        self.mem_cntr += 1
        self.episode_memory[index] = episode

    def get_past_trajectory(self, index, traj_len, safe=True):
        """
        返回某条经验所在 episode 的前 traj_len 条轨迹（包含 index 本身），
        不足部分自动补零。safe=True 时，若 index 非法则返回全 0 + 全 False mask。
        """
        if index >= self.mem_cntr or index < 0:
            if not safe:
                raise IndexError("Index exceeds number of stored transitions.")
            # 返回空轨迹
            return self._empty_trajectory(traj_len)

        episode_id = self.episode_memory[index]
        episode_indices = np.where(self.episode_memory[:self.mem_cntr] == episode_id)[0]
        pos_in_episode = np.where(episode_indices == index)[0][0]

        # 找到该条经验及其之前的轨迹索引（包含 index 本身）
        selected_indices = episode_indices[max(0, pos_in_episode - traj_len + 1):pos_in_episode + 1]
        valid_len = len(selected_indices)

        # 准备容器
        actor_states = [
            np.zeros((traj_len, self.actor_dims[agent_idx]), dtype=np.float32)
            for agent_idx in range(self.n_agents)
        ]
        actor_new_states = [
            np.zeros((traj_len, self.actor_dims[agent_idx]), dtype=np.float32)
            for agent_idx in range(self.n_agents)
        ]
        actions = [
            np.zeros((traj_len, self.n_actions), dtype=np.float32)
            for _ in range(self.n_agents)
        ]
        states = np.zeros((traj_len, self.state_memory.shape[1]), dtype=np.float32)
        new_states = np.zeros_like(states)
        rewards = np.zeros((traj_len, self.n_agents), dtype=np.float32)
        terminals = np.zeros((traj_len, self.n_agents), dtype=bool)
        mask = np.zeros(traj_len, dtype=bool)

        # 从末尾向前补齐（靠右对齐）
        if valid_len > 0:
            start = traj_len - valid_len
            for agent_idx in range(self.n_agents):
                actor_states[agent_idx][start:] = self.actor_state_memory[selected_indices, agent_idx,
                                                  :self.actor_dims[agent_idx]]
                actor_new_states[agent_idx][start:] = self.actor_new_state_memory[selected_indices, agent_idx,
                                                      :self.actor_dims[agent_idx]]
                actions[agent_idx][start:] = self.actor_action_memory[selected_indices, agent_idx]
            states[start:] = self.state_memory[selected_indices]
            new_states[start:] = self.new_state_memory[selected_indices]
            rewards[start:] = self.reward_memory[selected_indices]
            terminals[start:] = self.terminal_memory[selected_indices]
            mask[start:] = True

        return {
            'actor_states': actor_states,
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'actor_new_states': actor_new_states,
            'new_states': new_states,
            'terminals': terminals,
            'mask': mask,
            'indices': selected_indices.tolist()
        }

    def _empty_trajectory(self, traj_len):
        """返回一个全 0 + 全 False 的轨迹结构。"""
        actor_states = [
            np.zeros((traj_len, self.actor_dims[agent_idx]), dtype=np.float32)
            for agent_idx in range(self.n_agents)
        ]
        actor_new_states = [
            np.zeros((traj_len, self.actor_dims[agent_idx]), dtype=np.float32)
            for agent_idx in range(self.n_agents)
        ]
        actions = [
            np.zeros((traj_len, self.n_actions), dtype=np.float32)
            for _ in range(self.n_agents)
        ]
        states = np.zeros((traj_len, self.state_memory.shape[1]), dtype=np.float32)
        new_states = np.zeros_like(states)
        rewards = np.zeros((traj_len, self.n_agents), dtype=np.float32)
        terminals = np.zeros((traj_len, self.n_agents), dtype=bool)
        mask = np.zeros(traj_len, dtype=bool)

        return {
            'actor_states': actor_states,
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'actor_new_states': actor_new_states,
            'new_states': new_states,
            'terminals': terminals,
            'mask': mask,
            'indices': []
        }

    def sample_buffer(self):
        max_mem = min(self.mem_cntr, self.mem_size)

        if max_mem < self.batch_size:
            raise ValueError(f"Not enough samples in buffer: have {max_mem}, need {self.batch_size}")

        # 内存友好型采样：直接用 randint，无放回采样时用 torch.randperm 再截取
        if self.batch_size <= max_mem // 2:
            # 无放回采样，效率比 np.random.choice 好
            batch = torch.randperm(max_mem, device='cpu')[:self.batch_size].numpy()
        else:
            # 当 batch_size 接近 max_mem 时允许有放回采样，避免内存爆炸
            batch = np.random.randint(0, max_mem, size=self.batch_size)

        states = self.state_memory[batch]
        rewards = self.reward_memory[batch]
        safe_values = self.safe_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        actor_states = [self.actor_state_memory[batch, agent_idx, :self.actor_dims[agent_idx]]
                        for agent_idx in range(self.n_agents)]
        actor_new_states = [self.actor_new_state_memory[batch, agent_idx, :self.actor_dims[agent_idx]]
                            for agent_idx in range(self.n_agents)]
        actions = [self.actor_action_memory[batch, agent_idx]
                   for agent_idx in range(self.n_agents)]

        return actor_states, states, actions, rewards, actor_new_states, states_, terminal, safe_values

    def ready(self):
        return self.mem_cntr >= self.batch_size
