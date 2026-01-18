import os
import numpy as np
from collections import deque

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_

from trainer.scope.dynamics_models.DLinear import DLinear


# ===================== Normalizer =====================
class RunningStandardizer:
    def __init__(self, dim, eps=1e-5, momentum=0.01, device="cpu"):
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.device = device
        self.mean = torch.zeros(dim, device=device)
        self.var = torch.ones(dim, device=device)
        self.count = eps

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        if x.numel() == 0:
            return
        x = x.reshape(-1, self.dim)
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * (batch_count / tot_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta.pow(2) * (self.count * batch_count / tot_count)
        new_var = M2 / tot_count

        self.mean = (1 - self.momentum) * self.mean + self.momentum * new_mean
        self.var = (1 - self.momentum) * self.var + self.momentum * new_var
        self.count = tot_count

    def norm(self, x: torch.Tensor):
        return (x - self.mean) / torch.sqrt(self.var + self.eps)

    def denorm(self, x: torch.Tensor):
        return x * torch.sqrt(self.var + self.eps) + self.mean


# ===================== History Buffer =====================
class AgentHistory:
    def __init__(self, seq_len, obs_dim):
        self.seq_len = seq_len
        self.obs_dim = obs_dim
        self.history = deque(maxlen=seq_len)

    def add_obs(self, obs):
        self.history.append(obs)

    def is_ready(self):
        return len(self.history) == self.seq_len

    def get_input(self):
        return np.array(self.history, dtype=np.float32)  # [seq_len, obs_dim]


# ===================== SPDM =====================
class MultiAgentDynamicModel:
    def __init__(
        self,
        n_agents: int,
        seq_len: int,
        obs_dim: int,
        pred_len: int,
        chkpt_dir: str = "tmp/predictor",
        name: str = "DLinear.pt",
        lr: float = 5e-4,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        grad_clip: float = 1.0,
        use_individual: bool = False,
        use_norm: bool = True
    ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.n_agents = n_agents
        self.seq_len = seq_len
        self.obs_dim = obs_dim
        self.pred_len = pred_len

        self.model = DLinear(
            seq_len=seq_len,
            pred_len=pred_len,
            individual=use_individual,
            enc_in=obs_dim
        ).to(self.device)

        self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000, eta_min=lr*0.1)

        self.batch_size = batch_size
        self.grad_clip = grad_clip

        self.histories = [AgentHistory(seq_len, obs_dim) for _ in range(n_agents)]
        self.best_loss = float("inf")

        self.chkpt_dir = chkpt_dir
        self.name = name
        os.makedirs(self.chkpt_dir, exist_ok=True)

        self.use_norm = use_norm
        if self.use_norm:
            self.normers = [RunningStandardizer(obs_dim, device=self.device) for _ in range(n_agents)]
        else:
            self.normers = None

        # ----------- model save/load -----------
    def save_model(self, path=None):
        if not path:
            path = self.chkpt_dir
        path = str(os.path.join(path, self.name))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load_model(self):
        path = str(os.path.join(self.chkpt_dir, self.name))
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    # ----------- predict -----------
    @torch.no_grad()
    def update_and_predict(self, obs_list):
        self.model.eval()
        results = {}

        for agent_id, obs in enumerate(obs_list):
            self.histories[agent_id].add_obs(obs)

            if self.histories[agent_id].is_ready():
                inp = self.histories[agent_id].get_input()  # [seq_len, obs_dim]
                inp_t = torch.from_numpy(inp).unsqueeze(0).to(self.device)  # [1, seq_len, obs_dim]

                if self.use_norm:
                    self.normers[agent_id].update(inp_t)
                    inp_t = self.normers[agent_id].norm(inp_t)

                pred = self.model(inp_t)  # [1, pred_len, obs_dim]
                if self.use_norm:
                    pred = self.normers[agent_id].denorm(pred)

                results[agent_id] = pred.squeeze(0).cpu().numpy()
            else:
                results[agent_id] = np.zeros((self.pred_len, self.obs_dim), dtype=np.float32)

        return results

    # ----------- sample trajectory -----------
    def _sample_boundaries_in_episode(self, memory):
        if memory.mem_cntr <= (self.seq_len + self.pred_len):
            return None

        trial = 0
        while trial < 20:
            g = np.random.randint(0, min(memory.mem_cntr, memory.mem_size))
            ep_id = memory.episode_memory[g]
            ep_indices = np.where(memory.episode_memory[:memory.mem_cntr] == ep_id)[0]
            if len(ep_indices) < (self.seq_len + self.pred_len):
                trial += 1
                continue

            low = self.seq_len - 1
            high = len(ep_indices) - 1 - self.pred_len
            if high < low:
                trial += 1
                continue

            pos = np.random.randint(low, high + 1)
            idx = ep_indices[pos]
            return idx
        return None

    # ----------- training -----------
    def train_step(self, memory, updates: int = 1):
        self.model.train()
        total_loss = 0.0
        total_count = 0

        for _ in range(updates):
            idx_list = []
            for _b in range(self.batch_size):
                idx = self._sample_boundaries_in_episode(memory)
                if idx is not None:
                    idx_list.append(idx)

            if not idx_list:
                return 0.0

            hist_batch = []
            fut_batch = []

            for idx in idx_list:
                # history
                h_start = idx - self.seq_len + 1
                h_end = idx + 1

                # future
                f_start = idx + 1
                f_end = idx + 1 + self.pred_len

                for ag in range(self.n_agents):
                    # history
                    hist = memory.actor_state_memory[h_start:h_end, ag, :self.obs_dim]
                    # future
                    fut = memory.actor_state_memory[f_start:f_end, ag, :self.obs_dim]

                    if hist.shape[0] != self.seq_len or fut.shape[0] != self.pred_len:
                        continue

                    hist_batch.append(hist)
                    fut_batch.append(fut)

            if not hist_batch:
                continue

            hist_batch = torch.tensor(np.stack(hist_batch, axis=0), dtype=torch.float32, device=self.device)
            fut_batch = torch.tensor(np.stack(fut_batch, axis=0), dtype=torch.float32, device=self.device)
            if self.use_norm:
                with torch.no_grad():
                    for bi in range(0, hist_batch.shape[0], self.n_agents):
                        for ag in range(self.n_agents):
                            self.normers[ag].update(hist_batch[bi + ag])

                for bi in range(0, hist_batch.shape[0], self.n_agents):
                    for ag in range(self.n_agents):
                        h = hist_batch[bi + ag]
                        hist_batch[bi + ag] = self.normers[ag].norm(h)

            pred = self.model(hist_batch)
            loss = F.mse_loss(pred, fut_batch)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.sched.step()

            total_loss += loss.item()
            total_count += 1

        mean_loss = total_loss / max(total_count, 1)

        if mean_loss < self.best_loss:
            self.best_loss = mean_loss
            self.save_model()

        return mean_loss

