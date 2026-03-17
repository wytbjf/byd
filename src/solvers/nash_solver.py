from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from src.config import ExperimentConfig
from src.env import AIGameEnv
from src.rl_common import MLP, ReplayBuffer, soft_update


@dataclass
class TrainStats:
    episode: int
    ret_t: float
    ret_o: float


class NashSolver:
    """MADDPG-style approximation for continuous-action Nash game."""

    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.env = AIGameEnv(cfg.env, mode="nash")
        self.device = torch.device("cpu")
        obs_dim = self.env.observation_space.shape[0]
        self.actor_t = MLP(obs_dim, 1, cfg.train.hidden_dim).to(self.device)
        self.actor_o = MLP(obs_dim, 1, cfg.train.hidden_dim).to(self.device)
        self.critic_t = MLP(obs_dim + 2, 1, cfg.train.hidden_dim).to(self.device)
        self.critic_o = MLP(obs_dim + 2, 1, cfg.train.hidden_dim).to(self.device)
        self.target_actor_t = MLP(obs_dim, 1, cfg.train.hidden_dim).to(self.device)
        self.target_actor_o = MLP(obs_dim, 1, cfg.train.hidden_dim).to(self.device)
        self.target_critic_t = MLP(obs_dim + 2, 1, cfg.train.hidden_dim).to(self.device)
        self.target_critic_o = MLP(obs_dim + 2, 1, cfg.train.hidden_dim).to(self.device)
        self.target_actor_t.load_state_dict(self.actor_t.state_dict())
        self.target_actor_o.load_state_dict(self.actor_o.state_dict())
        self.target_critic_t.load_state_dict(self.critic_t.state_dict())
        self.target_critic_o.load_state_dict(self.critic_o.state_dict())
        self.opt_at = torch.optim.Adam(self.actor_t.parameters(), lr=cfg.train.actor_lr)
        self.opt_ao = torch.optim.Adam(self.actor_o.parameters(), lr=cfg.train.actor_lr)
        self.opt_ct = torch.optim.Adam(self.critic_t.parameters(), lr=cfg.train.critic_lr)
        self.opt_co = torch.optim.Adam(self.critic_o.parameters(), lr=cfg.train.critic_lr)
        self.replay = ReplayBuffer(cfg.train.replay_size)

    def act(self, s: np.ndarray, noise_scale: float = 0.1) -> np.ndarray:
        x = torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            a_t = self.actor_t(x).item()
            a_o = self.actor_o(x).item()
        return np.array([a_t + noise_scale * np.random.randn(), a_o + noise_scale * np.random.randn()], dtype=np.float32)

    def train(self) -> List[Dict[str, float]]:
        rows: List[Dict[str, float]] = []
        for ep in range(self.cfg.train.episodes):
            s, _ = self.env.reset()
            done = False
            rt = ro = 0.0
            while not done:
                a = self.act(s)
                s2, r, done, _, _ = self.env.step(a)
                self.replay.add(s, a, r, s2, float(done))
                s = s2
                rt += float(r[0])
                ro += float(r[1])
                if len(self.replay) > max(self.cfg.train.batch_size, self.cfg.train.warmup_steps):
                    self.update()
            rows.append({"episode": ep, "ret_t": rt, "ret_o": ro, "ret_total": rt + ro})
        return rows

    def update(self):
        b = self.replay.sample(self.cfg.train.batch_size, self.device)
        with torch.no_grad():
            a2_t = self.target_actor_t(b.s2)
            a2_o = self.target_actor_o(b.s2)
            a2 = torch.cat([a2_t, a2_o], dim=-1)
            y_t = b.r[:, 0:1] + (1 - b.d) * self.cfg.env.gamma * self.target_critic_t(torch.cat([b.s2, a2], dim=-1))
            y_o = b.r[:, 1:2] + (1 - b.d) * self.cfg.env.gamma * self.target_critic_o(torch.cat([b.s2, a2], dim=-1))
        q_t = self.critic_t(torch.cat([b.s, b.a], dim=-1))
        q_o = self.critic_o(torch.cat([b.s, b.a], dim=-1))
        loss_ct = F.mse_loss(q_t, y_t)
        loss_co = F.mse_loss(q_o, y_o)
        self.opt_ct.zero_grad(); loss_ct.backward(); self.opt_ct.step()
        self.opt_co.zero_grad(); loss_co.backward(); self.opt_co.step()

        a_t = self.actor_t(b.s)
        a_o_det = self.actor_o(b.s).detach()
        loss_at = -self.critic_t(torch.cat([b.s, torch.cat([a_t, a_o_det], dim=-1)], dim=-1)).mean()
        self.opt_at.zero_grad(); loss_at.backward(); self.opt_at.step()

        a_o = self.actor_o(b.s)
        a_t_det = self.actor_t(b.s).detach()
        loss_ao = -self.critic_o(torch.cat([b.s, torch.cat([a_t_det, a_o], dim=-1)], dim=-1)).mean()
        self.opt_ao.zero_grad(); loss_ao.backward(); self.opt_ao.step()

        soft_update(self.actor_t, self.target_actor_t, self.cfg.train.tau)
        soft_update(self.actor_o, self.target_actor_o, self.cfg.train.tau)
        soft_update(self.critic_t, self.target_critic_t, self.cfg.train.tau)
        soft_update(self.critic_o, self.target_critic_o, self.cfg.train.tau)
