from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from src.config import ExperimentConfig
from src.env import AIGameEnv
from src.rl_common import GRUPolicy, MLP, ReplayBuffer, soft_update


class CooperativeSolver:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.env = AIGameEnv(cfg.env, mode="cooperative")
        self.device = torch.device("cpu")
        obs_dim = self.env.observation_space.shape[0]
        if cfg.recurrent_policy:
            self.actor = GRUPolicy(obs_dim, 2, cfg.train.hidden_dim).to(self.device)
            self.recurrent = True
        else:
            self.actor = MLP(obs_dim, 2, cfg.train.hidden_dim).to(self.device)
            self.recurrent = False
        self.critic = MLP(obs_dim + 2, 1, cfg.train.hidden_dim).to(self.device)
        self.tar_actor = MLP(obs_dim, 2, cfg.train.hidden_dim).to(self.device)
        self.tar_critic = MLP(obs_dim + 2, 1, cfg.train.hidden_dim).to(self.device)
        if not self.recurrent:
            self.tar_actor.load_state_dict(self.actor.state_dict())
        self.tar_critic.load_state_dict(self.critic.state_dict())
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=cfg.train.actor_lr)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=cfg.train.critic_lr)
        self.replay = ReplayBuffer(cfg.train.replay_size)

    def act(self, s: np.ndarray, noise_scale: float = 0.1) -> np.ndarray:
        x = torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if self.recurrent:
                a, _ = self.actor(x.unsqueeze(1))
            else:
                a = self.actor(x)
        return (a.squeeze(0).cpu().numpy() + noise_scale * np.random.randn(2)).astype(np.float32)

    def train(self) -> List[Dict[str, float]]:
        rows = []
        for ep in range(self.cfg.train.episodes):
            s, _ = self.env.reset()
            done = False
            ret = 0.0
            while not done:
                a = self.act(s)
                s2, r, done, _, _ = self.env.step(a)
                self.replay.add(s, a, r, s2, float(done))
                s = s2
                ret += float(r[0])
                if len(self.replay) > max(self.cfg.train.batch_size, self.cfg.train.warmup_steps):
                    self.update()
            rows.append({"episode": ep, "ret_total": ret})
        return rows

    def update(self):
        b = self.replay.sample(self.cfg.train.batch_size, self.device)
        with torch.no_grad():
            if self.recurrent:
                a2, _ = self.actor(b.s2.unsqueeze(1))
            else:
                a2 = self.tar_actor(b.s2)
            y = b.r[:, 0:1] + (1 - b.d) * self.cfg.env.gamma * self.tar_critic(torch.cat([b.s2, a2], dim=-1))
        q = self.critic(torch.cat([b.s, b.a], dim=-1))
        risk = self.cfg.train.risk_lambda * torch.var(b.r[:, 0:1])
        lc = F.mse_loss(q, y)
        self.opt_c.zero_grad(); lc.backward(); self.opt_c.step()

        if self.recurrent:
            a, _ = self.actor(b.s.unsqueeze(1))
        else:
            a = self.actor(b.s)
        la = -self.critic(torch.cat([b.s, a], dim=-1)).mean() + risk
        self.opt_a.zero_grad(); la.backward(); self.opt_a.step()
        if not self.recurrent:
            soft_update(self.actor, self.tar_actor, self.cfg.train.tau)
        soft_update(self.critic, self.tar_critic, self.cfg.train.tau)
