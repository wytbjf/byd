from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from src.config import ExperimentConfig
from src.env import AIGameEnv
from src.rl_common import MLP, ReplayBuffer, soft_update


class StackelbergSolver:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.env = AIGameEnv(cfg.env, mode="stackelberg")
        self.device = torch.device("cpu")
        obs_dim = self.env.observation_space.shape[0]
        self.leader = MLP(obs_dim, 2, cfg.train.hidden_dim).to(self.device)  # E_T, theta logits
        self.follower = MLP(obs_dim + 2, 1, cfg.train.hidden_dim).to(self.device)  # s + leader action
        self.critic_l = MLP(obs_dim + 3, 1, cfg.train.hidden_dim).to(self.device)
        self.critic_f = MLP(obs_dim + 3, 1, cfg.train.hidden_dim).to(self.device)
        self.tar_leader = MLP(obs_dim, 2, cfg.train.hidden_dim).to(self.device)
        self.tar_follower = MLP(obs_dim + 2, 1, cfg.train.hidden_dim).to(self.device)
        self.tar_critic_l = MLP(obs_dim + 3, 1, cfg.train.hidden_dim).to(self.device)
        self.tar_critic_f = MLP(obs_dim + 3, 1, cfg.train.hidden_dim).to(self.device)
        for a, b in [(self.leader, self.tar_leader), (self.follower, self.tar_follower), (self.critic_l, self.tar_critic_l), (self.critic_f, self.tar_critic_f)]:
            b.load_state_dict(a.state_dict())
        self.opt_l = torch.optim.Adam(self.leader.parameters(), lr=cfg.train.actor_lr)
        self.opt_f = torch.optim.Adam(self.follower.parameters(), lr=cfg.train.actor_lr)
        self.opt_cl = torch.optim.Adam(self.critic_l.parameters(), lr=cfg.train.critic_lr)
        self.opt_cf = torch.optim.Adam(self.critic_f.parameters(), lr=cfg.train.critic_lr)
        self.replay = ReplayBuffer(cfg.train.replay_size)

    def _leader_action(self, s: torch.Tensor) -> torch.Tensor:
        out = self.leader(s)
        return out

    def act(self, s: np.ndarray, noise_scale: float = 0.1) -> np.ndarray:
        x = torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            lead = self.leader(x)
            a_f = self.follower(torch.cat([x, lead], dim=-1))
            a = torch.cat([lead[:, :1], a_f, lead[:, 1:2]], dim=-1)
        return (a.squeeze(0).cpu().numpy() + noise_scale * np.random.randn(3)).astype(np.float32)

    def train(self) -> List[Dict[str, float]]:
        rows = []
        for ep in range(self.cfg.train.episodes):
            s, _ = self.env.reset()
            done = False
            rt = ro = 0.0
            while not done:
                a = self.act(s)
                s2, r, done, _, _ = self.env.step(a)
                self.replay.add(s, a, r, s2, float(done))
                s = s2
                rt += float(r[0]); ro += float(r[1])
                if len(self.replay) > max(self.cfg.train.batch_size, self.cfg.train.warmup_steps):
                    self.update(ep)
            rows.append({"episode": ep, "ret_t": rt, "ret_o": ro, "ret_total": rt + ro})
        return rows

    def update(self, ep: int):
        b = self.replay.sample(self.cfg.train.batch_size, self.device)
        with torch.no_grad():
            l2 = self.tar_leader(b.s2)
            f2 = self.tar_follower(torch.cat([b.s2, l2], dim=-1))
            a2 = torch.cat([l2[:, :1], f2, l2[:, 1:2]], dim=-1)
            y_l = b.r[:, 0:1] + (1 - b.d) * self.cfg.env.gamma * self.tar_critic_l(torch.cat([b.s2, a2], dim=-1))
            y_f = b.r[:, 1:2] + (1 - b.d) * self.cfg.env.gamma * self.tar_critic_f(torch.cat([b.s2, a2], dim=-1))
        ql = self.critic_l(torch.cat([b.s, b.a], dim=-1)); qf = self.critic_f(torch.cat([b.s, b.a], dim=-1))
        lcl = F.mse_loss(ql, y_l); lcf = F.mse_loss(qf, y_f)
        self.opt_cl.zero_grad(); lcl.backward(); self.opt_cl.step()
        self.opt_cf.zero_grad(); lcf.backward(); self.opt_cf.step()

        if ep % self.cfg.train.alternating_interval != 0:
            lead_det = self.leader(b.s).detach()
            f = self.follower(torch.cat([b.s, lead_det], dim=-1))
            a = torch.cat([lead_det[:, :1], f, lead_det[:, 1:2]], dim=-1)
            lf = -self.critic_f(torch.cat([b.s, a], dim=-1)).mean()
            self.opt_f.zero_grad(); lf.backward(); self.opt_f.step()
        else:
            lead = self.leader(b.s)
            f_det = self.follower(torch.cat([b.s, lead.detach()], dim=-1)).detach()
            a = torch.cat([lead[:, :1], f_det, lead[:, 1:2]], dim=-1)
            ll = -self.critic_l(torch.cat([b.s, a], dim=-1)).mean()
            self.opt_l.zero_grad(); ll.backward(); self.opt_l.step()

        soft_update(self.leader, self.tar_leader, self.cfg.train.tau)
        soft_update(self.follower, self.tar_follower, self.cfg.train.tau)
        soft_update(self.critic_l, self.tar_critic_l, self.cfg.train.tau)
        soft_update(self.critic_f, self.tar_critic_f, self.cfg.train.tau)
