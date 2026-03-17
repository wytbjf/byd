from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import gymnasium as gym
import numpy as np

from src.config import EnvConfig, GameMode


@dataclass
class ParamDist:
    low: float
    high: float

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.low, self.high))


class AIGameEnv(gym.Env):
    metadata = {"render_modes": []}

    PARAMS = [
        "alpha", "beta", "delta", "epsilon", "a1", "a2", "TEU1", "TEU2", "k1", "k2", "omega1", "omega2", "C_AI"
    ]

    def __init__(self, cfg: EnvConfig, mode: GameMode):
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        self.rng = np.random.default_rng(cfg.seed)
        self.t = 0
        self.k = 0.0
        self._task_dists = self._build_param_dists()
        self.phi = self._sample_phi()
        self.hist: list[np.ndarray] = []

        act_dim = 2 if mode != "stackelberg" else 3
        self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(act_dim,), dtype=np.float32)
        obs_dim = 1 + 1 + 2 + len(self.PARAMS)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def _build_param_dists(self) -> Dict[str, Dict[str, ParamDist]]:
        routine = {
            "alpha": ParamDist(0.7, 0.9), "beta": ParamDist(0.4, 0.6), "delta": ParamDist(0.2, 0.3),
            "epsilon": ParamDist(0.15, 0.25), "a1": ParamDist(0.7, 0.8), "a2": ParamDist(0.5, 0.7),
            "TEU1": ParamDist(0.7, 0.85), "TEU2": ParamDist(0.55, 0.75), "k1": ParamDist(0.45, 0.6),
            "k2": ParamDist(0.35, 0.5), "omega1": ParamDist(0.5, 0.62), "omega2": ParamDist(0.3, 0.44),
            "C_AI": ParamDist(0.12, 0.18),
        }
        creative = {
            "alpha": ParamDist(0.55, 0.75), "beta": ParamDist(0.45, 0.7), "delta": ParamDist(0.1, 0.22),
            "epsilon": ParamDist(0.2, 0.32), "a1": ParamDist(0.6, 0.82), "a2": ParamDist(0.55, 0.8),
            "TEU1": ParamDist(0.62, 0.86), "TEU2": ParamDist(0.62, 0.88), "k1": ParamDist(0.42, 0.58),
            "k2": ParamDist(0.4, 0.56), "omega1": ParamDist(0.45, 0.6), "omega2": ParamDist(0.34, 0.5),
            "C_AI": ParamDist(0.14, 0.2),
        }
        return {"routine": routine, "creative": creative}

    def _sample_phi(self) -> Dict[str, float]:
        d = self._task_dists[self.cfg.task_type]
        for _ in range(200):
            phi = {k: dist.sample(self.rng) for k, dist in d.items()}
            if self.cfg.feasible_analytic_regime:
                phi["omega1"] = max(phi["omega1"], 0.51)
                phi["omega2"] = min(phi["omega2"], 0.42)
            if (1 - phi["epsilon"] * phi["a1"] * phi["TEU1"] > 0) and (1 - phi["epsilon"] * phi["a2"] * phi["TEU2"] > 0):
                if not self.cfg.feasible_analytic_regime or (2 * phi["omega1"] > phi["omega2"]):
                    return phi
        raise RuntimeError("Failed to sample feasible parameter set")

    def _obs(self) -> np.ndarray:
        mode_id = {"nash": 0.0, "stackelberg": 1.0, "cooperative": 2.0}[self.mode]
        task_id = 0.0 if self.cfg.task_type == "routine" else 1.0
        if self.cfg.uncertainty_mode == "hidden_parameter":
            phi_vec = np.zeros(len(self.PARAMS), dtype=np.float32)
        else:
            phi_vec = np.array([self.phi[k] for k in self.PARAMS], dtype=np.float32)
        return np.concatenate([np.array([self.k, self.t / self.cfg.horizon, mode_id, task_id], dtype=np.float32), phi_vec])

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self.k = float(self.rng.uniform(self.cfg.k0_low, self.cfg.k0_high))
        if self.cfg.uncertainty_mode in ["episode_random", "hidden_parameter"]:
            self.phi = self._sample_phi()
        self.hist = []
        return self._obs(), {}

    def _decode_action(self, action: np.ndarray) -> Tuple[float, float, float]:
        e_t = float(np.log1p(np.exp(action[0])))
        e_o = float(np.log1p(np.exp(action[1])))
        theta = 0.0
        if self.mode == "stackelberg":
            theta = float(1 / (1 + np.exp(-action[2])))
        return e_t, e_o, theta

    def step(self, action: np.ndarray):
        if self.cfg.uncertainty_mode == "step_random" and self.rng.random() < 0.25:
            nxt = self._sample_phi()
            for k in self.PARAMS:
                self.phi[k] = 0.9 * self.phi[k] + 0.1 * nxt[k]

        e_t, e_o, theta = self._decode_action(action)
        p = self.phi
        eta = 1 + p["epsilon"] * (0.5 * p["a1"] + 0.5 * p["a2"]) * (0.5 * p["TEU1"] + 0.5 * p["TEU2"])
        pi = p["k1"] * e_t + p["k2"] * e_o + eta * self.k - p["C_AI"]
        c_t = 0.5 * 1.1 * (1 - p["epsilon"] * p["a1"] * p["TEU1"]) * e_t ** 2
        c_o = 0.5 * 1.0 * (1 - p["epsilon"] * p["a2"] * p["TEU2"]) * e_o ** 2

        if self.mode == "nash":
            r_t = p["omega1"] * pi - c_t
            r_o = p["omega2"] * pi - c_o
            reward = np.array([r_t, r_o], dtype=np.float32)
        elif self.mode == "stackelberg":
            r_t = p["omega1"] * pi - c_t - theta * c_o
            r_o = p["omega2"] * pi - (1 - theta) * c_o
            reward = np.array([r_t, r_o], dtype=np.float32)
        else:
            r_g = (p["omega1"] + p["omega2"]) * pi - c_t - c_o
            reward = np.array([r_g], dtype=np.float32)

        noise = self.rng.normal(0, 1)
        self.k = self.k + self.cfg.dt * (p["alpha"] * e_t + p["beta"] * e_o - p["delta"] * self.k) + self.cfg.sigma_k * np.sqrt(self.cfg.dt) * noise
        self.t += 1
        done = self.t >= self.cfg.horizon
        obs = self._obs()
        info = {"e_t": e_t, "e_o": e_o, "theta": theta, "k": self.k, "phi": dict(self.phi)}
        return obs, reward, done, False, info
