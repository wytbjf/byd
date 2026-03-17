from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from src.config import EnvConfig
from src.env import AIGameEnv


@dataclass
class AnalyticResult:
    efforts: Dict[str, Dict[str, float]]
    trajectories: pd.DataFrame
    returns: Dict[str, float]


def closed_form_efforts(phi: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    q_t = 1.1 * (1 - phi["epsilon"] * phi["a1"] * phi["TEU1"])
    q_o = 1.0 * (1 - phi["epsilon"] * phi["a2"] * phi["TEU2"])
    nash_et = max(phi["omega1"] * phi["k1"] / q_t, 0.0)
    nash_eo = max(phi["omega2"] * phi["k2"] / q_o, 0.0)
    theta = np.clip((2 * phi["omega1"] - phi["omega2"]) / (2 * phi["omega1"] + 1e-6), 0, 1)
    stack_et = max(phi["omega1"] * phi["k1"] / q_t, 0.0)
    stack_eo = max((phi["omega2"] + theta * phi["omega1"]) * phi["k2"] / q_o, 0.0)
    coop_et = max((phi["omega1"] + phi["omega2"]) * phi["k1"] / q_t, 0.0)
    coop_eo = max((phi["omega1"] + phi["omega2"]) * phi["k2"] / q_o, 0.0)
    return {
        "nash": {"E_T": nash_et, "E_O": nash_eo},
        "stackelberg": {"E_T": stack_et, "E_O": stack_eo, "theta": float(theta)},
        "cooperative": {"E_T": coop_et, "E_O": coop_eo},
    }


def rollout_k(cfg: EnvConfig, phi: Dict[str, float], e_t: float, e_o: float) -> np.ndarray:
    k = (cfg.k0_high + cfg.k0_low) / 2
    ks = [k]
    for _ in range(cfg.horizon):
        k = k + cfg.dt * (phi["alpha"] * e_t + phi["beta"] * e_o - phi["delta"] * k)
        ks.append(k)
    return np.array(ks)


def compute_analytic_baseline(cfg: EnvConfig) -> AnalyticResult:
    env = AIGameEnv(cfg, mode="nash")
    phi = env._sample_phi()
    efforts = closed_form_efforts(phi)
    rows: List[Dict[str, float]] = []
    returns: Dict[str, float] = {}
    for mode in ["nash", "stackelberg", "cooperative"]:
        et = efforts[mode]["E_T"]
        eo = efforts[mode]["E_O"]
        theta = efforts[mode].get("theta", 0.0)
        ks = rollout_k(cfg, phi, et, eo)
        gamma = 1.0
        ret = 0.0
        for t, k in enumerate(ks[:-1]):
            eta = 1 + phi["epsilon"] * (0.5 * phi["a1"] + 0.5 * phi["a2"]) * (0.5 * phi["TEU1"] + 0.5 * phi["TEU2"])
            pi = phi["k1"] * et + phi["k2"] * eo + eta * k - phi["C_AI"]
            c_t = 0.5 * 1.1 * (1 - phi["epsilon"] * phi["a1"] * phi["TEU1"]) * et**2
            c_o = 0.5 * 1.0 * (1 - phi["epsilon"] * phi["a2"] * phi["TEU2"]) * eo**2
            if mode == "nash":
                r = phi["omega1"] * pi - c_t + phi["omega2"] * pi - c_o
            elif mode == "stackelberg":
                r = phi["omega1"] * pi - c_t - theta * c_o + phi["omega2"] * pi - (1 - theta) * c_o
            else:
                r = (phi["omega1"] + phi["omega2"]) * pi - c_t - c_o
            ret += gamma * r
            gamma *= cfg.gamma
            rows.append({"mode": mode, "t": t, "K": k, "E_T": et, "E_O": eo, "theta": theta})
        returns[mode] = ret
    return AnalyticResult(efforts=efforts, trajectories=pd.DataFrame(rows), returns=returns)


if __name__ == "__main__":
    cfg = EnvConfig(uncertainty_mode="fixed", task_type="routine")
    res = compute_analytic_baseline(cfg)
    print(res.trajectories.head())
    print(res.returns)
