from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional

UncertaintyMode = Literal["fixed", "episode_random", "step_random", "hidden_parameter"]
GameMode = Literal["nash", "stackelberg", "cooperative"]
TaskType = Literal["routine", "creative"]


@dataclass
class EnvConfig:
    horizon: int = 40
    dt: float = 0.1
    gamma: float = 0.98
    sigma_k: float = 0.02
    k0_low: float = 0.5
    k0_high: float = 1.5
    uncertainty_mode: UncertaintyMode = "fixed"
    task_type: TaskType = "routine"
    feasible_analytic_regime: bool = True
    seed: int = 42
    history_len: int = 4


@dataclass
class TrainConfig:
    episodes: int = 120
    batch_size: int = 64
    warmup_steps: int = 200
    replay_size: int = 50_000
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    tau: float = 0.01
    hidden_dim: int = 64
    alternating_interval: int = 5
    risk_lambda: float = 0.0
    robust_randomization: bool = False


@dataclass
class ExperimentConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    mode: GameMode = "nash"
    recurrent_policy: bool = False
    output_dir: str = "results"


def merge_dict(dc, updates: Dict):
    for k, v in updates.items():
        if hasattr(dc, k):
            cur = getattr(dc, k)
            if hasattr(cur, "__dataclass_fields__") and isinstance(v, dict):
                merge_dict(cur, v)
            else:
                setattr(dc, k, v)
    return dc
