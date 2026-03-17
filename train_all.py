from __future__ import annotations

import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
import yaml

from src.config import ExperimentConfig, merge_dict
from src.solvers.cooperative_solver import CooperativeSolver
from src.solvers.nash_solver import NashSolver
from src.solvers.stackelberg_solver import StackelbergSolver


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_solver(cfg: ExperimentConfig):
    if cfg.mode == "nash":
        return NashSolver(cfg)
    if cfg.mode == "stackelberg":
        return StackelbergSolver(cfg)
    return CooperativeSolver(cfg)


def main():
    ap = argparse.ArgumentParser(
        description="Train one solver from a yaml config."
    )
    ap.add_argument(
        "--config",
        default="configs/fixed.yaml",
        help="Path to yaml config (default: configs/fixed.yaml)",
    )
    args = ap.parse_args()
    if not os.path.exists(args.config):
        raise FileNotFoundError(
            f"Config file not found: {args.config}. "
            "Please pass a valid path, e.g. --config configs/fixed.yaml"
        )
    with open(args.config, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    cfg = merge_dict(ExperimentConfig(), data)
    os.makedirs(cfg.output_dir, exist_ok=True)
    set_seed(cfg.env.seed)
    solver = build_solver(cfg)
    rows = solver.train()
    out = os.path.join(cfg.output_dir, f"train_{cfg.mode}_{cfg.env.uncertainty_mode}_{cfg.env.task_type}.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"config: {args.config}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
