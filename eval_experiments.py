from __future__ import annotations

import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from analytic_baseline import compute_analytic_baseline
from src.config import ExperimentConfig, merge_dict


def load_cfg(path: str) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as f:
        return merge_dict(ExperimentConfig(), yaml.safe_load(f))


def summarize_train(csv_path: str) -> Dict[str, float]:
    df = pd.read_csv(csv_path)
    ret_col = "ret_total"
    vals = df[ret_col].values
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "worst10": float(np.quantile(vals, 0.1)),
    }


def plot_learning_curves(files: List[str], out: str):
    plt.figure(figsize=(7, 4))
    for f in files:
        df = pd.read_csv(f)
        y = df["ret_total"] if "ret_total" in df.columns else df[[c for c in df.columns if c.startswith("ret")]].sum(axis=1)
        plt.plot(df["episode"], y.rolling(10, min_periods=1).mean(), label=os.path.basename(f).replace("train_", "").replace(".csv", ""))
    plt.legend(); plt.xlabel("episode"); plt.ylabel("smoothed return"); plt.tight_layout(); plt.savefig(out, dpi=160)


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    cfg = load_cfg("configs/fixed.yaml")
    ana = compute_analytic_baseline(cfg.env)
    ana.trajectories.to_csv("results/analytic_trajectories.csv", index=False)
    pd.DataFrame([{"mode": k, "return": v} for k, v in ana.returns.items()]).to_csv("results/analytic_returns.csv", index=False)

    train_files = [
        "results/train_nash_fixed_routine.csv",
        "results/train_stackelberg_fixed_routine.csv",
        "results/train_cooperative_fixed_routine.csv",
        "results/train_cooperative_hidden_parameter_routine.csv",
        "results/train_cooperative_step_random_creative.csv",
    ]
    existing = [f for f in train_files if os.path.exists(f)]
    if existing:
        plot_learning_curves(existing[:3], "figures/learning_curves.png")

    rows = []
    for f in existing:
        s = summarize_train(f)
        s["file"] = os.path.basename(f)
        rows.append(s)
    pd.DataFrame(rows).to_csv("results/robustness_summary.csv", index=False)

    if os.path.exists("results/train_cooperative_fixed_routine.csv") and os.path.exists("results/train_cooperative_hidden_parameter_routine.csv"):
        a = pd.read_csv("results/train_cooperative_fixed_routine.csv")["ret_total"]
        b = pd.read_csv("results/train_cooperative_hidden_parameter_routine.csv")["ret_total"]
        plt.figure(figsize=(5, 4))
        plt.boxplot([a, b], labels=["MLP-fixed", "GRU-hidden"])
        plt.ylabel("Return")
        plt.tight_layout(); plt.savefig("figures/partial_observability_boxplot.png", dpi=160)

    ar = pd.read_csv("results/analytic_returns.csv")
    if len(existing) >= 3:
        vals = []
        for mode in ["nash", "stackelberg", "cooperative"]:
            f = f"results/train_{mode}_fixed_routine.csv"
            if os.path.exists(f):
                vals.append({"mode": mode, "rl_mean": pd.read_csv(f)["ret_total"].tail(20).mean(), "analytic": float(ar[ar.mode == mode]["return"].iloc[0])})
        out = pd.DataFrame(vals)
        out["abs_error"] = (out["rl_mean"] - out["analytic"]).abs()
        out.to_csv("results/fixed_recovery.csv", index=False)

        plt.figure(figsize=(6, 4))
        plt.bar(out["mode"], out["abs_error"])
        plt.ylabel("|RL-analytic|")
        plt.tight_layout(); plt.savefig("figures/fixed_recovery_error.png", dpi=160)


if __name__ == "__main__":
    main()
