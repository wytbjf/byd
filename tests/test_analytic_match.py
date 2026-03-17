from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analytic_baseline import compute_analytic_baseline
from src.config import EnvConfig


def test_closed_form_nonnegative() -> None:
    cfg = EnvConfig(uncertainty_mode="fixed", feasible_analytic_regime=True, seed=123)
    res = compute_analytic_baseline(cfg)
    for mode, d in res.efforts.items():
        assert d["E_T"] >= 0
        assert d["E_O"] >= 0
        if mode == "stackelberg":
            assert 0 <= d["theta"] <= 1


def test_returns_order_typical_case() -> None:
    cfg = EnvConfig(uncertainty_mode="fixed", feasible_analytic_regime=True, seed=8)
    res = compute_analytic_baseline(cfg)
    assert res.returns["cooperative"] >= res.returns["nash"]
