# Deep RL Differential Game Solver (AI scale-up strategy)

## Install
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install numpy torch gymnasium pyyaml matplotlib pandas
```

## Run training
```bash
python train_all.py --config configs/fixed.yaml
python train_all.py --config configs/stackelberg_fixed.yaml
python train_all.py --config configs/cooperative_fixed.yaml
python train_all.py --config configs/cooperative_hidden_gru.yaml
python train_all.py --config configs/cooperative_step_robust.yaml
```

## Run evaluation and figure generation
```bash
python eval_experiments.py
```

## Outputs
- `results/*.csv`: training curves and summary tables
- `figures/*.png`: learning curve, fixed recovery error, partial-observability boxplot
- `analytic_baseline.py`: fixed-parameter closed-form baseline for recovery target
- `tests/test_analytic_match.py`: sanity tests under fixed regime
