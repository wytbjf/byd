# Deep RL Differential Game Solver (AI scale-up strategy)

## 1) 环境准备
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## 2) 项目中“真正会用到”的文件
- 入口脚本：
  - `train_all.py`（统一训练入口）
  - `eval_experiments.py`（统一评估与画图）
  - `analytic_baseline.py`（解析基线）
- 核心代码：
  - `src/env.py`, `src/config.py`, `src/rl_common.py`
  - `src/solvers/nash_solver.py`
  - `src/solvers/stackelberg_solver.py`
  - `src/solvers/cooperative_solver.py`
- 配置：`configs/*.yaml`
- 测试：`tests/test_analytic_match.py`
- 文档：`ASSUMPTIONS.md`, `docs/method.md`

> `results/` 与 `figures/` 是运行后自动生成目录，仓库只保留空目录占位。

## 3) 训练
```bash
# 不传参数时，默认使用 configs/fixed.yaml
python train_all.py

# 指定其它配置
python train_all.py --config configs/fixed.yaml
python train_all.py --config configs/stackelberg_fixed.yaml
python train_all.py --config configs/cooperative_fixed.yaml
python train_all.py --config configs/cooperative_hidden_gru.yaml
python train_all.py --config configs/cooperative_step_robust.yaml
```

## 4) 评估与作图
```bash
python eval_experiments.py
```
会生成：
- `results/*.csv`
- `figures/*.png`

## 5) 单元测试
```bash
pytest -q tests/test_analytic_match.py
```

## 6) 快速检查（建议）
```bash
python -m py_compile $(find . -name '*.py')
```
