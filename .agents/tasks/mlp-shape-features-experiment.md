# Task: Fourier MLP with Cell Shape Features (`mlp_shape_features`)

**Date**: 2026-11-21
**Branch**: `exp/mlp_shape_features`
**Status**: Completed
**Scope**: Model Architecture, Training, Evaluation, Documentation

## Objective
Extend `MLPGridShiftModel` (`src/models/mlp.py`) to support per-axis cell shape features (`cat_dn`, `cat_de`) and S-Shape sample weighting (`s_shape_weight`), and evaluate whether either improves the champion `mlp_fourier_4x4` model on `RO_TEST_UNIFORM_4X4` and `RO_TEST_ADAPTIVE_3MM_MID`.

## Key Activities & Implementation
- Created `exp/mlp_shape_features` branch.
- Updated `MLPGridShiftModel` in [`src/models/mlp.py`](../../src/models/mlp.py) with `use_shape_features`, `shape_tol_mm`, and `s_shape_weight` parameters.
- Created [`experiments/mlp_shape_features/run_unit.py`](../../experiments/mlp_shape_features/run_unit.py) (`MLPShapeFeaturesUnit`) to run a controlled ablation across `3 mm`, `5 mm`, and `10 mm` shape inputs and `2x`/`3x` S-Shape sample weighting.
- Ran Dome vs. S-Shape error stratification on `mlp_fourier_4x4` (`ro_test_adaptive_3mm_mid`).
- Synchronized [`experiments/mlp_shape_features/report.md`](../../experiments/mlp_shape_features/report.md), [`experiments/README.md`](../../experiments/README.md), and [`docs/research/mlp_shape_features.md`](../../docs/research/mlp_shape_features.md).

## Verification & Results
- Executed `D:\anaconda3\envs\ml_transdat\python.exe -m experiments.mlp_shape_features.run_unit` (exit code 0).
- Discrete shape inputs (`3/5/10 mm`) degraded RMSE (`13.0` $\to$ `13.1–14.4 mm` uniform, `32.5` $\to$ `34.1–40.8 mm` adaptive).
- S-Shape sample weighting (`2x`, `3x`) trimmed `Max` outliers (`374.7` $\to$ `179.1–221.3 mm` uniform) without improving adaptive RMSE (`32.5` $\to$ `33.7–34.1 mm`).
