# Task: Fourier MLP with 3×3 Cell Neighbor Features & Center Distance (`mlp_neighbor_features`)

**Date**: 2026-11-21
**Branch**: `exp/mlp_neighbor_features`
**Status**: Completed
**Scope**: Model Architecture, Evaluation, Documentation

## Objective
Implement and evaluate `mlp_neighbor_features` to test whether adding 32 corner slopes, 40 slopes+twists, and/or continuous $C^0$ `[0, 1]` cell-midpoint distances improves the champion `mlp_fourier_4x4` model on valid interior $3 \times 3$ `.spg` cells.

## Key Activities & Implementation
- Added `CellNeighborFeatureExtractor.lookup_center_distances` in [`src/geometry/neighbor_features.py`](../../src/geometry/neighbor_features.py).
- Extended `MLPGridShiftModel` in [`src/models/mlp.py`](../../src/models/mlp.py) with `use_neighbor_features`, `neighbor_feature_mode` (`"slopes"` / `"all"`), training-set standardization, `NaN -> 0.0` fallback, and `use_cell_center_distance`.
- Created [`experiments/mlp_neighbor_features/run_unit.py`](../../experiments/mlp_neighbor_features/run_unit.py) filtering datasets to valid interior neighbor cells and [`experiments/mlp_neighbor_features/report.md`](../../experiments/mlp_neighbor_features/report.md).
- Added unit tests in [`tests/test_mlp_neighbor_features.py`](../../tests/test_mlp_neighbor_features.py).
- Synchronized [`experiments/README.md`](../../experiments/README.md), [`docs/research/mlp_neighbor_features.md`](../../docs/research/mlp_neighbor_features.md), and [`docs/research/roadmap.md`](../../docs/research/roadmap.md).

## Verification & Results
- **Unit Tests**: `D:\anaconda3\envs\ml_transdat\python.exe -m unittest tests.test_mlp_neighbor_features -v` → `3 tests OK`.
- **Experiment Execution**: `D:\anaconda3\envs\ml_transdat\python.exe -m experiments.mlp_neighbor_features.run_unit` → completed (`413.43s`), confirming `Dead End` (`13.0 mm` baseline vs. `13.7–15.5 mm` uniform RMSE; `32.5 mm` baseline vs. `42.0–47.3 mm` adaptive RMSE).
