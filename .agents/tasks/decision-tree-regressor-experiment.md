# Task: Decision Tree Spatial Regressor Experiment (Unit 05)

- **Branch**: `exp/decision_tree_regressor`
- **Status**: Completed
- **Date**: 2026-11-20

## Summary
Implemented, tested, and benchmarked `DecisionTreeSpatialRegressor` (`src/models/decision_tree_regressor.py`) for 2D coordinate distortion shifts `(st70_n, st70_e) -> (shift_dn_m, shift_de_m)`.

## Key Deliverables
- `src/models/decision_tree_regressor.py` & `src/models/__init__.py`: Multi-output `DecisionTreeRegressor` wrapper.
- `tests/test_decision_tree_regressor.py`: 4 unit tests (all 17 repo tests passing).
- `experiments/decision_tree_regressor/run_unit.py`: Unit 05 benchmark runner with `Agg` backend, `scorecard.csv`, `report.md`, `unit_brief.json`, and diagnostic plots.
- `data/grids/ro_grid_16x16.parquet` & `ro_grid_16x16.meta.json`: Generated 16x16 subgrid (586,063 valid points, 687.5 m spacing).

## Key Results
- **Base 11 km (2,689 pts)**: `0.0000 m` train RMSE, `0.0844 m` (4x4), `0.0817 m` (8x8), `0.0678 m` (corridor), `0.41 MB`.
- **4x4 (36,435 pts)**: `0.0246 m` (8x8), `0.0161 m` (corridor), `5.56 MB`.
- **8x8 (146,258 pts)**: `0.0080 m` (corridor), `22.32 MB`.
- **8x8 Pruned (`min_impurity_decrease=1e-8`)**: `0.0188 m` (corridor), `2.81 MB` (87.4% compression).
- **16x16 (586,063 pts, standalone)**: `0.0040 m` (4.01 mm corridor RMSE, `89.43 MB`); pruned `1e-8` gave `0.0187 m` at `2.80 MB`.
