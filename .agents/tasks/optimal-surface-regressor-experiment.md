# Task: Optimal Surface Regressor 10-Family Learnability Benchmark

**Date**: 2026-11-24
**Branch**: `exp/optimal_surface_regressor`
**Status**: Completed
**Scope**: Model Architecture | Evaluation | Visualization | Documentation

## Objective
Implement and benchmark 10 smooth surface regressor families (`RBFSurfaceRegressor` variants, `BSplineSurfaceRegressor`, `LocalPolynomialSurfaceRegressor`, `ParameterizedBicubicGridShiftModel`, `PolynomialSurfaceRegressor`, `CellBilinearShiftModel`) trained on `RO_GRID_4X4` and ranked by Stage-2 ML Surface Learnability metrics across `RO_GRID_4X4`, `RO_TEST_UNIFORM_16X16`, and `RO_TEST_ADAPTIVE_1MM_MID`.

## Key Activities & Implementation
- **New Reusable Surface Models (`src/models/`)**:
  - Added [`src/models/bspline_surface.py`](../../src/models/bspline_surface.py) (`BSplineSurfaceRegressor` wrapping `SmoothBivariateSpline`).
  - Added [`src/models/local_poly_surface.py`](../../src/models/local_poly_surface.py) (`LocalPolynomialSurfaceRegressor` 2D weighted LOESS with tricube/gaussian kernels).
  - Exported both in [`src/models/__init__.py`](../../src/models/__init__.py), documented in [`docs/architecture.md`](../../docs/architecture.md), and tested in [`tests/test_new_surface_models.py`](../../tests/test_new_surface_models.py).
- **4-Stage Screening & Champion Pipeline (`experiments/optimal_surface_regressor/`)**:
  - [`screen_families.py`](../../experiments/optimal_surface_regressor/screen_families.py): Stage 0 per-family subgrid calibration (`box10_interior`, `box10_dual`, `box20_west`), Stage 1 screening (`176` configs), and Stage 2 Top-3 + direct-full promotion (`29` models on Full Romania).
  - [`run_unit.py`](../../experiments/optimal_surface_regressor/run_unit.py): Stage 3 consolidated unit for the 10 Family Champions with cached `.pkl` checkpoints, `PlotGroup.SURFACE_DIAGNOSTIC`, Pareto learnability ranking plots, and 9-cell extreme scenario transects.
- **Documentation & Research Sync**:
  - Updated [`experiments/optimal_surface_regressor/report.md`](../../experiments/optimal_surface_regressor/report.md), [`experiments/README.md`](../../experiments/README.md), and [`docs/research/optimal_surface_regressor.md`](../../docs/research/optimal_surface_regressor.md).

## Verification & Results
- `tests/test_new_surface_models.py`: Both `test_bspline_surface` and `test_local_poly_surface` passed (`PASSED`).
- `experiments.optimal_surface_regressor.run_unit`: Completed in `248.96s` (`unit_brief.json` Schema v2, `learnability_ranking.csv`, 13 PNG plots).
- **Top 3 Shortlisted Champions**: `cubic_k25_s0` (`16.015 mm` Overall), `tps_deg2_k25_s1e-5` (`16.531 mm`), and `lpoly_deg2_k16_tric` (`16.696 mm`, ~3× faster inference).
