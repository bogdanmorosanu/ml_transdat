# Task: Hybrid Stage-1 Surface + Stage-2 Neural Feature Grid Experiment

**Date**: 2026-11-25
**Branch**: `exp/hybrid_surface_featgrid`
**Status**: Completed
**Scope**: Data Gen | Evaluation | Experiments | Documentation

## Objective
Implement, screen, and validate a two-stage hybrid architecture chaining the Top-3 `RO_GRID_4X4`-fitted smooth Stage-1 surfaces from `optimal_surface_regressor` (`cubic_k25_s0`, `tps_deg2_k25_s1e-5`, `lpoly_deg2_k16_tric`) with Stage-2 `NeuralFeatureGridShiftModel`, and promote the winning configuration to Full-Romania evaluation.

## Key Activities & Implementation
- **32x32 Data Infrastructure (`src/data/`)**:
  - Added `DEFAULT_SUBGRID_OFFSETS_32X32` in `src/data/samplers.py`, exported it in `src/data/__init__.py`, and registered `GridId.RO_GRID_32X32` in `src/data/catalog.py` (pre-cached `2,346,351` points to `.cache/grids/ro_grid_32x32.parquet`).
  - Added `generate_region_densified_dataset` in `src/data/regions.py` for fast on-the-fly regional density slicing.
- **Phase 1 — Subgrid Matrix Screening (`experiments/hybrid_surface_featgrid/screen_subgrid.py`)**:
  - Evaluated the full 48-model Cartesian matrix (`3` surfaces × `4` densities `4x4/8x8/16x16/32x32` × `2` widths `64/128` × `2` losses `mse/huber`) plus `3` Stage-1-only references and `3` `cyclic_cell_harmonics=4` checks at `32x32` on `box10_interior`.
- **Phase 2 — Full-Romania Champion Evaluation (`experiments/hybrid_surface_featgrid/run_unit.py`)**:
  - Evaluated `Hybrid_cubic_k25_s0_plus_featgrid_32x32` (`hidden_dim=128`, `loss_type="huber"`, `cyclic_cell_harmonics=2`, `batch_size=4096`, `max_epochs=120`) trained on `RO_GRID_32X32` alongside `Stage1_cubic_k25_s0_only` and `FeatGrid_32x32_only` on `RO_TEST_UNIFORM_16X16` and `RO_TEST_ADAPTIVE_1MM_MID`.
- **Synchronization**:
  - Added `experiments/hybrid_surface_featgrid/screen_subgrid.py` to `ALLOWED_EXTRA_EXP_FILES` in `tests/test_architecture_invariants.py`.
  - Synchronized `experiments/hybrid_surface_featgrid/report.md`, `experiments/README.md`, `docs/research/hybrid_surface_featgrid.md`, `docs/research/roadmap.md`, and `docs/research/narrative.md`.

## Verification & Results
- **Phase 1 (`box10_interior`)**: `cubic_k25_s0_32x32_h128_huber_k2` ranked #1 (`0.305 mm` `16x16` mean / `99.14% < 1 mm`; `0.351 mm` `Adapt 1mm` mean / `0.412 mm` RMSE / `98.87% < 1 mm`).
- **Phase 2 (Full Romania)**: `Hybrid_cubic_k25_s0_plus_featgrid_32x32` achieved `0.24 mm` mean (`99.3% < 1 mm`, `P99 = 0.9 mm`) on `RO_TEST_UNIFORM_16X16` and `84.6% < 1 mm` on `RO_TEST_ADAPTIVE_1MM_MID` (with `0.651 m` Stage-1 border spikes inflating `Adapt 1mm` RMSE to `14.84 mm` vs. `5.42 mm` for standalone `FeatGrid_32x32_only`).
- **Architectural Invariants**: `D:\anaconda3\envs\ml_transdat\python.exe -m unittest tests/test_architecture_invariants.py -v` — 5/5 tests passed (`OK`).
