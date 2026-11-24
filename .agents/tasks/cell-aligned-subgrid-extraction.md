# Task: Cell-Aligned Subgrid Extraction API (`src/data/`)

**Date**: 2026-11-24
**Branch**: `exp/optimal_surface_regressor`
**Status**: Completed
**Scope**: Data Gen | Architecture | Testing | Documentation

## Objective
Add reusable, `.spg` cell-aligned rectangular subgrid extraction to `src/data/` so any experiment can slice arbitrary `(row_start, col_start, n_rows, n_cols)` windows from canonical `GridId` datasets while preserving exact grid step, coordinates, and precomputed shifts, and validate the subgrid proxy workflow (`10×10`, `20×20`, `30×30`, SW Quarter) against Full Romania.

## Key Activities & Implementation
- **Reusable Subgrid API ([`src/data/catalog.py`](../../src/data/catalog.py), [`src/data/__init__.py`](../../src/data/__init__.py))**:
  - Added `CellRegion(row_start, col_start, n_rows, n_cols, name="")` with `bounds_st70()`, `padded(pad_cells=2)` (clamped to valid `.spg` bounds), and `from_center()`.
  - Added `GridDataset.slice_region(region, include_upper_boundary=True)` to extract exact `.spg`-aligned subgrids without re-interpolation.
  - Added `GridManager.get_region(grid_id, region)` with in-memory caching keyed by `(grid_id, region, include_upper_boundary)`.
  - Added `find_worst_case_region(size_cells, min_valid_fraction=0.80, pad_cells=2, priority_cells=None)` to locate high-curvature windows inside Romania.
- **Documentation & Tests ([`docs/architecture.md`](../../docs/architecture.md), [`tests/test_subgrid_extraction.py`](../../tests/test_subgrid_extraction.py))**:
  - Documented the new regional subgrid API in `docs/architecture.md` Section 2.
  - Added unit tests covering `CellRegion` alignment/clamping, `slice_region`, `GridManager.get_region` caching, and `find_worst_case_region`.

## Verification & Results
- Ran `tests/test_subgrid_extraction.py` via `D:\anaconda3\envs\ml_transdat\python.exe`: all 3 tests passed (`All 3 subgrid tests PASSED`).
- Validated subgrid proxy speedups and Top-K recall in `experiments/optimal_surface_regressor/proxy_validation.py`: `10×10` (`(19, 28)`) runs `15.3×` faster (`12.19s` vs `186.78s`), preserves within-family parameter trends, and identifies the Full Romania #1 candidate (`tps_deg2_k25_s1e-5`).
