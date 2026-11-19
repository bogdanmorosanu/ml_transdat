# Task: Spatial Grid Architecture Restructuring & Multi-Resolution Dataset Generation

**Date**: 2026-11-19
**Status**: Completed
**Scope**: Data Engineering | Grid Architecture | Refactoring | Invariant Verification

## Objective
Decouple grid datasets from trajectory benchmarks, transition from the legacy `ro_grid_3x3` nomenclature to clear subdivision/interval naming (`ro_grid_4x4`, `ro_grid_8x8`), extract clean valid SPG anchor nodes, and synchronize all research units and regression tests.

## Key Activities & Implementation

### 1. Dedicated Spatial Grid Infrastructure (`data/grids/`)
- Separated spatial grid datasets from evaluation benchmarks (`data/benchmarks/`), establishing `data/grids/` as the canonical store for spatial grids:
  - **`spg_nodes_valid.parquet`** (2,689 rows): Clean, pre-filtered valid land nodes extracted directly from `rom_grid3d_25.09.spg` (`grids.geodetic_shifts`). Contains precomputed $dN, dE$, vector magnitude, and corresponding bidirectional `pytransdat` coordinates.
  - **`ro_grid_4x4.parquet`** (36,435 rows): Canonical $2.75\,\text{km}$ subgrid (4 subdivisions per $11\,\text{km}$ SPG cell, step offsets $\{0.0, 0.25, 0.50, 0.75\}$), superseding `ro_grid_3x3.parquet`.
  - **`ro_grid_8x8.parquet`** (146,258 rows): Densified $1.375\,\text{km}$ subgrid (8 subdivisions per cell, step offsets $\{0.0, 0.125, \dots, 0.875\}$). Provides dense out-of-training interpolation evaluation points for upcoming tree and neural models.
- Built automated generator script in [`scripts/generate_grids.py`](../../scripts/generate_grids.py) to reproducibly regenerate all subgrid levels.

### 2. Codebase Refactoring & Nomenclature Synchronization
- Updated [`src/core/data_samplers.py`](../../src/core/data_samplers.py):
  - Added `DEFAULT_SUBGRID_OFFSETS_4X4` and `DEFAULT_SUBGRID_OFFSETS_8X8`.
  - Implemented generalized `build_grid_candidates(engine, offsets, z_normal)`.
  - Retained `build_3x3_grid_candidates` as a backward-compatible alias.
- Removed obsolete files `data/benchmarks/ro_grid_3x3.parquet` and `data/benchmarks/ro_grid_3x3.meta.json`.
- Maintained [`data/benchmarks/ro_corridor_snw.parquet`](../../data/benchmarks/ro_corridor_snw.parquet) strictly for directional trajectory and caching benchmarks.

### 3. Research Units & Living Documentation Alignment
- Updated benchmark paths from `data/benchmarks/ro_grid_3x3.parquet` to `data/grids/ro_grid_4x4.parquet` across:
  - Unit 01: [`experiments/01_spg_grid_analysis/run_unit.py`](../../experiments/01_spg_grid_analysis/run_unit.py), report, and README.
  - Unit 02: [`experiments/02_pyproj_baseline/run_unit.py`](../../experiments/02_pyproj_baseline/run_unit.py), report, and README.
  - Unit 03: [`experiments/03_helmert2d_baseline/run_unit.py`](../../experiments/03_helmert2d_baseline/run_unit.py), report, and README.
  - Living research docs: `docs/research/01_spg_grid_analysis.md`, `02_pyproj_baseline.md`, `03_helmert2d_baseline.md`.
- Re-executed all three research units; verified numerical equivalence and reproduced scorecards.

### 4. Verification & Regression Testing
- Updated unit tests in [`tests/test_ro_grid_benchmark.py`](../../tests/test_ro_grid_benchmark.py) and [`tests/test_eval_suite.py`](../../tests/test_eval_suite.py) to validate `ro_grid_4x4`, `ro_grid_8x8`, and `spg_nodes_valid`.
- Ran full test suite via `python -m unittest discover tests`: all 8 tests pass cleanly.
