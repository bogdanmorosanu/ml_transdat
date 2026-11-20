# Task: Continuous Spatial Benchmarks (Uniform Stratified & Adaptive Midpoints)

**Date**: 2026-11-20  
**Branch**: `infra/continuous-spatial-benchmarks`  
**Status**: Completed  
**Scope**: Data Gen | Evaluation | Testing | Documentation

## Objective
Implement a dual-benchmark suite of dynamic, deterministic continuous spatial test datasets to overcome spatial selection bias (evaluating solely on the 5% corridor) and regular-lattice bias (evaluating solely on grid nodes). Establish both area-weighted uniform stratified jittered generators ($4\times 4$, $8\times 8$, $16\times 16$) and complexity-weighted adaptive quadtree cell midpoint generators ($3\,\text{mm}$, $1\,\text{mm}$ tiers).

## Key Activities & Implementation
- **Valid Cell Detection & Caching (`src/data/samplers.py`)**:
  - Implemented `get_valid_spg_cells()` identifying the 2,297 valid interior SPG $11\,\text{km}$ cells in Romania using `pytransdat.trans_grid.Grid2D.interp` with in-memory caching.
- **Stratified Continuous Jittered Sampling (`src/data/samplers.py`)**:
  - Implemented `generate_stratified_continuous_df(strata_per_axis, seed)` placing exactly one uniformly distributed 2D pseudo-random point per subcell using cell-specific deterministic seeding (`(seed * 1000003 + r * 1009 + c * 9176) & 0xFFFFFFFF`).
  - Implemented `iter_stratified_continuous_chunks(...)` for lazy chunk streaming without storing intermediate DataFrames in RAM.
- **Adaptive Curvature Midpoint Sampling (`src/data/samplers.py`)**:
  - Implemented `generate_adaptive_midpoints_df(tier)` taking the exact midpoint `((n_min + n_max)/2, (e_min + e_max)/2)` of each leaf cell from existing adaptive quadtree files (`adaptive_3mm_dn.parquet`, `adaptive_1mm_dn.parquet`).
  - Tests maximum bilinear deviation without random sampling, landing strictly off grid nodes and concentrated in high-curvature deformation zones (Banat, Maramureș).
  - Implemented `iter_adaptive_midpoint_chunks(...)` for lazy streaming.
- **Catalog Integration (`src/data/catalog.py`)**:
  - Added `GridId.RO_TEST_UNIFORM_4X4`, `GridId.RO_TEST_UNIFORM_8X8`, `GridId.RO_TEST_UNIFORM_16X16`, `GridId.RO_TEST_ADAPTIVE_3MM_MID`, and `GridId.RO_TEST_ADAPTIVE_1MM_MID`.
  - Added `lazy_chunk_generator` hook to `_GridSpec` and `GridManager.iter_chunks`.
- **Package Exports (`src/data/__init__.py`)**:
  - Exported new generators and utility functions.
- **Verification & Unit Tests (`tests/test_continuous_spatial_benchmarks.py`)**:
  - 6 dedicated tests verifying point counts ($36,752$ for $4\times 4$; $176,350$ for $3\,\text{mm}$), zero NaNs, bit-for-bit reproducibility, chunk streaming, and `GridManager` resolution.
  - Full suite passed: 33 tests across repo in $14.1\,\text{s}$ with Ruff clean formatting.
