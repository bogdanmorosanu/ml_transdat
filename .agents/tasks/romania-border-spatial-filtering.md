# Task: Romania Border Spatial Filtering

**Date**: 2026-11-26
**Branch**: `exp/hybrid_surface_resnet`
**Status**: Completed
**Scope**: Data Gen | Evaluation | Documentation

## Objective
Enable evaluation grids to be filtered to points inside the Romanian border (plus a 300 m outward buffer) without duplicating densified grid datasets on disk, while keeping full rectangular `.spg` grids intact for training.

## Key Activities & Implementation
- Created `scripts/generate_boundary.py` to reproject the simplified Romania border polygon from `CRS84` to `EPSG:3844` (Stereo 70 meters) via `geopandas`/`pyproj`, apply a `300.0` m outward buffer, and emit static `data/boundaries/romania_border_st70.geojson` and `.meta.json`.
- Implemented `src/data/boundaries.py` (`BoundaryId.ROMANIA`, `SpatialBoundary`, `get_boundary`) using `shapely.prepare` and vectorized `shapely.contains_xy` (`[st70_n, st70_e]` → `(y, x)`).
- Integrated zero-duplication boundary filtering into `src/data/catalog.py` (`GridDataset.filter_boundary`, `GridManager.get(..., boundary=...)`, `_boundary_cache`, and virtual `*_BORDER` `GridId` aliases `RO_GRID_32X32_BORDER`, `RO_TEST_UNIFORM_4X4_BORDER`, `RO_TEST_UNIFORM_8X8_BORDER`, `RO_TEST_UNIFORM_16X16_BORDER`, `RO_TEST_ADAPTIVE_3MM_MID_BORDER`, `RO_TEST_ADAPTIVE_1MM_MID_BORDER` with `iter_chunks` support).
- Added `eval_boundary: BoundaryId | str | None = None` to `TransformationExperimentUnit` in `src/core/base_unit.py`.
- Updated `docs/architecture.md` and added `tests/test_boundaries.py`.

## Verification & Results
- `ruff check` passed with 0 errors across all modified and new files.
- `python -m unittest tests/test_architecture_invariants.py tests/test_boundaries.py` passed (9/9 tests).
- One-off SPG lattice diagnostic confirmed `spg_nodes_valid` retains 1,977/2,689 nodes (73.52%) and `ro_test_uniform_16x16` retains 506,311/588,032 points (86.10%), with at least 3 exterior SPG nodes (33 km) beyond the 300 m buffered border on all four cardinal edges.
