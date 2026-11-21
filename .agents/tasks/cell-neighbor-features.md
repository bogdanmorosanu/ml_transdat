# Task: 3×3 Grid Cell Neighbor Feature Extraction (`cell_neighbor_features`)

- **Date**: 2026-11-21
- **Status**: Completed

## Summary
- Implemented `src/geometry/neighbor_features.py` (`extract_cell_neighbor_features` and $O(1)$ `CellNeighborFeatureExtractor`).
- Implemented and executed `experiments/cell_neighbor_features/run_unit.py` (`AnalysisExperimentUnit`), extracting 40 continuous one-sided corner slopes and diagonal twists across 2,297 valid interior `11 km` `.spg` cells (`0` NaNs) to `data/features/cell_neighbor_features.parquet`.
- Synchronized `experiments/README.md`, `docs/research/cell_neighbor_features.md`, and `docs/research/roadmap.md` (re-anchoring feature-engineering units under `spg_grid_analysis`).
