# Master Tasks Log

Chronological execution log of engineering, data, and documentation tasks in `ml_transdat`. Read this file first to see what changed and in what order.

| Date | Branch | Task | Brief Summary |
| :--- | :--- | :--- | :--- |
| 2026-11-11 | `main` | [`project-setup`](project-setup.md) | Initialized `ml_transdat` repository, `pytransdat` ground-truth baseline, and core living docs. |
| 2026-11-13 | `main` | [`canonical-benchmarks-and-research-units`](canonical-benchmarks-and-research-units.md) | Built `src/core/` evaluation/viz pipeline, canonical benchmarks, and initial `spg_grid_analysis` & `pyproj_baseline` units. |
| 2026-11-14 | `main` | [`helmert2d-baseline-and-experiments-workflow`](helmert2d-baseline-and-experiments-workflow.md) | Implemented `src/models/helmert2d.py`, `helmert2d_baseline` unit, and `experiments/GEMINI.md` workflow rules. |
| 2026-11-14 | `main` | [`geodetic-ml-literature-survey`](geodetic-ml-literature-survey.md) | Completed geodetic ML literature survey, verified reference catalog, and search log under `literature/`. |
| 2026-11-19 | `main` | [`grid-restructuring-and-multiresolution-generation`](grid-restructuring-and-multiresolution-generation.md) | Decoupled `data/grids/` (`spg_nodes_valid`, `ro_grid_4x4`, `ro_grid_8x8`) from `data/benchmarks/` and updated tests/units. |
| 2026-11-19 | `main` | [`project-setup-modernization`](project-setup-modernization.md) | Added progressive-discovery personas, slug-based research/tasks/experiments, `master_tasks.md`, `narrative.md`, and modernized skills. |
| 2026-11-19 | `exp/knn_spatial_regressor` | [`knn-spatial-regressor-experiment`](knn-spatial-regressor-experiment.md) | Implemented and benchmarked KNN Spatial Regressor on 2D SPG shifts across multi-resolution grids (achieving 3.5 cm RMSE with K=4 IDW). |
| 2026-11-19 | `exp/knn_spatial_regressor` | [`bicubic-continuity-analysis-and-standards`](bicubic-continuity-analysis-and-standards.md) | Verified RomGEO piecewise bicubic C0/C1 slope kink (0.36 mm fidelity), surveyed international standards (NTv2, NADCON5), and updated docs/grid.md. |
| 2026-11-20 | `exp/decision_tree_regressor` | [`decision-tree-regressor-experiment`](decision-tree-regressor-experiment.md) | Implemented `DecisionTreeSpatialRegressor` (Unit 05), generated `ro_grid_16x16.parquet` (586,063 pts), and benchmarked 11 km / 4x4 / 8x8 / 16x16 + impurity pruning (`1e-8`). |
| 2026-11-20 | `exp/adaptive-quadtree-grid` | [`adaptive-quadtree-grid-experiment`](adaptive-quadtree-grid-experiment.md) | Generated pure `pytransdat` 2D adaptive Quadtree grids across 4 tiers (`2cm`, `1cm`, `3mm`, `1mm`), reorganized `data/grids/`, evaluated multi-tier corridor benchmark ($1.4\,\text{mm}$ at 3 mm, $0.5\,\text{mm}$ at 1 mm), and updated research roadmap/narrative. |
| 2026-11-20 | `infra/unified-architecture` | [`unified-architecture-refactoring`](unified-architecture-refactoring.md) | Refactored `ml_transdat` architecture: unified models/data/eval/viz, dynamic grid generation with caching, >3,300x Quadtree speedup, cross-model benchmarks suite, and Ruff clean. |
| 2026-11-20 | `infra/continuous-spatial-benchmarks` | [`continuous-spatial-benchmarks`](continuous-spatial-benchmarks.md) | Implemented dynamic continuous test benchmarks: area-weighted uniform stratified (4x4, 8x8, 16x16) and complexity-weighted adaptive quadtree midpoints (3mm, 1mm). |
| 2026-11-20 | `exp/mlp-spatial-regressor` | [`mlp-spatial-regressor-experiment`](mlp-spatial-regressor-experiment.md) | Implemented `MLPGridShiftModel` (Unit 06), solved spectral bias via Random Fourier Features, and benchmarked 11 km / 4x4 / 8x8 models on dual continuous test suites (reaching 1.3 cm nationwide RMSE). |
