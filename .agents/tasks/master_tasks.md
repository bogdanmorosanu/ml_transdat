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

- **2026-11-20 | `decision-tree-regressor-experiment`**: Implemented `DecisionTreeSpatialRegressor` (Unit 05), generated `ro_grid_16x16.parquet` (586,063 pts), and benchmarked 11 km / 4x4 / 8x8 / 16x16 + impurity pruning (`1e-8`). See [`.agents/tasks/decision-tree-regressor-experiment.md`](.agents/tasks/decision-tree-regressor-experiment.md).
