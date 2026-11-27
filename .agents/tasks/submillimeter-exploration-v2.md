# Task: Sub-Millimeter Exploration v2 (12-Track Model Landscape & Screening)

**Date**: 2026-11-27
**Branch**: `exp/submillimeter_exploration_v2`
**Status**: Completed
**Scope**: Literature | Model Architecture | Evaluation | Documentation

## Objective
Conduct an exhaustive pre-finalization model exploration across all plausible remaining neural INR, structured hybrid, non-neural ML, and geostatistical/signal-processing families on `box10_interior` (`10×10` interior SPG cells), profiling accuracy, `11 km` cell-wall kink stress, inference latency, and projected Full-Romania memory.

## Key Activities & Implementation
- Updated [`literature/README.md`](../../literature/README.md), [`literature/references.md`](../../literature/references.md) (`R4.1`–`R4.12`), and [`literature/search_log.md`](../../literature/search_log.md) (axes `H`, `I`, `J`) with 2021–2025 INR and geodetic ML references.
- Authored [`docs/exploration/submillimeter_v2_model_landscape.md`](../../docs/exploration/submillimeter_v2_model_landscape.md) auditing 15 tested paradigms (`[x]`) and defining 12 untested tracks (`[ ]`).
- Implemented 13 reusable `BaseGridShiftModel` classes across 4 new `src/models/` modules (zero new packages):
  - [`src/models/neural_inrs.py`](../../src/models/neural_inrs.py): `KANGridShiftModel` (`RBF`/`Fourier`), `WaveletINRGridShiftModel` (`WIRE`/`FINER`), `MFNGridShiftModel` (`Gabor`/`Fourier`), `SpatialMoEGridShiftModel` (`MoE`/`FBPINN`).
  - [`src/models/structured_neural_grids.py`](../../src/models/structured_neural_grids.py): `NeuralBicubicHeadShiftModel` (`grid`/`mlp`), `KPlanesGridShiftModel`.
  - [`src/models/non_neural_ml.py`](../../src/models/non_neural_ml.py): `RFFELMGridShiftModel`, `LSSVMGridShiftModel`, `XGBoostGridShiftModel`, `LinearLeafTreeShiftModel`.
  - [`src/models/geostat_and_wavelet.py`](../../src/models/geostat_and_wavelet.py): `LocalKrigingShiftModel` (with `predict_with_std`), `PartitionOfUnityRBFShiftModel`, `WaveletGridShiftModel`.
- Exported all new classes in [`src/models/__init__.py`](../../src/models/__init__.py) and added unit tests in [`tests/test_submillimeter_v2_models.py`](../../tests/test_submillimeter_v2_models.py).
- Implemented and executed [`experiments/submillimeter_exploration_v2/run_unit.py`](../../experiments/submillimeter_exploration_v2/run_unit.py) (25 models on `box10_interior` in `905.1s`), generating scorecard, Pareto plots, `report.md`, and Schema v2 `unit_brief.json`.
- Synchronized [`docs/architecture.md`](../../docs/architecture.md), [`experiments/README.md`](../../experiments/README.md), [`docs/research/submillimeter_exploration_v2.md`](../../docs/research/submillimeter_exploration_v2.md), [`docs/research/roadmap.md`](../../docs/research/roadmap.md), and [`docs/research/narrative.md`](../../docs/research/narrative.md).

## Verification & Results
- **Unit & Architecture Tests**: `D:\anaconda3\envs\ml_transdat\python.exe -m unittest tests/test_submillimeter_v2_models.py tests/test_architecture_invariants.py` — 7/7 tests passed.
- **Lint**: `D:\anaconda3\envs\ml_transdat\python.exe -m ruff check src/models/ experiments/submillimeter_exploration_v2/ tests/test_submillimeter_v2_models.py` — All checks passed.
- **Screening Highlights (`box10_interior`)**:
  - `V2_6b_catmull_featgrid_3lvl`: `0.111 mm` `U16` mean, `0.154 mm` `A1` RMSE (`100.00% < 1 mm`), `0.665 µs/pt`, `2.78 MB`.
  - `V2_6a_bilinear_featgrid_3lvl`: `0.165 mm` `U16` mean, `0.251 mm` `A1` RMSE (`99.98% < 1 mm`), `0.367 µs/pt`, `1.94 MB`.
  - `V2_2a_lpoly_wire` (`0 MB` lookup): `0.465 mm` `U16` mean, `0.945 mm` `A1` RMSE (`74.69% < 1 mm`).
  - Zero-PyTorch: `A3_Quadtree_1mm` (`0.555 mm` `A1` RMSE, `74.91 MB`), `V2_10a_lpoly_xgboost` (`1.771 mm`), `V2_12b_loess_deg3_16x16` (`0.268 mm` `U16` mean / `2.942 mm` `A1` RMSE).
- **Full-Romania Validation Check (`--full-ro`, `*_BORDER`)**:
  - `V2_6a_bilinear_featgrid_3lvl_FullRO`: `0.068 mm` `U16` mean, `0.098 mm` `A1` RMSE (`100.00% < 1 mm`, `max 0.789 mm`, `0.588 µs/pt`, `1.94 MB`).
  - `V2_6b_catmull_featgrid_3lvl_FullRO`: `0.068 mm` `U16` mean, `0.103 mm` `A1` RMSE (`100.00% < 1 mm`, `0.783 µs/pt`, `2.78 MB`).
  - `V2_4b_lpoly_fbpinn_FullRO` (`0 MB` lookup): `0.423 mm` `U16` mean, `0.893 mm` `A1` RMSE (`80.12% < 1 mm`, `5.26 MB`), outperforming `RBF-KAN` (`2.618 mm`) and `WIRE` (`3.260 mm`) nationwide.
  - Zero-PyTorch: `A3_Quadtree_1mm` (`0.599 mm` `A1` RMSE, `74.91 MB`), `V2_12b_loess_deg3_16x16_FullRO` (`1.402 mm` `A1` RMSE, `38.69 MB`), `V2_10a_lpoly_xgboost_FullRO` (`1.559 mm` `A1` RMSE, `59.40 MB`).
