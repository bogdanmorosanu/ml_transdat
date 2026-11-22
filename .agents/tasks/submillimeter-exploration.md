# Task: Multi-Track Sub-Millimeter Exploration & Benchmark Sync

**Date**: 2026-11-22
**Branch**: `exp/submillimeter_exploration`
**Status**: Completed
**Scope**: Data Gen | Model Architecture | Training | Evaluation | Documentation

## Objective
Create a persistent exploratory strategy roadmap (`docs/exploration/`), implement and screen 7 sub-millimeter improvement tracks under an apples-to-apples benchmark (`experiments/submillimeter_exploration/`), and promote validated champions into `benchmarks/grid_shift_models/`.

## Key Activities & Implementation
- Created [`docs/exploration/README.md`](../../docs/exploration/README.md) and [`docs/exploration/submillimeter_strategy_report.md`](../../docs/exploration/submillimeter_strategy_report.md) synthesizing bottlenecks and 7 prioritized improvement tracks.
- Extended [`src/models/torch_mlp.py`](../../src/models/torch_mlp.py) with multi-harmonic cyclic cell encoding (`cyclic_cell_harmonics`), multi-band Fourier scales (`fourier_scales`), full-grid training (`val_fraction=0.0`), `huber` / `charbonnier` losses, `L-BFGS` polishing (`lbfgs_steps`), and `SIREN` (`SineActivation`).
- Implemented [`src/models/composite_shift.py`](../../src/models/composite_shift.py) (`CellBilinearShiftModel`, `CompositeResidualShiftModel`), [`src/models/neural_feature_grid.py`](../../src/models/neural_feature_grid.py) (`NeuralFeatureGridShiftModel`), [`src/models/hist_gbr.py`](../../src/models/hist_gbr.py) (`HistGBRGridShiftModel`), and Matérn `3/2`/`5/2` local k-d tree RBFs in [`src/models/rbf_surface.py`](../../src/models/rbf_surface.py).
- Added `RO_TRAIN_ADAPTIVE_HYBRID` (off-dyadic `ADAPTIVE_1CM` probes + `RO_GRID_4X4`) and `RO_TRAIN_BORDER_ENRICHED_4X4` to [`src/data/samplers.py`](../../src/data/samplers.py) and [`src/data/catalog.py`](../../src/data/catalog.py).
- Implemented and ran [`experiments/submillimeter_exploration/run_unit.py`](../../experiments/submillimeter_exploration/run_unit.py) (19 model variants + Track 7 bidirectional `ETRS89 <-> Stereo70` check), synced [`docs/research/submillimeter_exploration.md`](../../docs/research/submillimeter_exploration.md), and promoted `T7_bilinear_plus_featgrid_8x8` and `T4_data_adaptive_hybrid` into [`benchmarks/grid_shift_models/`](../../benchmarks/grid_shift_models/summary_scorecard.md).

## Verification & Results
- **Unit Tests**: `D:\anaconda3\envs\ml_transdat\python.exe -m unittest tests/test_submillimeter_exploration_models.py` — 6/6 tests passed (including zero overlap between `RO_TRAIN_ADAPTIVE_HYBRID` and `RO_TEST_ADAPTIVE_3MM_MID`).
- **Lint**: `D:\anaconda3\envs\ml_transdat\python.exe -m ruff check src/models/ src/data/ experiments/submillimeter_exploration/ tests/test_submillimeter_exploration_models.py` — All checks passed.
- **Experiment & Benchmark Results**:
  - `T7_bilinear_plus_featgrid_8x8` (`0.66 MB`): `0.77 mm` uniform mean (`1.71 mm` RMSE, `76.5% < 1 mm`), `0.71 mm` mean / `0.84 mm` RMSE bidirectional on `ro_corridor_snw` (`78.3% < 1 mm`, `max 4.44 mm`).
  - `T4_data_adaptive_hybrid` (`0 MB` grid): `2.48 mm` uniform mean (`3.22 mm` RMSE), `1.89 mm` adaptive mean (`2.59 mm` RMSE), `3.08 mm` corridor RMSE.
