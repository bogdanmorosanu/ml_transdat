# Task: Deep PyTorch Residual Neural Network (`torch_mlp_regressor`)

**Date**: 2026-11-22
**Branch**: `exp/torch-mlp-regressor`
**Status**: Completed
**Scope**: Model Architecture, Training, Evaluation, Benchmarking, Documentation

## Objective
Upgrade the neural grid-shift baseline from `sklearn`'s `(256, 128)` `MLPRegressor` (`mlp_fourier_4x4`) to a GPU-accelerated PyTorch Residual Neural Network (`TorchMLPGridShiftModel`) and systematically test activations (`ReLU`, `SiLU`, `GELU`), scaled width/depth, smooth cyclic `(u, v)` intra-cell position, and raw/Hermite-weighted `3×3` neighbor slopes & twists.

## Key Activities & Implementation
- Implemented `TorchMLPGridShiftModel` and `TorchMLPNet` (`ResidualBlock` with `LayerNorm`, `AdamW`, `CosineAnnealingLR`) in `src/models/torch_mlp.py` and exported in `src/models/__init__.py`.
- Added smooth cyclic `(u, v)` cell-position encoding (`sin(2πu), cos(2πu), sin(2πv), cos(2πv)`) and 2D cubic Hermite basis weighting (`use_hermite_weights`, `hermite_sum`) in `src/models/torch_mlp.py`.
- Executed 3 experimental stages in `experiments/torch_mlp_regressor/run_unit.py` on `RO_GRID_4X4`, evaluating on `RO_TEST_UNIFORM_4X4` and `RO_TEST_ADAPTIVE_3MM_MID`.
- Synchronized experiment (`experiments/torch_mlp_regressor/report.md`, `unit_brief.json`, `experiments/README.md`, `docs/architecture.md`), research unit (`docs/research/torch_mlp_regressor.md`), and cross-model benchmark (`benchmarks/grid_shift_models/run_benchmark.py`, `benchmarks/README.md`).

## Verification & Results
- **Champion (`torch_res_relu_cellpos_4x4`)**:
  - `ro_test_uniform_4x4`: `6.8 mm` RMSE (`4.2 mm` Mean, `9.7 mm` P95, `95.6% < 1 cm`, `0` lookup table), vs. `13.0 mm` RMSE / `57.9% < 1 cm` for `mlp_fourier_4x4`.
  - `ro_test_adaptive_3mm_mid`: `44.2 mm` RMSE (`15.6 mm` Mean, `5.0 mm` p50, `71.3% < 1 cm`).
  - `benchmarks/grid_shift_models/` (`ro_corridor_snw`): `4.70 mm` RMSE, `3.91 mm` Mean, `96.84% < 1 cm` (vs. `10.35 mm` RMSE / `66.52% < 1 cm` for `Fourier MLP (4x4)`).
- **Neighbor Slopes/Twists**: Unweighted features degraded uniform RMSE (`10.8–12.9 mm`); Hermite-weighted slopes improved typical uniform points (`3.0 mm` Mean, `98.6% < 1 cm`) but worsened adaptive RMSE (`49.5 mm`) and Max outliers (`939.4 mm`), so they were not promoted.
