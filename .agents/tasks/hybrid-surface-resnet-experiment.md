# Task: Hybrid Stage-1 Smooth Surface + Stage-2 Grid-Free ResNet Experiment (`hybrid_surface_resnet`)

**Date**: 2026-11-26
**Branch**: `exp/hybrid_surface_resnet`
**Status**: Completed
**Scope**: Model Architecture | Training | Evaluation | Documentation

## Objective
Evaluate whether pairing smooth `RO_GRID_4X4` Stage-1 analytical surfaces (`cubic_k25_s0`, `tps_deg2_k25_s1e-5`, `lpoly_deg2_k16_tric`) with a compact, **grid-free** Stage-2 PyTorch ResNet (`TorchMLPGridShiftModel`, `0 MB` spatial lookup grid) via `CompositeResidualShiftModel` achieves sub-millimeter precision under the same two-phase protocol as `hybrid_surface_featgrid`, while optimizing PyTorch GPU training throughput for all future experiments.

## Key Activities & Implementation
- **GPU-Resident PyTorch Trainer Optimization (`src/models/_torch_trainer.py`)**:
  - Updated `train_torch_regressor` to keep CUDA training tensors resident on GPU with `torch.randperm` batch slicing, `optimizer.zero_grad(set_to_none=True)`, and once-per-epoch CPU loss synchronization (with automatic fallback to CPU `DataLoader` on `torch.cuda.OutOfMemoryError`).
- **Phase 1 Subgrid Screening (`experiments/hybrid_surface_resnet/screen_subgrid.py`)**:
  - Executed `56` runs on `box10_interior` (`3` Stage-1 references + `48` Option-B Cartesian runs across `3 surfaces × 4 densities × 2 widths (64, 128) × 2 activations (silu, relu)` with `num_res_blocks=3`, `loss_type="huber"`, `cyclic_cell_harmonics=2`, `val_fraction=0.0` + `3` `hidden_dim=256` checks at `32x32` + `2` `loss_type="mse"` checks on `cubic_k25_s0`).
  - Saved `results/subgrid_matrix.csv` and `results/subgrid_matrix.md`.
- **Phase 2 Full-Romania Scaling & Champion Evaluation (`experiments/hybrid_surface_resnet/run_unit.py`)**:
  - Implemented `HybridSurfaceResNetUnit(GridShiftExperimentUnit)` (`submillimeter_mode = True`, `render_extreme_scenarios = True`) on `RO_GRID_32X32` (`2,346,351` pts), evaluating `Stage1_lpoly_deg2_k16_tric_only`, `TorchMLP_32x32_only` (`h128`), and `Hybrid_lpoly_deg2_k16_tric_plus_resnet_32x32` across `h128`, `h256`, `h512`, and `h256_ms_rff` (`fourier_scales=(1.0, 3.0, 10.0)`, `64` freqs).
  - Promoted `Hybrid_lpoly_deg2_k16_tric_plus_resnet_32x32_h256_ms_rff` as `benchmark_candidate` in `unit_brief.json`.
- **Tests & Living Documentation**:
  - Registered `experiments/hybrid_surface_resnet/screen_subgrid.py` in `tests/test_architecture_invariants.py`.
  - Synced `experiments/hybrid_surface_resnet/report.md`, `experiments/README.md`, `docs/research/hybrid_surface_resnet.md`, `docs/research/roadmap.md`, and `docs/research/narrative.md`.

## Verification & Results
- **Phase 1 Champion (`box10_interior`)**: `lpoly_deg2_k16_tric_32x32_h128_silu_huber` (`1.002 mm` `U16` mean / `1.204 mm` RMSE / `59.47% < 1 mm`; `1.279 mm` `A1` mean / `1.577 mm` RMSE / `47.02% < 1 mm`).
- **Phase 2 Full-Romania (`RO_GRID_32X32` train)**:
  - `TorchMLP_32x32_only` (`h128`): `1.22 mm` `U16` mean (`50.89% < 1 mm`), `3.68 mm` `A1` mean (`23.04% < 1 mm`).
  - `Hybrid_..._h128`: `0.57 mm` `U16` mean (`89.71% < 1 mm`), `2.13 mm` `A1` mean (`62.13% < 1 mm`).
  - **`Hybrid_..._h256_ms_rff` (Champion)**: **`0.31 mm`** `U16` mean (**`98.39% < 1 mm`**), **`1.66 mm`** `A1` mean (**`79.31% < 1 mm`**), `4.082 MB` total (`0 MB` lookup grid).
- **Automated Tests**: `D:\anaconda3\envs\ml_transdat\python.exe -m unittest tests/test_architecture_invariants.py -v` passed (`Ran 5 tests in 0.449s`, `OK`).
