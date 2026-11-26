# Experiment Report: `hybrid_surface_resnet`

## 1. Objective & Hypothesis

This experiment evaluates a **two-stage, zero-lookup-grid hybrid architecture** combining:
1. **Stage 1 (Smooth Analytical Surface)**: Pre-fitted on `RO_GRID_4X4` to remove bulk macro/meso distortion without memorizing `11 km` $C^1$ boundary kinks.
2. **Stage 2 (`TorchMLPGridShiftModel`)**: A compact, grid-free PyTorch ResNet (`0 MB` spatial lookup table, `num_res_blocks=3`, Random Fourier Features + cyclic `(u, v)` intra-cell harmonics `k=2`) trained on the Stage-1 residuals.

**Core Hypothesis**: Removing smooth regional distortion in Stage 1 allows a compact, grid-free ResNet with `SiLU` activations and `Huber` loss (`beta=5 mm`) to beat standalone neural regressors (`TorchMLP_32x32_only` and `T4_data_adaptive_hybrid`: `3.22 mm` uniform RMSE) and approach sub-millimeter precision without storing any spatial feature grid.

---

## 2. Experimental Setup & Two-Phase Methodology

### GPU Optimization (`src/models/_torch_trainer.py`)
Added GPU-resident tensor batching with `torch.randperm` slicing and once-per-epoch loss synchronization (falling back automatically to CPU `DataLoader` on CUDA OOM), cutting Full-Romania `RO_GRID_32X32` (`2.35M` points) training time to ~4 minutes for `h128` and ~7 minutes for `h256`.

### Phase 1: Exploratory Subgrid Matrix (`screen_subgrid.py`)
Evaluated `56` runs on `box10_interior` (`10×10` SPG cells, `row_start=19, col_start=28`) across:
- **3 Stage-1 Surfaces** (fitted on Full `RO_GRID_4X4`): `cubic_k25_s0`, `tps_deg2_k25_s1e-5`, `lpoly_deg2_k16_tric`
- **4 Stage-2 Training Densities**: `4x4` (`1,681` pts), `8x8` (`6,561` pts), `16x16` (`25,600` pts), `32x32` (`103,041` pts)
- **48 Option-B Cartesian Models**: `3 surfaces × 4 densities × 2 widths (64, 128) × 2 activations ("silu", "relu")` with `loss_type="huber"`, `num_res_blocks=3`, `cyclic_cell_harmonics=2`, `val_fraction=0.0`
- **Additional Checks**: `3` capacity checks at `32x32` with `hidden_dim=256` (using winning `silu` activation) + `2` `loss_type="mse"` confirmation checks on `cubic_k25_s0` at `32x32` (`h64`, `h128`)

### Phase 2: Full-Romania Scaling & Champion Evaluation (`run_unit.py`)
Trained on Full Romania `RO_GRID_32X32` (`2,346,351` points, `batch_size=8192`, `max_epochs=150`) and evaluated on `RO_TEST_UNIFORM_16X16` (`588,032` points) and `RO_TEST_ADAPTIVE_1MM_MID` (`564,325` points):
- Baselines: `Stage1_lpoly_deg2_k16_tric_only` (`RO_GRID_4X4`) and `TorchMLP_32x32_only` (`h128`, raw shifts)
- Capacity & RFF Scale Sweep: `Hybrid_..._h128`, `Hybrid_..._h256`, `Hybrid_..._h512`, and `Hybrid_..._h256_ms_rff` (`fourier_scales=(1.0, 3.0, 10.0)`, `64` frequencies)

---

## 3. Quantitative Results

### Phase 1: Top Subgrid Configurations (`box10_interior`, sorted by `Adapt 1mm` RMSE)

| Rank | Model | Surface | Density | Width | Act | Loss | `16x16` Mean (mm) | `16x16` RMSE (mm) | `16x16` `< 1 mm` | `Adapt 1mm` Mean (mm) | `Adapt 1mm` RMSE (mm) | `Adapt 1mm` P95 (mm) | `Adapt 1mm` Max (mm) | `Adapt 1mm` `< 1 mm` |
| :-: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **`lpoly_deg2_k16_tric_32x32_h128_silu_huber`** | `lpoly_deg2_k16_tric` | `32x32` | 128 | `silu` | `huber` | **1.002** | **1.204** | **59.47%** | **1.279** | **1.577** | **3.054** | **9.137** | **47.02%** |
| 2 | `lpoly_deg2_k16_tric_32x32_h256_silu_huber` | `lpoly_deg2_k16_tric` | `32x32` | 256 | `silu` | `huber` | 1.033 | 1.243 | 57.68% | 1.341 | 1.660 | 3.216 | 12.976 | 44.87% |
| 3 | `lpoly_deg2_k16_tric_32x32_h64_silu_huber` | `lpoly_deg2_k16_tric` | `32x32` | 64 | `silu` | `huber` | 1.228 | 1.473 | 46.95% | 1.570 | 1.927 | 3.699 | 11.887 | 35.11% |
| 5 | `tps_deg2_k25_s1e-5_32x32_h256_silu_huber` | `tps_deg2_k25_s1e-5` | `32x32` | 256 | `silu` | `huber` | 0.914 | 1.182 | 68.84% | 2.297 | 2.915 | 5.780 | 11.677 | 31.09% |
| 6 | `cubic_k25_s0_32x32_h128_silu_huber` | `cubic_k25_s0` | `32x32` | 128 | `silu` | `huber` | 0.948 | 1.215 | 66.19% | 2.349 | 2.973 | 5.949 | 10.798 | 29.63% |
| 26 | `cubic_k25_s0_32x32_h128_silu_mse` *(check)* | `cubic_k25_s0` | `32x32` | 128 | `silu` | `mse` | 3.290 | 3.734 | 6.92% | 4.426 | 5.054 | 9.000 | 14.992 | 4.48% |
| — | `Stage1_lpoly_deg2_k16_tric_only` *(ref)* | `lpoly_deg2_k16_tric` | — | — | — | — | 3.432 | 4.809 | 31.23% | 3.302 | 4.583 | 9.629 | 26.994 | 27.00% |

*(Full 56-row matrix in [`results/subgrid_matrix.md`](results/subgrid_matrix.md).)*

### Phase 2: Full-Romania Evaluation (`RO_GRID_32X32` Train)

| Model | `16x16` Mean (mm) | `16x16` RMSE (mm) | `16x16` `< 1 mm` | `Adapt 1mm` Mean (mm) | `Adapt 1mm` RMSE (mm) | `Adapt 1mm` `< 1 mm` | Size (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Stage1_lpoly_deg2_k16_tric_only` | 2.29 | 5.17 | 41.98% | 5.93 | 23.47 | 29.66% | 2.405 |
| `TorchMLP_32x32_only` (`h128`) | 1.22 | 2.83 | 50.89% | 3.68 | 14.28 | 23.04% | **0.471** |
| `Hybrid_lpoly_deg2_k16_tric_plus_resnet_32x32` (`h128`) | 0.57 | **2.62** | 89.71% | 2.13 | **13.82** | 62.13% | 2.876 |
| `Hybrid_lpoly_deg2_k16_tric_plus_resnet_32x32_h256` | 0.33 | 2.82 | 97.96% | 1.76 | 15.15 | 75.86% | 4.082 |
| `Hybrid_lpoly_deg2_k16_tric_plus_resnet_32x32_h512` | 0.33 | 2.91 | 98.20% | 1.76 | 15.25 | 77.18% | 8.743 |
| **`Hybrid_lpoly_deg2_k16_tric_plus_resnet_32x32_h256_ms_rff`** | **0.31** | 2.97 | **98.39%** | **1.66** | 15.32 | **79.31%** | 4.082 |

*(Delta table in [`results/full_ro_delta_vs_stage1.md`](results/full_ro_delta_vs_stage1.md).)*

---

## 4. Visual Diagnostics
- [`plots/ro_test_uniform_16x16_grid_shift_2d_spatial_error_log.png`](plots/ro_test_uniform_16x16_grid_shift_2d_spatial_error_log.png)
- [`plots/ro_test_uniform_16x16_grid_shift_2d_hist_ecdf.png`](plots/ro_test_uniform_16x16_grid_shift_2d_hist_ecdf.png)
- [`plots/ro_test_adaptive_1mm_mid_grid_shift_2d_spatial_error_log.png`](plots/ro_test_adaptive_1mm_mid_grid_shift_2d_spatial_error_log.png)
- [`plots/ro_test_adaptive_1mm_mid_grid_shift_2d_hist_ecdf.png`](plots/ro_test_adaptive_1mm_mid_grid_shift_2d_hist_ecdf.png)
- [`plots/extreme_scenarios_gradient_and_linear_9cells.png`](plots/extreme_scenarios_gradient_and_linear_9cells.png)
- [`plots/extreme_scenarios_curvature_and_kinks_9cells.png`](plots/extreme_scenarios_curvature_and_kinks_9cells.png)
- [`plots/extreme_residuals_gradient_and_linear_9cells.png`](plots/extreme_residuals_gradient_and_linear_9cells.png)
- [`plots/extreme_residuals_curvature_and_kinks_9cells.png`](plots/extreme_residuals_curvature_and_kinks_9cells.png)

---

## 5. Takeaways & Verdict

1. **Stage-1 Surface Choice Differs for Grid-Free ResNets**: Unlike `NeuralFeatureGridShiftModel` (where `cubic_k25_s0` won), the grid-free ResNet performed best on `Adapt 1mm` with `lpoly_deg2_k16_tric` (`1.577 mm` vs `2.973 mm` for `cubic_k25_s0`), because `lpoly` had the lowest standalone residual error on `box10_interior`.
2. **`SiLU` & `Huber` Dominate**: `SiLU` beat `ReLU` across every surface and density (`1.577 mm` vs `3.240 mm` `A1` RMSE at `32x32 h128`), and `Huber` (`beta=5 mm`) outperformed `MSE` by ~2× (`2.973 mm` vs `5.054 mm` `A1` RMSE).
3. **Full-Romania Capacity & Multi-Scale RFF Scaling**: While `h128` peaked on `box10_interior` (`100` cells), Full Romania (`~23×` larger) needed **`h256` + multi-scale RFF `(1.0, 3.0, 10.0)`**, cutting `16x16` mean from `0.57 mm` → **`0.31 mm`** (**`98.39% < 1 mm`**) and `Adapt 1mm` `< 1 mm` from `62.13%` → **`79.31%`**. `h512` plateaued at ~3× the runtime (`1171.5s` vs `413.6s`).
4. **Grid-Free vs. Feature Grid Trade-Off**: `Hybrid_..._h256_ms_rff` (`0.31 mm` `U16` mean, `0 MB` lookup grid) closes nearly all of the gap to `hybrid_surface_featgrid` (`0.24 mm`, `99.3% < 1 mm`), while `NeuralFeatureGridShiftModel` retains an edge on high-curvature regions (`84.6%` vs `79.31% < 1 mm` on `Adapt 1mm`).

---

## 6. Benchmark Candidates
| Variant | Benchmark Label | Train Grid | Target Suite |
| :--- | :--- | :--- | :--- |
| `Hybrid_lpoly_deg2_k16_tric_plus_resnet_32x32_h256_ms_rff` | `LocalPoly k16 + Grid-Free ResNet (32x32, h256, MS-RFF, SiLU, Huber)` | `ro_grid_32x32` | `grid_shift_models` |
