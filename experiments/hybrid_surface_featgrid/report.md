# Experiment Report: `hybrid_surface_featgrid`

## 1. Objective & Hypothesis

This experiment evaluates a **two-stage hybrid architecture** combining:
1. **Stage 1 (Smooth Analytical Surface)**: Pre-fitted on `RO_GRID_4X4` (`36,435` points) to remove bulk macro/meso distortion without memorizing `11 km` $C^1$ boundary kinks.
2. **Stage 2 (`NeuralFeatureGridShiftModel`)**: A learnable 2-resolution 2D feature grid (`53×72` at `11 km` + `105×143` at `5.5 km`, `8` features/node) paired with cyclic intra-cell `(u, v)` encoding and a compact `SiLU` MLP, trained on the Stage-1 residuals.

**Core Hypothesis**: Removing smooth regional distortion in Stage 1 allows a compact Stage-2 neural feature grid to focus strictly on millimeter-scale intra-cell curvature and `11 km` $C^1$ border kinks—provided Stage 2 is trained on a denser grid than Stage 1's `RO_GRID_4X4` anchors.

---

## 2. Experimental Setup & Two-Phase Methodology

### Phase 1: Full Cartesian Subgrid Matrix (`screen_subgrid.py`)
Evaluated on `box10_interior` (`10×10` SPG cells, `row_start=19, col_start=28`) across:
- **3 Stage-1 Surfaces** (fitted on Full `RO_GRID_4X4`): `cubic_k25_s0`, `tps_deg2_k25_s1e-5`, `lpoly_deg2_k16_tric`
- **4 Stage-2 Training Densities**: `RO_GRID_4X4` (`1,681` pts), `RO_GRID_8X8` (`6,561` pts), `RO_TEST_UNIFORM_16X16` (`25,600` pts), `RO_GRID_32X32` (`103,041` pts)
- **2 MLP Widths**: `hidden_dim` $\in \{64, 128\}$
- **2 Loss Functions**: `mse` vs. `huber` (`SmoothL1Loss(beta=0.005)`)
- **Harmonics Check**: `cyclic_cell_harmonics` $\in \{2, 4\}$ at `32x32` for the winning `(hidden_dim, loss_type)`

### Phase 2: Full-Romania Champion Evaluation (`run_unit.py`)
Promoted the Phase-1 Champion to Full Romania, trained on `RO_GRID_32X32` (`2,346,351` points, `batch_size=4096`, `max_epochs=120`) and evaluated on `RO_TEST_UNIFORM_16X16` (`588,032` points) and `RO_TEST_ADAPTIVE_1MM_MID` (`564,325` points), alongside `Stage1_cubic_k25_s0_only` and `FeatGrid_32x32_only` references.

---

## 3. Quantitative Results

### Phase 1: Top Subgrid Configurations (`box10_interior`, sorted by `Adapt 1mm` RMSE)

| Rank | Model | Surface | Density | Width | Loss | `k` | `16x16` Mean (mm) | `16x16` RMSE (mm) | `16x16` `< 1 mm` | `Adapt 1mm` Mean (mm) | `Adapt 1mm` RMSE (mm) | `Adapt 1mm` P95 (mm) | `Adapt 1mm` Max (mm) | `Adapt 1mm` `< 1 mm` |
| :-: | :--- | :--- | :---: | :---: | :---: | :-: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **`cubic_k25_s0_32x32_h128_huber_k2`** | `cubic_k25_s0` | `32x32` | 128 | `huber` | 2 | **0.305** | **0.365** | **99.14%** | **0.351** | **0.412** | **0.766** | **1.901** | **98.87%** |
| 2 | `cubic_k25_s0_32x32_h64_huber_k2` | `cubic_k25_s0` | `32x32` | 64 | `huber` | 2 | 0.315 | 0.376 | 98.95% | 0.357 | 0.420 | 0.780 | 1.889 | 98.60% |
| 3 | `tps_deg2_k25_s1e-5_32x32_h128_huber_k2` | `tps_deg2_k25_s1e-5` | `32x32` | 128 | `huber` | 2 | 0.292 | 0.351 | 99.30% | 0.408 | 0.488 | 0.925 | 1.825 | 96.68% |
| 4 | `tps_deg2_k25_s1e-5_32x32_h64_huber_k2` | `tps_deg2_k25_s1e-5` | `32x32` | 64 | `huber` | 2 | 0.314 | 0.384 | 98.62% | 0.480 | 0.579 | 1.100 | 2.555 | 92.62% |
| 5 | `cubic_k25_s0_32x32_h64_mse_k2` | `cubic_k25_s0` | `32x32` | 64 | `mse` | 2 | 0.390 | 0.465 | 97.30% | 0.524 | 0.619 | 1.164 | 2.623 | 90.79% |
| 6 | `tps_deg2_k25_s1e-5_32x32_h128_huber_k4` | `tps_deg2_k25_s1e-5` | `32x32` | 128 | `huber` | 4 | 0.223 | 0.298 | 98.81% | 0.498 | 0.626 | 1.226 | 3.187 | 88.53% |
| 7 | `cubic_k25_s0_32x32_h128_huber_k4` | `cubic_k25_s0` | `32x32` | 128 | `huber` | 4 | 0.256 | 0.339 | 98.24% | 0.540 | 0.676 | 1.300 | 3.511 | 85.51% |
| 14 | `lpoly_deg2_k16_tric_32x32_h128_huber_k2` | `lpoly_deg2_k16_tric` | `32x32` | 128 | `huber` | 2 | 0.595 | 0.777 | 84.36% | 0.973 | 1.233 | 2.458 | 8.088 | 61.78% |
| — | `Stage1_cubic_k25_s0_only` *(ref)* | `cubic_k25_s0` | — | — | — | — | 4.415 | 5.716 | 12.16% | 4.558 | 5.872 | 11.958 | 27.154 | 11.15% |

*(Full 54-row matrix in [`results/subgrid_matrix.md`](results/subgrid_matrix.md).)*

### Phase 2: Full-Romania Champion Evaluation

| Model | `16x16` Mean (mm) | `16x16` RMSE (mm) | `16x16` P99 (mm) | `16x16` `< 1 mm` | `Adapt 1mm` Mean (mm) | `Adapt 1mm` RMSE (mm) | `Adapt 1mm` P95 (mm) | `Adapt 1mm` Max (m) | `Adapt 1mm` `< 1 mm` | Size (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Stage1_cubic_k25_s0_only` | 2.84 | 5.12 | 12.7 | 26.5% | 6.42 | 22.52 | 16.73 | 0.691 | 17.5% | 2.89 |
| `FeatGrid_32x32_only` | 0.39 | **1.09** | 1.3 | 97.0% | **0.98** | **5.42** | **1.94** | **0.244** | 82.5% | **0.66** |
| **`Hybrid_cubic_k25_s0_plus_featgrid_32x32`** | **0.24** | 2.36 | **0.9** | **99.3%** | 1.50 | 14.84 | 2.37 | 0.651 | **84.6%** | 3.55 |

*(Delta table in [`results/full_ro_delta_vs_stage1.md`](results/full_ro_delta_vs_stage1.md).)*

---

## 4. Visual Diagnostics
- [`plots/ro_test_uniform_16x16_grid_shift_2d_spatial_error.png`](plots/ro_test_uniform_16x16_grid_shift_2d_spatial_error.png)
- [`plots/ro_test_uniform_16x16_grid_shift_2d_hist_ecdf.png`](plots/ro_test_uniform_16x16_grid_shift_2d_hist_ecdf.png)
- [`plots/ro_test_adaptive_1mm_mid_grid_shift_2d_spatial_error.png`](plots/ro_test_adaptive_1mm_mid_grid_shift_2d_spatial_error.png)
- [`plots/ro_test_adaptive_1mm_mid_grid_shift_2d_hist_ecdf.png`](plots/ro_test_adaptive_1mm_mid_grid_shift_2d_hist_ecdf.png)
- [`plots/extreme_scenarios_gradient_and_linear_9cells.png`](plots/extreme_scenarios_gradient_and_linear_9cells.png)
- [`plots/extreme_scenarios_curvature_and_kinks_9cells.png`](plots/extreme_scenarios_curvature_and_kinks_9cells.png)

---

## 5. Takeaways & Verdict

1. **Stage-1 Learnability Ranking Vindicated**: `cubic_k25_s0` (ranked #1 in `optimal_surface_regressor` for lowest adaptive roughness) beat `tps_deg2_k25_s1e-5` and `lpoly_deg2_k16_tric` in Phase 1 (`0.412 mm` vs `0.488 mm` vs `1.233 mm` `Adapt 1mm` RMSE).
2. **Training Density & `Huber` + `h=128` Synergy**: Training Stage 2 on `4x4` fails (`7.6–11.2 mm`) because Stage 1 already interpolates `4x4`. Scaling to `32x32` with `huber` loss (`beta=5 mm`) unlocks `hidden_dim=128`, reaching **`98.87% < 1 mm`** on `box10_interior` `Adapt 1mm`.
3. **Full-Romania Interior vs. Border Trade-Off**: Nationwide, `Hybrid_cubic_k25_s0_plus_featgrid_32x32` achieves **`99.3% < 1 mm`** (`P99 = 0.9 mm`, `0.1–0.6 mm` across all 9-cell extreme scenarios), **but** `cubic_k25_s0`'s `~0.65 m` border extrapolation spikes inflate Full-Romania `RMSE` (`14.84 mm` vs. `5.42 mm` for standalone `FeatGrid_32x32_only`) because Huber loss down-weights large border outliers.

---

## 6. Benchmark Candidates
| Variant | Benchmark Label | Train Grid | Target Suite |
| :--- | :--- | :--- | :--- |
| `Hybrid_cubic_k25_s0_plus_featgrid_32x32` | `Cubic k25 + Neural Feature Grid (32x32, h128, Huber)` | `ro_grid_32x32` | `grid_shift_models` |
