# Exploratory Experiment Report: Sub-Millimeter Multi-Track Screening (`submillimeter_exploration`)

## 1. Objective & Protocol
This exploratory research unit screened **all 7 prioritized leads** from [`docs/exploration/submillimeter_strategy_report.md`](../../docs/exploration/submillimeter_strategy_report.md) under a controlled apples-to-apples benchmark protocol against our previous continuous neural champion (`B0_torch_res_relu_cellpos_4x4`).

- **Anchor Training Grid**: `RO_GRID_4X4` (36,435 points, ~2.75 km spacing) for all controlled comparisons, plus `RO_GRID_8X8` (146,258 points), `RO_TRAIN_ADAPTIVE_HYBRID` (disjoint off-dyadic `ADAPTIVE_1CM` probes + `RO_GRID_4X4`), and `RO_TRAIN_BORDER_ENRICHED_4X4` in Track 4/5/7.
- **Canonical Evaluation Benchmarks**:
  - `RO_TEST_UNIFORM_4X4` (36,752 off-lattice stratified points nationwide)
  - `RO_TEST_ADAPTIVE_3MM_MID` (176,350 high-curvature stress-test midpoints)
  - `CORRIDOR_SNW` (36,435 national corridor points)

---

## 2. Full Apples-to-Apples Attribution Scorecard (vs. `B0` Baseline)

| Model | Uniform Mean (mm) | Uniform RMSE (mm) | Δ Uniform RMSE (mm) | Uniform `<1mm` (%) | Adaptive Mean (mm) | Adaptive RMSE (mm) | Δ Adaptive RMSE (mm) | Corridor RMSE (mm) | Corridor `<1mm` (%) | Model+Grid Size (MB) | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`B0_torch_res_relu_cellpos_4x4`** *(Baseline)* | 4.23 | 6.80 | 0.00 | 6.49% | 15.55 | 44.18 | 0.00 | 4.70 | 6.71% | 2.214 | `Baseline` |
| `T1_harmonics4_4x4` | 44.69 | 53.38 | +46.58 | 0.11% | 47.91 | 67.34 | +23.16 | 49.99 | 0.13% | 2.229 | `Dead End` |
| `T1_multiscale_rff_4x4` | 4.97 | 9.29 | +2.49 | 5.53% | 22.62 | 61.78 | +17.60 | 5.29 | 5.79% | 2.214 | `Dead End` |
| `T1_harmonics4_multiscale_4x4` | 40.88 | 49.57 | +42.77 | 0.16% | 50.28 | 76.22 | +32.03 | 49.70 | 0.15% | 2.229 | `Dead End` |
| `T2_fullgrid_4x4` | 3.88 | 5.87 | **-0.93** | 7.59% | 13.19 | 34.15 | **-10.03** | 4.38 | 7.65% | 2.214 | `Validated` |
| `T2_huber_fullgrid_4x4` | 3.54 | 5.81 | **-0.99** | 10.60% | 13.14 | 37.64 | -6.54 | 4.00 | 10.87% | 2.214 | `Validated` |
| `T2_lbfgs_polish_4x4` | 3.88 | 5.87 | -0.93 | 7.53% | 13.19 | 34.16 | -10.03 | 4.38 | 7.61% | 2.214 | `Neutral` |
| `T3_tps11km_plus_resnet_4x4` | 11.01 | 14.54 | +7.74 | 2.00% | 12.75 | 22.12 | -22.07 | 14.66 | 2.07% | 2.375 | `Mixed` |
| `T3_bilinear_plus_resnet_4x4` | 14.57 | 18.52 | +11.73 | 1.32% | 14.56 | 20.33 | -23.85 | 18.51 | 1.29% | 2.288 | `Mixed` |
| `T3_coarse_to_fine_nn_4x4` | 10.57 | 13.72 | +6.92 | 1.62% | 17.99 | 36.91 | -7.27 | 12.68 | 1.65% | 2.530 | `Dead End` |
| `T4_data_8x8` | 10.75 | 12.80 | +6.01 | 0.77% | 21.34 | 41.36 | -2.83 | 12.14 | 0.82% | 2.214 | `Capacity Limit` |
| **`T4_data_adaptive_hybrid`** | **2.48** | **3.22** | **-3.58** | **16.18%** | **1.89** | **2.59** | **-41.60** | **3.08** | **14.55%** | **2.214** | **`Breakthrough`** |
| `T4_data_border_enriched` | 3.94 | 4.85 | -1.95 | 6.42% | 5.22 | 9.63 | -34.56 | 4.54 | 6.57% | 2.214 | `Validated` |
| `T5_neural_feature_grid_4x4` | 20.82 | 23.95 | +17.15 | 0.37% | 21.32 | 27.81 | -16.37 | 23.78 | 0.38% | 0.602 | `Needs 8x8` |
| **`T5_neural_feature_grid_8x8`** | **1.08** | **2.29** | **-4.51** | **56.23%** | **3.41** | **11.25** | **-32.93** | **1.12** | **58.81%** | **0.602** | **`Breakthrough`** |
| `T6_siren_4x4` | 375.44 | 449.56 | +442.76 | 0.00% | 398.78 | 495.51 | +451.33 | 365.61 | 0.00% | 2.082 | `Dead End` |
| **`T6_matern32_rbf_4x4`** | **2.93** | **4.66** | **-2.14** | **23.66%** | **10.06** | **33.58** | **-10.61** | **3.71** | **22.25%** | **2.405** | **`Validated`** |
| `T6_hist_gbr_4x4` | 41.01 | 52.39 | +45.59 | 0.06% | 89.45 | 144.33 | +100.15 | 43.42 | 0.05% | 6.822 | `Dead End` |
| **`T7_bilinear_plus_featgrid_8x8`** *(Champion)* | **0.77** | **1.71** | **-5.08** | **76.53%** | **1.81** | **7.84** | **-36.34** | **0.84** | **78.30%** | **0.661** | **`Breakthrough`** |

---

## 3. Track-by-Track Key Findings

1. **Track 5 + Track 3 Synthesis (`T7_bilinear_plus_featgrid_8x8`) — Sub-Millimeter Breakthrough (`0.77 mm` Mean / `0.84 mm` Corridor RMSE, `0.66 MB`)**:
   - Combining a **4-corner bilinear cell base (`CellBilinearShiftModel`)** with a **Learnable Multi-Resolution 2D Feature Grid + Tiny MLP (`NeuralFeatureGridShiftModel`)** trained on `RO_GRID_8X8` slashed nationwide uniform mean error to **`0.77 mm`** (`76.5% < 1 mm`, `99.9% < 1 cm`), adaptive stress-test mean error to **`1.81 mm`** (`98.3% < 1 cm`), and corridor RMSE to **`0.84 mm`** (`78.3% < 1 mm`, max `4.44 mm`) using only **`0.661 MB`** of weights and grid parameters.
   - Even without the bilinear base, **`T5_neural_feature_grid_8x8`** alone reached **`1.08 mm` uniform mean** (`56.2% < 1 mm`) in **`0.602 MB`**. (On sparse `4x4`, the `105x143` fine grid was under-constrained, confirming that learnable feature grids pair naturally with `8x8` or adaptive training density.)
2. **Track 4 (`T4_data_adaptive_hybrid`) — Zero-Lookup Neural Breakthrough (`2.48 mm` Uniform Mean, `2.59 mm` Adaptive RMSE)**:
   - Using the **exact same `2.21 MB` grid-free ResNet architecture as `B0`**, simply training on `RO_TRAIN_ADAPTIVE_HYBRID` (disjoint off-dyadic `ADAPTIVE_1CM` probes + `RO_GRID_4X4`) cut uniform RMSE by **53%** (`6.80 → 3.22 mm`) and collapsed the adaptive stress-test RMSE by **94%** (`44.18 → 2.59 mm`, `99.6% < 1 cm`)!
   - `T4_data_border_enriched` also helped (`44.18 → 9.63 mm` adaptive RMSE). In contrast, `T4_data_8x8` hurt the `256`-wide ResNet (`6.80 → 12.80 mm`) because 150 epochs with `batch_size=512` and uniform `8x8` points saturated global capacity.
3. **Track 2 (`T2_fullgrid_4x4`, `T2_huber_fullgrid_4x4`) — Free Optimization Gains**:
   - Training on 100% of the deterministic grid (`val_fraction=0.0`) consistently improved `B0` (`6.80 → 5.87 mm` uniform, `44.18 → 34.15 mm` adaptive). Adding Huber loss (`beta=0.005`) pushed `<1mm` pass rate from `6.5% → 10.6%` uniform (`4.70 → 4.00 mm` corridor). `L-BFGS` (40 steps after AdamW) did not improve beyond AdamW's cosine-annealed minimum.
4. **Track 6 (`T6_matern32_rbf_4x4` vs. `SIREN` vs. `HistGBR`)**:
   - **`Matérn 3/2` local RBF (`4x4`)** beat `B0` across every metric (`4.66 mm` uniform RMSE, `23.7% < 1 mm`, `3.71 mm` corridor RMSE) because its finite $C^1$ smoothness matches the grid better than infinitely smooth kernels.
   - **`HistGBR`** (`52.4 mm` uniform RMSE) and **`SIREN`** (`449.6 mm` uniform RMSE without RFF) are confirmed dead ends for this problem.
5. **Track 1 (`T1_harmonics4`, `T1_multiscale_rff`) & Track 3 (ResNet Residuals on `4x4`)**:
   - Adding 4th-order cell harmonics (`k=1..4`) and cross-terms into the `256`-wide ResNet destabilized optimization (`6.80 → 53.38 mm`), likely due to high-frequency aliasing in `ReLU` layers.
   - Note that Two-Stage Residuals *did* cut adaptive RMSE in half for ResNets (`44.18 → 20.33 mm`), and when paired with the **Feature Grid** (`T7_bilinear_plus_featgrid_8x8`), unlocked our **`0.77 mm`** overall champion.

---

## 4. Track 7: Full Bidirectional `ETRS89 <-> Stereo70` Verification

Wrapping the winning shift model (`T7_bilinear_plus_featgrid_8x8`) inside `ResidualCoordinateTransformer(TwoStepHelmert2DTransformer(), best_model)` on `CORRIDOR_SNW` (36,435 pts):

| Direction | Mean 2D (mm) | RMSE 2D (mm) | P95 (mm) | Max 2D (mm) | Pass `< 1 mm` (%) | Pass `< 1 cm` (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `etrs89_to_st70` | **0.7068** | **0.8436** | 1.5892 | 4.4339 | **78.28%** | **100.0%** |
| `st70_to_etrs89` | **0.7070** | **0.8438** | 1.5896 | 4.4456 | **78.30%** | **100.0%** |

---

## 5. Visual Diagnostics (`T7_bilinear_plus_featgrid_8x8`)

- [`plots/T7_bilinear_plus_featgrid_8x8_ro_test_uniform_4x4_grid_shift_2d_spatial_error.png`](plots/T7_bilinear_plus_featgrid_8x8_ro_test_uniform_4x4_grid_shift_2d_spatial_error.png)
- [`plots/T7_bilinear_plus_featgrid_8x8_ro_test_uniform_4x4_grid_shift_2d_hist_ecdf.png`](plots/T7_bilinear_plus_featgrid_8x8_ro_test_uniform_4x4_grid_shift_2d_hist_ecdf.png)
- [`plots/T7_bilinear_plus_featgrid_8x8_ro_test_adaptive_3mm_mid_grid_shift_2d_spatial_error.png`](plots/T7_bilinear_plus_featgrid_8x8_ro_test_adaptive_3mm_mid_grid_shift_2d_spatial_error.png)
- [`plots/T7_bilinear_plus_featgrid_8x8_ro_test_adaptive_3mm_mid_grid_shift_2d_hist_ecdf.png`](plots/T7_bilinear_plus_featgrid_8x8_ro_test_adaptive_3mm_mid_grid_shift_2d_hist_ecdf.png)

---

## 6. Takeaways & Verdict
- **Verdict**: `Validated` (Exploratory Screening Complete).
- **Top Leads for Dedicated Deep-Dive Units**:
  1. **Micro-Grid Sub-Millimeter Champion (`T7_bilinear_plus_featgrid_8x8`)**: `0.77 mm` uniform mean (`1.71 mm` RMSE, `76.5% < 1 mm`), `0.84 mm` corridor RMSE (`78.3% < 1 mm`), and `0.71 mm` mean / `0.84 mm` RMSE bidirectional `ETRS89 <-> Stereo70` in `0.66 MB`.
  2. **Zero-Lookup Neural Champion (`T4_data_adaptive_hybrid`)**: `2.48 mm` uniform mean (`3.22 mm` RMSE) and `1.89 mm` adaptive mean (`2.59 mm` RMSE, `99.6% < 1 cm`) with `0 MB` lookup table.

---

## 7. Benchmark Candidates
| Variant | Benchmark Label | Train Grid | Target Suite |
| :--- | :--- | :--- | :--- |
| `T7_bilinear_plus_featgrid_8x8` | `Bilinear + Neural Feature Grid (8x8)` | `ro_grid_8x8` | `grid_shift_models` |
| `T4_data_adaptive_hybrid` | `PyTorch ResNet + Cyclic CellPos (Adaptive Hybrid)` | `ro_train_adaptive_hybrid` | `grid_shift_models` |
