# Empirical Report: Optimal Surface Regressor (`optimal_surface_regressor`)

## 1. Executive Summary & Unified 4-Stage Methodology

This research unit establishes an exhaustive, **learnability-driven benchmark** across **10 smooth surface regressor families** trained on **`RO_GRID_4X4`** (36,435 points, ~2.75 km spacing) and evaluated across **`RO_GRID_4X4`**, **`RO_TEST_UNIFORM_16X16`** (588,032 points), and **`RO_TEST_ADAPTIVE_1MM_MID`** (564,325 points).

Rather than optimizing raw RMSE alone (which can reward memorizing 11 km bicubic boundary kinks), candidates are ranked by **Stage-2 ML Surface Learnability Score** (`cRMSE_2D`, `IQR_2D`, KD-tree `Roughness Grad RMS`, and `P95`) using a unified 4-stage funnel implemented in [`screen_families.py`](screen_families.py) and [`run_unit.py`](run_unit.py):

1. **Stage 0 — Per-Family Subgrid Calibration**: Fit 3 diverse anchor configs on Full `RO_GRID_4X4` and compare their Full-Romania ordering against 3 dense `CellRegion` subgrids (`box10_interior`, `box10_dual`, `box20_west`) to pick the fastest faithful proxy per family.
2. **Stage 1 — Per-Family Screening (`persist=False`, `plots=None`)**: Sweep **174 parameter configurations** across 7 trainable families on their Stage-0-verified subgrids (`results/screening_<family>.csv`).
3. **Stage 2 — Top-3 Full-Romania Promotion**: Promote the **Top 3** configs per swept family + **8 Direct-Full** reference models (**29 models total**, `results/stage2_full_shortlist.csv`) to Full Romania to select **1 Champion per family**.
4. **Stage 3 — Consolidated Champions (`run_unit.py`)**: Persist `.pkl` checkpoints and render unified `PlotGroup.SURFACE_DIAGNOSTIC`, Pareto Learnability Ranking, and 9-Cell Extreme Scenario visuals for the **10 Family Champions**.

---

## 2. Stage 0 Subgrid Calibration Verdicts

| Family | Configs Screened | Selected Subgrid | Top-1 Match | Spearman $\rho$ |
| :--- | :---: | :--- | :---: | :---: |
| `tps` | 16 | `box10_interior` `(19, 28)` | Yes | `1.000` |
| `matern` | 48 | `box20_west` `(3, 15)` | Yes | `0.500` |
| `multiquadric` | 48 | `box10_interior` `(19, 28)` | Yes | `1.000` |
| `gaussian` | 24 | `box10_interior` `(19, 28)` | No | `-0.500` |
| `polyharmonic` | 18 | `box20_west` `(3, 15)` | Yes | `1.000` |
| `bspline` | 10 | `box20_west` `(3, 15)` | No | `0.500` |
| `local_poly` | 12 | `box10_interior` `(19, 28)` | Yes | `1.000` |

---

## 3. Stage 3 Cross-Family Champions Scorecard (`learnability_ranking.csv`)

| Rank | Family | Champion Model | Overall Score (mm) | `16x16` cRMSE (mm) | `16x16` RoughRMS (mm/km) | `16x16` Score (mm) | `Adapt 1mm` cRMSE (mm) | `Adapt 1mm` RoughRMS (mm/km) | `Adapt 1mm` Score (mm) |
| :-: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Polyharmonic** | **`cubic_k25_s0`** | **16.015** | 5.120 | 2.941 | 6.589 | 22.515 | **13.135** | **25.440** |
| **2** | **Thin Plate Spline** | **`tps_deg2_k25_s1e-5`** | **16.531** | **4.724** | **2.798** | **6.059** | **20.620** | 15.762 | 27.003 |
| **3** | **Local Polynomial** | **`lpoly_deg2_k16_tric`** | **16.696** | 5.171 | 3.051 | 6.489 | 23.470 | 14.557 | 26.903 |
| **4** | **Matérn ($C^1$/$C^2$)** | **`matern32_k16_eps5_s1e-5`** | **17.387** | 5.029 | 2.864 | 6.336 | 24.700 | 14.210 | 28.439 |
| **5** | **11 km Grid Spline** | `grid11km_natural_spline_c2` | 22.001 | 16.379 | 4.883 | 18.171 | 19.681 | 12.345 | 25.830 |
| **6** | **Multiquadric / IMQ** | `imq_deg2_k16_eps25_s1e-4` | 23.758 | 5.616 | 3.470 | 7.011 | 27.812 | 28.548 | 40.506 |
| **7** | **Bilinear Reference** | `cell_bilinear_11km` | 25.622 | 21.697 | 4.623 | 23.166 | 22.753 | 9.281 | 28.079 |
| **8** | **Gaussian RBF** | `gauss_deg2_k16_eps40_s1e-4` | 38.174 | 6.121 | 4.387 | 8.069 | 29.758 | 60.798 | 68.279 |
| **9** | **Global Polynomial** | `global_poly_deg12` | 206.707 | 170.926 | 14.893 | 162.158 | 254.766 | 26.944 | 251.257 |
| **10** | **Tensor B-Spline** | `bspline_deg5_s0p05` | 904.784 | 193.731 | 108.565 | 204.933 | 1586.629 | 927.438 | 1604.636 |

### Key Scientific Takeaways
1. **Top 4 Stage-1 Surface Champions (`16.0–17.4 mm`)**:
   * **`cubic_k25_s0`** (`16.015 mm`) emerged as **Rank #1 overall** because its local cubic polyharmonic kernel (`k=25`) achieves the lowest `Adaptive 1mm` roughness (`13.135 mm/km`) among all 4x4-trained continuous surfaces.
   * **`tps_deg2_k25_s1e-5`** (`16.531 mm`, Rank #2) remains the **best on `16x16` (`6.059 mm` Score, `2.798 mm/km` RoughRMS)** and best `Adaptive 1mm` `cRMSE` (`20.620 mm`).
   * **`lpoly_deg2_k16_tric`** (`16.696 mm`, Rank #3, `LocalPolynomialSurfaceRegressor`) is a **3× faster** (`~10s` full eval) local quadratic LOESS alternative with nearly identical learnability.
   * **`matern32_k16_eps5_s1e-5`** (`17.387 mm`, Rank #4) confirms that Matérn 3/2 ($C^1$ smoothness) with wider reach (`eps=5`) outperforms Matérn 5/2 on `Adaptive 1mm` roughness (`14.210 mm/km`).
2. **Why `bspline_deg5_s0p05` (`904.78 mm`) and `global_poly_deg12` (`206.71 mm`) are Dead Ends**:
   * `SmoothBivariateSpline` requires a rectangular domain; Romania's concave border leaves empty corners where tensor-product B-splines extrapolate wildly (`57 m` max spike). Global degree-12 polynomials lack local capacity (`206.7 mm`).

---

## 4. Visual Diagnostics

* **Cross-Family Learnability Pareto & Cost Ranking**:
  * [`plots/surface_learnability_ranking_16x16.png`](plots/surface_learnability_ranking_16x16.png)
  * [`plots/surface_learnability_ranking_adaptive_1mm.png`](plots/surface_learnability_ranking_adaptive_1mm.png)
* **Residual Roughness (Gradient Map, Variogram, Cell-Wall Profile)**:
  * [`plots/ro_test_uniform_16x16_grid_shift_2d_residual_roughness.png`](plots/ro_test_uniform_16x16_grid_shift_2d_residual_roughness.png)
  * [`plots/ro_test_adaptive_1mm_mid_grid_shift_2d_residual_roughness.png`](plots/ro_test_adaptive_1mm_mid_grid_shift_2d_residual_roughness.png)
* **Spatial Error & Distribution**:
  * [`plots/ro_test_uniform_16x16_grid_shift_2d_spatial_error.png`](plots/ro_test_uniform_16x16_grid_shift_2d_spatial_error.png)
  * [`plots/ro_test_uniform_16x16_grid_shift_2d_hist_ecdf.png`](plots/ro_test_uniform_16x16_grid_shift_2d_hist_ecdf.png)
  * [`plots/ro_test_adaptive_1mm_mid_grid_shift_2d_spatial_error.png`](plots/ro_test_adaptive_1mm_mid_grid_shift_2d_spatial_error.png)
  * [`plots/ro_test_adaptive_1mm_mid_grid_shift_2d_hist_ecdf.png`](plots/ro_test_adaptive_1mm_mid_grid_shift_2d_hist_ecdf.png)
* **9-Cell (99 km) Extreme Scenario Transects**:
  * [`plots/extreme_scenarios_gradient_and_linear_9cells.png`](plots/extreme_scenarios_gradient_and_linear_9cells.png)
  * [`plots/extreme_scenarios_curvature_and_kinks_9cells.png`](plots/extreme_scenarios_curvature_and_kinks_9cells.png)

---

## 5. Methodology Insights & Proxy Validation Findings

1. **Always Fit on Full `RO_GRID_4X4`, Subgrid Only for Evaluation**:
   * Fitting RBF/Matérn/LOESS on all 36,435 `RO_GRID_4X4` points takes only `0.006–0.007s`—nearly 100% of runtime is evaluation.
   * Training on a smaller box silently shrinks `normalize_coords=True`'s half-span (`~77 km` vs. `~350 km`), which rescales `smoothing` and `epsilon` and can invert parameter rankings. Fitting on Full `RO_GRID_4X4` avoids this scale trap completely.
2. **Contiguous `CellRegion` vs. Point-Thinning**:
   * Thinning test grids (10%–20% stratified) widens neighbor spacing, artificially lowering `Roughness Grad RMS` for rough models (`matern52_k25_eps2p0_s0` jumped from #5 to #2 when thinned).
   * Contiguous `CellRegion` windows (`box10_interior` `15.3×` faster, `box20_west` `5.4×` faster) preserve 100% of sub-cell point density so roughness and S-shape dynamics remain intact.

---

## 6. Takeaways & Shortlisted Stage-1 Candidates

* **Verdict**: `Validated` — Top-3 Stage-1 base surfaces locked in for Stage-2 residual ML training:
  1. **`cubic_k25_s0`** (`16.015 mm` Overall Learnability): Best overall; lowest `Adaptive 1mm` roughness (`13.135 mm/km`).
  2. **`tps_deg2_k25_s1e-5`** (`16.531 mm` Overall Learnability): Best on `16x16` (`6.059 mm` Score, `2.798 mm/km` RoughRMS) and best `Adaptive 1mm` `cRMSE` (`20.620 mm`).
  3. **`lpoly_deg2_k16_tric`** (`16.696 mm` Overall Learnability): **~3× faster** inference (`~10s` full eval) using local quadratic LOESS rather than RBFs.
  *(Optional #4: **`matern32_k16_eps5_s1e-5`** at `17.387 mm` as a $C^1$ Matérn representative.)*
