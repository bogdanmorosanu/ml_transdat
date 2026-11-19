# Research Unit 03: 2D 4-Parameter Helmert Baseline (No Grid)

## Executive Summary
This unit evaluates a **grid-free 2-step 2D coordinate transformation** (`StereoProj` $\leftrightarrow$ `Helmert2D`) between **ETRS89** and **Stereo 70**, omitting the `Grid2D` bicubic distortion grid. We fit optimal 4-parameter 2D conformal Helmert parameters $(t_N, t_E, dm, R_z)$ via closed-form linear least squares on the **2,222 valid SPG grid nodes** (`is_node == True` in `ro_grid_4x4.parquet`) and evaluate bidirectionally across all **36,435 points** of `data/grids/ro_grid_4x4.parquet`.

---

## Fitted vs. Embedded 2D Helmert Parameters (`st70_to_os`)

| Parameter | Embedded (`rom_grid3d_25.09.spg`) | Refitted (2,222 SPG Nodes) | Delta (Fitted $-$ Embedded) |
|:---|---:|---:|---:|
| **Translation North ($t_N$)** | $-31.8051\,\text{m}$ | $-31.6399\,\text{m}$ | $+0.1652\,\text{m}$ |
| **Translation East ($t_E$)** | $-119.7358\,\text{m}$ | $-119.6209\,\text{m}$ | $+0.1149\,\text{m}$ |
| **Scale Correction ($dm$)** | $-0.1156\,\text{ppm}$ | $-0.4074\,\text{ppm}$ | $-0.2918\,\text{ppm}$ |
| **Rotation ($R_z$)** | $+0.2274''$ | $+0.2192''$ | $-0.0082''$ |

---

## Comparative Scorecard (`ro_grid_4x4.parquet`, $N = 36,435$)

### `ETRS89 -> Stereo 70`

| Metric | Embedded `Helmert2D` (No Grid) | Refitted `Helmert2D` (No Grid) | PyProj `EPSG:3844` (Unit 02) |
|:---|:---:|:---:|:---:|
| **2D Mean Error** | $0.3716\,\text{m}$ | $0.3743\,\text{m}$ | $0.3601\,\text{m}$ |
| **2D Std Dev** | $0.2488\,\text{m}$ | $0.2368\,\text{m}$ | $0.2437\,\text{m}$ |
| **2D RMSE** | $0.4472\,\text{m}$ | $0.4429\,\text{m}$ | $0.4348\,\text{m}$ |
| **Median (p50)** | $0.3170\,\text{m}$ | $0.3291\,\text{m}$ | $0.3058\,\text{m}$ |
| **95th Percentile (p95)** | $0.8784\,\text{m}$ | $0.8515\,\text{m}$ | $0.8460\,\text{m}$ |
| **99th Percentile (p99)** | $1.1498\,\text{m}$ | $1.1123\,\text{m}$ | $1.1367\,\text{m}$ |
| **Maximum 2D Error** | $2.7570\,\text{m}$ | **$2.6811\,\text{m}$** | $2.8207\,\text{m}$ |
| **Signed Bias ($dN, dE$)** | $+0.0043, -0.0120\,\text{m}$ | **$-0.0011, -0.00005\,\text{m}$** | $+0.0087, -0.0160\,\text{m}$ |
| **Pass Rate $< 1\,\text{m}$** | $97.8\%$ | **$98.1\%$** | $98.0\%$ |

### `Stereo 70 -> ETRS89` (Refitted `Helmert2D`)
- **2D Mean / RMSE**: $0.3743\,\text{m}$ / $0.4429\,\text{m}$
- **p50 / p95 / p99 / Max**: $0.3291\,\text{m}$ / $0.8514\,\text{m}$ / $1.1117\,\text{m}$ / $2.6794\,\text{m}$
- **Out-of-Sample Interior Only ($34,213$ points)**: Mean = $0.3741\,\text{m}$, RMSE = $0.4426\,\text{m}$, Max = $2.6811\,\text{m}$ (`etrs89_to_st70`).
- **Round-Trip Precision**: Max `st70 -> etrs89 -> st70` error = $5.24 \times 10^{-9}\,\text{m}$; max `etrs89 -> st70 -> etrs89` error = $7.77 \times 10^{-16}\,\text{rad}$.

---

## Visual Gallery

### 1. Spatial Error Heatmaps (2D Error in Meters)

| ETRS89 $\rightarrow$ Stereo 70 | Stereo 70 $\rightarrow$ ETRS89 |
|:---:|:---:|
| ![Spatial Inv](plots/fitted_helmert2d_etrs89_to_st70_spatial_error.png) | ![Spatial Fwd](plots/fitted_helmert2d_st70_to_etrs89_spatial_error.png) |

### 2. Error Distributions (Histogram & ECDF)

| ETRS89 $\rightarrow$ Stereo 70 | Stereo 70 $\rightarrow$ ETRS89 |
|:---:|:---:|
| ![Dist Inv](plots/fitted_helmert2d_etrs89_to_st70_hist_ecdf.png) | ![Dist Fwd](plots/fitted_helmert2d_st70_to_etrs89_hist_ecdf.png) |

### 3. Signed Residual Scatter & Sub-Cell Profiles

| Signed Residuals ($dE$ vs $dN$) | Sub-Cell $(u, v)$ Error Profile |
|:---:|:---:|
| ![Scatter](plots/fitted_helmert2d_etrs89_to_st70_signed_scatter.png) | ![Subcell](plots/fitted_helmert2d_etrs89_to_st70_subcell_heatmap.png) |

---

## Comparison vs. Experiment 01 (SPG 2D Distortion Grid)

Because `pytransdat`'s full pipeline is `StereoProj` $\to$ `Embedded Helmert2D` $\to$ `Grid2D`, omitting `Grid2D` while keeping the embedded `Helmert2D` parameters reproduces the **exact negated SPG distortion field** profiled in [Unit 01](../01_spg_grid_analysis/report.md) ($\text{Mean} = 0.3716\,\text{m}$, $\text{p50} = 0.3170\,\text{m}$, $\text{p95} = 0.8784\,\text{m}$, $\text{p99} = 1.1498\,\text{m}$, $\text{Max} = 2.7570\,\text{m}$).

Refitting the 4-parameter `Helmert2D` on the 2,222 valid SPG grid nodes shifts and scales the 2D conformal frame slightly ($\Delta dm = -0.2918\,\text{ppm}$, $\Delta R_z = -0.0082''$) so that the residual field is **centered at zero mean** and **reduces the extreme border tails**:

| Metric (`ro_grid_4x4`, $N = 36,435$) | Unit 01: SPG Grid Shifts (`Grid2D`) | Unit 03: Refitted `Helmert2D` (`etrs89_to_st70`) | Delta (Refitted $-$ Grid) |
|:---|:---:|:---:|:---:|
| **Mean Signed Bias ($dN, dE$)** | $-0.0043, +0.0120\,\text{m}$ | **$-0.0011, -0.00005\,\text{m}$** | Centered near $(0, 0)$ |
| **2D Mean Magnitude** | $0.3716\,\text{m}$ | $0.3743\,\text{m}$ | $+0.0027\,\text{m}$ |
| **Median (p50)** | $0.3170\,\text{m}$ | $0.3291\,\text{m}$ | $+0.0121\,\text{m}$ |
| **95th Percentile (p95)** | $0.8784\,\text{m}$ | **$0.8515\,\text{m}$** | **$-0.0269\,\text{m}$** |
| **99th Percentile (p99)** | $1.1498\,\text{m}$ | **$1.1123\,\text{m}$** | **$-0.0375\,\text{m}$** |
| **Maximum 2D Magnitude** | $2.7570\,\text{m}$ | **$2.6811\,\text{m}$** | **$-0.0759\,\text{m}$** |

### Visual Comparison: Error Distribution & Signed Residual Scatter

| Unit 01: SPG Grid Distribution (`Grid2D`) | Unit 03: Refitted `Helmert2D` Distribution |
|:---:|:---:|
| ![Unit 01 Hist ECDF](../01_spg_grid_analysis/plots/spg_grid_shifts_2d_bicubic_hist_ecdf.png) | ![Unit 03 Hist ECDF](plots/fitted_helmert2d_etrs89_to_st70_hist_ecdf.png) |

| Unit 01: SPG Grid Signed Scatter (`dE` vs `dN`) | Unit 03: Refitted `Helmert2D` Signed Scatter |
|:---:|:---:|
| ![Unit 01 Signed Scatter](../01_spg_grid_analysis/plots/spg_grid_shifts_2d_bicubic_signed_scatter.png) | ![Unit 03 Signed Scatter](plots/fitted_helmert2d_etrs89_to_st70_signed_scatter.png) |

*Note on sign convention*: Unit 01 plots the SPG shift $\Delta = \text{GroundTruth} - \text{PreGrid}$, whereas Unit 03 plots the prediction error $\varepsilon = \text{Pred} - \text{GroundTruth} = -\Delta_{\text{residual}}$ (mirrored across the origin). Notice how the refitted 2D Helmert pulls the extreme Banat/West lobe inward from $2.757\,\text{m}$ to $2.681\,\text{m}$ while keeping the core distribution tightly centered at $(0, 0)$.

---

## Scientific Takeaways
1. **Optimal 2D Linear Floor**: Refitting the 4-parameter 2D Helmert on the `25.09` SPG nodes eliminates global signed bias, improves tail metrics (p95, p99, Max = $2.68\,\text{m}$ vs. PyProj's $2.82\,\text{m}$), and lifts `< 1 m` pass rate to $98.1\%$, using only 4 closed-form parameters on `StereoProj`.
2. **Physical Ceiling Confirmed**: Because the remaining residuals are localized non-linear triangulation network distortions (up to $\sim 2.68\,\text{m}$), **no** global 4-parameter 2D conformal map can push RMSE below $\sim 0.44\,\text{m}$; non-linear spatial models (e.g., MLPs, RBFs, Fourier networks) are required for sub-centimeter accuracy.
