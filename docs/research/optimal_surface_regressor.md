# Research Unit: Optimal Stage-1 Surface Regressors (`optimal_surface_regressor`)

**Status**: `Validated`  
**Primary Sources**: [`experiments/optimal_surface_regressor/report.md`](../../experiments/optimal_surface_regressor/report.md), [`experiments/optimal_surface_regressor/results/learnability_ranking.csv`](../../experiments/optimal_surface_regressor/results/learnability_ranking.csv), [`src/models/bspline_surface.py`](../../src/models/bspline_surface.py), [`src/models/local_poly_surface.py`](../../src/models/local_poly_surface.py), [`.agents/tasks/cell-aligned-subgrid-extraction.md`](../../.agents/tasks/cell-aligned-subgrid-extraction.md)

---

## 1. Scientific Question & Hypothesis
Can we identify an optimal **Stage-1 smooth base surface** trained on `RO_GRID_4X4` (~2.75 km spacing, which resolves linear, dome/bowl, and S-shape cell dynamics) that removes the bulk of the `.spg` distortion shift without memorizing legacy 11 km bicubic boundary kinks—leaving **low-amplitude, low-roughness residuals** for Stage-2 ML models?

---

## 2. Methodology & Key Pivots
1. **Learnability-First Ranking**: Replaced raw RMSE with **Stage-2 ML Surface Learnability Score** (`cRMSE_2D`, `IQR_2D`, KD-tree `Roughness Grad RMS`, `P95`). Raw RMSE alone misranked rough local kernels (e.g., `matern52_k25_eps2p0_s0` had `19.65 mm` adaptive `cRMSE` but `23.91 mm/km` roughness).
2. **Full-Grid Fit + Contiguous `CellRegion` Screening**:
   - Fitting on `RO_GRID_4X4` takes `0.006–0.007s` and avoids the coordinate-normalization scale trap (`normalize_coords=True` shrinks half-span from `~350 km` to `~77 km` if trained on a subgrid, distorting `smoothing`/`epsilon`).
   - Point-thinning (10%–20% stratified) widens neighbor spacing and under-estimates `Roughness Grad RMS`. Contiguous `CellRegion` subgrids (`box10_interior` `15.3×` faster, `box20_west` `5.4×` faster) preserve 100% of sub-cell density.
3. **4-Stage Exhaustive Pipeline**: Stage 0 per-family 3-anchor subgrid calibration $\to$ Stage 1 screening (`176` configs across 7 swept families) $\to$ Stage 2 Top-3 + 8 direct-full Full-Romania evaluation (`29` models) $\to$ Stage 3 consolidated Family Champions (`10` models).

---

## 3. Verdicts & Shortlisted Stage-1 Champions

| Rank | Family | Champion | Overall Score (mm) | `16x16` Score (mm) | `Adapt 1mm` Score (mm) | Verdict |
| :-: | :--- | :--- | :---: | :---: | :---: | :--- |
| **1** | Polyharmonic | `cubic_k25_s0` | **16.015** | 6.589 | **25.440** | **Shortlisted (#1)** |
| **2** | Thin Plate Spline | `tps_deg2_k25_s1e-5` | **16.531** | **6.059** | 27.003 | **Shortlisted (#2)** |
| **3** | Local Polynomial | `lpoly_deg2_k16_tric` | **16.696** | 6.489 | 26.903 | **Shortlisted (#3, 3× faster)** |
| **4** | Matérn ($C^1$/$C^2$) | `matern32_k16_eps5_s1e-5` | 17.387 | 6.336 | 28.439 | Validated backup |
| **9** | Global Polynomial | `global_poly_deg12` | 206.707 | 162.158 | 251.257 | `Dead End` (insufficient local capacity) |
| **10** | Tensor B-Spline | `bspline_deg5_s0p05` | 904.784 | 204.933 | 1604.636 | `Dead End` (rectangular extrapolation spikes up to `57 m`) |

**Next Step**: Train Stage-2 residual ML models (e.g., PyTorch ResNet / Neural Feature Grid) on top of **`cubic_k25_s0`**, **`tps_deg2_k25_s1e-5`**, and **`lpoly_deg2_k16_tric`** to measure end-to-end sub-millimeter learnability.
