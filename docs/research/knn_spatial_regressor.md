# Research Unit: knn_spatial_regressor

**Slug**: `knn_spatial_regressor`  
**Status**: `Validated`  
**Date**: 2026-11-19  
**Parent Unit**: [`helmert2d_baseline`](helmert2d_baseline.md)  
**Primary Artifact**: [`experiments/knn_spatial_regressor/report.md`](../../experiments/knn_spatial_regressor/report.md)  

---

## 1. Scientific Objective
Evaluate whether local spatial partition / neighborhood models (**K-Nearest Neighbors**) can learn the 2D planimetric distortion shifts ($(N, E) \mapsto (\delta N, \delta E)$) directly from the official $2,689$ control nodes (`spg_nodes_valid.parquet`) and interpolate continuous physical space inside $11\,\text{km}$ grid cells without requiring end-to-end geographic-to-grid mapping.

---

## 2. Hypothesis & Rationale
* **Hypothesis 1 (Nodal Memorization vs. Interior Interpolation)**: An unweighted $K=1$ model will achieve $0.0000\,\text{m}$ error on training nodes but create sharp step-plateaus (Voronoi polygons) with severe boundary cliff errors inside the cells.
* **Hypothesis 2 (4-Corner IDW Sweet Spot)**: Because each grid cell has 4 bounding corners, $K=4$ Inverse Distance Weighting will provide the optimal balance between exact nodal recovery and smooth interior blending. Expanding $K$ to 8 or 16 will over-smooth local gradients because positive-only distance weights pull predictions toward distant nodes.

---

## 3. Key Findings & Scorecard

- **Baseline Floor (Zero Shift)**: $\text{RMSE} = 0.4472\,\text{m}$
- **Champion Configuration**: **$K=4$ Distance-Weighted (IDW)**
  - **Training Nodes**: $\text{RMSE} = 0.0000\,\text{m}$ (exact nodal recovery)
  - **Validation 4x4 Grid (2.75 km spacing)**: **$\text{RMSE} = 0.0353\,\text{m}$ ($3.5\,\text{cm}$)**, $\text{Max} = 0.4783\,\text{m}$ ($92.1\%$ reduction over baseline floor).
  - **Generalization 8x8 Grid (1.375 km spacing)**: **$\text{RMSE} = 0.0346\,\text{m}$ ($3.4\,\text{cm}$)**, $\text{Max} = 0.4972\,\text{m}$.
  - **Sub-Centimeter Share**: $19.3\%$ of points achieve $< 1\,\text{cm}$ error.
- **Why $K=8$ and $K=16$ Degrade**:
  - $K=4$ Distance: $\text{RMSE} = 0.0353\,\text{m}$
  - $K=8$ Distance: $\text{RMSE} = 0.0601\,\text{m}$
  - $K=16$ Distance: $\text{RMSE} = 0.0921\,\text{m}$
  Positive-only IDW weights cannot compute directional derivative slopes like the 16-point bicubic spline; instead, they dilute local curvature toward the regional average.
- **Discovery of Piecewise $C^0/C^1$ Kink**: 1D 66 km transects revealed that the official Romanian `.spg` / RomGEO bicubic grid has real first-derivative slope kinks ($\Delta\text{slope} \approx 4.2 \times 10^{-5}$) at $11\,\text{km}$ cell boundaries due to local one-sided finite-difference stencil shifts (verified against desktop RomGEO to $< 0.36\,\text{mm}$).

---

## 4. Impact on Future Modeling
1. Proves that isolating the grid shift transformation allows a simple spatial model to drop error from $45\,\text{cm}$ down to $3.5\,\text{cm}$.
2. Highlights the **inductive bias mismatch**: purely smooth global models (RBF, Neural Networks) will naturally smooth out the artificial piecewise kinks of the legacy grid unless cell-interior coordinates $(u, v)$ are provided.
