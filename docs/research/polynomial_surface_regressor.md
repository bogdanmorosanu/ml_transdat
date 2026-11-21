# Research Unit: 2D Polynomial Surface Regressor

**Status**: `Validated`  
**Domain**: 2D Continuous Coordinate Distortion Shifts ($(st70_n, st70_e) \to (shift_n, shift_e)$)  
**Parent / Predecessor Units**: [`helmert2d_baseline`](helmert2d_baseline.md)  
**Related Experiment**: [`experiments/polynomial_surface_regressor/`](../../experiments/polynomial_surface_regressor/)  
**Task Log**: [`.agents/tasks/polynomial-surface-regressor-experiment.md`](../../.agents/tasks/polynomial-surface-regressor-experiment.md)

---

## 1. Scientific Hypothesis & Motivation

Our foundational geometric baseline, **Helmert 2D** ([`helmert2d_baseline`](helmert2d_baseline.md)), establishes an optimal 4-parameter linear/conformal floor of $\text{RMSE} = 0.4429\,\text{m}$ (max error $2.68\,\text{m}$). Geometrically, Helmert 2D defines a **flat, rigid plane** with rotation, uniform scale, and translation. Because it cannot bend, it leaves nationwide regional curvatures unmodeled.

### Core Hypothesis
By generalizing the flat linear plane into a **2D bivariate polynomial surface** of degree $D$, we can introduce continuous mathematical curvature (quadratic bowls, saddles, and cubic S-curves). This will absorb nationwide macroscopic distortion trends, cutting the Helmert baseline error from $\sim 44\,\text{cm}$ down toward $\sim 15\text{--}20\,\text{cm}$ with a closed-form algebraic formula requiring zero lookup tables and only tens of parameters.

Furthermore, sweeping degrees $D = 1 \dots 14$ will empirically reveal the exact degree at which global polynomials reach diminishing returns or encounter boundary oscillations (Runge's phenomenon).

---

## 2. Experimental Methodology & Dual-Benchmark Protocol

All polynomial models were evaluated on the standardized nationwide continuous benchmarks:
1. **`RO_TEST_UNIFORM_4X4`** (36,752 off-lattice stratified points across all 2,297 valid cells): Area-weighted national survey accuracy.
2. **`RO_TEST_ADAPTIVE_3MM_MID`** (176,350 midpoints in high-curvature zones): Curvature stress test in Banat, Maramureș, and Dobrogea.

### Model Configurations Evaluated
* **`poly_deg1_11km`**: Linear flat plane ($D=1$, 3 terms per axis). Equivalent to Helmert/Affine baseline.
* **`poly_deg2_11km`**: Quadratic bowl / saddle ($D=2$, 6 terms per axis).
* **`poly_deg3_11km`**: Classical geodetic cubic surface ($D=3$, 10 terms per axis, inflection/asymmetry).
* **`poly_deg5_11km`**: Classical geodetic quintic surface ($D=5$, 21 terms per axis).
* **`poly_deg8_11km`**: Multi-harmonic polynomial surface ($D=8$, 45 terms per axis).
* **`poly_deg12_11km`**: High-order polynomial limit ($D=12$, 91 terms per axis).
* **`poly_deg5_4x4` / `poly_deg12_4x4`**: Polynomial models trained on the densified $4\times 4$ subgrid (36,435 points, ~2.7 km).

---

## 3. Quantitative Verdicts & Performance

| Model Architecture | Degree (D) | Terms / Axis | Training Grid | Eval Dataset | Mean 2D (mm) | RMSE 2D (mm) | P95 (mm) | Max 2D (mm) | < 5 cm (%) | < 10 cm (%) |
| :--- | :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `poly_deg1_11km` | 1 | 3 | 11 km nodes | Uniform 4x4 | 360.0 | 427.6 | 804.1 | 2550.9 | 1.9% | 7.9% |
| `poly_deg1_11km` | 1 | 3 | 11 km nodes | Adaptive 3mm | 388.5 | 478.1 | 886.9 | 2602.9 | 2.1% | 8.0% |
| `poly_deg2_11km` | 2 | 6 | 11 km nodes | Uniform 4x4 | 271.2 | 330.1 | 637.9 | 2632.1 | 4.0% | 14.4% |
| `poly_deg2_11km` | 2 | 6 | 11 km nodes | Adaptive 3mm | 325.8 | 421.5 | 787.4 | 2685.8 | 2.6% | 10.6% |
| `poly_deg3_11km` | 3 | 10 | 11 km nodes | Uniform 4x4 | 246.6 | 294.8 | 559.9 | 2377.9 | 4.3% | 15.6% |
| `poly_deg3_11km` | 3 | 10 | 11 km nodes | Adaptive 3mm | 291.8 | 372.7 | 668.3 | 2430.4 | 3.6% | 12.8% |
| `poly_deg5_11km` | 5 | 21 | 11 km nodes | Uniform 4x4 | 224.9 | 270.0 | 512.6 | 1839.4 | 4.4% | 17.6% |
| `poly_deg5_11km` | 5 | 21 | 11 km nodes | Adaptive 3mm | 277.5 | 349.2 | 678.0 | 1885.8 | 3.7% | 14.3% |
| `poly_deg8_11km` | 8 | 45 | 11 km nodes | Uniform 4x4 | 176.3 | 216.1 | 408.6 | 1112.4 | 8.8% | 29.3% |
| `poly_deg8_11km` | 8 | 45 | 11 km nodes | Adaptive 3mm | 251.7 | 316.2 | 632.2 | 1650.8 | 5.5% | 19.0% |
| `poly_deg12_11km` | 12 | 91 | 11 km nodes | Uniform 4x4 | **158.9** | **191.8** | **369.4** | **893.3** | **9.9%** | **32.3%** |
| `poly_deg12_11km` | 12 | 91 | 11 km nodes | Adaptive 3mm | **219.2** | **269.4** | **529.2** | **1343.1** | **6.1%** | **21.2%** |
| `poly_deg5_4x4` | 5 | 21 | 4x4 subgrid | Uniform 4x4 | 183.6 | 224.0 | 409.9 | 2026.1 | 7.1% | 26.2% |
| `poly_deg5_4x4` | 5 | 21 | 4x4 subgrid | Adaptive 3mm | 290.5 | 380.5 | 766.4 | 2077.4 | 4.4% | 16.1% |
| `poly_deg12_4x4` | 12 | 91 | 4x4 subgrid | Uniform 4x4 | **140.5** | **170.8** | **322.9** | **932.2** | **13.0%** | **39.5%** |
| `poly_deg12_4x4` | 12 | 91 | 4x4 subgrid | Adaptive 3mm | **217.5** | **273.6** | **544.2** | **1551.9** | **7.1%** | **23.3%** |
| `poly_deg5_sparse_lasso` | 5 | 13 dN / 11 dE | 11 km nodes | Uniform 4x4 | 230.4 | 281.3 | 551.7 | 2024.6 | 5.9% | 19.1% |
| `poly_deg5_sparse_lasso` | 5 | 13 dN / 11 dE | 11 km nodes | Adaptive 3mm | 280.3 | 354.7 | 697.9 | 2075.5 | 4.7% | 15.2% |
| `poly_deg12_sparse_lasso` | 12 | 21 dN / 25 dE | 11 km nodes | Uniform 4x4 | 205.9 | 251.4 | 475.7 | 1618.2 | 5.8% | 22.6% |
| `poly_deg12_sparse_lasso` | 12 | 21 dN / 25 dE | 11 km nodes | Adaptive 3mm | 269.3 | 341.6 | 687.7 | 1704.0 | 4.2% | 17.1% |

---

## 4. Key Scientific Findings & Takeaways

1. **Curvature Upgrades the Geometric Baseline**:
   - Upgrading from a flat plane ($D=1$, $42.8\,\text{cm}$) to a quadratic bowl ($D=2$, $33.0\,\text{cm}$) immediately reduces nationwide error by **$23\%$** by adding just 3 parameters per axis ($X^2, XY, Y^2$).
   - Higher odd degrees ($D=3, 5$) introduce inflection S-curves, reducing uniform RMSE to **$27.0\,\text{cm}$** and slashing worst-case maximum error from **$2.60\,\text{m} \to 1.84\,\text{m}$**.
   - Pushing to Degree 12 achieves **$19.2\,\text{cm}$ uniform RMSE** and cuts maximum nationwide error below 1 meter (**$89.3\,\text{cm}$**).
2. **Minimalist Sparse Formula (Lasso Feature Selection)**:
   - L1 regularization prunes Degree 5 from 21 terms down to **13 terms for $\Delta N$ and 11 terms for $\Delta E$**, maintaining **$28.1\,\text{cm}$ uniform RMSE** ($95\%$ of full polynomial capacity).
   - Analysis of selected terms reveals that cross-terms ($u^3 v, u v^3, u v$) and Easting powers ($v^3, v^5$) dominate, directly reflecting the sharp West-to-East distortion drop in Romania.
3. **Asymptotic Limits of Pure Global Polynomials**:
   - The scaling curve demonstrates diminishing returns past degree 10: uniform RMSE plateaus around **$17\text{--}18\,\text{cm}$**.
   - Because monomials are non-local (global), fitting sharp localized curvature (Banat dome) introduces ripple oscillations (Runge's phenomenon) along national borders.
   - This validates why classical geodesy historically preferred 3rd or 5th-degree surfaces: they capture broad curvature while avoiding edge instabilities.
4. **Bridge to Level 3 (Radial Basis Functions / Splines)**:
   - Global polynomial surfaces succeed at capturing the macroscopic curved trend of the country.
   - Pushing to centimeter or millimeter precision requires localized basis functions (like RBF / Thin Plate Splines) that can bend locally without perturbing distant regions.

---

## 5. Visual Diagnostics

* **3D Perspective Surface Progression** ([`plots/polynomial_3d_surfaces_comparison.png`](../../experiments/polynomial_surface_regressor/plots/polynomial_3d_surfaces_comparison.png)): 6-panel 3D perspective visualization showing the physical surface bending from flat plane (Deg 1) to quadratic bowl (Deg 2), cubic S-curve (Deg 3), quintic undulation (Deg 5), and high-order fit (Deg 12) vs. Ground Truth.
* **2D Plan-View Distortion Contours** ([`plots/polynomial_2d_heatmaps_comparison.png`](../../experiments/polynomial_surface_regressor/plots/polynomial_2d_heatmaps_comparison.png)): 6-panel plan-view contour map across Romania ($0.25\,\text{m}$ to $2.50\,\text{m}$ contours).
* **Degree Scaling Law** ([`plots/polynomial_degree_scaling.png`](../../experiments/polynomial_surface_regressor/plots/polynomial_degree_scaling.png)): Quantitative decay and asymptotic plateau curve across degrees $1 \dots 14$.
* **66 km Multi-Cell Transect** ([`plots/cell_transect_66km_polynomial.png`](../../experiments/polynomial_surface_regressor/plots/cell_transect_66km_polynomial.png)): 1D cross-section comparing polynomial interpolation across 6 SPG cells against bicubic ground truth.
