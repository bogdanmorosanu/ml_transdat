# Research Unit 07 Report: 2D Polynomial Surface Regressor

## 1. Executive Summary & Core Objective

Research Unit 07 investigates whether **classical 2D bivariate polynomial surfaces** can upgrade our foundational geometric baseline—moving from the **flat rigid plane** of the 2D Helmert transformation ($44.3\,\text{cm}$ RMSE floor) to a **curved mathematical surface** capable of absorbing nationwide regional distortion trends.

All polynomial models were evaluated against our two standardized continuous nationwide benchmarks:
1. **`RO_TEST_UNIFORM_4X4`** (36,752 off-lattice stratified points across all 2,297 valid cells): Unbiased area-weighted national survey accuracy.
2. **`RO_TEST_ADAPTIVE_3MM_MID`** (176,350 midpoints concentrated in high-curvature zones): Mathematical fidelity in steep distortion gradients (Banat dome and Maramureș shear).

```
Geometric Progression Beyond the Flat Plane:
  [Helmert 2D Floor] ──> [Quadratic Bowl (Deg 2)] ──> [Cubic Surface (Deg 3)] ──> [Quintic Surface (Deg 5)] ──> [High-Order (Deg 12)]
      (Flat Board)               (6 Terms)                 (10 Terms)                 (21 Terms)                  (91 Terms)
   RMSE: 42.8 - 47.8 cm       RMSE: 33.0 - 42.1 cm       RMSE: 29.5 - 37.3 cm       RMSE: 27.0 - 34.9 cm        RMSE: 19.2 - 26.9 cm
   Max:  2.55 - 2.60 m        Max:  2.63 - 2.69 m        Max:  2.38 - 2.43 m        Max:  1.84 - 1.89 m         Max:  0.89 - 1.34 m
```

---

## 2. Mathematical Architecture

Coordinates are normalized to centered aspect-ratio preserved coordinates $(u, v) \in [-1, 1]$:
$$u = \frac{N - N_0}{S}, \quad v = \frac{E - E_0}{S}$$
where $S = \frac{1}{2} \max(\text{span}_N, \text{span}_E)$.

Two independent polynomial surfaces of degree $D$ are fitted for the 2D shifts using Ridge regularization ($\alpha = 10^{-8}$):
$$\Delta N(u, v) = \sum_{p+q \le D} c^{(n)}_{pq} u^p v^q, \quad \Delta E(u, v) = \sum_{p+q \le D} c^{(e)}_{pq} u^p v^q$$
The number of monomials per axis is exactly $\frac{(D+1)(D+2)}{2}$.

---

## 3. Results & Performance Scorecard

All errors are reported in millimeters ($\text{mm}$) on continuous points never seen during training:

| Model Architecture | Degree (D) | Terms / Axis | Training Grid | Test Dataset | Mean 2D (mm) | RMSE 2D (mm) | P95 (mm) | Max 2D (mm) | < 5 cm (%) | < 10 cm (%) |
| :--- | :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`poly_deg1_11km`** | 1 | 3 | 11 km nodes | Uniform 4x4 | 360.0 | 427.6 | 804.1 | 2550.9 | 1.9% | 7.9% |
| **`poly_deg1_11km`** | 1 | 3 | 11 km nodes | Adaptive 3mm | 388.5 | 478.1 | 886.9 | 2602.9 | 2.1% | 8.0% |
| **`poly_deg2_11km`** | 2 | 6 | 11 km nodes | Uniform 4x4 | 271.2 | 330.1 | 637.9 | 2632.1 | 4.0% | 14.4% |
| **`poly_deg2_11km`** | 2 | 6 | 11 km nodes | Adaptive 3mm | 325.8 | 421.5 | 787.4 | 2685.8 | 2.6% | 10.6% |
| **`poly_deg3_11km`** | 3 | 10 | 11 km nodes | Uniform 4x4 | 246.6 | 294.8 | 559.9 | 2377.9 | 4.3% | 15.6% |
| **`poly_deg3_11km`** | 3 | 10 | 11 km nodes | Adaptive 3mm | 291.8 | 372.7 | 668.3 | 2430.4 | 3.6% | 12.8% |
| **`poly_deg5_11km`** | 5 | 21 | 11 km nodes | Uniform 4x4 | 224.9 | 270.0 | 512.6 | 1839.4 | 4.4% | 17.6% |
| **`poly_deg5_11km`** | 5 | 21 | 11 km nodes | Adaptive 3mm | 277.5 | 349.2 | 678.0 | 1885.8 | 3.7% | 14.3% |
| **`poly_deg8_11km`** | 8 | 45 | 11 km nodes | Uniform 4x4 | 176.3 | 216.1 | 408.6 | 1112.4 | 8.8% | 29.3% |
| **`poly_deg8_11km`** | 8 | 45 | 11 km nodes | Adaptive 3mm | 251.7 | 316.2 | 632.2 | 1650.8 | 5.5% | 19.0% |
| **`poly_deg12_11km`** | 12 | 91 | 11 km nodes | Uniform 4x4 | **158.9** | **191.8** | **369.4** | **893.3** | **9.9%** | **32.3%** |
| **`poly_deg12_11km`** | 12 | 91 | 11 km nodes | Adaptive 3mm | **219.2** | **269.4** | **529.2** | **1343.1** | **6.1%** | **21.2%** |
| **`poly_deg5_4x4`** | 5 | 21 | 4x4 subgrid | Uniform 4x4 | 183.6 | 224.0 | 409.9 | 2026.1 | 7.1% | 26.2% |
| **`poly_deg5_4x4`** | 5 | 21 | 4x4 subgrid | Adaptive 3mm | 290.5 | 380.5 | 766.4 | 2077.4 | 4.4% | 16.1% |
| **`poly_deg12_4x4`** | 12 | 91 | 4x4 subgrid | Uniform 4x4 | **140.5** | **170.8** | **322.9** | **932.2** | **13.0%** | **39.5%** |
| **`poly_deg12_4x4`** | 12 | 91 | 4x4 subgrid | Adaptive 3mm | **217.5** | **273.6** | **544.2** | **1551.9** | **7.1%** | **23.3%** |
| **`poly_deg5_sparse_lasso`** | 5 | 13 dN / 11 dE | 11 km nodes | Uniform 4x4 | 230.4 | 281.3 | 551.7 | 2024.6 | 5.9% | 19.1% |
| **`poly_deg5_sparse_lasso`** | 5 | 13 dN / 11 dE | 11 km nodes | Adaptive 3mm | 280.3 | 354.7 | 697.9 | 2075.5 | 4.7% | 15.2% |
| **`poly_deg12_sparse_lasso`** | 12 | 21 dN / 25 dE | 11 km nodes | Uniform 4x4 | 205.9 | 251.4 | 475.7 | 1618.2 | 5.8% | 22.6% |
| **`poly_deg12_sparse_lasso`** | 12 | 21 dN / 25 dE | 11 km nodes | Adaptive 3mm | 269.3 | 341.6 | 687.7 | 1704.0 | 4.2% | 17.1% |

---

## 4. Key Scientific Insights

### A. The Curvature Breakthrough: Bending the Helmert Plane
1. **Degree 1 $\to$ Degree 2 (Quadratic Bowl)**:
   - Introducing quadratic terms ($X^2, XY, Y^2$) bends the flat Helmert board into a nationwide paraboloid and hyperbolic saddle.
   - This alone drops uniform nationwide RMSE from **$42.8\,\text{cm} \to 33.0\,\text{cm}$**—a **$23\%$ reduction in error** achieved by adding just 3 extra parameters per axis.
2. **Degree 2 $\to$ Degree 3 & 5 (S-Curves and Regional Waves)**:
   - Odd-degree polynomials introduce inflection points, allowing the surface to rise in the Western border and dip toward the Black Sea.
   - Degree 3 drops uniform RMSE to **$29.5\,\text{cm}$**.
   - Degree 5 drops uniform RMSE to **$27.0\,\text{cm}$** and slashes worst-case max error by **$76\,\text{cm}$** ($2.60\,\text{m} \to 1.84\,\text{m}$).
3. **Degree 12 (High-Order Capacity)**:
   - Reaches **$19.2\,\text{cm}$ uniform RMSE** and brings nationwide maximum error below 1 meter (**$89.3\,\text{cm}$**) using 91 parameters.

### B. The Minimalist Formula Analysis (L1 Lasso Feature Selection)
Using L1 Lasso regularization on Degree 5 and Degree 12 reveals the **mathematical core** of Romania's distortion field:
* **Degree 5 Sparsity**: Pruning from 21 terms down to **13 terms for $\Delta N$** and **11 terms for $\Delta E$** achieves **$28.1\,\text{cm}$ uniform RMSE** (within $1.1\,\text{cm}$ of the full 21-term Ridge model).
* **Dominance of Cross-Terms and Easting Powers**:
  - The highest-magnitude terms are cross-terms $u^3 v, u v^3, u v$ (torsional twists) and odd Easting powers $v^3, v^5$.
  - In Romania, the distortion drops steeply from West to East ($2.76\,\text{m} \to 0.2\,\text{m}$), meaning Easting terms ($v$) require higher-order flexibility than North-South terms ($u$).
* **Exact Minimalist Degree 5 Equation**:
  $$\Delta N(u, v) = +4.01 u^3 v + 2.91 u v^3 - 2.26 u v - 1.38 v^2 - 1.27 u^4 + 1.25 v^4 - 1.24 u^2 v + 0.79 u^2 - 0.65 u^3 + 0.42 u + 0.20 v^3 + 0.13 + 0.12 v$$
  $$\Delta E(u, v) = +1.00 u v + 0.51 v^3 + 0.40 u^2 - 0.36 u v^3 - 0.31 u^4 - 0.25 u^3 v - 0.22 v - 0.13 u^5 - 0.02 v^2 - 0.01 + 0.01 u$$
  This single clean pair of formulas drops Helmert's $42.8\,\text{cm}$ error down to $28.1\,\text{cm}$ using only hand-writable polynomial monomials!

### B. The Asymptotic Limit of Global Polynomials & Runge's Phenomenon
* **The Asymptotic Ceiling**: As shown in the scaling analysis, pushing polynomial degree from $D=12$ up to $D=16\dots 20$ yields minimal gain (stabilizing around $\sim 17 - 18\,\text{cm}$ nationwide).
* **The Non-Local Problem**: A polynomial is inherently a global function. If you adjust a high-order coefficient to fit the steep local distortion dome in Banat, that monomial has non-zero values across the entire country, creating unintentional ripples (Runge-like oscillations) in Dobrogea and along national borders.
* **Why Classical Geodesy Chose Degrees 3 to 5**: This explains why national mapping agencies historically stopped around 3rd or 5th-degree surfaces: degrees 3–5 capture almost all macroscopic nationwide curvature ($27\text{--}29\,\text{cm}$) without introducing dangerous edge oscillations.

### C. The Bridge to Level 3: Radial Basis Functions (RBF)
* Global polynomials excel at establishing the broad **curved macroscopic foundation** of the country.
* However, sub-centimeter or sub-decimeter precision requires **localized degrees of freedom** that can bend locally without affecting distant regions.
* This is precisely what **Radial Basis Functions (RBF) and Thin Plate Splines (TPS)** provide: they mathematically combine this exact low-order polynomial trend with localized radial bumps!

---

## 5. Visual Diagnostics

### 1. 2D Polynomial Surface Degree Scaling Curve
![Polynomial Degree Scaling](plots/polynomial_degree_scaling.png)
The scaling curve confirms a steep exponential error reduction from Degree 1 ($42.8\,\text{cm}$) through Degree 8 ($21.6\,\text{cm}$), followed by asymptotic leveling off near $\sim 18\,\text{cm}$ as degree approaches 14.

### 2. 9-Cell (99 km) Extreme Scenario Transects
![Gradient & Linear Extreme Scenarios](plots/extreme_scenarios_gradient_and_linear_9cells.png)
![Curvature & Kink Extreme Scenarios](plots/extreme_scenarios_curvature_and_kinks_9cells.png)
The transect clearly illustrates the geometric progression:
* **Degree 1 (Flat Plane)** cuts straight through the landscape with zero curvature.
* **Degrees 2 and 3** introduce macroscopic bowing that approximates the general trend line.
* **Degree 5 and Degree 12** track regional undulations much more closely, but cannot replicate the local high-frequency peaks of the bicubic ground truth.

### 3. 3D Perspective Surfaces Comparison (2x3 Grid)
![3D Surfaces Comparison](plots/polynomial_3d_surfaces_comparison.png)
The 3D perspective surface visualizes the actual 3D elevation of 2D distortion magnitude across Romania:
* **Panel 1 (Ground Truth)** shows the true topography: the steep $2.76\,\text{m}$ dome in Banat/West, the sharp Maramureș slope, and the valleys.
* **Panel 2 (Degree 1)** shows the rigid flat board (Helmert equivalent).
* **Panel 3 (Degree 2)** bends the board into a paraboloid bowl and hyperbolic saddle.
* **Panel 4 (Degree 3)** adds inflection S-curves.
* **Panel 5 (Degree 5)** introduces multi-wave undulations.
* **Panel 6 (Degree 12)** achieves a detailed macroscopic fit, but remains smooth compared to the localized ground-truth peaks.

### 4. 2D Plan-View Heatmaps Comparison (2x3 Grid)
![2D Heatmaps Comparison](plots/polynomial_2d_heatmaps_comparison.png)
The 2D plan-view contour map shows the geographic contours and contours of distortion across Romania with consistent color scale and contour lines ($0.25\,\text{m}$ to $2.50\,\text{m}$).

---

## 6. Benchmark Candidates
Promoted to `benchmarks/global_analytical_models/` via `specs_from_unit(PolynomialSurfaceRegressorUnit, suite="global_analytical_models")`:

| Variant | Benchmark Label | Train Grid | Exact Hyperparameters |
| :--- | :--- | :--- | :--- |
| `poly_deg1_11km` | `Linear Affine Plane (Deg 1, 6T)` | `spg_nodes_valid` | `degree=1, alpha=1e-8` |
| `poly_deg2_11km` | `Quadratic Bowl (Deg 2, 12T)` | `spg_nodes_valid` | `degree=2, alpha=1e-8` |
| `poly_deg3_11km` | `Cubic Surface (Deg 3, 20T)` | `spg_nodes_valid` | `degree=3, alpha=1e-8` |
| `poly_deg5_11km` | `Quintic Surface (Deg 5, 42T)` | `spg_nodes_valid` | `degree=5, alpha=1e-8` |
| `poly_deg8_11km` | `Multi-Harmonic (Deg 8, 90T)` | `spg_nodes_valid` | `degree=8, alpha=1e-8` |
| `poly_deg12_11km` | `High-Order Limit (Deg 12, 182T)` | `spg_nodes_valid` | `degree=12, alpha=1e-8` |
| `poly_deg5_sparse_lasso` | `Sparse Lasso (Deg 5, 24T)` | `spg_nodes_valid` | `degree=5, alpha=5e-4, regressor="lasso"` |
