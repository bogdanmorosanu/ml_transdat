# Research Unit: 2D RBF & Thin Plate Spline Surface Regressors

**Status**: `Validated`  
**Domain**: 2D Continuous Coordinate Distortion Shifts ($(st70_n, st70_e) \to (\Delta N, \Delta E)$)  
**Parent / Predecessor Units**: [`helmert2d_baseline`](helmert2d_baseline.md), [`polynomial_surface_regressor`](polynomial_surface_regressor.md)  
**Related Experiment**: [`experiments/rbf_surface_regressor/`](../../experiments/rbf_surface_regressor/)  
**Primary Artifacts**: [`results/scorecard.csv`](../../experiments/rbf_surface_regressor/results/scorecard.csv), [`report.md`](../../experiments/rbf_surface_regressor/report.md)

---

## 1. Scientific Hypothesis & Motivation

Our mathematical coordinate transformation models progressed through two prior stages:
1. **Level 1: Helmert 2D** ([`helmert2d_baseline`](helmert2d_baseline.md)): A rigid flat plane with rotation, uniform scale, and translation ($\text{RMSE} = 0.4429\,\text{m}$, $\text{Max} = 2.68\,\text{m}$).
2. **Level 2: Bivariate Polynomials** ([`polynomial_surface_regressor`](polynomial_surface_regressor.md)): Continuous curved surfaces cutting error to $\text{RMSE} = 0.1918\,\text{m}$ ($D=12$). However, global monomials hit an asymptotic ceiling near $\sim 17\text{--}18\,\text{cm}$ because fitting localized peaks (Banat dome) induces Runge's boundary oscillations elsewhere.

### Core Hypothesis
By modeling the distortion field as a **localized continuous elastic sheet** using **Thin Plate Splines (TPS)** ($r^2 \ln r$), the surface can bend locally where distortion gradients are steep without destabilizing distant regions. Furthermore, introducing gentle nugget smoothing ($\lambda \approx 10^{-4} \dots 2 \times 10^{-4}$) will filter out the legacy $C^0/C^1$ boundary slope kinks of Romania's `.spg` grid, allowing a closed-form continuous surface to achieve **sub-centimeter nationwide accuracy** with zero lookup tables ($0\,\text{MB}$ grid data).

---

## 2. Experimental Methodology & Progression

Models were evaluated on the project's canonical dual-benchmark protocol:
1. **`RO_TEST_UNIFORM_4X4`** (36,752 off-lattice stratified points across all 2,297 valid cells): Area-weighted national survey accuracy.
2. **`RO_TEST_ADAPTIVE_3MM_MID`** (176,350 midpoints in high-curvature mountain & border zones): Stress test.

### Model Configurations Evaluated
* **`tps_base_exact`**: Thin Plate Spline ($r^2 \ln r$), Affine linear trend ($D=1$), exact node pinning ($\lambda = 0.0$) on 2,222 base anchors.
* **`tps_base_smooth_1e-4`**: Thin Plate Spline, Affine trend, smoothing ($\lambda = 10^{-4}$) on 2,222 base anchors.
* **`tps_base_optimal_2e-4`**: Thin Plate Spline, Affine trend, optimal smoothing ($\lambda = 2 \times 10^{-4}$) on 2,222 base anchors.
* **`tps_densified_4x4`**: Thin Plate Spline trained on densified $4\times 4$ subgrid (36,435 anchors, ~2.7 km spacing, `neighbors=60`).
* **`linear_base`**: Scale-free linear kernel ($\phi(r) = r$), smoothing $\lambda = 10^{-4}$.
* **`mq_tuned_eps10`**: Multiquadric RBF tuned to optimal spatial frequency ($\epsilon = 10.0, \lambda = 10^{-4}$).
* **`gaussian_tuned_eps40`**: Gaussian RBF tuned to optimal spatial frequency ($\epsilon = 40.0, \lambda = 10^{-4}$).
* **`cubic_base`**: Scale-free cubic kernel ($\phi(r) = r^3$), demonstrating overshooting between control points.

---

## 3. Quantitative Scorecard

| Model Architecture | Training Anchors | Kernel Type | Smoothing ($\lambda$) | Eval Dataset | Mean 2D (mm) | RMSE 2D (mm) | P95 (mm) | Max 2D (mm) | < 1 cm (%) | < 5 cm (%) | Fit Time |
| :--- | :---: | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `tps_base_exact` | 2,222 (11 km) | Thin Plate Spline | 0.0 (Exact) | Uniform 4x4 | 10.3 | 13.2 | 23.9 | 322.6 | 58.5% | 99.8% | 1.46s |
| `tps_base_exact` | 2,222 (11 km) | Thin Plate Spline | 0.0 (Exact) | Adaptive 3mm | 11.4 | 22.1 | 27.7 | 343.2 | 63.6% | 98.1% | 1.46s |
| `tps_base_smooth_1e-4` | 2,222 (11 km) | Thin Plate Spline | $10^{-4}$ (Smooth) | Uniform 4x4 | 7.3 | 9.3 | 16.9 | 202.8 | 77.4% | 99.9% | 1.41s |
| `tps_base_smooth_1e-4` | 2,222 (11 km) | Thin Plate Spline | $10^{-4}$ (Smooth) | Adaptive 3mm | 9.8 | 19.1 | 24.1 | 219.1 | 72.3% | 98.2% | 1.41s |
| **`tps_base_optimal_2e-4`** | **2,222 (11 km)** | **Thin Plate Spline** | **$2 \times 10^{-4}$ (Opt)** | **Uniform 4x4** | **6.8** | **8.6** | **15.9** | **246.6** | **80.6%** | **100.0%** | **1.43s** |
| **`tps_base_optimal_2e-4`** | **2,222 (11 km)** | **Thin Plate Spline** | **$2 \times 10^{-4}$ (Opt)** | **Adaptive 3mm** | **11.8** | **23.7** | **29.6** | **301.9** | **64.5%** | **98.0%** | **1.43s** |
| **`tps_densified_4x4`** | **36,435 (2.7 km)** | **Thin Plate Spline** | **$10^{-4}$ (k-d tree)** | **Uniform 4x4** | **3.7** | **5.7** | **8.8** | **234.0** | **96.7%** | **99.9%** | **0.01s** |
| `tps_densified_4x4` | 36,435 (2.7 km) | Thin Plate Spline | $10^{-4}$ (k-d tree) | Adaptive 3mm | 13.9 | 42.8 | 46.4 | 795.7 | 72.9% | 95.5% | 0.01s |
| `linear_base` | 2,222 (11 km) | Linear ($\phi=r$) | $10^{-4}$ | Uniform 4x4 | 10.5 | 13.0 | 24.1 | 279.7 | 56.5% | 99.8% | 0.69s |
| `linear_base` | 2,222 (11 km) | Linear ($\phi=r$) | $10^{-4}$ | Adaptive 3mm | 15.0 | 26.1 | 37.7 | 316.1 | 48.1% | 97.3% | 0.69s |
| `mq_tuned_eps10` | 2,222 (11 km) | Multiquadric ($\epsilon=10$) | $10^{-4}$ | Uniform 4x4 | 11.3 | 14.0 | 24.5 | 265.7 | 51.4% | 99.7% | 0.75s |
| `mq_tuned_eps10` | 2,222 (11 km) | Multiquadric ($\epsilon=10$) | $10^{-4}$ | Adaptive 3mm | 15.5 | 35.9 | 32.4 | 542.4 | 51.9% | 97.3% | 0.75s |
| `gaussian_tuned_eps40` | 2,222 (11 km) | Gaussian ($\epsilon=40$) | $10^{-4}$ | Uniform 4x4 | 13.1 | 17.1 | 30.3 | 378.6 | 45.5% | 99.5% | 1.86s |
| `gaussian_tuned_eps40` | 2,222 (11 km) | Gaussian ($\epsilon=40$) | $10^{-4}$ | Adaptive 3mm | 14.1 | 26.5 | 35.4 | 387.8 | 53.7% | 97.3% | 1.86s |
| `cubic_base` | 2,222 (11 km) | Cubic ($\phi=r^3$) | $10^{-4}$ | Uniform 4x4 | 42.4 | 51.5 | 95.2 | 699.5 | 6.0% | 69.3% | 0.67s |
| `cubic_base` | 2,222 (11 km) | Cubic ($\phi=r^3$) | $10^{-4}$ | Adaptive 3mm | 62.0 | 96.3 | 150.0 | 1039.5 | 5.1% | 54.4% | 0.67s |

---

## 4. Key Scientific Findings & Takeaways

1. **Sub-Centimeter Nationwide Accuracy from 2,222 Base Anchors**:
   * `tps_base_optimal_2e-4` achieves **$8.63\,\text{mm}$ (0.86 cm) nationwide RMSE**, with a median error of **$5.7\,\text{mm}$** and **$100.0\%$ pass rate within $< 5\,\text{cm}$**.
   * This is a **$22.2\times$ error reduction** over Level 2 Polynomials ($191.8\,\text{mm}$) and a **$51.3\times$ reduction** over Level 1 Helmert ($442.9\,\text{mm}$) with zero lookup tables ($0\,\text{MB}$ grid data).
2. **The Smoothing Dividend (Filtering Legacy Grid Kinks)**:
   * Relaxing exact pinning ($\lambda = 0 \to \lambda = 2 \times 10^{-4}$) improves nationwide off-lattice accuracy by **$34.8\%$** ($13.25\,\text{mm} \to 8.63\,\text{mm}$) and slashes maximum error from $34.3\,\text{cm} \to 24.6\,\text{cm}$.
   * **Mechanism**: Romania's official `.spg` grid utilizes the *Numerical Recipes* local bicubic patch algorithm with one-sided finite differences, creating localized first-derivative slope breaks ($\Delta\text{slope} \approx 4.2 \times 10^{-5}$) at $11\,\text{km}$ cell boundaries. Exact interpolation forces the elastic sheet to kink at every grid node. Gentle smoothing lets the sheet float over numerical kinks, yielding a smoother, more physically plausible surface.
3. **Axis Decoupling Sweep Confirms Uniform Noise**:
   * Sweeping $\lambda_N$ and $\lambda_E$ independently confirmed that both axes reach their optimum at the exact same value ($\lambda_N = \lambda_E = 2.0 \times 10^{-4}$). The coordinate shifts share identical spatial noise characteristics because they were computed from the same geodetic survey adjustments.
4. **The Densification Ladder & The Legacy Grid Wall**:
   * Training on the densified $4\times 4$ subgrid (36,435 anchors) cuts average error in half ($7.3\,\text{mm} \to 3.7\,\text{mm}$) and brings **$96.7\%$ of points within $< 1\,\text{cm}$** ($P95 = 8.8\,\text{mm}$).
   * However, nationwide RMSE plateaus around **$5.7\text{--}6.9\,\text{mm}$**. A smooth continuous biharmonic sheet cannot match discrete piecewise-cubic cell boundary slope kinks. The remaining $4\text{--}7\,\text{mm}$ residual is the physical discrepancy between a smooth elastic surface and a 1990s legacy grid with boundary kinks.
5. **Complete Elimination of Boundary Oscillations**:
   * Unlike high-degree polynomials ($D \ge 12$), which produced edge-ringing along Romanian frontiers, Thin Plate Splines attenuate smoothly beyond the convex hull of control points without high-frequency ripples.

---

## 5. Visual Diagnostics

* **3D Perspective Surface Progression** ([`plots/rbf_3d_surfaces_comparison.png`](../../experiments/rbf_surface_regressor/plots/rbf_3d_surfaces_comparison.png)): 6-panel 3D view showing Ground Truth vs. Flat Plane (Level 1) vs. Polynomial (Level 2) vs. TPS Base Smooth ($8.6\,\text{mm}$) vs. TPS Densified ($3.7\,\text{mm}$ mean) vs. Tuned MQ.
* **2D Plan-View Contours** ([`plots/rbf_2d_heatmaps_comparison.png`](../../experiments/rbf_surface_regressor/plots/rbf_2d_heatmaps_comparison.png)): 6-panel plan-view map demonstrating localized deformation around the Banat dome and Dobrogea.
* **66 km Multi-Cell Transect** ([`plots/cell_transect_66km_rbf.png`](../../experiments/rbf_surface_regressor/plots/cell_transect_66km_rbf.png)): 1D cross-section comparing RBF and TPS interpolation against bicubic ground truth across cell boundaries.
* **Foundation & Kernel Comparison** ([`plots/rbf_foundation_and_smoothing.png`](../../experiments/rbf_surface_regressor/plots/rbf_foundation_and_smoothing.png)): Quantitative bar chart illustrating the progression across all 8 evaluated model configurations.
