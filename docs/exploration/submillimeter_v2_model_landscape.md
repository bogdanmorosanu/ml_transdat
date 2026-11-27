# Sub-Millimeter Exploration v2: Exhaustive Model Landscape & Pre-Finalization Roadmap

- **Status**: `Active`
- **Target Goal**: Identify and screen every plausible remaining model family (neural and non-neural) for **sub-millimeter (`< 1 mm`)** `ETRS89 ↔ Stereo 70` distortion modeling before locking in the final champion for fine-tuning and publication.
- **Environment Constraints**: `torch 2.14.1+cu126` (CUDA enabled), `scipy`, `scikit-learn`, `xgboost`, `numpy`, `pandas` are already installed in `ml_transdat`. All 12 proposed tracks below require **zero new package installations**.
- **Primary References**: [`docs/research/roadmap.md`](../research/roadmap.md), [`docs/exploration/submillimeter_strategy_report.md`](submillimeter_strategy_report.md), [`experiments/hybrid_surface_featgrid/report.md`](../../experiments/hybrid_surface_featgrid/report.md), [`experiments/hybrid_surface_resnet/report.md`](../../experiments/hybrid_surface_resnet/report.md), [`literature/README.md`](../../literature/README.md)

---

## 1. Where We Stand Post-`hybrid_surface_featgrid` & `hybrid_surface_resnet`

Across 20 research units, we broke the `1 mm` barrier in the interior of Romania using two-stage residual decomposition (`Stage 1 smooth surface + Stage 2 Huber-trained residual corrector on RO_GRID_32X32`):

| Deployment Tier | Current Champion | `16x16` Mean / `< 1 mm` | `Adapt 1mm` Mean / `< 1 mm` | `Adapt 1mm` Max | Lookup / Total Size | Key Bottleneck |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Compact Micro-Grid Hybrid** | `cubic_k25_s0 + FeatGrid (32x32, h128, Huber)` | **`0.24 mm` / `99.3%`** | `1.50 mm` / **`84.6%`** | `0.651 m` | `0.66 MB` / `3.55 MB` | `~0.65 m` concave-border Stage-1 RBF spikes |
| **Zero-Lookup Neural Hybrid** | `lpoly_deg2_k16_tric + ResNet (32x32, h256, MS-RFF, SiLU, Huber)` | `0.31 mm` / `98.39%` | `1.66 mm` / `79.31%` | — | **`0 MB`** / `4.08 MB` | Steep adaptive zones (`79.3%` vs `84.6%`) + `15.3 mm` RMSE |
| **Pure Non-Neural Lookup** | `Adaptive Quadtree (1mm tier)` | `0.50 mm` / `93.8%` *(corridor)* | — | `5.77 mm` *(corridor)* | `74.9 MB` | Large lookup table |
| **Pure Non-Neural Continuous** | `Matérn 3/2 RBF (4x4)` / `TPS (4x4)` | `4.66 mm` (`U4`) / `3.49 mm` *(corridor)* | `19.1 mm` (`TPS`) | `247 mm` | `~3 MB` / `0 MB` | Smooths over `11 km` $C^1$ kinks; slow evaluation |

### Remaining Scientific Gaps Before Champion Finalization
1. **Concave-Border Stage-1 Extrapolation (`0.65–0.69 m` spikes)**: `cubic_k25_s0` and `lpoly_deg2_k16_tric` extrapolate sharply outside Romania's concave border, and `Huber` loss down-weights those rare spikes (`14.8–15.3 mm` `Adapt 1mm` RMSE vs. `5.42 mm` for standalone `FeatGrid_32x32_only`).
2. **High-Curvature Adaptive Pass Rate (`79–85% < 1 mm`)**: Pushing `Adapt 1mm` `< 1 mm` pass rate toward `> 95%` without exploding model size.
3. **Model-Family Completeness**: Before declaring a final winner, we want scientific data validating or invalidating modern coordinate INRs (`KAN`, `WIRE`, `FINER`, `MFN`, `FBPINN`/`MoE`), structured geometric heads (`Neural Bicubic`, `K-Planes`, `Hash/Multi-Res Grids`), and **non-neural** ML/geostatistical methods (`ELM`, `LS-SVM`, `XGBoost` on Stage-2 residuals, `GPR/Kriging`, `PU-RBF`, `Wavelets`).

---

## 2. Complete Audit of Tested Paradigms (`[x]` Checklist)

- [x] **Global Linear / Conformal Baselines (`pyproj_baseline`, `helmert2d_baseline`)**: `43.5–44.3 cm` RMSE; `StereoProj + Helmert2D` locked in as exact analytically invertible Stage-0 geometric base.
- [x] **Global Bivariate Polynomials (`polynomial_surface_regressor`, deg 1–12)**: `42.8 cm → 19.2 cm` RMSE; Runge's edge oscillation ceiling (`Dead End` for Stage-1).
- [x] **Tensor-Product B-Splines (`optimal_surface_regressor`)**: `904.8 mm` learnability score due to `57 m` rectangular-domain corner spikes (`Dead End`).
- [x] **K-Nearest Neighbors (`knn_spatial_regressor`, K=1..16)**: `35.3 mm` RMSE with `K=4` IDW; reveals $C^0/C^1$ cell boundary kinks.
- [x] **Decision Trees (`decision_tree_regressor`)**: `81.7 mm` (`11 km`) → `4.0 mm` (`16x16` at `89.4 MB`); axis-aligned step plateaus.
- [x] **Adaptive Quadtree (`adaptive_quadtree_grid`)**: `0.50 mm` mean (`93.8% < 1 mm`) at `1 mm` tier (`74.9 MB` lookup); pure geometric baseline.
- [x] **11 km Analytical Splines (`bicubic_interpolation_forensics`)**: `Catmull-Rom`, `PCHIP`, `Natural Spline C²`; proves `5.6×–11.5×` smooth-ML learnability penalty from `TransDatRO` $C^1$ kinks.
- [x] **Localized Analytical Surfaces (`rbf_surface_regressor`, `optimal_surface_regressor`)**: `TPS`, `Polyharmonic (cubic)`, `Matérn 3/2 & 5/2`, `Multiquadric/IMQ`, `Gaussian RBF`, `Local Polynomial LOESS (deg=2)`; `cubic_k25_s0`, `tps_deg2_k25_s1e-5`, and `lpoly_deg2_k16_tric` shortlisted as Top-3 Stage-1 bases (`16.0–16.7 mm` learnability).
- [x] **Fourier MLP & PyTorch ResNet (`mlp_spatial_regressor`, `torch_mlp_regressor`)**: RFF + cyclic `(u, v)` cuts uniform RMSE to `6.8 mm` (`4x4`).
- [x] **Discrete Shape & 3×3 Neighbor Feature Concatenation (`mlp_shape_features`, `mlp_neighbor_features`)**: Degraded Fourier MLP (`Dead End`) due to discontinuous/high-dim boundary jumps.
- [x] **Standalone `SIREN` (`submillimeter_exploration` T6)**: `449.6 mm` RMSE on raw shifts without RFF or Stage-1 decomposition (`Dead End`).
- [x] **`HistGradientBoostingRegressor` (`submillimeter_exploration` T6)**: `52.4 mm` uniform / `144.3 mm` adaptive RMSE on raw shifts (`Dead End`).
- [x] **Curvature-Adaptive Training + Huber/L-BFGS (`submillimeter_exploration` T2, T4)**: `T4_data_adaptive_hybrid` cut adaptive ResNet RMSE by 94% (`44.18 → 2.59 mm`).
- [x] **Stage-1 Smooth Surface + Stage-2 `NeuralFeatureGridShiftModel` (`hybrid_surface_featgrid`)**: `0.24 mm` `16x16` mean (`99.3% < 1 mm`), exposing `0.651 m` Stage-1 border spikes.
- [x] **Stage-1 Smooth Surface + Stage-2 Grid-Free `TorchMLPGridShiftModel` (`hybrid_surface_resnet`)**: `0.31 mm` `16x16` mean (`98.39% < 1 mm`, `0 MB` lookup) with `lpoly` + `h256` + `MS-RFF` + `SiLU` + `Huber`.

---

## 3. Exhaustive Catalog of Untested Candidate Tracks (`[ ]` Proposed for `submillimeter_exploration_v2`)

### Family I — Modern Implicit Neural Representations (`0 MB` Lookup Grid)

- [ ] **`V2-1`: Kolmogorov-Arnold Networks (`KAN` / `Fourier-KAN` / `RBF-KAN`) as Stage-2 Residual Corrector**
  - **Mechanism**: Replaces fixed node activations (`ReLU`/`SiLU`) with learnable univariate functions on network edges—parameterized via B-splines (original `KAN`), truncated Fourier series (`Fourier-KAN`), or Gaussian RBFs (`Fast-KAN` / `RBF-KAN`).
  - **Literature Grounding**: [R4.3](../../literature/references.md#category-4-modern-implicit-neural-representations-scientific-ml--advanced-spatial-regressors-20232026-supplement) (KAN), [R4.4](../../literature/references.md#category-4-modern-implicit-neural-representations-scientific-ml--advanced-spatial-regressors-20232026-supplement) (Fourier-KAN & MLP-KAN hybrids), [R4.5](../../literature/references.md#category-4-modern-implicit-neural-representations-scientific-ml--advanced-spatial-regressors-20232026-supplement) (GeoKAN).
  - **Fit to `ml_transdat`**: `TransDatRO` intra-cell residuals are smooth low-degree bivariate polynomials (plus `11 km` $C^1$ creases). Spline/RBF edge activations can represent piecewise-cubic shapes with far fewer parameters than deep `SiLU` MLPs, and `GeoKAN`-style coordinate warping aligns naturally with steep distortion zones (Banat/Maramureș).
  - **Honest Doubt / Risk**: B-spline `KAN` layers are noticeably slower per step on GPU than `Linear + SiLU`; `RBF-KAN` or `Fourier-KAN` (pure PyTorch tensor ops) is much faster and avoids needing `pykan`.
  - **Implementation Path**: Pure PyTorch `FourierKANLayer` / `RBFKANLayer` (~40 lines) inside a `Stage2` wrapper on `CompositeResidualShiftModel`.

- [ ] **`V2-2`: Wavelet INRs (`WIRE` — Complex Gabor) & Variable-Periodic (`FINER`) on Stage-2 Residuals**
  - **Mechanism**: `WIRE` uses continuous complex Gabor wavelets $\psi(z) = \exp(j \omega_0 z - |s_0 z|^2)$, simultaneously localizing in both space and frequency. `FINER` uses variable-periodic activations $\sin(\omega_0 (|z| + 1) z)$ to dynamically tune spectral bias without hand-tuning multi-scale RFF bands.
  - **Literature Grounding**: [R4.6](../../literature/references.md#category-4-modern-implicit-neural-representations-scientific-ml--advanced-spatial-regressors-20232026-supplement) (WIRE, CVPR 2023), [R4.7](../../literature/references.md#category-4-modern-implicit-neural-representations-scientific-ml--advanced-spatial-regressors-20232026-supplement) (FINER).
  - **Fit to `ml_transdat`**: Directly tests whether localized wavelets (`WIRE`) or variable-periodic sines (`FINER`) capture `11 km` cell-wall $C^1$ kinks better than `MS-RFF + SiLU` (and rehabilitates periodic INRs after standalone `SIREN` failed on raw `±4.37 m` shifts).
  - **Honest Doubt / Risk**: Complex-valued layers double activation memory (`WIRE`), and periodic activations can ring near steep gradients if `omega_0` is too high.
  - **Implementation Path**: Pure PyTorch activation/layer variants in `src/models/` evaluated on `lpoly_deg2_k16_tric` Stage-2 residuals.

- [ ] **`V2-3`: Multiplicative Filter Networks (`MFN`: `GaborNet` / `FourierNet`) on Stage-2 Residuals**
  - **Mechanism**: Instead of deep compositional activations, `MFN` applies linear layers multiplied element-wise by spatial Fourier or Gabor filters at each stage ($z_{k+1} = (W_k z_k + b_k) \odot g_k(x)$), forming a direct linear combination over an exponential basis of localized filters.
  - **Literature Grounding**: [R4.8](../../literature/references.md#category-4-modern-implicit-neural-representations-scientific-ml--advanced-spatial-regressors-20232026-supplement) (Fathony et al., ICLR 2021).
  - **Fit to `ml_transdat`**: `GaborNet` provides spatially localized frequency filters—ideal for isolated distortion domes (Banat, Maramureș, Dobrogea) without global ringing.
  - **Honest Doubt / Risk**: May need more filter width than `h256` to cover all ~2,200 `11 km` cells across Romania.
  - **Implementation Path**: Pure PyTorch `GaborNet` / `FourierNet` (~50 lines).

- [ ] **`V2-4`: Spatial Mixture-of-Experts (`MoE`) & Domain-Decomposed Overlapping Subdomain MLPs (`FBPINN`-style)**
  - **Mechanism**: `FBPINN` partitions Romania into overlapping 2D spatial subdomains with smooth cosine/bump window weights, assigning a tiny dedicated local MLP to each subdomain. Spatial `MoE` uses a lightweight gating network on $(N, E)$ to route points to $K$ specialized residual experts (e.g., flat plains vs. steep curvature domes vs. borders).
  - **Literature Grounding**: [R4.9](../../literature/references.md#category-4-modern-implicit-neural-representations-scientific-ml--advanced-spatial-regressors-20232026-supplement) (FBPINNs, Spatial MoE).
  - **Fit to `ml_transdat`**: Directly solves the `box10_interior` vs. Full-Romania capacity gap (`h128` worked on 100 cells, Full Romania is ~23× larger) by giving each region its own local basis network without storing a dense `32x32` feature table.
  - **Honest Doubt / Risk**: If `FBPINN` uses too many subdomains (e.g., 10×10), inference requires evaluating overlapping neighbors cleanly; `MoE` routing can suffer load-imbalance without an auxiliary balance loss.
  - **Implementation Path**: Pure PyTorch vectorized `FBPINN` (batched `einsum` over local tile MLPs) and top-2 `SpatialMoE`.

---

### Family II — Structured Neural-Geometric & Tensor Hybrids

- [ ] **`V2-5`: Neural Bicubic / Local Polynomial-Basis Head (`(u, v)` Monomial Parameterizer)**
  - **Mechanism**: Instead of asking a `ReLU`/`SiLU` MLP to approximate products like $u^2 v$ or $u^3 v^2$ inside each `11 km` cell, a compact network (or small feature grid) predicts the **16 bicubic coefficients** (or 10 cubic polynomial coefficients) for `(u, v) ∈ [0, 1]^2` and evaluates the exact polynomial $\sum_{i,j=0}^3 c_{ij} u^i v^j$ analytically.
  - **Literature Grounding**: Extension of `submillimeter_strategy_report.md` Rank 5.2 and `TransDatRO` patch structure ([R3.5](../../literature/references.md#category-3-national-geodetic-grid-interpolation-standards--algorithms)).
  - **Fit to `ml_transdat`**: Matches the *exact* algebraic form of `TransDatRO`'s `Grid.BiInterp` inside every `11 km` cell! A grid-free network predicting cell-level coefficients (keyed by cell center `(N_c, E_c)` + tiny global residual) could theoretically achieve near-exact `0.01–0.1 mm` interior precision.
  - **Honest Doubt / Risk**: Evaluating a global MLP on cell-center coordinates `(N_c, E_c)` still requires learning the discontinuous jump in 3rd derivatives between adjacent cells; however, a small `(53×72)` coefficient grid or cell-center ResNet avoids per-point high-frequency evaluation.
  - **Implementation Path**: Pure PyTorch `Tensor` monomial basis head (`16` outputs per axis `dN, dE`) paired with `TorchMLP` or `NeuralFeatureGrid`.

- [ ] **`V2-6`: Border-Safe Stage-1 Base + Multi-Scale `NeuralFeatureGrid` (3–4 Resolutions / Hash-Style)**
  - **Mechanism**: Addresses the `0.651 m` Stage-1 border spike found in `hybrid_surface_featgrid`. Replaces `cubic_k25_s0` with a **border-safe Stage-1 base** (`CellBilinearShiftModel`, `Catmull-Rom`, or `lpoly` with fallback/clamping) or adds a 3rd/4th finer grid level (e.g. `210×286`) / hash-indexed residual table (`Instant-NGP` style) trained with `Huber` + border-weighted sampling.
  - **Literature Grounding**: [R4.10](../../literature/references.md#category-4-modern-implicit-neural-representations-scientific-ml--advanced-spatial-regressors-20232026-supplement) (Instant-NGP multiresolution feature grids) + `hybrid_surface_featgrid` diagnosis.
  - **Fit to `ml_transdat`**: Directly attacks the #1 flaw of our current overall accuracy champion (`0.24 mm` `U16` mean, `14.84 mm` `Adapt 1mm` RMSE due to Stage-1 border spikes).
  - **Honest Doubt / Risk**: Adding finer grid levels increases memory (`0.66 MB → ~2–3 MB`), though still tiny compared to `74.9 MB` Quadtree.
  - **Implementation Path**: Reuse `NeuralFeatureGridShiftModel` + `grid_bicubic_variants.py` / `CellBilinearShiftModel` in `CompositeResidualShiftModel`.

- [ ] **`V2-7`: Low-Rank Tensor / Factorized Plane Decomposition (`K-Planes` / `SVD` + Residual)**
  - **Mechanism**: Represents a dense 2D distortion residual grid (e.g., `16x16` or `32x32` over Romania) as a sum of rank-$R$ outer products of 1D/2D factorized feature vectors (`K-Planes` / `SVD` factorization) plus a tiny MLP decoder.
  - **Literature Grounding**: [R4.10](../../literature/references.md#category-4-modern-implicit-neural-representations-scientific-ml--advanced-spatial-regressors-20232026-supplement) (`K-Planes` spatial factorization).
  - **Fit to `ml_transdat`**: Tests whether Romania's distortion field has exploitable low-rank separable structure along Northing/Easting axes, compressing a `32x32` lookup grid into `< 100 KB`.
  - **Honest Doubt / Risk**: Localized anomalies (Banat, Maramureș) are non-separable diagonally, so pure low-rank factorization may need a high rank $R$ or a tiny sparse correction.
  - **Implementation Path**: Pure PyTorch `nn.Parameter` 1D axis-plane factors with linear interpolation + 2-layer MLP.

---

### Family III — Non-Neural Machine Learning (Literature Gaps & Tree Rehabilitation)

- [ ] **`V2-8`: Extreme Learning Machines (`RFF-ELM` / `Kernel-ELM` Closed-Form Ridge Solve)**
  - **Mechanism**: Single-hidden-layer network with fixed random weights (or Random Fourier Features) where output weights are solved **analytically in one closed-form Ridge/Cholesky step** ($\mathbf{W}_{\text{out}} = (\mathbf{H}^T\mathbf{H} + \lambda \mathbf{I})^{-1}\mathbf{H}^T \mathbf{Y}$)—zero iterative gradient descent.
  - **Literature Grounding**: [R2.8](../../literature/references.md#2c-extreme-learning-machines-elm), [R2.9](../../literature/references.md#2c-extreme-learning-machines-elm), [R4.12](../../literature/references.md#category-4-modern-implicit-neural-representations-scientific-ml--advanced-spatial-regressors-20232026-supplement) (ELM for datum conversions).
  - **Fit to `ml_transdat`**: Explicitly cited in geodetic literature as matching/beating BPNNs in fractions of a second. Applying `RFF-ELM` (e.g., 2,000–4,000 features + cyclic `(u, v)`) on Stage-2 residuals tests whether a **zero-PyTorch, closed-form** linear solve can reach `1–2 mm`!
  - **Honest Doubt / Risk**: Single hidden layer lacks compositional depth, so it may hit a capacity wall around `3–8 mm` on Full Romania compared to a 3-block ResNet.
  - **Implementation Path**: `scipy.linalg.solve` / `sklearn.linear_model.Ridge` on `MS-RFF + cyclic (u, v)` features.

- [ ] **`V2-9`: Least-Squares SVM (`LS-SVM`) & Local Kernel Ridge / Nyström `SVR`**
  - **Mechanism**: `LS-SVM` solves a linear KKT system with an RBF/Matérn kernel (mathematically equivalent to Kernel Ridge Regression). Global $N \times N$ matrices are intractable at `2.35M` points, so we evaluate **Nyström Kernel Ridge** (on Full Romania) and **exact local/subgrid `LS-SVM`** (on `box10_interior`).
  - **Literature Grounding**: [R2.10](../../literature/references.md#2d-support-vector-machines-svm--ls-svm--gaussian-processes-gp), [R4.1](../../literature/references.md#category-4-modern-implicit-neural-representations-scientific-ml--advanced-spatial-regressors-20232026-supplement), [R4.12](../../literature/references.md#category-4-modern-implicit-neural-representations-scientific-ml--advanced-spatial-regressors-20232026-supplement) (`LS-SVM` for geodetic coordinate residuals).
  - **Fit to `ml_transdat`**: Closes a direct literature gap (`R2.10`) by benchmarking `LS-SVM` / Kernel Ridge on Stage-2 residuals.
  - **Honest Doubt / Risk**: Likely bounded by the same smooth-kernel wall as `TPS`/`Matérn` (`~3–5 mm`) unless paired with `(u, v)` cell features in the kernel.
  - **Implementation Path**: `sklearn.kernel_approximation.Nystroem` + `Ridge` (or `sklearn.svm.SVR` / `KernelRidge` on `box10_interior`).

- [ ] **`V2-10`: `XGBoost` on Millimeter-Scale Stage-2 Residuals + Piecewise-Linear Model Trees**
  - **Mechanism**: In `submillimeter_exploration`, `HistGBR` failed (`52.4 mm`) because it was trained on **raw `±4.37 m` shifts** using only global `(N, E)` coordinates—forcing axis-aligned step plateaus to approximate steep ramps. Here we test: (a) **`XGBoost`** (installed!) trained on **Stage-2 millimeter residuals** with `(N, E, u, v)` features, and (b) **Piecewise-Linear Model Trees** (or `DecisionTree` partitioning + local linear/quadratic fit per leaf) so leaves are smooth tilted planes instead of flat steps.
  - **Literature Grounding**: [R2.13](../../literature/references.md#2f-tree-based-ensembles-xgboost-random-forest-rfsi), [R4.1](../../literature/references.md#category-4-modern-implicit-neural-representations-scientific-ml--advanced-spatial-regressors-20232026-supplement) (`XGBoost` for 3D datum conversions).
  - **Fit to `ml_transdat`**: Gives a fair, apples-to-apples scientific verdict on tree ensembles when given the same residual framing and `(u, v)` features as our neural models.
  - **Honest Doubt / Risk**: `XGBoost` still outputs piecewise-constant steps, so even on Stage-2 residuals it will likely plateau around `2–5 mm` unless very deep/large; linear-leaf model trees should be much sharper.
  - **Implementation Path**: `xgboost.XGBRegressor` (GPU `tree_method="hist"` supported!) + lightweight `DecisionTreeRegressor` leaf-ID linear/quadratic fitter in `numpy`/`scipy`.

---

### Family IV — Geostatistical, Spline & Signal-Processing Non-Neural Methods

- [ ] **`V2-11`: Local Anisotropic Gaussian Process Regression / Universal Kriging**
  - **Mechanism**: Models Stage-2 residuals as a Gaussian Process with an anisotropic Matérn `3/2` or `5/2` kernel (separate lengthscales for `N`, `E` or cell-local `(u, v)`), solved on $K$-nearest local neighborhoods (`k=25–49`) to produce both a prediction and **calibrated predictive uncertainty $\sigma(N, E)$**.
  - **Literature Grounding**: [R4.12](../../literature/references.md#category-4-modern-implicit-neural-representations-scientific-ml--advanced-spatial-regressors-20232026-supplement) (GPR/Kriging for geodetic distortion grids).
  - **Fit to `ml_transdat`**: Directly tests whether anisotropic Matérn Kriging on `RO_GRID_8X8`/`16X16` beats isotropic `Matérn 3/2` (`4.66 mm`) and provides an uncertainty map highlighting border/kink risk zones.
  - **Honest Doubt / Risk**: Neighborhood Kriging without `(u, v)` boundary awareness still smooths across `11 km` walls; inference is slower than neural nets (`~20–80 µs/pt`).
  - **Implementation Path**: `scipy.spatial.cKDTree` + analytical local Matérn GP solve (or `sklearn.gaussian_process` on `box10_interior`).

- [ ] **`V2-12`: Discontinuity-Aware Partition of Unity RBF (`PU-RBF`), Cell-Aligned Cubic `LOESS (deg=3)` on Dense Grids, & 2D Wavelet Compression (`DWT`)**
  - **Mechanism**: Three compact classical baselines:
    1. **`PU-RBF` (Partition of Unity RBF)**: Overlapping Wendland-weighted local RBF patches aligned to cell blocks so patches do not straddle severe kinks blindly.
    2. **Cell-Aligned Cubic `LOESS (deg=3)` on `8x8`/`16x16`**: `optimal_surface_regressor` only tested `lpoly_deg2` on `4x4`. Testing `deg=3` (cubic) on denser grids (`8x8`/`16x16`) or cell-interior neighborhoods directly matches `TransDatRO`'s cubic degree!
    3. **2D Wavelet / Multiresolution Residual Compression**: Decomposing the `32x32` residual grid via 2D Haar/Biorthogonal wavelets (pure `numpy` lifting/filter bank) and thresholding small coefficients to measure the pure signal-processing Pareto frontier (size vs. mm error).
  - **Literature Grounding**: [R4.11](../../literature/references.md#category-4-modern-implicit-neural-representations-scientific-ml--advanced-spatial-regressors-20232026-supplement) (`PU-RBF` / `NL-PUM`).
  - **Fit to `ml_transdat`**: Establishes the exact non-neural signal-processing and local-cubic-regression baselines to compare against `FeatGrid` (`0.66 MB`) and `ResNet` (`0 MB` lookup).
  - **Honest Doubt / Risk**: Wavelet thresholding requires a uniform grid (rectangular bounding box); `LOESS deg=3` has higher inference cost (`10` terms per solve).
  - **Implementation Path**: Pure `numpy` / `scipy` (`LocalPolynomialSurfaceRegressor(degree=3)` already exists in `src/models/local_poly_surface.py`!).

---

## 4. Master Priority & Feasibility Matrix

| ID | Candidate Model / Track | Family | Lookup Grid? | PyTorch? | Sub-mm Potential | Effort | Wave |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`V2-6`** | **Border-Safe Stage-1 + Multi-Res Feature Grid (`32x32`)** | Structured Hybrid | Small (`~1–3 MB`) | Yes | **Very High** (`<0.3 mm` + fixes border spikes) | Low | **Wave 1** |
| **`V2-5`** | **Neural Bicubic / `(u,v)` Monomial-Basis Head** | Structured Hybrid | Optional (`0–1 MB`) | Yes | **Very High** (matches `TransDatRO` math) | Medium | **Wave 1** |
| **`V2-1`** | **Kolmogorov-Arnold Networks (`Fourier-KAN` / `RBF-KAN`)** | Modern INR | `0 MB` | Yes | **High** (compact edge splines) | Medium | **Wave 1** |
| **`V2-2`** | **`WIRE` (Complex Gabor) & `FINER` on Stage-2 Residuals** | Modern INR | `0 MB` | Yes | **High** (localized wavelets at kinks) | Low–Med | **Wave 1** |
| **`V2-4`** | **Spatial `MoE` / `FBPINN` Subdomain MLPs** | Modern INR | `0 MB` | Yes | **High** (solves Full-RO capacity) | Medium | **Wave 1** |
| **`V2-3`** | **Multiplicative Filter Networks (`GaborNet` / `FourierNet`)** | Modern INR | `0 MB` | Yes | **Medium–High** | Low–Med | **Wave 1** |
| **`V2-12`** | **Dense Cubic `LOESS (deg=3)` / `PU-RBF` / 2D Wavelets** | Classical / SigProc | `0–2 MB` | **No** | **High** (`deg=3` matches cubic!) | Low–Med | **Wave 2** |
| **`V2-10`** | **`XGBoost` (Stage-2 Residuals) & Linear-Leaf Trees** | Tree / Hybrid | `0 MB` (`model`) | **No** | **Medium** (linear-leaf) / Low (`XGB`) | Low–Med | **Wave 2** |
| **`V2-8`** | **Extreme Learning Machine (`MS-RFF ELM` Ridge)** | Closed-Form ML | `0 MB` | **No** | **Medium** (ultra-fast baseline) | Low | **Wave 2** |
| **`V2-7`** | **Low-Rank `K-Planes` / `SVD` Factorized Grid** | Structured Hybrid | Tiny (`<0.2 MB`) | Yes | **Medium** (compression test) | Low–Med | **Wave 2** |
| **`V2-11`** | **Local Anisotropic `GPR` / Universal Kriging** | Geostatistics | Nodes / KD-Tree | **No** | **Low–Med** (uncertainty map) | Low–Med | **Wave 2** |
| **`V2-9`** | **`LS-SVM` / Nyström Kernel Ridge** | Kernel ML | Landmarks | **No** | **Low** (scientific validation) | Low | **Wave 2** |

---

## 5. Recommended Fast 2-Phase Screening Protocol

To avoid wasting hours on Full-Romania (`2.35M`-point) runs for weak candidates, we will reuse the exact two-phase funnel validated in `hybrid_surface_featgrid` and `hybrid_surface_resnet`:

1. **Phase 1 — Fast Subgrid Screening (`box10_interior` + Border-Stress `CellRegion`)**:
   - Fit Stage-1 bases on Full `RO_GRID_4X4` (to avoid coordinate-normalization scale distortion).
   - Train and evaluate all Wave 1 / Wave 2 candidates on `box10_interior` (`10×10` cells, `RO_GRID_32X32` = `103k` pts, ~15–60s per model) plus a **border-stress `CellRegion`** (e.g., `box20_west` or concave-border cells) to catch Stage-1 extrapolation spikes early.
   - **Promotion Gate**: Candidates must beat or rival our current tier benchmarks (`< 1.0 mm` `16x16` mean or `< 1.6 mm` `Adapt 1mm` RMSE for `0 MB` lookup; `< 0.4 mm` for micro-grid; or `< 3.0 mm` for zero-PyTorch non-neural) to advance to Phase 2.
2. **Phase 2 — Full-Romania Promotion & Bidirectional Gate**:
   - Train promoted winners on Full Romania (`RO_GRID_32X32`, `2.35M` pts).
   - Evaluate on `RO_TEST_UNIFORM_16X16`, `RO_TEST_ADAPTIVE_1MM_MID`, **and** their `*_BORDER` masked counterparts (`eval_boundary=True`), plus full bidirectional `ETRS89 ↔ Stereo70` round-trip verification (`ResidualCoordinateTransformer`) on `ro_corridor_snw`.
