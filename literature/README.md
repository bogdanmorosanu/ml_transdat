# Literature Survey: Machine Learning for Geodetic Coordinate Transformations

> [!NOTE]
> **Purpose**: This report summarizes published methodologies for applying Machine Learning (ML) and Artificial Neural Networks (ANNs) to geodetic coordinate transformations. All cited sources are indexed with clickable links in [references.md](references.md), and all executed queries (including negative results) are logged in [search_log.md](search_log.md).

---

## Executive Summary

Across geodetic literature, ML is used to overcome the limitations of rigid parametric models (e.g., 7-parameter Bursa-Wolf, Molodensky-Badekas, 2D affine/conformal) when legacy local networks exhibit **non-linear, spatially varying distortions**. 

Two overarching paradigms emerge:
1. **End-to-End Surrogates**: Training an ANN directly on $(\varphi, \lambda, h)$ or $(X, Y)$ to predict target datum coordinates.
2. **Hybrid / Residual-Based Models (Most Relevant to `ml_transdat`)**: Computing a deterministic geometric baseline (e.g., conformal/Helmert/projection) and training an ML model strictly on the **spatial residuals** ($\Delta N, \Delta E$ or $\Delta\varphi, \Delta\lambda$).

---

## Category 1: Closely Related Transformations (Stereographic Projections & Shift/Distortion Grids)

### 1.1 Hybrid / Residual-Based Neural Network Correction (RBNC)
- **Core Idea**: Instead of learning the entire coordinate transformation from scratch, a classical geometric/projection model handles the global shift, and a neural network (MLP or RBFNN) is trained solely to predict the remaining local distortion residuals:
  $$\text{Coordinate}_{\text{final}} = \text{Coordinate}_{\text{geometric model}} + \text{NN}_{\text{residuals}}(\text{position})$$
- **Why It Matters**: Dramatically reduces model complexity, prevents erratic extrapolation outside control-point boundaries, and stabilizes training on sparse or unevenly distributed control points.
- **References**: [R1.4](references.md#category-1-closely-related-stereographic-projections--distortionshift-grids), [R1.5](references.md#category-1-closely-related-stereographic-projections--distortionshift-grids)

### 1.2 Neural Surrogates for Distortion & Shift Grids (NTv2 / NADCON Analogues)
- **Core Idea**: Replacing or approximating discrete binary shift grids (`.gsb` / NTv2 / NADCON) with continuous neural approximators (**BPANN**, **RBFNN**, **ANFIS**) trained on dense control-point meshes.
- **Key Takeaways**:
  - Neural networks act as compressed, continuous function mappers for irregular regional deformations (crustal motion, legacy survey distortions).
  - **Coordinate Encoding**: Literature highlights that feeding raw angular coordinates can introduce boundary/wrapping artifacts; trigonometric encoding ($\sin/\cos$) or local projected coordinates $(x, y)$ are preferred.
- **References**: [R1.1](references.md#category-1-closely-related-stereographic-projections--distortionshift-grids), [R1.2](references.md#category-1-closely-related-stereographic-projections--distortionshift-grids), [R1.3](references.md#category-1-closely-related-stereographic-projections--distortionshift-grids)

### 1.3 Conformal Mapping & Physics-Informed Neural Networks (CoCo-PINNs)
- **Core Idea**: Embedding conformal mappings into PINN architectures to map complex physical domains onto canonical computational domains (e.g., unit circle/rectangle) while preserving angles and enforcing differential operators via the Jacobian.
- **Relevance**: Demonstrated for 2D PDEs and inverse material design rather than geodetic datum shifts, but conceptually relevant if enforcing conformality or metric-tensor constraints in future ML projection models.
- **References**: [R1.6](references.md#category-1-closely-related-stereographic-projections--distortionshift-grids), [R1.7](references.md#category-1-closely-related-stereographic-projections--distortionshift-grids)

---

## Category 2: Broader ML Coordinate Transformation Literature

### 2.1 Multi-Layer Perceptrons (MLP / Backpropagation ANNs)
- **Usage**: The most widely published baseline across regional datum studies (e.g., Ghana, Egypt, Turkey, Iraq) converting between local legacy datums and WGS84/ITRF.
- **Key Findings**:
  - Consistently outperforms standard 7-parameter Bursa-Wolf or low-order polynomials in RMSE when local networks contain unmodeled distortions.
  - Recent studies emphasize **$K$-fold cross-validation** and **robust outlier filtering** to prevent overfitting on sparse control networks.
- **References**: [R2.1](references.md#2a-multi-layer-perceptrons-mlp--bpann--cross-validation), [R2.2](references.md#2a-multi-layer-perceptrons-mlp--bpann--cross-validation), [R2.3](references.md#2a-multi-layer-perceptrons-mlp--bpann--cross-validation), [R2.4](references.md#2a-multi-layer-perceptrons-mlp--bpann--cross-validation)

### 2.2 Radial Basis Function Neural Networks (RBFNN) & Thin Plate Splines (TPS)
- **Usage**: Heavily used for 2D/3D datum transformations (WGS84 / ETRS89 ↔ local grids) due to strong localized learning via radial kernels (Gaussian, Multiquadric, Thin Plate Spline) + optional low-order polynomial trend.
- **Hybrid Variant**: **TLS-RBFNN** (Total Least Squares + RBFNN) combines stochastic error handling with non-linear radial basis interpolation.
- **References**: [R2.5](references.md#2b-radial-basis-function-networks-rbfnn--thin-plate-splines-tps), [R2.6](references.md#2b-radial-basis-function-networks-rbfnn--thin-plate-splines-tps), [R2.7](references.md#2b-radial-basis-function-networks-rbfnn--thin-plate-splines-tps)

### 2.3 Extreme Learning Machines (ELM)
- **Usage**: Single-hidden-layer feedforward networks with randomly initialized hidden weights and analytically solved output weights (Moore-Penrose generalized inverse).
- **Key Findings**: Achieves comparable or superior accuracy to BPNN and RBFNN for 2D cadastral and regional datum transformations while training in fractions of a second without iterative gradient descent.
- **References**: [R2.8](references.md#2c-extreme-learning-machines-elm), [R2.9](references.md#2c-extreme-learning-machines-elm)

### 2.4 Support Vector Machines (LS-SVM) & Gaussian Processes (GP / Kriging)
- **LS-SVM**: Reformulates SVM into linear equations using RBF/Gaussian kernels; often outperforms BPNN on small control-point datasets with strong generalization.
- **Gaussian Processes (GP)**: Models spatial autocorrelation directly via covariance kernels (e.g., Matérn, squared exponential) and uniquely provides **predictive variance (uncertainty quantification)** alongside transformed coordinates.
- **References**: [R2.10](references.md#2d-support-vector-machines-svm--ls-svm--gaussian-processes-gp)

### 2.5 Vertical Datums / Geoid Modeling & Neural Fields
- **GRNN for Geoid Undulation**: Generalized Regression Neural Networks combined with spatial interpolation to model local geoid/height transformations.
- **GeodesyNets**: Neural fields on arXiv learning continuous 3D gravitational/shape representations and frame mappings for celestial bodies.
- **References**: [R2.11](references.md#2e-vertical-datums--geoid-modeling--neural-fields), [R2.12](references.md#2e-vertical-datums--geoid-modeling--neural-fields)

### 2.6 Tree-Based Ensembles (XGBoost, Random Forest, RFSI)
- **Usage**: Separate per-axis models (e.g., one for $dN$, one for $dE$) trained on coordinates and spatial features, either end-to-end or as residual correctors. *Random Forest Spatial Interpolation (RFSI)* adds distances to neighboring control points as features.
- **Pros**: Fast to train, no feature-scaling headaches, strong on tabular data, and resistant to outliers.
- **Cons for `ml_transdat`**: Decision trees produce **piecewise-constant ("staircase")** predictions rather than smooth, continuous, differentiable surfaces; they extrapolate poorly outside the training bounding box; and large ensembles can be bulky compared to compact neural networks.
- **References**: [R2.13](references.md#2f-tree-based-ensembles-xgboost-random-forest-rfsi), [R2.14](references.md#2f-tree-based-ensembles-xgboost-random-forest-rfsi), [R2.15](references.md#2f-tree-based-ensembles-xgboost-random-forest-rfsi)

---

## Searched & Not Found (Knowledge Gaps)

1. **ETRS89 ↔ Stereo 70 (Romania) via Machine Learning**: No published papers, preprints, or technical reports were found applying ML/ANNs to `Stereo 70`, `TransDatRO`, or Romania's `.spg` distortion grid.
2. **Production Replacement of NTv2 / NADCON Binary Grids**: No published national geodetic agency standard or production benchmark replacing `.gsb` (NTv2) or NADCON grids with neural networks was found.
3. **PINNs for Geodetic Map Projections**: `CoCo-PINNs` exist for PDEs on complex 2D domains, but no published work was found applying PINNs to geodetic map projections or conformal datum transformations.
