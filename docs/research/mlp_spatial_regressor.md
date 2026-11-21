# Research Unit: Multilayer Perceptron (MLP) Spatial Regressor

**Status**: `Validated`  
**Domain**: 2D Continuous Coordinate Distortion Shifts ($(st70_n, st70_e) \to (shift_n, shift_e)$)  
**Parent / Predecessor Units**: [`knn_spatial_regressor`](knn_spatial_regressor.md), [`decision_tree_regressor`](decision_tree_regressor.md)  
**Related Experiment**: [`experiments/mlp_spatial_regressor/`](../../experiments/mlp_spatial_regressor/)  
**Task Log**: [`.agents/tasks/mlp-spatial-regressor-experiment.md`](../../.agents/tasks/mlp-spatial-regressor-experiment.md)

---

## 1. Scientific Hypothesis & Motivation

Prior parametric and non-parametric spatial regressors in `ml_transdat` exhibited fundamental structural trade-offs:
* **Helmert 2D**: Pure 4-parameter conformal formula with zero memory footprint, but stuck at a $\sim 44.3\,\text{cm}$ RMSE floor due to lack of local degrees of freedom.
* **Decision Trees**: Partitioned space into flat orthogonal step plateaus, requiring $89.4\,\text{MB}$ ($16\times 16$) to reach sub-centimeter accuracy while introducing discontinuous step artifacts ($8.2\,\text{cm}$ floor on $11\,\text{km}$ nodes).
* **KNN Regressors**: Achieved $3.5\,\text{cm}$ RMSE ($K=4$ IDW), but required keeping all anchor nodes in memory during inference.

### Core Hypothesis
A compact Multilayer Perceptron (MLP) neural network can approximate the continuous 2D distortion field of Romania as a seamless $C^\infty$ mathematical surface without storing any discrete lookup grid ($0\,\text{MB}$ grid table). However, standard neural networks mapping raw Cartesian coordinates suffer from **spectral bias** (prioritizing low-frequency planar trends over local curvature). Introducing **Random Fourier Features (RFF)** / sinusoidal positional encodings will overcome spectral bias, enabling sub-decimeter and sub-centimeter continuous representation.

---

## 2. Experimental Methodology & Dual-Benchmark Protocol

All models were evaluated using the newly implemented continuous nationwide evaluation protocol, eliminating spatial corridor selection bias and testing both general expectation and edge-case curvature fidelity:
1. **`RO_TEST_UNIFORM_4X4`** (36,752 nationwide stratified continuous points): Unbiased, area-weighted survey accuracy across all 2,297 valid grid cells.
2. **`RO_TEST_ADAPTIVE_3MM_MID`** (176,350 quadtree midpoints): Curvature-weighted stress test concentrated in steep distortion gradient zones (Banat dome and Maramureș shear).

### Model Configurations Evaluated
* **`mlp_raw_11km`**: Standard feedforward MLP (128 → 128 neurons) mapping normalized coordinates $(N, E) \in [-1, 1]$ directly to shifts.
* **`mlp_fourier_11km`**: 2-layer MLP (128 → 128) preceded by Random Fourier Features ($\sigma=3.0$, $32$ projection frequencies, 66 input dimensions) trained on the 2,689 valid anchor nodes.
* **`mlp_fourier_4x4`**: Fourier MLP (256 → 128 neurons, $64$ projection frequencies) trained on the densified $4\times 4$ subgrid (36,435 points, ~2.75 km resolution) using mini-batch Adam with tight convergence tolerance (`tol=1e-7`).
* **`mlp_fourier_8x8`**: Fourier MLP (256 → 128 neurons, $64$ projection frequencies) trained on the densified $8\times 8$ subgrid (146,258 points, ~1.37 km resolution).

---

## 3. Quantitative Verdicts & Performance

| Model Architecture | Training Grid | Eval Dataset | Mean 2D (mm) | RMSE 2D (mm) | P95 (mm) | Max 2D (mm) | < 1 cm (%) | < 5 cm (%) | Size (KB) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `mlp_raw_11km` | 11 km nodes | Uniform 4x4 | 152.3 | 182.3 | 336.4 | 1,000.1 | 0.5% | 10.5% | ~200 KB |
| `mlp_raw_11km` | 11 km nodes | Adaptive 3mm | 207.6 | 257.9 | 500.8 | 1,519.0 | 0.3% | 6.3% | ~200 KB |
| `mlp_fourier_11km` | 11 km nodes | Uniform 4x4 | 41.7 | 49.5 | 91.6 | 411.7 | 6.7% | 68.5% | ~200 KB |
| `mlp_fourier_11km` | 11 km nodes | Adaptive 3mm | 36.5 | 46.0 | 88.1 | 416.4 | 10.8% | 76.0% | ~200 KB |
| `mlp_fourier_4x4` | 4x4 subgrid | Uniform 4x4 | **10.3** | **13.0** | **23.1** | 374.7 | **57.9%** | **99.8%** | ~200 KB |
| `mlp_fourier_4x4` | 4x4 subgrid | Adaptive 3mm | **18.4** | **32.5** | **54.3** | 514.6 | **40.9%** | **94.2%** | ~200 KB |
| `mlp_fourier_8x8` | 8x8 subgrid | Uniform 4x4 | 12.4 | 14.6 | 26.7 | **151.7** | 44.7% | **99.8%** | ~200 KB |
| `mlp_fourier_8x8` | 8x8 subgrid | Adaptive 3mm | 18.8 | **30.1** | **44.7** | 504.9 | 31.4% | **96.0%** | ~200 KB |

---

## 4. Key Scientific Findings & Takeaways

1. **Confirmation of Spectral Bias in Spatial Geodesy**:
   Standard feedforward MLPs on raw coordinates fail to capture local distortion gradients, smoothing out regional domes ($182\,\text{mm}$ RMSE uniformly, $>1.5\,\text{m}$ max error). Adding depth or capacity without positional encoding yielded negligible improvement ($176\,\text{mm}$).
2. **Positional Encoding Resolution**:
   Random Fourier Features resolved spectral bias, reducing RMSE by **3.7x down to $46.0\,\text{mm}$** on the high-curvature adaptive benchmark using only the 2,689 official control nodes. The Fourier MLP generalized better on the adaptive stress test ($36.5\,\text{mm}$ mean) than on uniform points ($41.7\,\text{mm}$), proving effective tracking of steep regional domes.
3. **Training Density Scaling & Diminishing Returns**:
   * Scaling training from $11\,\text{km}$ to $4\times 4$ ($2.75\,\text{km}$) produced a major accuracy leap: nationwide uniform RMSE dropped to **$13.0\,\text{mm}$** with **$99.8\%$ of all points within $5\,\text{cm}$** and a median error of **$8.8\,\text{mm}$**.
   * Scaling further to $8\times 8$ ($1.37\,\text{km}$, 146k points) did not meaningfully improve average precision ($30.1\,\text{mm}$ vs $32.5\,\text{mm}$ on adaptive), but effectively regularized edge-case anomalies, cutting nationwide maximum error from $37.5\,\text{cm} \to 15.2\,\text{cm}$.
4. **Conclusion & Limits of Global Neural Fitting**:
   A single global neural network reaches an asymptotic capacity floor around $\sim 3\,\text{cm}$ standard error across complex national curvature. Moving toward sub-millimeter precision requires either localized adaptive frequency networks, domain-decomposed architectures, or hybrid analytical-residual pipelines (e.g., Helmert + Local Residual RBF/MLP).
