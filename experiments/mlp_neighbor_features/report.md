# Experiment Report: Fourier MLP with 3×3 Cell Neighbor Features & Center Distance (`mlp_neighbor_features`)

## 1. Objective & Hypothesis
This unit tests whether injecting the **40 continuous $3 \times 3$ cell neighbor geometry features** (extracted in [`cell_neighbor_features`](../cell_neighbor_features/report.md) via `CellNeighborFeatureExtractor` in `src/geometry/neighbor_features.py`) and/or a **continuous $C^0$ intra-cell midpoint distance** (`lookup_center_distances`: $0$ at cell center, $1$ at cell edges) improves the champion Fourier MLP spatial regressor (`mlp_fourier_4x4` from `mlp_spatial_regressor`) without changing its architecture or training hyperparameters.

---

## 2. Experimental Setup & Methodology
- **Baseline**: `mlp_fourier_4x4` (`hidden_layer_sizes=(256, 128)`, `use_fourier_features=True`, `fourier_scale=3.0`, `num_fourier_frequencies=64`, `solver="adam"`, `batch_size=128`, `learning_rate_init=0.001`, `tol=1e-7`, `n_iter_no_change=25`, `max_iter=300`, `alpha=1e-6`), trained on `RO_GRID_4X4`.
- **Stage 1 (Neighbor Features Only)**:
  - `mlp_fourier_neighbor_slopes_4x4`: +32 standardized corner slopes.
  - `mlp_fourier_neighbor_all_4x4`: +40 standardized slopes & twists.
- **Stage 2 (Continuous Midpoint Distance + Neighbor Features)**:
  - `mlp_fourier_center_dist_4x4`: +2 continuous `[0, 1]` midpoint distances (`2|u-0.5|, 2|v-0.5|`).
  - `mlp_fourier_center_dist_slopes_4x4`: +2 midpoint distances + 32 standardized slopes.
  - `mlp_fourier_center_dist_all_4x4`: +2 midpoint distances + 40 standardized slopes & twists.
- **Clean Interior Filtering**: All training and evaluation datasets are filtered strictly to points inside the 2,297 valid $3 \times 3$ interior cells (`36,752` points on `RO_TEST_UNIFORM_4X4`; `176,350` points on `RO_TEST_ADAPTIVE_3MM_MID`).

---

## 3. Quantitative Results

### 3.1 Nationwide Uniform (`ro_test_uniform_4x4`, 36,752 pts)

| Model | Mean 2D (mm) | RMSE 2D (mm) | P95 (mm) | Max 2D (mm) | < 1 cm (%) | Fit Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `mlp_fourier_4x4` (Baseline) | **10.3** | **13.0** | 23.1 | 374.7 | **57.9%** | 108.2 s |
| `mlp_fourier_center_dist_4x4` (+Center Dist) | 10.4 | 13.7 | **22.1** | 320.7 | 57.7% | 125.0 s |
| `mlp_fourier_neighbor_slopes_4x4` (+32 Slopes)* | 11.5 | 14.8 | 27.8 | 233.6 | 55.0% | 139.7 s |
| `mlp_fourier_center_dist_slopes_4x4` (+Center Dist + 32 Slopes) | 12.4 | 15.5 | 28.9 | **214.0** | 47.7% | 65.4 s |
| `mlp_fourier_neighbor_all_4x4` (+40 All)* | 11.6 | 15.4 | 28.3 | 330.9 | 54.3% | 112.1 s |
| `mlp_fourier_center_dist_all_4x4` (+Center Dist + 40 All) | 10.9 | 14.5 | 26.3 | 259.5 | 57.5% | 97.3 s |

### 3.2 Adaptive 3 mm Stress Test (`ro_test_adaptive_3mm_mid`, 176,350 pts)

| Model | Mean 2D (mm) | RMSE 2D (mm) | P95 (mm) | Max 2D (mm) | < 1 cm (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `mlp_fourier_4x4` (Baseline) | **18.4** | **32.5** | **54.3** | **514.6** | **40.9%** |
| `mlp_fourier_center_dist_4x4` (+Center Dist) | 21.1 | 42.0 | 61.4 | 584.3 | 38.9% |
| `mlp_fourier_neighbor_slopes_4x4` (+32 Slopes)* | 24.7 | 49.0 | 67.1 | 768.7 | 34.0% |
| `mlp_fourier_center_dist_slopes_4x4` (+Center Dist + 32 Slopes) | 25.7 | 47.3 | 67.1 | 755.3 | 28.0% |
| `mlp_fourier_neighbor_all_4x4` (+40 All)* | 25.7 | 51.4 | 73.2 | 904.8 | 32.2% |
| `mlp_fourier_center_dist_all_4x4` (+Center Dist + 40 All) | 24.4 | 47.0 | 67.8 | 835.7 | 33.7% |

*\*Rows marked with `*` are from Stage 1 (same baseline/data).*

---

## 4. Visual Diagnostics
- [`plots/extreme_scenarios_gradient_and_linear_9cells.png`](plots/extreme_scenarios_gradient_and_linear_9cells.png) & [`plots/extreme_scenarios_curvature_and_kinks_9cells.png`](plots/extreme_scenarios_curvature_and_kinks_9cells.png): 9-cell (99 km) extreme scenario transects comparing the baseline Fourier MLP against the center-distance and neighbor-feature variants.

---

## 5. Takeaways & Verdict
- **Verdict**: `Dead End` (for concatenating cell-level $3 \times 3$ neighbor features and/or symmetric midpoint distances onto global `sklearn` Fourier MLP inputs).
- **Key Findings**:
  1. Adding `center_dist` modestly improved the 40-feature model (`15.4 -> 14.5 mm` uniform; `51.4 -> 47.0 mm` adaptive), confirming that intra-cell position helps the network partially use neighbor features.
  2. However, none of the variants beat the bare `mlp_fourier_4x4` baseline (`13.0 mm` uniform, `32.5 mm` adaptive). Symmetric midpoint distance (`2|u-0.5|`) cannot distinguish left-of-center from right-of-center, and an additive `ReLU(Wx + b)` MLP still cannot cleanly multiply local coordinates by corner derivatives.
  3. Future neural experiments aiming to leverage local cell geometry should move to a dedicated PyTorch architecture with explicit **multiplicative/bilinear interactions** or **residual local polynomial/bicubic layers** rather than feature concatenation in `sklearn`.
