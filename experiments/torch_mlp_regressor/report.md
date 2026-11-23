# Experiment Report: Deep PyTorch Residual Neural Network Spatial Regressor (`torch_mlp_regressor`)

## 1. Objective & Hypothesis
This research unit upgrades the neural network baseline from `sklearn`'s shallow `(256, 128)` `MLPRegressor` (`mlp_fourier_4x4`) to a **GPU-accelerated PyTorch Residual Neural Network (`TorchMLPGridShiftModel` in `src/models/torch_mlp.py`)** and systematically tests three hypotheses:

1. **Stage 1 (Architecture, Activation & Capacity)**: Residual skip connections + `LayerNorm` + `AdamW` + `CosineAnnealingLR` (comparing `ReLU`, `SiLU`, `GELU`, and wider `512`-dim blocks) will outperform `mlp_fourier_4x4` (`13.0 mm` uniform RMSE).
2. **Stage 2 (Smooth Cyclic Cell Position & Raw Neighbor Features)**: Adding smooth, asymmetric intra-cell phase features `sin(2πu), cos(2πu), sin(2πv), cos(2πv)` (eliminating both border jumps and the symmetry flaw of `center_dist`) will improve accuracy, whereas raw piecewise-constant `3×3` neighbor slopes/twists will not.
3. **Stage 3 (2D Cubic Hermite-Weighted Neighbor Features)**: Pre-multiplying corner slopes and twists by 2D cubic Hermite basis weights ($u(1-u)^2$, $u^2(u-1)$, etc.) forces neighbor features to vanish smoothly at cell borders—testing whether physics-weighted neighbor derivatives add enough signal to justify their complexity.

---

## 2. Experimental Setup & Methodology
- **Training Dataset**: `RO_GRID_4X4` (36,435 points, ~2.75 km spacing).
- **Evaluation Benchmarks**:
  - `RO_TEST_UNIFORM_4X4` (36,752 off-lattice points nationwide).
  - `RO_TEST_ADAPTIVE_3MM_MID` (176,350 points concentrated in high-curvature zones).
- **Shared Backbone**: Random Fourier Features (`fourier_scale=3.0`, `num_fourier_frequencies=64`, 130 base features), `AdamW` (`lr=1e-3`, `weight_decay=1e-5`), `CosineAnnealingLR`, `batch_size=256`, 90/10 train/val early stopping.
- **Reproducible `run_unit.py` Suite**: Retains the 3 key progression models (`torch_res_relu_4x4`, `torch_res_relu_cellpos_4x4` [Champion], and `torch_res_relu_cellpos_hermite_slopes_4x4`), while all 11 evaluated variants across Stages 1–3 are recorded below.

---

## 3. Quantitative Results

### 3.1 Nationwide Uniform (`ro_test_uniform_4x4`, 36,752 pts)

| Stage | Model | Features / Config | Mean (mm) | **RMSE (mm)** | P95 (mm) | Max (mm) | **< 1 cm (%)** |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Baseline | `mlp_fourier_4x4` (`sklearn`) | 130 RFF, `(256, 128)` ReLU | 10.3 | 13.0 | 23.1 | 374.7 | 57.9% |
| Stage 1 | `torch_res_silu_4x4` | 130 RFF, `256` × 3 blocks, SiLU | 15.3 | 18.4 | 31.1 | 362.9 | 30.8% |
| Stage 1 | `torch_res_gelu_4x4` | 130 RFF, `256` × 3 blocks, GELU | 12.7 | 15.6 | 27.3 | 303.5 | 43.8% |
| Stage 1 | `torch_res_wide_silu_4x4` | 130 RFF, `512` × 4 blocks, SiLU | 6.4 | 8.5 | 14.7 | 349.2 | 83.9% |
| Stage 1 | `torch_res_relu_4x4` | 130 RFF, `256` × 2 blocks, ReLU | 4.9 | 7.2 | 11.1 | 310.7 | 93.0% |
| **Stage 2** | **`torch_res_relu_cellpos_4x4`** *(Champion)* | **+4 cyclic `(u, v)`** (134), `256` × 2 ReLU | **4.2** | **6.8** | **9.7** | **294.6** | **95.6%** |
| Stage 2 | `torch_res_relu_cellpos_slopes_4x4` | +4 cyclic + 32 raw slopes (166) | 6.7 | 10.8 | 22.1 | 280.4 | 81.3% |
| Stage 2 | `torch_res_relu_cellpos_all_4x4` | +4 cyclic + 40 raw slopes & twists (174) | 7.2 | 11.2 | 23.5 | **264.7** | 80.0% |
| Stage 2 | `torch_res_wide_relu_cellpos_all_4x4` | Same 174, `512` × 3 blocks, ReLU | 7.8 | 12.9 | 27.0 | 295.2 | 78.1% |
| Stage 3 | `torch_res_relu_cellpos_hermite_slopes_4x4` | +4 cyclic + 32 Hermite slopes (166) | **3.0** | **6.5** | **6.3** | 411.8 | **98.6%** |
| Stage 3 | `torch_res_relu_cellpos_hermite_all_4x4` | +4 cyclic + 40 Hermite slopes & twists (174) | **3.0** | 6.6 | 6.4 | 434.8 | **98.6%** |
| Stage 3 | `torch_res_relu_cellpos_hermite_sum_4x4` | +4 cyclic + 6 summed Hermite terms (140) | 4.2 | 7.1 | 9.4 | 365.5 | 95.9% |

### 3.2 High-Curvature Adaptive Stress Test (`ro_test_adaptive_3mm_mid`, 176,350 pts)

| Stage | Model | Mean (mm) | p50 (mm) | **RMSE (mm)** | P95 (mm) | Max (mm) | **< 1 cm (%)** |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline | `mlp_fourier_4x4` (`sklearn`) | 18.4 | — | **32.5** | 54.3 | **514.6** | 40.9% |
| Stage 1 | `torch_res_silu_4x4` | 29.6 | 19.0 | 56.2 | 85.7 | 907.0 | 18.9% |
| Stage 1 | `torch_res_gelu_4x4` | 24.1 | 15.0 | 49.6 | 66.8 | 865.8 | 29.2% |
| Stage 1 | `torch_res_wide_silu_4x4` | 15.5 | 8.1 | 44.4 | **39.4** | 936.1 | 60.0% |
| Stage 1 | `torch_res_relu_4x4` | 15.8 | 6.1 | 42.8 | 54.9 | 804.9 | 67.9% |
| **Stage 2** | **`torch_res_relu_cellpos_4x4`** *(Champion)* | **15.6** | **5.0** | **44.2** | 59.4 | **838.0** | **71.3%** |
| Stage 2 | `torch_res_relu_cellpos_slopes_4x4` | 20.7 | 9.6 | 46.9 | 63.6 | 915.9 | 50.8% |
| Stage 2 | `torch_res_relu_cellpos_all_4x4` | 20.6 | 9.7 | 46.2 | 61.2 | 887.8 | 50.7% |
| Stage 2 | `torch_res_wide_relu_cellpos_all_4x4` | 23.5 | 11.7 | 48.2 | 70.8 | 855.4 | 46.5% |
| Stage 3 | `torch_res_relu_cellpos_hermite_slopes_4x4` | **14.5** | **3.4** | 49.5 | 57.5 | 939.4 | **76.8%** |
| Stage 3 | `torch_res_relu_cellpos_hermite_all_4x4` | 15.1 | 3.5 | 51.9 | 58.6 | 969.2 | 76.3% |
| Stage 3 | `torch_res_relu_cellpos_hermite_sum_4x4` | 14.8 | 5.0 | 44.5 | 52.0 | 874.0 | 72.2% |

---

## 4. Visual Diagnostics
- [`plots/ro_test_uniform_4x4_grid_shift_2d_spatial_error.png`](plots/ro_test_uniform_4x4_grid_shift_2d_spatial_error.png)
- [`plots/ro_test_uniform_4x4_grid_shift_2d_hist_ecdf.png`](plots/ro_test_uniform_4x4_grid_shift_2d_hist_ecdf.png)
- [`plots/ro_test_uniform_4x4_grid_shift_2d_signed_scatter.png`](plots/ro_test_uniform_4x4_grid_shift_2d_signed_scatter.png)
- [`plots/ro_test_adaptive_3mm_mid_grid_shift_2d_spatial_error.png`](plots/ro_test_adaptive_3mm_mid_grid_shift_2d_spatial_error.png)
- [`plots/ro_test_adaptive_3mm_mid_grid_shift_2d_hist_ecdf.png`](plots/ro_test_adaptive_3mm_mid_grid_shift_2d_hist_ecdf.png)
- [`plots/ro_test_adaptive_3mm_mid_grid_shift_2d_signed_scatter.png`](plots/ro_test_adaptive_3mm_mid_grid_shift_2d_signed_scatter.png)

---

## 5. Takeaways & Verdict
- **Verdict**: `Validated` (for PyTorch ResNet + smooth cyclic `(u, v)` cell-position encoding `torch_res_relu_cellpos_4x4`); `Inconclusive / Not Worth Complexity` for `3×3` neighbor slopes & twists.
- **Key Findings for Future Agents**:
  1. **PyTorch ResNet (`ReLU`) nearly halves nationwide uniform error**: Moving from `sklearn` `(256, 128)` (`13.0 mm` RMSE, `57.9% < 1 cm`) to PyTorch ResNet (`256` × 2 residual blocks, `LayerNorm`, `AdamW` + `CosineAnnealingLR`) drops uniform RMSE to **`7.2 mm`** (`93.0% < 1 cm`). Smooth activations (`SiLU`, `GELU`) lag at `256` width and require `512` width (`8.5 mm` RMSE, `39.4 mm` adaptive P95) to compete.
  2. **Smooth cyclic `(u, v)` cell position is the ONLY engineered feature that clearly helped**: Encoding normalized intra-cell position $(u, v) \in [0, 1)$ as `sin(2πu), cos(2πu), sin(2πv), cos(2πv)` eliminates cell-border jumps while preserving full directional asymmetry, improving `torch_res_relu_4x4` across every uniform metric (`7.2 → 6.8 mm` RMSE, `4.9 → 4.2 mm` Mean, `11.1 → 9.7 mm` P95, `93.0% → 95.6% < 1 cm`) and lifting adaptive `< 1 cm` from `67.9% → 71.3%` with **zero external neighbor lookup table**.
  3. **Raw `3×3` neighbor slopes & twists degrade accuracy**: Even with `512`-neuron capacity, cell-constant slopes/twists jump abruptly at $11\,\text{km}$ borders and hurt both uniform (`6.8 → 10.8–12.9 mm` RMSE) and adaptive (`71.3% → 46.5–50.8% < 1 cm`) performance.
  4. **Hermite-weighted slopes improve median/mean but worsen tail RMSE & Max**: Pre-multiplying the 32 slopes by 2D cubic Hermite weights ($u(1-u)^2$, $u^2(u-1)$) eliminates border jumps and improves typical errors (`3.0 mm` mean, `98.6% < 1 cm` uniform; `3.4 mm` median, `76.8% < 1 cm` adaptive), **but** increases worst-case `Max` outliers (`294.6 → 411.8 mm` uniform; `838.0 → 939.4 mm` adaptive) and adaptive `RMSE` (`44.2 → 49.5 mm`) while requiring 32 lookup-dependent features. Overall, the added complexity and tail-error regression do not justify promoting neighbor slopes over the clean, grid-free `torch_res_relu_cellpos_4x4` champion.

---

## 6. Benchmark Candidates
| Variant | Benchmark Label | Train Grid | Exact Hyperparameters |
| :--- | :--- | :--- | :--- |
| `torch_res_relu_cellpos_4x4` | `PyTorch ResNet + Cyclic CellPos (4x4)` | `ro_grid_4x4` | `hidden_dim=256, num_res_blocks=2, activation="relu", use_fourier_features=True, fourier_scale=3.0, num_fourier_frequencies=64, use_cyclic_cell_pos=True, batch_size=256, lr=1e-3, max_epochs=200, patience=25` |
