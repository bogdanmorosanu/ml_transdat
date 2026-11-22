# Research Unit: Deep PyTorch Residual Neural Network (`torch_mlp_regressor`)

- **Status**: `Validated`
- **Primary Experiment**: [`experiments/torch_mlp_regressor/report.md`](../../experiments/torch_mlp_regressor/report.md)
- **Implementation**: [`src/models/torch_mlp.py`](../../src/models/torch_mlp.py)
- **Upstream Dependencies**: [`mlp_spatial_regressor`](mlp_spatial_regressor.md), [`mlp_neighbor_features`](mlp_neighbor_features.md), [`cell_neighbor_features`](cell_neighbor_features.md)

---

## 1. Scientific Question
Can a GPU-accelerated **PyTorch Residual Neural Network** (`TorchMLPGridShiftModel`) with scaled depth/width, `LayerNorm`, modern activations (`ReLU`, `SiLU`, `GELU`), and `AdamW` + `CosineAnnealingLR` break past the `sklearn` `mlp_fourier_4x4` baseline (`13.0 mm` uniform / `32.5 mm` adaptive RMSE)—and can smooth asymmetric cell-position encoding plus raw or 2D cubic Hermite-weighted `3×3` neighbor slopes/twists add further signal?

---

## 2. Hypotheses & Findings
1. **PyTorch ResNet Architecture & Activation Sweep (RFF Coordinates Only)**:
   - *Hypothesis*: Residual skip connections (`num_res_blocks` ∈ `{2, 3, 4}`), `LayerNorm`, and smooth activations (`SiLU`, `GELU`) will outperform `sklearn`'s `(256, 128)` `ReLU` MLP on `RO_GRID_4X4`.
   - *Verdict*: **Validated (with `ReLU` winning)**. `torch_res_relu_4x4` (`256` × 2 blocks) cut nationwide uniform RMSE by ~45% (`13.0 → 7.2 mm`, `93.0% < 1 cm`). Smooth `SiLU`/`GELU` underperformed at `256` width (`18.4 / 15.6 mm`) and required `512` width (`torch_res_wide_silu_4x4`: `8.5 mm` uniform RMSE, `39.4 mm` adaptive P95) to compete.
2. **Smooth Cyclic `(u, v)` Intra-Cell Position (`sin(2πu), cos(2πu), sin(2πv), cos(2πv)`)**:
   - *Hypothesis*: Encoding normalized cell coordinates $(u, v) \in [0, 1)$ as 2D cyclic phases eliminates sawtooth jumps at $11\,\text{km}$ borders while preserving the directional asymmetry lost by `center_dist`.
   - *Verdict*: **Validated (New Champion)**. `torch_res_relu_cellpos_4x4` improved every uniform metric (`7.2 → 6.8 mm` RMSE, `4.9 → 4.2 mm` Mean, `11.1 → 9.7 mm` P95, `93.0% → 95.6% < 1 cm`) and lifted adaptive `< 1 cm` from `67.9% → 71.3%` with **zero** external neighbor lookup table.
3. **Raw & 2D Cubic Hermite-Weighted `3×3` Neighbor Slopes/Twists**:
   - *Hypothesis*: Weighting corner slopes/twists by 2D cubic Hermite basis functions ($u(1-u)^2$, $u^2(u-1)$) forces them to vanish smoothly at cell edges and models corner-distance decay.
   - *Verdict*: **Not Worth Complexity (`Dead End` for promotion)**. Raw neighbor features degraded uniform RMSE (`6.8 → 10.8–12.9 mm`) even at `512` width. Hermite weighting eliminated the border-jump penalty on typical points (`3.0 mm` mean, `98.6% < 1 cm` uniform; `76.8% < 1 cm` adaptive), **but** worsened adaptive `RMSE` (`44.2 → 49.5 mm`) and worst-case `Max` outliers (`294.6 → 411.8 mm` uniform; `838.0 → 939.4 mm` adaptive) while adding 32 lookup-dependent features.

---

## 3. Strategic Takeaway
- **What Worked**: Upgrading to a PyTorch ResNet (`ReLU`, `LayerNorm`, `AdamW` + cosine LR) and adding **smooth cyclic `(u, v)` cell position** (`torch_res_relu_cellpos_4x4`, `6.8 mm` uniform RMSE, `95.6% < 1 cm`) without any neighbor lookup table.
- **What Did Not**: Across all tested neural feature-engineering variants, `3×3` cell-level slopes and twists—whether raw, unweighted, summed, or Hermite-weighted—did not provide enough net signal over tail-error and complexity costs to justify replacing the grid-free `torch_res_relu_cellpos_4x4` champion.
