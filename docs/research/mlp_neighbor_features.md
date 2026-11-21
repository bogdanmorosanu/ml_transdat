# Research Unit: Fourier MLP with 3×3 Cell Neighbor Features & Center Distance (`mlp_neighbor_features`)

- **Status**: `Dead End`
- **Primary Experiment**: [`experiments/mlp_neighbor_features/report.md`](../../experiments/mlp_neighbor_features/report.md)
- **Implementation**: [`src/models/mlp.py`](../../src/models/mlp.py) | [`src/geometry/neighbor_features.py`](../../src/geometry/neighbor_features.py)
- **Upstream Dependencies**: [`mlp_spatial_regressor`](mlp_spatial_regressor.md), [`cell_neighbor_features`](cell_neighbor_features.md), [`mlp_shape_features`](mlp_shape_features.md)

---

## 1. Scientific Question
Can the champion Fourier MLP (`mlp_fourier_4x4`, `13.0 mm` uniform RMSE / `32.5 mm` adaptive 3 mm RMSE) break past its global capacity ceiling when given **continuous $3 \times 3$ cell neighbor geometry features** (32 one-sided corner slopes and 8 diagonal twists from `cell_neighbor_features`) and/or a **continuous $C^0$ intra-cell midpoint distance** (`2|u - 0.5|, 2|v - 0.5|`)?

---

## 2. Hypotheses & Findings
1. **Continuous Neighbor Slopes (32) & Twists (40) as Standardized Inputs**:
   - *Hypothesis*: Unlike discrete one-hot shape tags (`mlp_shape_features`), continuous corner slopes and twists capture the exact quantitative derivatives used by bicubic interpolation without categorical quantization.
   - *Verdict*: **Rejected (`Dead End`)**. On valid interior cells (`36,752` uniform / `176,350` adaptive), adding 32 slopes degraded RMSE (`13.0 mm` $\to$ `14.8 mm` uniform; `32.5 mm` $\to$ `49.0 mm` adaptive), and adding all 40 slopes + twists degraded it slightly further (`15.4 mm` uniform; `51.4 mm` adaptive).
2. **Continuous $C^0$ Cell-Midpoint Distance (`center_dist`)**:
   - *Hypothesis*: Supplying a continuous triangle-wave distance from each $11\,\text{km}$ cell midpoint (`0.0` at center, `1.0` at edges) provides intra-cell position without vertical boundary jumps, letting the MLP combine local position with corner derivatives.
   - *Verdict*: **Rejected (`Dead End`)**. Adding `center_dist` modestly improved the 40-feature model (`15.4` $\to$ `14.5 mm` uniform; `51.4` $\to$ `47.0 mm` adaptive), confirming that intra-cell position helps partially unlock neighbor-feature signal, but still trailed the bare baseline (`13.0 mm` uniform; `32.5 mm` adaptive). Adding `center_dist` alone also hurt (`13.7 mm` uniform; `42.0 mm` adaptive).

---

## 3. Strategic Takeaway
- **Root Cause**: Cell-level neighbor features remain **piecewise-constant** within each $11 \times 11\,\text{km}$ cell (acting as cell-level memorization keys under `4x4` subgrid sampling) and an additive `ReLU(Wx + b)` MLP cannot cleanly multiply local coordinates by corner derivatives.
- **Next Direction**: Future neural attempts to use neighbor geometry should move to a dedicated **PyTorch** architecture (with its own clean baseline first) using explicit **multiplicative/bilinear interactions** or **local residual polynomial/bicubic layers** over asymmetric $(u, v) \in [0, 1]^2$.
