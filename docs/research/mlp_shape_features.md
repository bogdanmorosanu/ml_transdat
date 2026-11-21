# Research Unit: Fourier MLP with Cell Shape Features (`mlp_shape_features`)

- **Status**: `Dead End`
- **Primary Experiment**: [`experiments/mlp_shape_features/report.md`](../../experiments/mlp_shape_features/report.md)
- **Implementation**: [`src/models/mlp.py`](../../src/models/mlp.py)
- **Upstream Dependencies**: [`mlp_spatial_regressor`](mlp_spatial_regressor.md), [`cell_planar_features`](cell_planar_features.md)

---

## 1. Scientific Question
Can the champion Fourier MLP (`mlp_fourier_4x4`, `13.0 mm` uniform RMSE / `32.5 mm` adaptive 3 mm RMSE) break past its global capacity ceiling if given the **per-axis cell shape categories** (`cat_dn`, `cat_de`: Linear, Dome/Bowl, S-Shape) engineered in `cell_planar_features`?

---

## 2. Hypotheses & Findings
1. **Discrete Shape Categories as Input Features (`3 mm`, `5 mm`, `10 mm`)**:
   - *Hypothesis*: One-hot encoding `cat_dn` and `cat_de` (6 binary columns) and concatenating them after Fourier-encoded `(N, E)` will help the MLP specialize its fit for parabolic Dome vs. cubic S-Shape cells.
   - *Verdict*: **Rejected (`Dead End`)**. Across all three tolerances (`3 mm`, `5 mm`, `10 mm`), RMSE degraded (`13.0 mm` $\to$ `13.1–14.4 mm` on `ro_test_uniform_4x4`; `32.5 mm` $\to$ `34.1–40.8 mm` on `ro_test_adaptive_3mm_mid`). Piecewise-constant cell labels introduce artificial discontinuities at $11\,\text{km}$ cell boundaries and trigger Adam early stopping prematurely (`49–84 s` vs. `105 s`).
2. **Baseline Error Stratification (`ro_test_adaptive_3mm_mid`)**:
   - *Finding*: `mlp_fourier_4x4` has nearly double the RMSE on 5 mm S-Shape cells (`39.4 mm`, $n=97,108$) compared to Dome/Bowl cells (`21.1 mm`, $n=79,242$).
3. **S-Shape Sample Weighting (`2x`, `3x`)**:
   - *Hypothesis*: Upweighting S-Shape training points during `fit()` will focus network capacity on 3rd-order curves without introducing border cliffs.
   - *Verdict*: **Rejected (`Dead End`)**. Upweighting trimmed worst-case `Max` outliers (`374.7 mm` $\to$ `179.1 mm` uniform) and nudged uniform RMSE (`13.0 mm` $\to$ `12.5 mm`), but slightly degraded `ro_test_adaptive_3mm_mid` RMSE (`32.5 mm` $\to$ `33.7–34.1 mm`).

---

## 3. Strategic Takeaway
Discrete cell-level shape labels do not provide useful signal to a continuous global Fourier MLP. Future neural improvements should focus on **continuous local residual decomposition** (`shift - base_shift` from `(u, v)`) or **multi-scale Fourier features** in PyTorch.
