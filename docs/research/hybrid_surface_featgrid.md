# Hybrid Stage-1 Surface + Stage-2 Neural Feature Grid (`hybrid_surface_featgrid`)

- **Status**: `Validated`
- **Spawned From**: [`optimal_surface_regressor`](optimal_surface_regressor.md), [`submillimeter_exploration`](submillimeter_exploration.md)
- **Linked Sources**: [`experiments/hybrid_surface_featgrid/`](../../experiments/hybrid_surface_featgrid/report.md), [`src/models/composite_shift.py`](../../src/models/composite_shift.py), [`src/models/neural_feature_grid.py`](../../src/models/neural_feature_grid.py), [`src/data/catalog.py`](../../src/data/catalog.py)

---

## Current Verdict
Chaining `optimal_surface_regressor`'s #1 Stage-1 base (`cubic_k25_s0`, fitted on `RO_GRID_4X4`) with `NeuralFeatureGridShiftModel` (`((53, 72), (105, 143))`, `hidden_dim=128`, `huber` loss, `cyclic_cell_harmonics=2`) trained on `RO_GRID_32X32` achieves **`0.351 mm` mean / `0.412 mm` RMSE (`98.87% < 1 mm`)** on unseen `Adapt 1mm` points in `box10_interior`, and **`0.24 mm` mean (`99.3% < 1 mm`, `P99 = 0.9 mm`)** on Full-Romania `16x16`. However, `cubic_k25_s0`'s `~0.65–0.69 m` concave-border extrapolation spikes—left unpenalized by `huber` loss—inflate Full-Romania `Adapt 1mm` `RMSE` (`14.84 mm` vs. `5.42 mm` for standalone `FeatGrid_32x32_only`).

---

## Premise & Rationale
In `submillimeter_exploration`, `T7_bilinear_plus_featgrid_8x8` broke `1 mm` on top of a simple `11 km` bilinear base. Here we tested whether replacing that bilinear base with the Top-3 `4x4` smooth surfaces (`cubic_k25_s0`, `tps_deg2_k25_s1e-5`, `lpoly_deg2_k16_tric`) via `CompositeResidualShiftModel` yields even sharper sub-millimeter precision across `4x4`, `8x8`, `16x16`, and `32x32` Stage-2 training densities.

---

## Key Strategic Findings
- **Stage-1 Learnability Ranking Vindicated**: In the 48-model Cartesian matrix on `box10_interior`, `cubic_k25_s0` beat `tps_deg2_k25_s1e-5` and `lpoly_deg2_k16_tric` on unseen `Adapt 1mm` (`0.412 mm` vs. `0.488 mm` vs. `1.233 mm` RMSE), confirming `optimal_surface_regressor`'s roughness-based ranking ([`subgrid_matrix.md`](../../experiments/hybrid_surface_featgrid/results/subgrid_matrix.md)).
- **Density & `Huber` + `h128` Synergy**: Training Stage 2 on `4x4` fails (`7.6–11.2 mm`) because Stage 1 already interpolates `4x4`. Scaling to `RO_GRID_32X32` with `huber` loss (`beta=5 mm`) unlocks `hidden_dim=128` (`0.412 mm` vs. `0.936 mm` under `mse`), while `cyclic_cell_harmonics=2` generalizes better than `4` (`0.412 mm` vs. `0.676 mm`).
- **Full-Romania Interior vs. Border Trade-Off**: Nationwide, `Hybrid_cubic_k25_s0_plus_featgrid_32x32` delivers our sharpest interior accuracy (`99.3% < 1 mm`, `0.1–0.6 mm` on 9-cell extreme scenarios), while exposing Stage-1 RBF border extrapolation (`0.651 m` max) as the primary bottleneck vs. standalone `FeatGrid_32x32_only` (`0.244 m` max, `5.42 mm` `Adapt 1mm` RMSE) ([`report.md`](../../experiments/hybrid_surface_featgrid/report.md)).
