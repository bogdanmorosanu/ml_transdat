# Multi-Track Sub-Millimeter Exploration (`submillimeter_exploration`)

- **Status**: `Validated`
- **Spawned From**: [`torch_mlp_regressor`](torch_mlp_regressor.md), [`rbf_surface_regressor`](rbf_surface_regressor.md), [`adaptive_quadtree_grid`](adaptive_quadtree_grid.md)
- **Linked Sources**: [`experiments/submillimeter_exploration/`](../../experiments/submillimeter_exploration/report.md), [`docs/exploration/submillimeter_strategy_report.md`](../exploration/submillimeter_strategy_report.md), [`src/models/neural_feature_grid.py`](../../src/models/neural_feature_grid.py), [`src/models/composite_shift.py`](../../src/models/composite_shift.py), [`src/data/samplers.py`](../../src/data/samplers.py)

---

## Current Verdict
Screening 7 improvement tracks (19 model variants) under an apples-to-apples benchmark broke through our previous `4–7 mm` continuous ceiling on two distinct fronts:
1. **Micro-Grid Sub-Millimeter Champion (`T7_bilinear_plus_featgrid_8x8`)**: Pairing a 4-corner bilinear cell base (`CellBilinearShiftModel`) with a learnable multi-resolution 2D feature grid + tiny MLP (`NeuralFeatureGridShiftModel`, `0.66 MB` total) trained on `RO_GRID_8X8` achieves **`0.77 mm` uniform mean (`1.71 mm` RMSE, `76.5% < 1 mm`)**, **`0.84 mm` corridor RMSE (`78.3% < 1 mm`)**, and **`0.71 mm` mean / `0.84 mm` RMSE** in full bidirectional `ETRS89 <-> Stereo70` verification.
2. **Zero-Lookup Neural Breakthrough (`T4_data_adaptive_hybrid`)**: Training the existing `2.21 MB` grid-free ResNet on `RO_TRAIN_ADAPTIVE_HYBRID` (disjoint off-dyadic `ADAPTIVE_1CM` probes + `RO_GRID_4X4`) cuts uniform RMSE by **53%** (`6.80 → 3.22 mm`) and collapses adaptive stress-test RMSE by **94%** (`44.18 → 2.59 mm`, `99.6% < 1 cm`).

---

## Premise & Rationale
Prior continuous models (`tps_densified_4x4`, `torch_res_relu_cellpos_4x4`) plateaued at `4–7 mm` uniform RMSE and `19–44 mm` on adaptive high-curvature stress tests due to three bottlenecks: (1) smoothing over `11 km` $C^1$ cell-boundary slope kinks, (2) dynamic-range overload across `±4.37 m` raw shifts, and (3) uniform `4x4` training density starving steep distortion zones. We designed a 7-track screening suite to isolate the exact millimeter attribution of feature encoding, training optimization, two-stage residual decomposition, curvature-adaptive datasets, learnable 2D feature grids, and untested model families (`SIREN`, `Matérn 3/2` RBF, `HistGBR`).

---

## Key Strategic Findings

- **Bilinear Base + Learnable Feature Grid Matches `11 km` Continuity (`Breakthrough`)**: Aligning a learnable `53×72` + `105×143` bilinear feature grid to the `.spg` lattice naturally models $C^0$ values with $C^1$ cell-border bends; with `8x8` density, `T7_bilinear_plus_featgrid_8x8` (`0.66 MB`) reaches `0.77 mm` uniform mean and `0.84 mm` bidirectional `ETRS89 <-> Stereo70` RMSE ([`experiments/submillimeter_exploration/report.md`](../../experiments/submillimeter_exploration/report.md)).
- **Curvature-Adaptive Training Data Solves Neural Stress-Test Failure (`Breakthrough`)**: `T4_data_adaptive_hybrid` (`0 MB` lookup) proves the `256`-wide ResNet was data-starved in steep regions rather than capacity-limited (`44.18 → 2.59 mm` adaptive RMSE).
- **100% Grid Fit + Huber Loss (`Validated`) & `Matérn 3/2` RBF (`Validated`)**: Removing the `10%` validation holdout on deterministic grids + Huber loss improves ResNet uniform RMSE (`6.80 → 5.81 mm`, `10.6% < 1 mm`), and localized `Matérn 3/2` RBF improves over TPS on `4x4` (`4.66 mm` uniform RMSE, `23.7% < 1 mm`).
- **Confirmed Screening `Dead Ends`**: High-order `k=1..4` cell harmonics (`53.38 mm`), standalone `SIREN` without RFF (`449.56 mm`), and `HistGradientBoostingRegressor` (`52.39 mm`) all degraded accuracy relative to `B0`.
