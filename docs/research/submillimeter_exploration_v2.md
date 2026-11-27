# Sub-Millimeter Exploration v2 (`submillimeter_exploration_v2`)

- **Status**: `Validated` (Phase 1 Screening on `box10_interior`; Full-Romania promotion pending)
- **Spawned From**: [`hybrid_surface_featgrid`](hybrid_surface_featgrid.md), [`hybrid_surface_resnet`](hybrid_surface_resnet.md), [`submillimeter_exploration`](submillimeter_exploration.md)
- **Linked Sources**: [`experiments/submillimeter_exploration_v2/report.md`](../../experiments/submillimeter_exploration_v2/report.md), [`docs/exploration/submillimeter_v2_model_landscape.md`](../exploration/submillimeter_v2_model_landscape.md), [`literature/README.md`](../../literature/README.md), [`src/models/neural_inrs.py`](../../src/models/neural_inrs.py), [`src/models/structured_neural_grids.py`](../../src/models/structured_neural_grids.py), [`src/models/non_neural_ml.py`](../../src/models/non_neural_ml.py), [`src/models/geostat_and_wavelet.py`](../../src/models/geostat_and_wavelet.py)

---

## Current Verdict
Screening **25 models** (`4` anchors + `21` candidate variants across all `12` previously untested tracks) on `box10_interior` (`10×10` interior SPG cells) established three decisive findings:
1. **Border-Safe Micro-Grid Breakthrough (`V2-6`)**: Replacing `cubic_k25_s0` with border-safe `Catmull-Rom` or `CellBilinear` Stage-1 bases paired with a **3-level `NeuralFeatureGrid`** (`53×72`, `105×143`, `209×285`) reaches **`0.111 mm` `U16` mean / `0.154 mm` `A1` RMSE (`100.00% < 1 mm`, `0.665 µs/pt`, `2.78 MB`)** (`V2_6b_catmull_featgrid_3lvl`) and **`0.165 mm` `U16` mean / `0.251 mm` `A1` RMSE (`99.98% < 1 mm`, `0.367 µs/pt`, `1.94 MB`)** (`V2_6a_bilinear_featgrid_3lvl`)—2×–3× more accurate and 50×–100× faster than `A1_Champion_FeatGrid_32x32`.
2. **Zero-Lookup (`0 MB`) Neural Champion (`V2-2a` `WIRE`)**: Complex Gabor Wavelet INRs (`WIRE`) broke `1.0 mm` `A1` RMSE (`0.945 mm`, `74.69% < 1 mm`, `0.465 mm` `U16` mean), outperforming `A2_Champion_ResNet_32x32` (`1.200 mm`), followed closely by `FINER` (`1.023 mm`), `RBF-KAN` (`1.058 mm`), and `FBPINN`/`MoE` (`1.106–1.116 mm`). `MFN` (`3.56–5.21 mm`) is a confirmed `Dead End`.
3. **Zero-PyTorch Deployment Frontier**: `Adaptive Quadtree (1mm)` (`0.555 mm` `A1` RMSE, `1.47 µs/pt`, `74.9 MB`) remains the top Zero-PyTorch accuracy model; `LocalPoly deg=3` (`0.268 mm` `U16` mean / `2.942 mm` `A1` RMSE), `XGBoost` on residuals (`1.771 mm` `A1` RMSE), `LocalKriging` (`2.199 mm`), `PU-RBF` (`2.231 mm`), and `LinearLeafTree` (`2.818 mm`) form the `1–3 mm` Zero-PyTorch tier, while `RFF-ELM` (`4.07 mm`), `LS-SVM` (`4.54 mm`), and `2D Haar Wavelets` (`11.45 mm`) are confirmed `Dead End`s.

---

## Next Step
The fast Full-Romania validation check (`--full-ro`, `*_BORDER`) confirmed:
- **`V2_6a_bilinear_featgrid_3lvl_FullRO`** (`0.068 mm` `U16` mean, `0.098 mm` `A1` RMSE, `100.00% < 1 mm`, `max 0.789 mm`, `0.588 µs/pt`, `1.94 MB`) and **`V2_6b_catmull_featgrid_3lvl_FullRO`** (`0.068 mm` / `0.103 mm`, `2.78 MB`) are our strongest overall sub-millimeter candidates.
- **`V2_4b_lpoly_fbpinn_FullRO`** (`0.423 mm` `U16` mean, `0.893 mm` `A1` RMSE, `80.12% < 1 mm`, `5.26 MB`) is our strongest `0 MB`-lookup neural candidate, overtaking `WIRE` (`3.260 mm`) and `RBF-KAN` (`2.618 mm`) nationwide.
- **`LocalPoly deg=3 (16x16)`** (`0.245 mm` `U16` mean / `1.402 mm` `A1` RMSE, `38.69 MB`) and **`XGBoost`** (`1.559 mm` `A1` RMSE, `59.40 MB`) trail `Adaptive Quadtree (1mm)` (`0.599 mm` `A1` RMSE, `74.91 MB`) nationwide.

Proceed to dedicated champion finalization & fine-tuning (plus bidirectional `ETRS89 <-> Stereo70` verification).
