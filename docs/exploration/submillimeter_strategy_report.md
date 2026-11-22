# Exploratory Strategic Report: Paths Toward 1 mm Precision

- **Status**: `Active`
- **Target Goal**: Replicate the official `pytransdat` / RomGEO mathematical transformation down to **~1 mm precision**
- **Scope**: Full ML pipeline review (problem framing, training datasets, feature engineering, training optimization, and model architectures)
- **Primary References**: [`docs/research/roadmap.md`](../research/roadmap.md), [`docs/research/narrative.md`](../research/narrative.md), [`benchmarks/grid_shift_models/summary_scorecard.md`](../../benchmarks/grid_shift_models/summary_scorecard.md), [`literature/geodetic_grid_interpolation_standards.md`](../../literature/geodetic_grid_interpolation_standards.md)

---

## 1. Executive Summary & Where We Stand Today

Across 14 research units and 2 cross-model benchmark suites, we have progressed from **~44 cm** global linear errors down to **~4–7 mm** with continuous zero-lookup models, and **0.5 mm** with pure geometric lookup tables.

### Plain-English Summary of Our Journey
1. **Flat & Curved Global Formulas Hit a Wall (`44 cm` → `19 cm`)**:
   - Standard GIS (`pyproj_baseline`: `43.5 cm` RMSE) and our refitted 2D Helmert baseline (`helmert2d_baseline`: `44.3 cm` RMSE) act like a flat pane of glass tilted over Romania.
   - Bending that glass into a single nationwide polynomial curve (`polynomial_surface_regressor`, up to Degree 12) cuts error roughly in half to `19.2 cm`, but pushing higher causes severe edge-wobbling along the borders.
2. **Spatial Partitioning & Pure Lookup Tables (`3.5 cm` → `0.5 mm`)**:
   - Local 4-corner distance blending (`knn_spatial_regressor`, `K=4` IDW) drops error to `3.5 cm` (`2.85 cm` on the corridor benchmark).
   - Orthogonal step-plateaus (`decision_tree_regressor`) require `89.4 MB` (`16x16` grid) to reach `4.0 mm`.
   - Our **Adaptive Quadtree** (`adaptive_quadtree_grid`) proves that 1 mm precision is already achievable via a **pure geometric lookup table**: `0.50 mm` mean error (`0.60 mm` RMSE, `93.8% < 1 mm`) at the `1 mm` tier (`74.9 MB` grid lookup), and `1.43 mm` mean (`1.77 mm` RMSE) at the `3 mm` tier (`23.0 MB`).
3. **Continuous Zero-Lookup Models (`1.3 cm` → `3.5–6.8 mm`)**:
   - **Thin Plate Splines (`rbf_surface_regressor`)**: Acting like a flexible elastic sheet pinned at control points, TPS achieved **`8.6 mm` uniform RMSE** (`8.05 mm` corridor RMSE) from just the 2,222 base `11 km` nodes, and **`5.7 mm` uniform RMSE** (`4.27 mm` corridor RMSE, `3.49 mm` mean, `97.5% < 1 cm`) when trained on the `4x4` (`2.75 km`) subgrid.
   - **Fourier MLP & Deep PyTorch ResNet (`mlp_spatial_regressor` → `torch_mlp_regressor`)**: Solving spectral bias with Random Fourier Features brought the `sklearn` MLP to `13.0 mm` uniform RMSE. Upgrading to a GPU-accelerated PyTorch ResNet (`ReLU`, `LayerNorm`, `AdamW` + Cosine LR) plus **smooth cyclic cell-position encoding** (`sin/cos(2πu), sin/cos(2πv)`) dropped uniform RMSE to **`6.8 mm`** (`4.70 mm` corridor RMSE, `3.91 mm` mean, `96.8% < 1 cm`) with **zero lookup grid** (`~2.3 MB` weights).
4. **What Failed (`Dead Ends`) & Why**:
   - Feeding discrete cell shape tags (`mlp_shape_features`: Linear, Dome, S-Shape) or raw `3x3` corner slopes and twists (`mlp_neighbor_features`) directly into a global neural network *degraded* accuracy because those features jump abruptly at every `11 km` cell border.
   - Even when smoothed with 2D cubic Hermite weights (`torch_mlp_regressor` Stage 3), neighbor slopes improved typical uniform accuracy (`3.0 mm` mean, `98.6% < 1 cm`) but worsened worst-case tail outliers (`Max` error grew from `838 mm` to `939 mm` on the adaptive stress test).

### Current Champion Snapshot (`benchmarks/grid_shift_models/summary_scorecard.md`)

| Model | Paradigm | Corridor Mean / RMSE (mm) | Pass < 1 mm (%) | Pass < 1 cm (%) | Grid Lookup Size |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `Adaptive Quadtree (1mm tier)` | Geometric Lookup | **0.50 / 0.60** | **93.8%** | **100.0%** | 74.9 MB |
| `Adaptive Quadtree (3mm tier)` | Geometric Lookup | 1.43 / 1.77 | 39.2% | 99.9% | 23.0 MB |
| `Thin Plate Spline (4x4, 2.7km)` | Continuous Elastic Sheet | 3.49 / 4.27 | 10.0% | 97.5% | 0 MB |
| `PyTorch ResNet + Cyclic CellPos (4x4)` | Continuous Neural Net | 3.91 / 4.70 | 6.7% | 96.8% | 0 MB |
| `Thin Plate Spline (11km)` | Continuous Elastic Sheet | 6.74 / 8.05 | 2.2% | 81.5% | 0 MB |
| `Fourier MLP (4x4, 2.7km)` | Continuous Neural Net | 8.83 / 10.35 | 1.2% | 66.5% | 0 MB |

---

## 2. Diagnosis: Why Our Best Continuous Models Plateau at ~4–7 mm

Before picking new models or features, it helps to understand **why** our best continuous models currently stall around `4–7 mm` nationwide RMSE (and `19–44 mm` on the high-curvature adaptive stress test):

1. **The "Origami vs. Rubber Sheet" Conflict (The 11 km Grid Kink Wall)**
   - As documented in [`literature/geodetic_grid_interpolation_standards.md`](../../literature/geodetic_grid_interpolation_standards.md), the official RomGEO / `pytransdat` ground truth is **not** one globally smooth surface. Inside each `11 km × 11 km` cell, it builds a local cubic patch from a surrounding `4 × 4` neighborhood using one-sided finite differences.
   - At every `11 km` cell border, the surface values meet cleanly (no vertical step jump), **but the slope bends sharply**.
   - Smooth global models (Thin Plate Splines and standard Fourier MLPs) act like a seamless rubber sheet stretched over a surface that actually has subtle "origami creases" every `11 km`. Trying to smooth over those creases creates a permanent **3–5 mm error floor** near borders and in steep curvature zones.
2. **Dynamic Range & Scale Overload (Meters vs. Millimeters)**
   - Right now, a single model is asked to predict raw distortion shifts spanning **`±4,370 mm` (`±4.37 m`)** down to **`1 mm`**—more than a **4,000-to-1 dynamic range**.
   - A neural network spending most of its capacity holding up a `2.7 m` distortion dome in Banat struggles to simultaneously polish `1 mm` ripples in central Romania.
3. **Training Data Blind Spots**
   - Our best continuous models (`tps_densified_4x4`, `torch_res_relu_cellpos_4x4`) are trained on `RO_GRID_4X4` (36,435 points at `2.75 km` spacing).
   - That gives only 16 points per `11 km` cell (7 of which lie directly on cell edges), and it samples flat plains and steep mountain distortion zones at the exact same density. Meanwhile, `pytransdat` is a deterministic mathematical oracle—we can generate **infinite free training data** wherever the model struggles most.

---

## 3. Prioritized Next Steps (Ordered by Value Toward 1 mm)

The following recommendations cover **every part of the pipeline**—problem framing, training datasets, feature engineering, training optimization, and new model families—ordered from highest expected value to longer-term exploration.

---

### Rank 1 (Highest Value): Coarse-to-Fine "Two-Stage Residual" Decomposition
- **Pipeline Component**: Problem Framing & Target Engineering (Zero new dependencies).
- **Plain-English Idea**:
  - Instead of forcing one model to learn the entire `±4,370 mm` distortion field from scratch, split the job into a **macro stage** and a **micro stage**:
    1. **Stage 1 (Macro Sheet)** removes the broad `98–99%` of the distortion field.
    2. **Stage 2 (Micro Corrector)** is trained **only on the leftover millimeter residuals** (typically `±10 to ±30 mm`).
  - **Two concrete ways to test this**:
    - *Option A — Grid-Free Two-Stage Neural / TPS+NN Boost*: Use our `11 km` Thin Plate Spline (`8.6 mm`, `151 KB`) or a low-frequency neural network as Stage 1, freeze it, and train a second high-frequency PyTorch network exclusively on Stage 1's leftover errors.
    - *Option B — Corner-Bilinear Residual*: Subtract the simple 4-corner bilinear shift of each `11 km` cell (which we already explored in `cell_planar_features`) so the neural network only predicts the internal cubic "belly" or "S-curve" (`0` at all 4 corners, peaking at a few millimeters inside the cell).
- **Why It Brings the Most Value**: Shrinking the target numbers by `100×` (from meters to millimeters) removes dynamic-range overload and lets gradient descent focus 100% of its effort on the final millimeters.

---

### Rank 2: Smarter Training Datasets (Curvature-Weighted & Dense Off-Lattice Sampling)
- **Pipeline Component**: Dataset Creation & Sampling Strategy (`src/data/`).
- **Plain-English Idea**:
  - We have an exact ground-truth oracle (`pytransdat`), yet our neural models have mostly been trained on the rigid `4x4` lattice (`36,435` points).
  - **Three high-impact dataset upgrades**:
    1. **Train PyTorch GPU ResNets on `RO_GRID_8X8` (146k pts) and `RO_GRID_16X16` (586k pts)**: We tested `8x8` only on the old `sklearn` CPU MLP (which hit a capacity wall at `30 mm` adaptive), and never tested `8x8` or `16x16` on our GPU PyTorch ResNet or localized TPS.
    2. **Curvature-Adaptive Training Set (Using Our Quadtree)**: We already built `ADAPTIVE_3MM` and `ADAPTIVE_1MM` quadtrees for *testing* and *lookup*, which place `10×` more points in Banat, Maramureș, and Dobrogea where the surface bends sharply. **We have never trained a neural network on an adaptive, curvature-weighted dataset.** Mixing a uniform `4x4` base with adaptive midpoints feeds the model dense examples exactly where its `30–44 mm` errors currently live.
    3. **Continuous Jittered & Border-Enriched Sampling**: Instead of fixed grid fractions (`0.0, 0.25, 0.50, 0.75`), sample random continuous positions inside cells—plus extra samples right next to cell edges (`u, v` near `0.02` and `0.98`)—so the network explicitly learns how slopes bend across `11 km` borders.
- **Why It Brings High Value**: Fixes the mismatch between where the model is tested (`RO_TEST_ADAPTIVE_3MM_MID`) and where it is trained, at near-zero architectural complexity.

---

### Rank 3: Multi-Scale Fourier Features & Grid-Aligned Periodic Harmonics
- **Pipeline Component**: Feature Engineering / Input Encoding (`src/models/torch_mlp.py`).
- **Plain-English Idea**:
  - Right now, our Fourier MLP picks 64 **random** wave frequencies around one single scale (`fourier_scale=3.0`). Random waves do not line up with Romania's exact `11 km × 11 km` cell lattice, and a single scale is too coarse to capture sub-kilometer creases.
  - Notice why `use_cyclic_cell_pos` (`sin(2πu), cos(2πu), sin(2πv), cos(2πv)`) was the *only* engineered feature that improved `torch_mlp_regressor`: it gave the network exact `11 km` grid-aligned waves!
  - **Two upgrades to test**:
    1. **Multi-Harmonic Cell-Aligned Waves**: Expand `cyclic_cell_pos` from just the 1st wave (`k=1`) to **harmonics** (`k = 1, 2, 3, 4`: `sin(2πku), cos(2πku)`) plus simple 2D cross-terms (`sin(2πu)sin(2πv)`, etc.). Because any smooth cubic curve inside a cell can be built from its first few sine/cosine harmonics, this gives the network the exact building blocks of intra-cell curves with **zero lookup table**.
    2. **Multi-Band Global Fourier Bank**: Replace the single `σ=3.0` random wave bank with a multi-scale bank (for example, combining `σ = 1.0, 3.0, 10.0, 25.0`) so the network simultaneously has nationwide rulers (`100 km`), regional rulers (`20 km`), and local rulers (`2 km`).
- **Why It Brings High Value**: Directly addresses spectral bias at the exact spatial scales where the `3–7 mm` residuals live, without needing any external neighbor lookup table.

---

### Rank 4: Training Optimization, Loss Functions & Second-Order Polishing
- **Pipeline Component**: Training Pipeline & Objective Function (`src/models/torch_mlp.py`).
- **Plain-English Idea**:
  - Three quiet bottlenecks exist in how our PyTorch model currently trains:
    1. **Don't Punch Holes in the Training Grid**: Right now, `TorchMLPGridShiftModel` holds out `10%` of the training grid for early stopping. Because `pytransdat` has zero measurement noise, holding out `10%` of the `4x4` grid leaves random holes in the map. Train on **100% of the grid** and monitor validation on a separate off-lattice sample!
    2. **Stop Outliers from Starving Typical Points**: Standard `MSELoss` squares every error. A single `300 mm` border outlier produces `90,000×` more gradient pull than a `1 mm` point, causing the optimizer to ignore the `95%` of Romania sitting at `3–4 mm`. Testing **Smooth-L1 / Huber / Charbonnier Loss** (or training with MSE first and fine-tuning with L1) balances both.
    3. **AdamW → L-BFGS Polishing**: In scientific coordinate networks (like PINNs and neural fields), `AdamW` is great for finding the right neighborhood, but a second-order optimizer (**L-BFGS**) run for the final steps routinely cuts residual approximation error by `2× to 5×`.
- **Why It Brings High Value**: Extracts substantially more accuracy from the *exact same* network and dataset, often turning a `6.8 mm` model into a `2–4 mm` model.

---

### Rank 5: Hybrid Architectures & Compact Learnable Feature Grids
- **Pipeline Component**: Model Architecture (`src/models/`).
- **Plain-English Idea**:
  - Our two strongest families have opposite strengths:
    - **Thin Plate Spline** is ultra-accurate from sparse nodes (`8.6 mm` from 2,222 nodes) and smooth, but slow to evaluate when densified (`43–74 µs/pt`).
    - **PyTorch ResNet** is fast (`2.5–7.4 µs/pt`) and handles GPU batches easily, but struggles with `11 km` border kinks.
  - **Two promising hybrids**:
    1. **Learnable Multi-Resolution Feature Grid + Tiny MLP (Instant-NGP style)**: Instead of storing raw coordinate shifts in a `75 MB` quadtree, store a small learnable grid of 4–8 numbers per node (just a few megabytes) and feed bilinearly interpolated features into a tiny 2-layer PyTorch MLP. Because bilinear grid lookup naturally bends at cell borders (`C⁰` continuous with sharp `C¹` slope changes), it matches RomGEO's exact continuity structure while curving smoothly inside each cell!
    2. **Localized Patch-Basis / Neural Bicubic Head**: Have a neural network predict the local polynomial shape coefficients inside `(u, v)` and evaluate `u^i v^j` directly, giving the network explicit multiplicative power instead of forcing `ReLU` layers to approximate products like `u²v`.

---

### Rank 6: Untested Pure-ML Model Families (SIREN, Matérn GPs, Boosted Trees)
- **Pipeline Component**: New Model Exploration (`src/models/`).
- **Plain-English Assessment**:
  1. **SIREN (Sinusoidal Representation Networks)** — *Worth Testing*: Replaces `ReLU`/`SiLU` activations with periodic sine waves (`sin(ωx)`) in every layer. Widely used in scientific ML for learning continuous spatial fields and their derivatives to sub-millimeter precision. Easy to add as an option in `TorchMLPGridShiftModel`.
  2. **Gaussian Processes / Matérn RBF Kernels** — *Worth Testing on Subsets*: Unlike Thin Plate Splines (which assume very smooth bending) or Gaussian RBFs (infinitely smooth), a **Matérn `3/2` or `5/2` kernel** explicitly models surfaces with finite smoothness—much closer to Romania's `C⁰/C¹` field—and also provides uncertainty estimates.
  3. **Gradient Boosted Trees (`HistGradientBoostingRegressor` / `XGBoost`)** — *Low Priority*: Unlike a single Decision Tree (`89 MB` for `4 mm`), an ensemble of hundreds of shallow boosted trees sums overlapping steps into a much finer staircase. However, it still produces flat micro-steps rather than smooth curves and extrapolates poorly, so continuous neural/spline approaches remain better aligned with `1 mm` precision.

---

### Rank 7: End-to-End Bidirectional & 3D Verification
- **Pipeline Component**: Full Pipeline Integration (`ResidualCoordinateTransformer`).
- **Plain-English Idea**:
  - Once a 2D shift model breaks below **`2 mm` RMSE**, plug it into `ResidualCoordinateTransformer` and benchmark the **full round-trip** (`ETRS89 → Stereo70 → ETRS89`), verifying that iterative inverse lookup (`etrs89_to_st70`) converges cleanly and checking how it pairs with the 1D quasigeoid height lookup.

---

## 4. Summary Priority Matrix

| Rank | Strategy | Pipeline Component | Expected Impact | Implementation Effort | Lookup Table Needed? |
| :---: | :--- | :--- | :---: | :---: | :---: |
| **1** | **Coarse-to-Fine Two-Stage Residual** | Target Framing | **Very High** (`100×` smaller targets) | Low–Medium | Optional (0 MB if NN+NN or TPS+NN) |
| **2** | **Adaptive + `8x8`/`16x16` + Border Training Data** | Dataset & Sampling | **High** (fixes adaptive blind spots) | Low | No (`0 MB`) |
| **3** | **Harmonic Cell Waves + Multi-Band Fourier** | Feature Engineering | **High** (matches `11 km` cell math) | Low | No (`0 MB`) |
| **4** | **100% Grid Fit + Huber/L1 + L-BFGS Polish** | Training & Loss | **Medium–High** (`2×` tighter fit) | Low | No (`0 MB`) |
| **5** | **Learnable Feature Grid + Tiny MLP / Poly Head** | Hybrid Architecture | **High** (natively models `C¹` kinks) | Medium | Small (`1–5 MB` learnable grid) |
| **6** | **SIREN Activations / Matérn Kernels** | New Model Families | **Medium** (better smooth/derivative fit) | Low–Medium | No (`0 MB`) |
| **7** | **Full Bidirectional & 3D Validation** | System Integration | **Required Gate** before declaring 1 mm | Low | Depends on chosen model |

---

## 5. Honest Feasibility Verdict: Can We Actually Reach 1 mm?

A fair exploratory review also has to ask: *are we near the theoretical limit of what is possible, or is there still real headroom?*

**The honest answer is that `1 mm` is achievable—and we have not hit the ceiling yet—because our current `4–7 mm` plateau is caused by solvable mismatches rather than a fundamental impossibility:**

1. **If a compact lookup table (`5–75 MB`) is acceptable**:
   - We **already reached `0.50 mm` mean (`0.60 mm` RMSE, `93.8% < 1 mm`)** with `Adaptive Quadtree (1mm tier)` (`74.9 MB`).
   - Replacing flat bilinear quadtree leaves with a **Learnable Multi-Resolution Feature Grid + Tiny MLP** or **local quadratic/cubic patches** could plausibly keep `~1 mm` accuracy in under **`5 MB`**.
2. **If we require a pure continuous model (`0 MB` grid lookup)**:
   - A single *globally smooth* rubber-sheet model (like plain TPS or a basic MLP) *does* hit a physical wall around **`3–5 mm`** because it cannot bend sharply at `11 km` cell borders.
   - **However**, those borders sit on a known, regular `11 km` lattice. By combining **Coarse-to-Fine Residual decomposition (Rank 1)**, **Adaptive + Dense training data (Rank 2)**, **Cell-Aligned Harmonic Waves (Rank 3)**, and **L-BFGS / Smooth-L1 polishing (Rank 4)**, a zero-lookup neural model does not have to fight the `11 km` lattice—it can align with it. Pushing continuous zero-lookup RMSE from `6.8 mm` down into the **`1–2 mm` range** is a realistic, testable target.

---

## 6. Recommended Immediate Action Plan (Next 3 Experiments)

If we want the fastest path to testing whether continuous ML can cross below `2 mm` toward `1 mm`:

1. **Experiment A — Training & Encoding Quick Wins on `TorchMLPGridShiftModel`**:
   - Train on `RO_GRID_8X8` + `ADAPTIVE_3MM` midpoints (instead of just `4x4`), use `100%` of the grid (off-lattice validation), add multi-harmonic cyclic cell waves (`k=1..4`) + multi-band global Fourier features (`σ ∈ {1, 3, 10, 25}`), and test Smooth-L1 + L-BFGS polishing.
2. **Experiment B — Coarse-to-Fine Two-Stage Residual Pipeline**:
   - Use `tps_base_optimal_2e-4` (`151 KB`, `8.6 mm`) or a Stage-1 low-frequency ResNet as the macro base, and train a Stage-2 high-frequency PyTorch ResNet strictly on the leftover millimeter residuals.
3. **Experiment C — Learnable 2D Feature Grid + Tiny MLP (or SIREN Comparison)**:
   - Compare a `SIREN` continuous network (`0 MB` grid) against a compact learnable multi-resolution 2D feature grid + tiny MLP (`~2–5 MB`) to see which breaks the `1 mm` barrier cleanly across both `RO_TEST_UNIFORM_4X4` and `RO_TEST_ADAPTIVE_3MM_MID`.

---

## 7. Empirical Screening Verdicts (`experiments/submillimeter_exploration/`)

All 7 tracks were screened under a unified benchmark in [`experiments/submillimeter_exploration/report.md`](../../experiments/submillimeter_exploration/report.md) (19 model variants evaluated across `RO_TEST_UNIFORM_4X4`, `RO_TEST_ADAPTIVE_3MM_MID`, and `CORRIDOR_SNW`).

### What Broke Through (High-Value Leads)
1. **Bilinear Base + Learnable Multi-Resolution 2D Feature Grid (`T7_bilinear_plus_featgrid_8x8`) — `0.77 mm` Uniform Mean / `0.84 mm` Corridor RMSE (`0.66 MB`)**:
   - Combining a simple 4-corner bilinear base with a compact learnable 2D feature grid (`53x72` + `105x143`, `feature_dim=8`) + tiny 2-layer MLP reached **`76.5% < 1 mm`** nationwide (`78.3% < 1 mm` on corridor, `98.3% < 1 cm` on adaptive stress test) and passed full bidirectional `ETRS89 <-> Stereo70` verification (`0.84 mm` RMSE in both directions).
2. **Curvature-Adaptive Training Data on Zero-Lookup ResNet (`T4_data_adaptive_hybrid`) — `2.48 mm` Uniform Mean / `2.59 mm` Adaptive RMSE (`0 MB` lookup)**:
   - Training the exact same `2.21 MB` grid-free ResNet on `RO_TRAIN_ADAPTIVE_HYBRID` (disjoint off-dyadic `ADAPTIVE_1CM` probes + `RO_GRID_4X4`) cut adaptive stress-test RMSE by **94%** (`44.18 → 2.59 mm`) and uniform RMSE by **53%** (`6.80 → 3.22 mm`).
3. **100% Grid Training + Huber Loss (`T2_huber_fullgrid_4x4`) & Matérn 3/2 Local RBF (`T6_matern32_rbf_4x4`)**:
   - Removing the 10% validation holdout + using Huber loss improved ResNet uniform RMSE (`6.80 → 5.81 mm`, `<1mm` from `6.5% → 10.6%`).
   - Switching RBF from Thin Plate Spline to localized **Matérn 3/2** improved uniform RMSE to **`4.66 mm`** (`23.7% < 1 mm`) and corridor RMSE to **`3.71 mm`**.

### What Did Not Work (`Dead Ends` in Screening)
- **High-order `k=1..4` cell harmonics (`T1_harmonics4`)**: Degraded ResNet RMSE (`6.80 → 53.38 mm`) due to high-frequency aliasing.
- **Standalone `SIREN` (`T6_siren_4x4`)**: Failed to converge well without RFF (`449.6 mm` uniform RMSE).
- **`HistGradientBoostingRegressor` (`T6_hist_gbr_4x4`)**: Piecewise-constant trees stalled at `52.4 mm` uniform RMSE (`144.3 mm` adaptive).

### Next Deep-Dive Candidates
- **Path A (Micro-Grid Champion, `< 1 mm`)**: Train `T7_bilinear_plus_featgrid` on `RO_TRAIN_ADAPTIVE_HYBRID` (or `16x16` + adaptive) + Huber loss to push `< 1 mm` pass rate from `76.5%` toward `> 95%` and eliminate remaining `185 mm` edge outliers while staying under `1 MB`.
- **Path B (Pure Zero-Lookup Champion, `0 MB` grid)**: Combine `T4_data_adaptive_hybrid` + `T2_huber_fullgrid` + wider/deeper ResNet capacity to test how close a 100% grid-free neural network can get to `1 mm`.

