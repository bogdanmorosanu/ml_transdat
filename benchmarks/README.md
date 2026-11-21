# Benchmarks Registry & Cross-Model Taxonomy

This directory hosts **macro-level, cross-model benchmark suites** for `ml_transdat`.

While `experiments/<slug>/` houses self-contained, single-hypothesis research units and `docs/research/` tracks scientific evolution, `benchmarks/` compares **validated candidate models side-by-side** on canonical test beds using standardized metrics, units, and visualizations.

---

## 1. Benchmark Suites Taxonomy & Splitting Criteria

To prevent scale distortion (e.g., $440\,\text{mm}$ global models flattening $0.5\,\text{mm}$ local models into invisible chart slivers) and reflect different operational roles, models are separated into distinct benchmark suites according to three criteria:

1. **Locality & Inductive Bias**:
   - **Global Models**: A single nationwide closed-form equation evaluates every coordinate $(u, v) \to (\Delta N, \Delta E)$ without nearest-neighbor search or localized cell lookups.
   - **Local / Discretized & Neural Models**: Space is partitioned into cells, quadtrees, k-d trees, or continuous neural coordinate networks to capture steep local gradients.
2. **Architecture & Storage Footprint**:
   - **Pure Closed-Form Equations**: $0\,\text{MB}$ lookup data; stores only $3\text{--}91$ algebraic coefficients.
   - **Lookup-Backed / Weight Tensors**: Retains control stations (KNN), binary split trees (Decision Trees), quadtree cell arrays (Adaptive Quadtrees), or neural weight matrices (Fourier MLPs).
3. **Precision & Measurement Scale**:
   - **Decimeter Tier** ($15\text{--}45\,\text{cm}$): Plotted with centimeter-scale clarity.
   - **Sub-Centimeter / Millimeter Tier** ($0.5\text{--}35\,\text{mm}$): Plotted with millimeter-scale precision.

---

## 2. Active Benchmark Suites

| Suite Directory | Modeling Paradigm | Target Accuracy | Models Evaluated | Key Outputs |
| :--- | :--- | :---: | :--- | :--- |
| [`global_analytical_models/`](global_analytical_models/) | **Global Closed-Form Mathematical Surfaces**: Algebraic polynomials & conformal planes with zero lookup tables ($0\,\text{MB}$). | **Decimeter**<br>($160\text{--}365\,\text{mm}$ on corridor) | • Helmert 2D Conformal (4P)<br>• Linear Affine Plane (Deg 1, 6T)<br>• Quadratic Bowl & Saddle (Deg 2, 12T)<br>• Cubic Surface (Deg 3, 20T)<br>• Quintic Surface (Deg 5, 42T)<br>• Multi-Harmonic (Deg 8, 90T)<br>• High-Order Limit (Deg 12, 182T)<br>• Sparse Lasso (Deg 5, 24T) | [`summary_scorecard.md`](global_analytical_models/summary_scorecard.md)<br>[`plots/`](global_analytical_models/plots/)<br>[`results/`](global_analytical_models/results/) |
| [`grid_shift_models/`](grid_shift_models/) | **Local Spatial Partitioning & Continuous Neural Fields**: High-precision discrete partitions and continuous parametric neural networks. | **Millimeter to Centimeter**<br>($0.5\text{--}35\,\text{mm}$ on corridor) | • KNN ($K=4, 16$ IDW, 11km)<br>• Decision Tree (11km base & 8x8 pruned)<br>• Adaptive Quadtree (3mm & 1mm tiers)<br>• Fourier MLP (4x4 subgrid, 2.7km)<br>• Thin Plate Spline (11km base & 4x4 subgrid) | [`summary_scorecard.md`](grid_shift_models/summary_scorecard.md)<br>[`plots/`](grid_shift_models/plots/)<br>[`results/`](grid_shift_models/results/) |

---

## 3. Standardized Evaluation Views

Every benchmark suite evaluates models uniformly in a single inference pass per workload across three dimensions:

1. **Precision (mm)**: Mean 2D, Std Dev, RMSE, Min, P50, P90, P95, P99, Max, Signed Bias ($dN, dE$), and tolerance pass rates (`<1mm`, `<1cm`, `<5cm`, `<10cm`, `<1m`) — reported in **millimeters (mm)**.
2. **Performance Speed & Throughput**: Evaluation time ($\text{s}$), throughput ($\text{pts/s}$), per-point latency ($\mu\text{s/pt}$), and fit/index-build time ($\text{s}$) across named workloads (`corridor_36k`, `uniform_4x4`, `single_cell_10k`).
3. **4-Component Memory & Storage Breakdown**:
   - **Code Size**: Model implementation file on disk.
   - **Library Dependencies**: Installed package footprint of external runtime dependencies (e.g., `scikit-learn`, `numpy`).
   - **Model Instance / Weights**: In-memory parameters, split trees, or network weights.
   - **Grid Lookup Data**: Retained control anchors (KNN) or indexed quadtree cell tables ($0\,\text{MB}$ for analytical and neural models).

---

## 4. How to Run the Benchmarks

All benchmark suites are runnable from the workspace root using the project conda environment:

```powershell
# 1. Run Global Analytical Models Benchmark (Helmert / Polynomials / Sparse Lasso)
D:\anaconda3\envs\ml_transdat\python.exe -m benchmarks.global_analytical_models.run_benchmark

# 2. Run High-Precision Grid Shift Models Benchmark (Quadtrees / Trees / KNN / Fourier MLP)
D:\anaconda3\envs\ml_transdat\python.exe -m benchmarks.grid_shift_models.run_benchmark
```
