# Benchmarks Registry

This directory hosts **macro-level, cross-model benchmark suites** for `ml_transdat`.

While `experiments/<slug>/` houses self-contained, single-hypothesis research units and `docs/research/` tracks scientific evolution, `benchmarks/` compares **validated candidate models side-by-side** on canonical test beds using standardized metrics, units, and visualizations.

---

## Active Benchmark Suites

| Suite | Scope | Canonical Workloads | Key Outputs |
| :--- | :--- | :--- | :--- |
| [`grid_shift_models/`](grid_shift_models/) | Cross-family comparison of `BaseGridShiftModel` regressors (KNN, Decision Tree, Adaptive Quadtree) | `corridor_36k` (`ro_corridor_snw`), `uniform_4x4` (`ro_grid_4x4`), `single_cell_10k` (`single_cell_10k`) | [`summary_scorecard.md`](grid_shift_models/summary_scorecard.md), [`plots/`](grid_shift_models/plots/), [`results/`](grid_shift_models/results/) |

---

## Standardized Views

Every benchmark suite evaluates models uniformly in a single inference pass per workload across three dimensions:
1. **Precision (mm)**: Mean 2D, Std Dev, RMSE, Min, P50, P90, P95, P99, Max, Signed Bias ($dN, dE$), and tolerance pass rates (`<1mm`, `<1cm`, `<5cm`, `<10cm`, `<1m`) — reported in **millimeters (mm)** by default.
2. **Performance Speed & Throughput**: Evaluation time ($\text{s}$), throughput ($\text{pts/s}$), per-point latency ($\mu\text{s/pt}$), and fit/index-build time ($\text{s}$) across named workloads (`corridor_36k`, `uniform_4x4`, `single_cell_10k`).
3. **4-Component Memory & Storage Breakdown**:
   - **Code Size**: Model implementation file on disk.
   - **Library Dependencies**: Installed package footprint of external runtime dependencies (e.g., `scikit-learn`, `pyproj`).
   - **Model Instance / Weights**: In-memory weights, split trees, or index metadata.
   - **Grid Lookup Data**: Retained control anchors (e.g., KNN) or indexed lookup tables (e.g., Quadtree lattice arrays).

---

## How to Run

From the workspace root:

```powershell
D:\anaconda3\envs\ml_transdat\python.exe -m benchmarks.grid_shift_models.run_benchmark
```
