# Task: Canonical Benchmarks, PyProj Evaluation, and Research-Unit Architecture

**Date**: 2026-11-14
**Status**: Completed
**Scope**: Data Gen | Evaluation | Architecture | Documentation

## Objective
Establish high-density, nationwide canonical test datasets across Romania, implement standard evaluation metrics and diagnostic visualizations, benchmark open-source `pyproj` (`EPSG:3844`) against ground-truth `pytransdat`, and organize experiments into a modular, self-contained Research-Unit Centric Architecture.

---

## Key Activities & Implementation

### 1. Canonical Nationwide Benchmarks (`data/benchmarks/`)
- Generated **`ro_grid_3x3.parquet`** and metadata manifest:
  - Samples all valid $11\,\text{km} \times 11\,\text{km}$ SPG grid cells at normalized sub-grid offsets $(u, v) \in \{0.0, 0.25, 0.50, 0.75\}^2$ ($2.75\,\text{km}$ spacing).
  - Filters to strictly bidirectionally valid points: **36,435 points** (2,222 exact nodes, 34,213 interior points across 2,297 cells).
  - Maximum round-trip error in `pytransdat` is $< 0.47\,\text{mm}$ horizontal, $0.0\,\text{mm}$ vertical.
- Generated **`ro_corridor_snw.parquet`**:
  - Samples **36,435 points** ordered along a 22 km (2-grid-cell) wide corridor traversing South $\to$ Northeast $\to$ West Romania to evaluate spatial locality and caching.

### 2. Standardized Reusable Core Framework (`src/core/`)
- **`src/core/metrics.py`**:
  - Computes per-point 2D/3D residuals ($\Delta N, \Delta E, \Delta H$), converting ETRS89 angular differences into metric distances via GRS80 principal radii of curvature ($M, N$).
  - Produces overall and node-vs-interior stratified scorecards (Mean, Std, RMSE, p50, p95, p99, Max, signed bias, pass rates).
- **`src/core/viz.py`**:
  - Borderless square spatial heatmaps (`marker="s"`, `s=6.5`, `linewidths=0`).
  - Dual-panel Histogram & ECDF distributions.
  - Signed residual vector scatter ($dE$ vs $dN$) with bias and tolerance circles.
  - Sub-cell $(u, v)$ error heatmaps across the $4 \times 4$ intra-cell grid.
  - Vector field quiver plots showing spatial flow of distortion corrections.
- **`src/core/data_samplers.py`**:
  - Parameterized nationwide grid sampling and directional corridor generation.
- **`src/core/base_unit.py`**:
  - Abstract `ResearchUnit` base class standardizing execution, plot export, and `unit_brief.json` serialization.

### 3. Established Self-Contained Research Units (`experiments/`)
- **Unit 01: SPG 2D Distortion Grid Analysis (`experiments/01_spg_grid_analysis/`)**:
  - Profiles the physical 2D distortion field from `rom_grid3d_25.09.spg`.
  - Generates component heatmaps ($dN, dE$), combined magnitude, vector quiver, and diagnostic distributions.
  - Includes executable runner (`run_unit.py`), human report (`report.md`), agent brief (`README_AI.md`), and machine scorecard (`unit_brief.json`).
- **Unit 02: PyProj Baseline Evaluation (`experiments/02_pyproj_baseline/`)**:
  - Benchmarks bidirectional `pyproj` (`EPSG:4258` $\leftrightarrow$ `EPSG:3844`) against `pytransdat`.
  - Includes executable runner (`run_unit.py`), human narrative report (`report.md`), agent brief (`README_AI.md`), and metric JSON outputs (`results/`).
- **Experiments Registry (`experiments/README.md`)**:
  - Central index detailing unit lifecycle states, execution guidelines, and the standardized blueprint for adding future units.

### 4. Living Documentation Synchronization (`docs/research/`)
- Updated **`docs/research/roadmap.md`** with the research topology DAG and status index.
- Created branch journals **`docs/research/01_spg_grid_analysis.md`** and **`docs/research/02_pyproj_baseline.md`**.
- Created **`scratch/README.md`** establishing ground rules for disposable prototypes.
- Updated **`.gitignore`** to ensure canonical benchmark files (`data/benchmarks/`) are tracked in git while ephemeral datasets and caches remain ignored.

---

## Baseline & Distortion Evaluation Results

### Research Unit 01: Ground-Truth SPG Distortion Field (`rom_grid3d_25.09.spg`)
- **Planimetric Shift Magnitude**:
  - Mean: $0.3716\,\text{m}$ | Median (p50): $0.2642\,\text{m}$
  - p95: $1.1420\,\text{m}$ | p99: $1.6841\,\text{m}$ | Max: $2.7570\,\text{m}$
- **Components**:
  - Northing Shift ($dN$): Min = $-2.7364\,\text{m}$, Mean = $-0.0043\,\text{m}$, Max = $+1.2293\,\text{m}$
  - Easting Shift ($dE$): Min = $-0.9421\,\text{m}$, Mean = $+0.0120\,\text{m}$, Max = $+1.1808\,\text{m}$
- **Spatial Vectors**:
  - Strong, coherent southwestward displacement concentrated in Banat/West and northward in Maramureș.

### Research Unit 02: PyProj (`EPSG:3844`) Performance
- **Dataset**: `ro_grid_3x3.parquet` (36,435 points nationwide)
- **Transformation Directions**:
  - `etrs89_to_st70`: Mean 2D = $0.3601\,\text{m}$ | RMSE = $0.4348\,\text{m}$ | Max = $2.8207\,\text{m}$
  - `st70_to_etrs89`: Mean 2D = $0.3601\,\text{m}$ | RMSE = $0.4347\,\text{m}$ | Max = $2.8189\,\text{m}$
- **Systematic Bias**: $dN = +0.0087\,\text{m}$, $dE = -0.0160\,\text{m}$ (negligible global datum translation).
- **Pass Rates**: $< 10\,\text{cm}$: $8.4\%$ | $< 1\,\text{m}$: $98.0\%$ | $< 1\,\text{cm}$: $0.1\%$.
- **Throughput**: $> 3.5 \times 10^6\,\text{pts/s}$ (`etrs89_to_st70`), $> 1.8 \times 10^6\,\text{pts/s}$ (`st70_to_etrs89`).
- **Conclusion**: Standard PROJ resolves the conformal projection and 7-parameter Helmert shift, but errors match the unmodeled SPG distortion grid ($0.36\,\text{m}$ mean, $2.82\,\text{m}$ max). This defines the baseline target for future ML and hybrid models.

---

## Verification & Regression Status
- Executed `experiments.01_spg_grid_analysis.run_unit` $\to$ Success.
- Executed `experiments.02_pyproj_baseline.run_unit` $\to$ Success.
- Executed full test suite:
  ```powershell
  D:\anaconda3\envs\ml_transdat\python.exe -m unittest discover -s tests -v
  ```
  **Result: 6 tests passed in 5.12s (OK)**.
