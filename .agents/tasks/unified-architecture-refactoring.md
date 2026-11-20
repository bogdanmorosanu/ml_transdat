# Task: Unified Architecture Refactoring & Cross-Model Benchmarking

**Date**: 2026-11-20  
**Branch**: `infra/unified-architecture`  
**Status**: Completed  
**Scope**: Model Architecture | Data Gen | Evaluation | Refactoring | Documentation

## Objective
Refactor and unify the architecture across `ml_transdat` into clean, decoupled, high-performance layers. Establish standard abstractions for coordinate transformations and 2D grid shifts, implement dynamic on-the-fly grid generation with chunking and disk caching, unify evaluation and declarative visualization (standardizing on millimeters `mm`), optimize model inference (achieving >3,300x speedup for `AdaptiveQuadtreeRegressor`), establish a cross-model benchmark suite under `benchmarks/`, resolve all linting/formatting errors with Ruff, and synchronize living documentation.

---

## Key Activities & Implementation

### 1. Abstract Model Hierarchy & Memory Profiling (`src/models/`)
- **`BaseSpatialModel`**: Abstract base class defining universal contracts: `fit`, `predict`, `evaluate`, and `memory_breakdown`. Introduced centralized code-size inspection (`_code_size_bytes()`).
- **`BaseGridShiftModel`**: Standard contract for predicting 2D shift components `(dN, dE)` given `(N, E)` in Stereo70.
- **`BaseCoordinateTransformer`**: Contract for full forward/inverse 3D coordinate transformations.
- **`ResidualCoordinateTransformer`**: Hybrid composition coupling an analytical base transformer (e.g. `PyprojEpsg3844Transformer` or `Helmert2D`) with a residual `BaseGridShiftModel`. Fixed sign conventions to match ground-truth `pytransdat/trans_ro.py`:
  - Direct ($ETRS89 \to Stereo70$): $N = N_{\text{approx}} + dN$, $E = E_{\text{approx}} + dE$.
  - Inverse ($Stereo70 \to ETRS89$): $N_{\text{approx}} = N - dN$, $E_{\text{approx}} = E - dE$.
- **`MemoryBreakdown`**: Structured breakdown tracking `model_weights_bytes`, `lookup_data_bytes`, `library_overhead_bytes` (measured dynamically via `importlib.metadata`), and total bytes.
- **Model Implementations & Optimizations**:
  - `KNNSpatialRegressor`: Standardized on `(N, 2)` float64 arrays, restored `kneighbors()` query support, declared `"scikit-learn"` runtime dependency.
  - `DecisionTreeSpatialRegressor`: Standardized on `(N, 2)` arrays, modern reST docstrings and typing.
  - `AdaptiveQuadtreeRegressor`: Re-architected inference from slow recursive tree walking ($64.6\,\text{s}$) to a vectorized regular-lattice index ($0.0195\,\text{s}$, a **>3,300x speedup**). Grouped cells by dimension level, hashed $(N, E)$ coordinates to packed 64-bit integer keys, and resolved nearest leaf cells via `np.searchsorted`.
  - `PyprojEpsg3844Transformer`: Extracted and standardized into `src/models/pyproj_transformer.py`.

### 2. Dynamic Grid Generation & Lazy Streaming (`src/data/`)
- **`catalog.py`**:
  - Enumerated `GridId` identifiers (`SPG_NODES_VALID`, `RO_GRID_4X4`, `RO_GRID_8X8`, `RO_GRID_16X16`, `RO_GRID_32X32`, `QUADTREE_2CM`, `QUADTREE_1CM`, `QUADTREE_3MM`, `QUADTREE_1MM`).
  - Implemented `GridManager` with local disk caching in `.cache/grids/` (git-ignored), transparently generating densified grids on-the-fly when not cached.
  - Implemented `get_default_grid_manager()`.
- **`samplers.py`**:
  - `generate_densified_grid_df`: Vectorized computation of densified grid coordinates and ground-truth shift lookups via `pytransdat.trans_ro.TransRO.direct_shift_at`.
  - `iter_densified_grid_chunks`: Memory-safe chunked generator streaming arbitrary-resolution grid partitions.
  - `generate_single_cell_workload_df`: Generates localized test workloads within individual SPG cells to evaluate spatial caching and locality.
- **Storage Cleanup**: Deleted all large redundant pre-generated parquet files in `data/grids/densified/` (`~50 MB`), reducing repository footprint.

### 3. Unified Metrics & Evaluation Framework (`src/evaluation/`)
- **`metrics.py`**:
  - Enumerated `MetricType` (MEAN, MEDIAN, STD, MAX, MIN, P95, RMSE) and defined `MetricMeta` with explicit unit symbols (`mm`, `m`, `sec`, `pts/sec`, `bytes`, `MB`). Standardized precision reporting on millimeters (`mm`).
  - Defined `MetricGroup` (PRECISION_2D, RESIDUAL_COMPONENTS, PERFORMANCE_SPEED, MODEL_COMPLEXITY).
  - Implemented `NaN`-safe `_summarize_slice()` and `dataframe_to_markdown()`.
- **`evaluator.py`**:
  - Implemented `ModelEvaluator` with `evaluate_model()`, `evaluate_models()`, `tabulate_results()`, and `compare_models()`.
  - Standardized `EvaluationResult.to_scorecard_row()` with configurable precision scaling (defaulting to `"mm"`).

### 4. Declarative Visualization Framework (`src/viz/`)
- **`registry.py`**: Enumerated `PlotType` (`MODEL_COMPARISON`, `SPEED_WORKLOADS`, `MEMORY_BREAKDOWN_STACKED`) and `PlotGroup`.
- **`dispatcher.py`**: Extensible `PlotDispatcher` with registered plot handlers and strict `ValueError` handling for unknown plot types.
- **`plots.py`**:
  - `plot_model_comparison`: Multi-model precision bar charts across components with clean millimetric scaling.
  - `plot_speed_workloads`: Multi-workload throughput and inference latency charts.
  - `plot_memory_breakdown_stacked`: Two-panel stacked bar chart separating Model + Lookup data (panel 1, log scale) from Full Runtime Deployment including framework overhead (panel 2, linear MB scale).

### 5. Experiment Unit Lifecycle & Clean Base Units (`src/core/base_unit.py`)
- Refactored `TransformationExperimentUnit` and `GridShiftExperimentUnit` inheriting from `BaseExperimentUnit`.
- Standardized unit lifecycle: `setup()`, `run()`, `evaluate()`, `visualize()`, and `teardown()`.
- Migrated all 6 experiment units:
  - `spg_grid_analysis`
  - `pyproj_baseline`
  - `helmert2d_baseline`
  - `knn_spatial_regressor`
  - `decision_tree_regressor`
  - `adaptive_quadtree_grid` (removed redundant duplicate `run.py`)

### 6. Top-Level Cross-Model Benchmarks (`benchmarks/grid_shift_models/`)
- Established dedicated top-level `benchmarks/` root directory.
- Implemented `benchmarks/grid_shift_models/run_benchmark.py`:
  - Evaluated `KNNSpatialRegressor (K=4, IDW)`, `DecisionTreeSpatialRegressor (depth=22)`, and `AdaptiveQuadtreeRegressor (1mm threshold)`.
  - Assessed accuracy against the `ro_corridor_dense` benchmark (8,518 points) in millimeters (`mm`).
  - Benchmarked inference speeds across 3 workloads: Single Point (1 pt), Local SPG Cell (1,000 pts), and Corridor Benchmark (8,518 pts).
  - Profiling memory breakdown for all four components.
  - Generated `summary_scorecard.md`, `results/benchmark_summary.json`, and 3 high-resolution publication-quality plots:
    1. `01_precision_comparison.png`
    2. `02_performance_speed.png`
    3. `03_memory_breakdown_stacked.png`

### 7. Code Quality, Ruff Audits & Documentation Synchronization
- Installed and ran Ruff: fixed all 85 identified linting issues (type annotations, docstrings, imports, formatting) down to 0 errors.
- Created `docs/architecture.md` detailing module boundaries, dataflow, and extension guides.
- Created `src/GEMINI.md` to instruct future AI agents on architecture invariants and synchronization requirements.
- Updated `benchmarks/README.md`, `data/grids/README.md`, `experiments/GEMINI.md`, `experiments/README.md`, `.agents/personas/ml-architect.md`, and project `GEMINI.md`.

---

## Verification & Results

### Automated Test Suite
- Executed full test suite: `python -m unittest discover -s tests -v`
  - **Result**: `27/27 tests passed` in `8.43s`.
  - Verified tests across `test_unified_architecture.py`, `test_knn_regressor.py`, `test_decision_tree_regressor.py`, `test_ro_grid_benchmark.py`, and `test_eval_suite.py`.

### Linting & Formatting
- `ruff check src benchmarks experiments tests scripts`: **All checks passed! (0 errors)**
- `ruff format --check src benchmarks experiments tests scripts`: **51 files already formatted.**

### Cross-Model Benchmark Suite
- Executed: `python -m benchmarks.grid_shift_models.run_benchmark`
  - Exit code `0`.
  - **Precision Summary (Corridor Dense Benchmark, 8,518 points)**:
    - `KNN (K=4, IDW)`: Mean 2D Error = **34.78 mm**, Max = **250.60 mm**, RMSE = **44.97 mm**
    - `Decision Tree (depth=22)`: Mean 2D Error = **17.91 mm**, Max = **117.85 mm**, RMSE = **23.51 mm**
    - `Adaptive Quadtree (1mm)`: Mean 2D Error = **0.50 mm**, Max = **1.70 mm**, RMSE = **0.57 mm**
  - **Inference Speed (Corridor Dense Benchmark, 8,518 points)**:
    - `Adaptive Quadtree (1mm)`: **0.0195 s** (~437,000 pts/sec)
    - `Decision Tree (depth=22)`: **0.0210 s** (~405,000 pts/sec)
    - `KNN (K=4, IDW)`: **0.0630 s** (~135,000 pts/sec)
