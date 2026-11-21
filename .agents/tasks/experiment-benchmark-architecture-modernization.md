# Task: Experiment & Benchmark Architecture Modernization

**Date**: 2026-11-21
**Branch**: `main`
**Status**: Completed
**Scope**: Refactoring | Architecture | Documentation

## Objective
Audit and modernize the `experiments/`, `benchmarks/`, and supporting `src/` layers to support both modeling and non-modeling (`analysis`) experiments cleanly, eliminate `src/`-to-`experiments/` layering inversions and duplicated plotting/benchmarking code, prevent hyperparameter drift between experiments and benchmarks, and introduce `experiment-sync` and `benchmark-sync` skills.

## Key Activities & Implementation
- **3-Category Experiment Hierarchy (`src/core/base_unit.py`)**:
  - Added `ExperimentType` (`grid_shift`, `transformation`, `analysis`), replaced colliding numeric `unit_id`s with `slug`, and standardized `unit_brief.json` Schema v2 (`schema_version`, `slug`, `experiment_type`, `status`, `runtime_s`, `key_metrics`, `artifacts`, `benchmark_candidates`, `extra`).
  - Added `AnalysisExperimentUnit` for exploratory/feature-engineering units (`spg_grid_analysis`, `cell_planar_features`).
- **Drift-Free Benchmark Engine (`src/core/base_benchmark.py`)**:
  - Added `BenchmarkSuite`, `BenchmarkModelSpec`, and `specs_from_unit(...)` to pull promoted `ExperimentModelSpec(..., benchmark_candidate=True)` variants directly from source experiments.
  - Refactored `benchmarks/global_analytical_models/run_benchmark.py` and `benchmarks/grid_shift_models/run_benchmark.py` to subclass `BenchmarkSuite`, aligning `Fourier MLP (4x4, 2.7km)` with `mlp_spatial_regressor`.
- **`src/` Isolation & Externalized Features**:
  - Extracted `plot_surface_comparison_grids` into `src/viz/plots.py`.
  - Refactored `src/geometry/plane.py` to use `SpgReader()` instead of direct `pickle.load()`.
  - Updated `src/geometry/shape_classifier.py` to read bounds from `SpgReader()` and default to `data/features/cell_shape_metrics.parquet`.
- **Retroactive Cleanup Across All 10 Experiments**:
  - Unified `cell_planar_features` into `run_unit.py` (`AnalysisExperimentUnit`), exporting shared tables to `data/features/`.
  - Migrated `adaptive_quadtree_grid/README.md` to `report.md`, removed stray `decision_tree_regressor/scorecard.csv`, audited and retired all 6 `README_AI.md` files after merging unique insights into `report.md`, and added **Benchmark Candidates** sections to promoted reports.
- **Skills & Living Docs**:
  - Created `.agents/skills/experiment-sync/SKILL.md` and `.agents/skills/benchmark-sync/SKILL.md`.
  - Updated `experiments/GEMINI.md`, `experiments/README.md`, `benchmarks/README.md`, `docs/architecture.md`, and `.agents/personas/ml-architect.md`.

## Verification & Results
- **Unit Tests**: `python -m unittest discover -s tests -p "test_*.py" -v` — **48/48 passed**.
- **Lint**: `ruff check` passed with 0 errors.
- **End-to-End Runs**: Re-ran all 10 experiment units and both benchmark suites from scratch with exit code 0.
- **Static Audit**: Confirmed 10/10 experiments conform to `run_unit.py` + `report.md` + Schema v2 `unit_brief.json`, zero `README_AI.md` files remain, and `data/features/*.parquet` exist.
