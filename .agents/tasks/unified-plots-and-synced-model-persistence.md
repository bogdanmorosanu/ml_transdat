# Task: Unified Multi-Model Row Plotting, Synced Model Persistence & Architecture Refinement

**Date**: 2026-11-23
**Branch**: `main`
**Status**: Completed
**Scope**: Refactoring | Model Architecture | Evaluation | Documentation

## Objective
Refine the experiment and benchmark architecture across `src/`, `experiments/`, `benchmarks/`, and agent skills/docs to:
1. Replace per-model diagnostic PNGs with unified 1-row-per-model figures (`{dataset_id}_{direction}_{plot_type}.png`) via `PlotDispatcher.render_grouped()`.
2. Persist fitted models by default (`persist_model=True`) to gitignored `experiments/<slug>/models/<spec_name>.pkl` with deterministic SHA-256 code+config+grid fingerprinting (`compute_sync_fingerprint`, `load_or_fit`) shared across experiments and benchmarks.
3. Resolve all 6 DRY and consistency audit findings across `src/` and `experiments/`.

## Key Activities & Implementation
- **Unified Row Plotting (`src/viz/plots.py`, `src/viz/dispatcher.py`)**: Implemented `plot_spatial_error_rows`, `plot_hist_ecdf_rows`, `plot_signed_scatter_rows`, `plot_subcell_heatmap_rows`, `plot_vector_quiver_rows`, `plot_multi_dataset_comparison`, and `PlotDispatcher.render_grouped()`. Removed 228 legacy per-model diagnostic PNGs and updated `report.md` links.
- **Fingerprint-Synced Model Persistence (`src/models/base.py`, `src/core/base_unit.py`, `src/core/base_benchmark.py`)**: Added `get_config_dict()`, `_source_code_sha256()`, `compute_sync_fingerprint()`, `save_checkpoint()`, and `load_or_fit()`; wired `persist_model=True` into `ExperimentModelSpec` and `cache_path` into `BenchmarkModelSpec` / `specs_from_unit()`. Fixed PyTorch `__getstate__` (`copy.deepcopy(self.net).cpu()`) to prevent in-place CUDA-to-CPU mutation during checkpointing.
- **Shared Geometry & PyTorch Trainer (`src/geometry/cell_uv.py`, `src/models/_torch_trainer.py`)**: Extracted `get_spg_grid_bounds`, `compute_spg_cell_uv`, and `encode_cyclic_cell_uv` for `TorchMLPGridShiftModel`, `NeuralFeatureGridShiftModel`, and `HistGBRGridShiftModel`; extracted `train_torch_regressor` and picklable loss functions for PyTorch models; added `load_embedded_spg_helmert_params()` to `src/models/helmert2d.py`.
- **Evaluator & Orchestrator Upgrades (`src/evaluation/evaluator.py`, `src/core/base_unit.py`)**: Added `fit_time_sec` propagation and `tabulate_delta_vs_baseline()`; added `TransformationModelSpec` and template-method `run()` on `TransformationExperimentUnit`; upgraded `experiments/bicubic_boundary_analysis/run_unit.py` to subclass `AnalysisExperimentUnit`.
- **Living Docs & Skills Sync**: Updated `.gitignore` (`experiments/*/models/`), `docs/architecture.md`, `experiments/GEMINI.md`, `experiments/README.md`, `.agents/personas/ml-architect.md`, and `experiment-sync`, `benchmark-sync`, `code-audit`, and `task-commit` skills.

## Verification & Results
- **Fingerprint Sync & Cache Invalidation**: Verified cache save, `[CACHE HIT]` load without retraining, and `[CACHE INVALIDATED]` on hyperparameter mutation.
- **Experiments & Benchmarks**: All 14 modified experiments and both benchmark suites passed (`global_analytical_models`: 7/7 `[CACHE HIT]` in `4.26s`; `grid_shift_models`: 10/10 `[CACHE HIT]` in `23.62s`; `torch_mlp_regressor` re-run: 3/3 `[CACHE HIT]`).
- **Hygiene & Lint**: Verified 0 broken `report.md` plot links, 17/17 valid Schema v2 `unit_brief.json` files, 0 `.pkl` files tracked in `git status`, and `ruff check` passed on all modified files.
