# Task: Codebase Audit & Architectural Refinement

**Date**: 2026-11-24
**Branch**: `infra/codebase-refinement`
**Status**: Completed
**Scope**: Refactoring | Model Architecture | Evaluation | Documentation

## Objective
Conduct a full-codebase architectural audit of `src/`, `experiments/`, `benchmarks/`, and `tests/` to eliminate god-modules, cross-experiment code duplication, and stale legacy artifacts, while establishing automated architectural guardrails and synced living documentation for scalable future experiments.

## Key Activities & Implementation
- **Phase 1 — Baseline Health, Tooling & Repo Hygiene**:
  - Created `infra/codebase-refinement` branch and added `pyproject.toml` pinning `[tool.ruff]` (`py312`, `line-length = 100`, rules `E, F, I, UP, B, RUF`).
  - Resolved all 11 pre-existing Ruff lint errors across `src/` and `experiments/`.
  - Flattened `tests/unit/test_rbf_surface.py` → `tests/test_rbf_surface.py`, removed `scratch/test_axis_decoupling.py`, and removed `experiments/helmert2d_baseline/__init__.py`.
- **Phase 2 — `src/viz/` Decomposition, `transect_66km` Retirement & Core Primitives**:
  - Split `src/viz/plots.py` (2,025 lines) into 4 cohesive domain modules (`src/viz/diagnostics.py`, `src/viz/scenarios.py`, `src/viz/surfaces.py`, `src/viz/benchmarks.py`) and removed `src/viz/plots.py`.
  - Retired `PlotType.TRANSECT_66KM`, `plot_transect_66km`, and `transect_models` across `src/viz/`, `src/core/base_unit.py`, and `tests/test_unified_architecture.py` (fixing the failing `test_plot_registry_and_dispatcher` test).
  - Extracted `src/data/spg_cache.py` (`get_default_grid2d`, `get_default_spg_reader`) and `src/data/regions.py` (`CellRegion`, `find_worst_case_region`, `concat_datasets`).
  - Unified `_extract_model_config()` in `src/core/base_unit.py` to delegate to `BaseSpatialModel.get_config_dict()`, added `extreme_scenario_label` on `ExperimentModelSpec`, and replaced the `__probe__` directory hack with `probe_only=True` in `ResearchUnit` / `specs_from_unit()`.
- **Phase 3 — Cross-Experiment DRY & Legacy Transect Plot Replacement**:
  - Rendered canonical 9-cell extreme scenario plots (`extreme_scenarios_gradient_and_linear_9cells.png` & `extreme_scenarios_curvature_and_kinks_9cells.png`) across all 9 trainable `grid_shift` units from cached `.pkl` checkpoints, deleted all 8 legacy `transect_66km` / `cell_transect` PNGs, and updated `report.md` and `unit_brief.json` references.
  - Wired `experiments/bicubic_boundary_analysis/run_unit.py` and `experiments/bicubic_interpolation_forensics/run_unit.py` to import shared scenarios and `eval_pytransdat_1d_safe` from `src.viz.scenarios`.
  - Extracted `SpatialFeatureEncoder` (`src/models/_features.py`) and refactored `MLPGridShiftModel` (`src/models/mlp.py`) and `TorchMLPGridShiftModel` (`src/models/torch_mlp.py`) to compose it with identical `__init__` signatures.
  - Cleaned up duplicate helper and `__probe__` check in `experiments/submillimeter_exploration/run_unit.py`.
- **Phase 4 — Automated Guardrail Tests & Living Docs Sync**:
  - Added `tests/test_architecture_invariants.py` (5 automated checks: `src/` layering isolation, Schema v2 `unit_brief.json` & experiment structure, zero hardcoded absolute paths, zero `transect_66km` regressions, and exported model config/fingerprint contracts) and `tests/test_spatial_feature_encoder.py`.
  - Updated `docs/architecture.md`, `README.md`, `experiments/GEMINI.md`, `.agents/personas/ml-architect.md`, `.agents/skills/code-audit/SKILL.md`, and `.agents/skills/experiment-sync/SKILL.md`.

## Verification & Results
- **Ruff Lint**: `ruff check src benchmarks experiments tests scripts` — **All checks passed! (0 errors)**.
- **Unit Tests**: `unittest discover -s tests -p "test_*.py"` — **70/70 passed** in `31.26s` (up from 56/57).
- **End-to-End Smoke Runs**:
  - `experiments.knn_spatial_regressor.run_unit` — Passed (`28.60s`, 7/7 `[CACHE HIT]`).
  - `experiments.bicubic_boundary_analysis.run_unit` — Passed (`4.16s`).
  - `benchmarks.global_analytical_models.run_benchmark` — Passed (`6.73s`, 7/7 `[CACHE HIT]`).
