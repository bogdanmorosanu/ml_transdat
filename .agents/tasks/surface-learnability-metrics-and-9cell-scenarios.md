# Task: Surface Learnability Metrics, Residual Roughness Visuals & 9-Cell Extreme Scenario Transects

**Date**: 2026-11-23
**Branch**: `exp/optimal_surface_regressor`
**Status**: Completed
**Scope**: Evaluation | Visualization | Core Architecture | Refactoring | Documentation

## Objective
Implement a dedicated suite of Stage-2 ML surface learnability metrics, residual roughness/variogram and Pareto ranking visualizations, and standardize all grid-shift experiments to render the canonical 9-cell (99 km) extreme scenario transects (`Gradient & Linear` + `Curvature & Kinks`) instead of the legacy 66 km single-row transect.

## Key Activities & Implementation
- **Stage-2 ML Surface Learnability Metrics ([`src/evaluation/metrics.py`](../../src/evaluation/metrics.py), [`src/evaluation/evaluator.py`](../../src/evaluation/evaluator.py), [`src/evaluation/__init__.py`](../../src/evaluation/__init__.py))**:
  - Added 6 `MetricType` members and `MetricGroup.SURFACE_LEARNABILITY`:
    - `CRMSE_2D_M`: Vector centered RMSE ($\sqrt{\text{Var}(e_N) + \text{Var}(e_E)}$).
    - `IQR_2D_M`: Bulk interquartile range ($P_{75} - P_{25}$).
    - `ROUGHNESS_GRAD_RMS_MM_KM` & `ROUGHNESS_GRAD_P95_MM_KM`: Vectorized $k=4$ nearest-neighbor KD-tree (`scipy.spatial.cKDTree`) residual gradient energy ($\text{mm/km}$), with per-point `grad_2d_mm_km` stored directly on `err_df`.
    - `LOCAL_JITTER_RATIO`: Dimensionless high-frequency neighbor oscillation ratio ($\text{RMS}(\|\mathbf{e}_i - \bar{\mathbf{e}}_{\mathcal{N}_k(i)}\|_2) / \text{cRMSE}_{2D}$).
    - `LEARNABILITY_SCORE_MM`: Composite Stage-2 learnability cost ($0.50 \cdot \text{cRMSE}_{2D\,\text{(mm)}} + 0.30 \cdot (2.75\,\text{km} \cdot \text{Roughness}_{\text{RMS}}) + 0.20 \cdot P_{95, 2D\,\text{(mm)}}$).
  - Updated `EvaluationResult.to_scorecard_row`, `format_scorecard_table`, and added `ModelEvaluator.rank_surface_candidates` for multi-dataset candidate ranking.
- **Residual Roughness, Pareto Ranking & 9-Cell Extreme Scenario Visuals ([`src/viz/registry.py`](../../src/viz/registry.py), [`src/viz/plots.py`](../../src/viz/plots.py), [`src/viz/dispatcher.py`](../../src/viz/dispatcher.py), [`src/viz/__init__.py`](../../src/viz/__init__.py))**:
  - Added `PlotType.RESIDUAL_ROUGHNESS`, `PlotType.EXTREME_SCENARIOS_9CELLS`, `PlotType.SURFACE_LEARNABILITY_RANKING`, and `PlotGroup.SURFACE_DIAGNOSTIC`.
  - Implemented `plot_residual_roughness_rows` (1-row-per-model 3-panel figure: Spatial Gradient Energy Map, Multi-Scale Empirical Variogram $h \in [0.5, 22]\,\text{km}$, and Distance-to-11km-Cell-Wall Profile).
  - Implemented `plot_surface_learnability_ranking` (2-panel Amplitude vs. Roughness Pareto frontier + Top-K decomposed Learnability Cost bar chart).
  - Implemented `plot_extreme_scenarios_9cells` using canonical `EXTREME_SCENARIOS_GROUP1_GRADIENT_LINEAR` (3 scenarios) and `EXTREME_SCENARIOS_GROUP2_CURVATURE_KINKS` (4 scenarios) over 99 km (4 cells left + target `[0, 1]` + 4 cells right).
- **Core Orchestration & Experiment Standardization ([`src/core/base_unit.py`](../../src/core/base_unit.py), [`docs/architecture.md`](../../docs/architecture.md), `experiments/*/run_unit.py`)**:
  - Updated `summarize_eval_results` to emit the new learnability metrics into Schema v2 `unit_brief.json`.
  - Added `render_extreme_scenarios: bool = True` and `get_extreme_scenario_models()` hook to `GridShiftExperimentUnit.run()` to automatically generate `extreme_scenarios_gradient_and_linear_9cells.png` and `extreme_scenarios_curvature_and_kinks_9cells.png`.
  - Migrated all existing grid-shift units (`knn_spatial_regressor`, `decision_tree_regressor`, `polynomial_surface_regressor`, `rbf_surface_regressor`, `mlp_spatial_regressor`, `mlp_shape_features`, `mlp_neighbor_features`, `torch_mlp_regressor`, `submillimeter_exploration`) from `plot_transect_66km` to `get_extreme_scenario_models`.
  - Synchronized [`docs/architecture.md`](../../docs/architecture.md).

## Verification & Results
- Verified clean imports across `src.evaluation`, `src.viz`, and `src.core.base_unit` via `D:\anaconda3\envs\ml_transdat\python.exe` (`Imports OK`).
- Note: Execution of `experiments/optimal_surface_regressor` was intentionally deferred per user request as that unit is being iterated on separately.
