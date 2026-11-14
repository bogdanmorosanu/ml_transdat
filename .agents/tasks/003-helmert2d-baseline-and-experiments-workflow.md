# Task: 2D 4-Parameter Helmert Baseline (Unit 03) & Experiments Workflow

**Date**: 2026-11-14
**Status**: Completed
**Scope**: Model Architecture | Evaluation | Documentation

## Objective
1. Implement and evaluate a grid-free 2-step 2D coordinate transformation (`StereoProj` $\leftrightarrow$ `Helmert2D`) between **ETRS89** and **Stereo 70** by refitting the 4 conformal Helmert parameters via closed-form OLS on the valid nodes of `rom_grid3d_25.09.spg`.
2. Extract reusable Helmert 2D fitting and transformation utilities into `src/models/` and establish `experiments/GEMINI.md` workflow rules.

## Key Activities & Implementation
- **Reusable Model (`src/models/helmert2d.py`, `src/models/__init__.py`)**:
  - Implemented `Helmert2DParams`, `fit_helmert2d` (closed-form `numpy.linalg.lstsq` centered at $(500000, 500000)$), `FittedHelmert2D` (exact analytical inverse), and `TwoStepHelmert2DTransformer`.
- **Research Unit 03 (`experiments/03_helmert2d_baseline/`)**:
  - Implemented `run_unit.py` (`Helmert2DBaselineUnit`) fitting on the 2,222 valid SPG grid nodes (`is_node == True` in `data/benchmarks/ro_grid_3x3.parquet`) and evaluating bidirectionally on all 36,435 points.
  - Generated `unit_brief.json`, `results/helmert2d_params.json`, per-direction metric JSONs, diagnostic plots, `README_AI.md`, and `report.md` (including a direct comparison against Unit 01's SPG distortion grid).
- **Workflow & Documentation Sync**:
  - Created `experiments/GEMINI.md` enforcing `src/` reusable-code isolation, `ResearchUnit` structure, and automatic registry/roadmap updates.
  - Updated `experiments/README.md`, `docs/research/roadmap.md`, and created `docs/research/03_helmert2d_baseline.md`.

## ML Experiment / Evaluation Results
- **Model Architecture**: 2-step 2D conformal transformation (`StereoProj` + refitted 4-parameter `FittedHelmert2D`, no `Grid2D`).
- **Fitted Parameters (`st70_to_os`)**: $t_N = -31.6399\,\text{m}$, $t_E = -119.6209\,\text{m}$, $dm = -0.4074\,\text{ppm}$, $R_z = +0.2192''$ (vs. embedded: $t_N = -31.8051\,\text{m}$, $t_E = -119.7358\,\text{m}$, $dm = -0.1156\,\text{ppm}$, $R_z = +0.2274''$).
- **Transformation Direction**: Bidirectional (`etrs89_to_st70` and `st70_to_etrs89`).
- **Dataset Size / Distribution**: 2,222 SPG grid nodes for fitting; 36,435 points (`data/benchmarks/ro_grid_3x3.parquet`) for evaluation (34,213 out-of-sample interior points).
- **Accuracy vs. Baselines (`etrs89_to_st70`)**:
  - **Refitted `Helmert2D`**: Mean = $0.3743\,\text{m}$, RMSE = $0.4429\,\text{m}$, p95 = $0.8515\,\text{m}$, p99 = $1.1123\,\text{m}$, Max = $2.6811\,\text{m}$, $< 1\,\text{m}$ = $98.13\%$, Bias = $(-0.0011, -0.00005)\,\text{m}$.
  - **Embedded `Helmert2D` (no grid)**: Mean = $0.3716\,\text{m}$, RMSE = $0.4472\,\text{m}$, Max = $2.7570\,\text{m}$, $< 1\,\text{m}$ = $97.80\%$.
  - **Round-Trip Precision**: Max `st70 -> etrs89 -> st70` error = $5.24 \times 10^{-9}\,\text{m}$.
- **Inference Speed**: $871,627.8\,\text{pts/s}$ (`etrs89_to_st70`), $403,936.6\,\text{pts/s}$ (`st70_to_etrs89`).
