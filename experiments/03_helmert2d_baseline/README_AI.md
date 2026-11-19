# Research Unit 03 Briefing: 2D 4-Parameter Helmert Baseline

## Purpose
Evaluates a grid-free 2-step 2D coordinate transformation (`StereoProj` $\leftrightarrow$ `Helmert2D`) with 4 conformal parameters refitted via closed-form OLS on the 2,222 valid SPG grid nodes (`is_node == True` in `ro_grid_4x4.parquet`).

## Core Metadata
- **Unit ID**: `03`
- **Unit Slug**: `helmert2d_baseline`
- **Fitting Set**: 2,222 valid SPG grid nodes (`ro_grid_4x4.parquet[is_node]`)
- **Evaluation Set**: 36,435 points (`ro_grid_4x4.parquet`), plus 34,213 interior-only out-of-sample check
- **Fitted Parameters (`st70_to_os`)**: $t_N = -31.6399\,\text{m}$, $t_E = -119.6209\,\text{m}$, $dm = -0.4074\,\text{ppm}$, $R_z = +0.2192''$

## Quantitative Performance (`etrs89_to_st70`)
- **Refitted `Helmert2D`**: Mean = $0.3743\,\text{m}$, RMSE = $0.4429\,\text{m}$, p50 = $0.3291\,\text{m}$, p95 = $0.8515\,\text{m}$, p99 = $1.1123\,\text{m}$, Max = $2.6811\,\text{m}$, $< 1\,\text{m}$ = $98.1\%$
- **Embedded `Helmert2D` (no grid)**: Mean = $0.3716\,\text{m}$, RMSE = $0.4472\,\text{m}$, Max = $2.7570\,\text{m}$, $< 1\,\text{m}$ = $97.8\%$
- **Round-Trip Max Error**: $5.24 \times 10^{-9}\,\text{m}$ (`st70 -> etrs89 -> st70`)

## Artifact Index
- [`src/models/helmert2d.py`](../../src/models/helmert2d.py): Reusable closed-form 4-parameter OLS fitter and exact analytical inverse transformer.
- `run_unit.py`: Standalone execution script.
- `report.md`: Comparative tables and diagnostic figures.
- `unit_brief.json`: Machine-readable metrics and parameters payload.
- `results/`: Parameter comparison JSON and per-direction scorecard JSONs.
- `plots/`: Diagnostic heatmaps, hist/ecdf, signed scatter, and subcell plots.
