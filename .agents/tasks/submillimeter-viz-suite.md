# Task: Sub-Millimeter Diagnostic Visualization Suite

**Date**: 2026-11-25
**Branch**: `exp/hybrid_surface_featgrid`
**Status**: Completed
**Scope**: Evaluation / Refactoring / Visualization

## Objective
Add opt-in, reusable visualization tools tailored for sub-millimeter (`< 1 mm`) coordinate transformation models—specifically zoomed 9-cell extreme residual transects with twin-axis True Bicubic context and log-scaled spatial 2D error maps—integrated cleanly into `src/viz/` and `TransformationExperimentUnit` / `GridShiftExperimentUnit`.

## Key Activities & Implementation
- **Zoomed 9-Cell Extreme Residual Plots (`src/viz/scenarios.py`, `src/viz/__init__.py`)**:
  - Refactored shared 9-cell window setup, mm metric formatting, and Group 1 / Group 2 figure rendering into `_compute_9cell_window`, `_format_9cell_diff_line`, and `_render_extreme_groups`.
  - Added `plot_extreme_residuals_9cells` and `_draw_9cell_extreme_residual_panel` displaying signed residuals vs. True Bicubic in `mm` (left axis, clamped to `±ylim_mm` with `±1 mm` band and `0 mm` line) alongside the darkened gray True Bicubic macro shape in `m` (`#475569`, `alpha=0.55`, right `twinx()` axis).
- **Log-Scale Spatial Error Heatmap (`src/viz/diagnostics.py`, `src/viz/registry.py`, `src/viz/dispatcher.py`)**:
  - Extended `_draw_spatial_error_ax`, `plot_spatial_error`, and `plot_spatial_error_rows` with `log_scale: bool = False`, `vmin_mm: float = 0.05`, and `vmax_mm: float | None = None` using `cmap="turbo"` and `matplotlib.colors.LogNorm`.
  - Added `PlotType.SPATIAL_ERROR_LOG = "spatial_error_log"`, `PlotGroup.SUBMILLIMETER_DIAGNOSTIC`, and `_handle_spatial_error_log_rows` in `PlotDispatcher`.
- **Experiment-Level Opt-In (`src/core/base_unit.py`, `experiments/hybrid_surface_featgrid/run_unit.py`)**:
  - Added `submillimeter_mode: bool = False` and `_resolve_plots_spec()` on `TransformationExperimentUnit` (inherited by `GridShiftExperimentUnit`), plus `render_extreme_residuals: bool | None = None` and `extreme_residual_ylim_mm: float = 5.0`.
  - When `submillimeter_mode = True`, default `PlotGroup.STANDARD_DIAGNOSTIC` specs are promoted to `PlotGroup.SUBMILLIMETER_DIAGNOSTIC` and `extreme_residuals_9cells` plots are rendered automatically.
  - Enabled `submillimeter_mode: bool = True` on `HybridSurfaceFeatGridUnit` and cleaned up temporary scratch scripts (`scratch/proto_spatial_error_scales.*`, `scratch/test_axis_decoupling.py`).

## Verification & Results
- Visually verified rendered PNGs from cached `hybrid_surface_featgrid` models:
  - `experiments/hybrid_surface_featgrid/plots/extreme_residuals_gradient_and_linear_9cells.png`
  - `experiments/hybrid_surface_featgrid/plots/extreme_residuals_curvature_and_kinks_9cells.png`
  - `experiments/hybrid_surface_featgrid/plots/ro_test_adaptive_1mm_mid_grid_shift_2d_spatial_error_log.png`
- Ran `D:\anaconda3\envs\ml_transdat\python.exe -m unittest tests/test_architecture_invariants.py tests/test_unified_architecture.py -v` -> **Ran 18 tests in 2.384s — OK**.
