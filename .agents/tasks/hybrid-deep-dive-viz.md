# Task: Hybrid Deep-Dive Visualizations

**Date**: 2026-11-27
**Branch**: `main`
**Status**: Completed
**Scope**: Evaluation | Visualization | Refactoring | Documentation

## Objective
Design, prototype, and promote a dedicated `PlotGroup.HYBRID_DEEP_DIVE` visualization suite for `CompositeResidualShiftModel` champions in sub-millimeter experiments, enabling stage-by-stage visual diagnosis of Stage 1 surface errors, Stage 2 neural network residual corrections, and their compound hybrid output.

## Key Activities & Implementation
- Added `PlotType.HYBRID_EXTREME_RESIDUALS_9CELLS`, `PlotType.HYBRID_CANCELLATION_SCATTER`, `PlotType.HYBRID_SPATIAL_3PANEL`, `PlotType.HYBRID_SUBCELL_PHASE_HEATMAP`, and `PlotGroup.HYBRID_DEEP_DIVE` to `src/viz/registry.py`.
- Implemented `src/viz/hybrid.py` (exported in `src/viz/__init__.py`) with:
  - `plot_hybrid_extreme_residuals_9cells`: 9-cell extreme transects plotting Stage 1 error, Stage 2 NN correction, and Hybrid error with dynamic per-panel vertical scaling.
  - `plot_hybrid_cancellation_scatter`: Stage 1 Error vs. Stage 2 NN Correction scatter against the `y = -x` ideal cancellation diagonal and `±1 mm` band.
  - `plot_hybrid_spatial_3panel`: Centered vertical 3-panel spatial map (`|Stage 1 Error|`, `|Stage 2 NN Correction|`, `|Hybrid Final Error|`) with independent `p99`-scaled colorbars.
  - `plot_hybrid_subcell_phase_heatmap`: 3×2 normalized `[0, 1]²` sub-cell phase heatmap (Mean & 95th-Percentile across all 3 stages) with `4×4` dashed and `8×8` dotted reference lines and repeated y-axes.
  - `select_hybrid_eval_grid` & `render_hybrid_deep_dive`: Selects the champion's uniform evaluation grid when available (falling back to the first eval grid).
- Integrated `render_hybrid_deep_dive` and `_select_champion_hybrid` into `GridShiftExperimentUnit` (`src/core/base_unit.py`), automatically enabled when `submillimeter_mode=True`.
- Added unit tests in `tests/unit/test_hybrid_viz.py` and updated `docs/architecture.md`.

## Verification & Results
- `ruff check src/viz/ src/core/base_unit.py tests/unit/test_hybrid_viz.py`: passed (`All checks passed!`).
- `python -m unittest tests.test_architecture_invariants tests.unit.test_hybrid_viz -v`: 9/9 tests passed.
- Backfilled both `experiments/hybrid_surface_featgrid` (`143.68s`) and `experiments/hybrid_surface_resnet` (`174.23s`) from cached `.pkl` checkpoints, generating all 5 `hybrid_champion_*.png` plots in each experiment's `plots/` directory.
