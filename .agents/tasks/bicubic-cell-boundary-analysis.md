# Task: Bicubic Cell Boundary Extension Analysis & `TransDatRO` Literature Verification

**Date**: 2026-11-22
**Branch**: `exp/bicubic-cell-boundary`
**Status**: Completed
**Scope**: Evaluation | Documentation

## Objective
Diagnose the $C^1$ slope discontinuities ("kinks") observed at $11\,\text{km}$ SPG cell boundaries in 1D transect plots (`plot_transect_66km`), visualize per-cell bicubic polynomials extended into neighboring cells, and verify `pytransdat`'s `Grid.BiInterp` equations against official published `TransDatRO` papers.

## Key Activities & Implementation
- Implemented [`experiments/bicubic_boundary_analysis/run_unit.py`](../../experiments/bicubic_boundary_analysis/run_unit.py) generating:
  - A **198 km (18-cell)** Easting transect (`row 24, cols 12–30`) centered on the `plot_transect_66km` section (`cols 18–24`), with per-cell `Grid.BiInterp` polynomials extended to `[-1, 2]` in distinct dashed colors and exact `.spg` nodal values marked.
  - An **isolated single-cell zoom** (`row 24, cols 20–21`) showing the left ($u=0$, `cols 20–22`) and right ($u=1$, `cols 19–21`) Hermite tangent lines, column/value annotations, and the extended `pytransdat` true curve.
  - A **full-grid stress test** (`row 23, cols 36–37` strongest S-shape; `row 23, cols 35–36` strongest bowl) comparing `pytransdat` against standard central-difference Catmull-Rom.
- Processed the primary PDFs in `articles/` into [`literature/transdatro_mathematical_model.md`](../../literature/transdatro_mathematical_model.md) and indexed them in [`literature/README.md`](../../literature/README.md) and [`literature/references.md`](../../literature/references.md).
- Updated [`docs/core_engine.md`](../../docs/core_engine.md) and [`docs/grid.md`](../../docs/grid.md) to document the verified `TransDatRO` one-sided 3-point finite-difference formulas.
- Synchronized [`experiments/bicubic_boundary_analysis/unit_brief.json`](../../experiments/bicubic_boundary_analysis/unit_brief.json), [`experiments/bicubic_boundary_analysis/report.md`](../../experiments/bicubic_boundary_analysis/report.md), and [`experiments/README.md`](../../experiments/README.md).

## Verification & Results
- Executed `D:\anaconda3\envs\ml_transdat\python.exe -m experiments.bicubic_boundary_analysis.run_unit` (exit code 0), producing all 3 diagnostic plots in `experiments/bicubic_boundary_analysis/plots/`.
- Confirmed term-for-term match between `pytransdat/trans_grid.py` (`Grid.BiInterp`) and Avramiuc, Dragomir, & Rus (2009, *RevCAD*, p. 112).
