# Task: Bicubic Boundary Analysis & Extreme Scenarios Expansion

**Date**: 2026-11-23  
**Branch**: `main`  
**Status**: Completed  
**Scope**: Evaluation & Documentation  

## Objective
Diagnose why the initial S-shape scenario appeared flat/unbent, formulate rigorous mathematical criteria for identifying genuine geometric extremes across both shift layers ($dN$ and $dE$), extend visualizations to 5-cell windows ($u \in [-2, 3]$) with true vertical-axis scaling, and reorganize output plots into clean functional groups (Gradient/Linear and Curvature/Kinks).

## Key Activities & Implementation
- **Mathematical S-Shape Formulation**: Identified that selecting cells by raw step difference yielded steep linear ramps ($a_3 \approx 0.026$). Formulated the genuine S-shape condition using the second derivative zero-crossing ($P''(u_{\text{inf}}) = 0$ with $u_{\text{inf}} \in [0.2, 0.8]$ and opposite boundary curvatures), finding `row 23, col 36, dN` ($u_{\text{inf}} = 0.48, |a_3| = 0.587$).
- **Vertical-Axis Magnification Fix**: Diagnosed that unconstrained polynomial extrapolation over 5 cells reached $\pm 10.5\,\text{m}$, which naively auto-scaled the y-axis to $21\,\text{m}$ and squashed the true $1\,\text{m}$ shift curve into a flat line. Clamped vertical limits strictly to true curve values and node coordinates ($18\%$ margin) to render the S-bend at the same magnified resolution as bowls and domes.
- **Grouped Output Plots**: Replaced single-cell zoom and mixed plots with two high-resolution multi-panel figures:
  - `plots/extreme_scenarios_gradient_and_linear.png` (3 panels): Steepest Fall (`row 23, col 34, dN`, $-0.536\,\text{m}$ drop), Steepest Easting Rise (`row 45, col 38, dE`, $+0.453\,\text{m}$ jump), and Flat-Sag Bowing (`row 45, col 35, dN`, $\Delta g = 1.5\,\text{mm}$ with $38.0\,\text{mm}$ bowing).
  - `plots/extreme_scenarios_curvature_and_kinks.png` (4 panels): True S-Shape (`row 23, col 36, dN`), Strongest Bowl (`row 23, col 35, dN`, $74.7\,\text{mm}$ dip), Strongest Dome (`row 3, col 34, dN`, $41.1\,\text{mm}$ crest), and Worst-Case $C^1$ Kink (`row 10, col 16, dN`, $0.788\,\text{m/step}$ tangent break).
- **Code & Asset Cleanup**: Purged obsolete plotting code and images (`single_cell_zoom_col20.png`, `extreme_scenarios_s_and_bowl.png`, `extreme_scenarios_expanded.png`) and updated `run_unit.py` and `unit_brief.json` (Schema v2).
- **Research & Experiment Sync**: Created [`docs/research/bicubic_boundary_analysis.md`](../../docs/research/bicubic_boundary_analysis.md), updated [`docs/research/roadmap.md`](../../docs/research/roadmap.md) (DAG, table, index), appended Phase 6 to [`docs/research/narrative.md`](../../docs/research/narrative.md), and synchronized [`experiments/README.md`](../../experiments/README.md) and [`experiments/bicubic_boundary_analysis/report.md`](../../experiments/bicubic_boundary_analysis/report.md).

## Verification & Results
- Ran `python -m experiments.bicubic_boundary_analysis.run_unit` successfully: generated all 3 diagnostic figures (`bicubic_cell_boundary_extension.png`, `extreme_scenarios_gradient_and_linear.png`, `extreme_scenarios_curvature_and_kinks.png`).
- Verified all plots display the cell of interest in the center flanked by 2 neighboring cells on each side ($u \in [-2, 3]$), with magnified vertical scaling and annotated coordinates.
