# Bicubic Cell Boundary & Extreme Scenarios Analysis (`bicubic_boundary_analysis`)

- **Status**: `Validated`
- **Spawned From**: [`spg_grid_analysis`](spg_grid_analysis.md), [`cell_planar_features`](cell_planar_features.md), [`cell_neighbor_features`](cell_neighbor_features.md)
- **Linked Sources**: [`experiments/bicubic_boundary_analysis/`](../../experiments/bicubic_boundary_analysis/report.md), [`pytransdat/trans_grid.py`](../../pytransdat/trans_grid.py), [`docs/core_engine.md`](../core_engine.md), [`docs/grid.md`](../grid.md), [`literature/transdatro_mathematical_model.md`](../../literature/transdatro_mathematical_model.md)

---

## Current Verdict
1. **Mathematical Verification of Official Ground Truth (`TransDatRO` Match)**:
   - Verified that `pytransdat`'s `Grid.BiInterp` implementation matches the official Romanian `TransDatRO` geodetic specification published in **Avramiuc, Dragomir, & Rus (2009, p. 112)** verbatim.
   - The visible slope changes ("kinks" or "creases") observed at $11\,\text{km}$ cell borders are **inherent to the official algorithm**: because boundary partial derivatives ($f_N, f_E$) are estimated using **one-sided 3-point finite-difference stencils** (forward at 0, backward at 1), adjacent cells evaluate different boundary tangents, producing exact $C^0$ value continuity but an unavoidable $C^1$ first-derivative slope discontinuity.
2. **Comprehensive Taxonomy of Extreme Boundary Scenarios**:
   - Established a 7-scenario taxonomy across both Northing ($dN$) and Easting ($dE$) distortion layers, evaluated across 5-cell extended windows ($u \in [-2, 3]$) with the target cell in the center flanked by neighbor cells:
     - **Group 1 (Gradient & Linear Extremes)**:
       - *Scenario 1A (Steepest Fall)*: $\Delta dN = -0.536\,\text{m}$ drop over $11\,\text{km}$ (`row 23, col 34, dN`).
       - *Scenario 1B (Steepest Rise in Easting)*: $\Delta dE = +0.453\,\text{m}$ jump over $11\,\text{km}$ (`row 45, col 38, dE`).
       - *Scenario 1C (Flat Endpoints with High Internal Sag)*: Nodal difference $< 1.5\,\text{mm}$ appears flat, but outer stencil slopes force **$38.0\,\text{mm}$ of internal bowing** (`row 45, col 35, dN`).
     - **Group 2 (Curvature & Boundary Discontinuity Extremes)**:
       - *Scenario 2A (True S-Shape)*: Sigmoidal inflection at $u_{\text{inf}} = 0.48$ with opposite boundary curvature signs ($|a_3| = 0.587$, `row 23, col 36, dN`), requiring true-curve vertical axis scaling to properly resolve.
       - *Scenario 2B (Strongest Bowl / Trough)*: Maximum downward dip of $74.7\,\text{mm}$ below the nodal chord line (`row 23, col 35, dN`).
       - *Scenario 2C (Strongest Dome / Peak)*: Maximum upward crest of $41.1\,\text{mm}$ above the nodal chord line (`row 3, col 34, dN`).
       - *Scenario 2D (Worst-Case $C^1$ Boundary Discontinuity)*: Maximum boundary tangent break of $\Delta\text{slope} = 0.788\,\text{m/step}$ across an $11\,\text{km}$ border in active territory (`row 10, col 16, dN`).

---

## Premise & Rationale
During 1D transect profiling of `pytransdat`'s ground-truth shift curve (`plot_transect_66km`), unexpected slope changes were observed whenever a profile crossed an $11\,\text{km}$ grid cell boundary. Furthermore, initial attempts to identify worst-case S-curves by raw nodal jump $|g_6 - g_5|$ selected cells that acted as steep monotonic ramps where the S-bend was visually imperceptible due to auto-scaled vertical axes and small cubic coefficients. We sought to:
1. Formulate exact mathematical criteria to discover genuine curvature archetypes (S-shapes with interior inflection points, bowls, domes, flat-sag bridges, and worst-case boundary slope breaks).
2. Group and display these scenarios across extended 5-cell windows with true vertical-axis scaling so that subtle centimeter-level internal bends are clearly visible alongside steep macroscopic drops.
3. Formally verify whether the boundary slope kinks are algorithmic bugs or intrinsic geodetic invariants of the official standard.

---

## Key Strategic Findings

- **True S-Shape Formulation Requires Curvature Zero-Crossing**: Selecting extreme cells by raw step difference fails to isolate S-shapes; a genuine S-shape requires the second derivative $P''(u) = 2 a_2 + 6 a_3 u$ to have an interior inflection point $u_{\text{inf}} = -\frac{a_2}{3 a_3} \in [0.2, 0.8]$ and opposite curvature signs at boundaries ($P''(0) \cdot P''(1) < 0$).
- **Vertical-Axis Magnification Resolves Internal Features**: Unconstrained polynomial extrapolation over 5 cells ($u \in [-2, 3]$) blows up cubic terms to $> 20\,\text{m}$, which compresses the true $1\,\text{m}$ ground-truth curve into a flat line under naive auto-scaling. Clamping the vertical axis to the true curve and node values ($18\%$ margin) magnifies the internal S-bend, revealing its true sigmoidal geometry.
- **Deceptive Flat-Line Bending**: Two bounding nodes with nearly zero difference ($< 1.5\,\text{mm}$) can still exhibit up to $38.0\,\text{mm}$ of internal sag due to steep outer stencil nodes influencing the one-sided tangents.
- **Implications for ML Models**: Because official ground truth has non-smooth $C^1$ creases at every $11\,\text{km}$ cell boundary, continuous global ML models (such as standard RBFs or global MLPs) will naturally smooth over these creases unless equipped with localized cell-position priors or micro-grid feature representations.
