# Bicubic Cell Boundary Extension Analysis (`bicubic_boundary_analysis`)

## 1. Objective & Hypothesis
- **Objective**: Diagnose why the `pytransdat` ground-truth shift curve exhibits visible slope changes ("kinks") at $11\,\text{km}$ SPG cell boundaries in 1D transect plots (`plot_transect_66km`), verify whether `pytransdat`'s `Grid.BiInterp` implementation matches the official published `TransDatRO` mathematical specification, and establish a grouped suite of extreme scenarios across both Northing ($dN$) and Easting ($dE$) shift grids.
- **Hypothesis**: The slope kinks arise because `Grid.BiInterp` computes each cell's local cubic Hermite polynomial using **one-sided 3-point finite-difference** corner slopes rather than symmetric central differences, causing adjacent cells to disagree on the tangent at shared $11\,\text{km}$ boundary nodes. Furthermore, evaluating extreme scenarios in an expanded 5-cell window with proper vertical-axis scaling allows clear visualization of both subtle geometric features (S-shapes, domes, bowls, and flat-sag bowing) and dramatic gradient shifts.

---

## 2. Experimental Setup & Methodology
1. **198 km (18-Cell) Easting Transect**: Extended the exact `plot_transect_66km` transect (`row 24`, `cols 18–24`) by 6 cells before and 6 cells after (`cols 12–30`), overlaying each cell's `Grid.BiInterp` polynomial evaluated over $u \in [-1, 2]$ in distinct dashed colors against the true `pytransdat` curve and exact `.spg` nodal values.
2. **Group 1: Gradient & Linear Extremes (3-Panel Output Plot: `plots/extreme_scenarios_gradient_and_linear.png`)**:
   - **Scenario 1A: Steepest Left-to-Right Fall** (`row 23, col 34, dN`): Maximum drop across an $11\,\text{km}$ cell width ($\Delta dN = -0.536\,\text{m}$).
   - **Scenario 1B: Steepest Left-to-Right Rise in Easting** (`row 45, col 38, dE`): Maximum horizontal Easting jump across an $11\,\text{km}$ cell width ($\Delta dE = +0.453\,\text{m}$).
   - **Scenario 1C: Flat Endpoints with High Internal Sag** (`row 45, col 35, dN`): Bounding nodes differ by only $1.5\,\text{mm}$, yet the true bicubic curve bows by $38.0\,\text{mm}$ in the cell interior due to external stencil slopes.
3. **Group 2: Curvature & Boundary Discontinuity Extremes (4-Panel Output Plot: `plots/extreme_scenarios_curvature_and_kinks.png`)**:
   - **Scenario 2A: True S-Shape** (`row 23, col 36, dN`): Sigmoidal inflection point at $u_{\text{inf}} = 0.48$ with opposite curvature signs at boundaries and high slope contrast ($|a_3| = 0.587$), magnified to the true vertical scale ($\sim 1.0\,\text{m}$ span) so the internal S-bend is clearly visible.
   - **Scenario 2B: Strongest Bowl / Trough** (`row 23, col 35, dN`): Downward trough dipping $\Delta = 74.7\,\text{mm}$ below the nodal chord line.
   - **Scenario 2C: Strongest Dome / Peak** (`row 3, col 34, dN`): Upward crest arching $\Delta = 41.1\,\text{mm}$ above the boundary chord line.
   - **Scenario 2D: Worst-Case $C^1$ Boundary Discontinuity** (`row 10, col 16, dN`): Tangent break across an $11\,\text{km}$ cell border of $\Delta\text{slope} = 0.788\,\text{m/step}$.
4. **5-Cell Extended Window Visualization**: Each scenario is displayed across a 5-cell span ($u \in [-2, 3]$) with the target cell in the center ($[0, 1]$, highlighted in yellow) flanked by neighbor cells, comparing the target cell's extended polynomial against true piecewise bicubic interpolation and Catmull-Rom. The vertical axis is strictly scaled to the true curve and nodes to ensure maximum visual clarity without auto-scaling distortion.

---

## 3. Quantitative Results & Key Findings

| Group & Scenario | Cell & Layer | Nodal Values $(g_4, g_5, g_6, g_7)\,\text{m}$ | Characteristic Metric | Visual Behavior |
| :--- | :--- | :--- | :--- | :--- |
| **1A. Steepest Fall** | Row 23, Col 34 ($dN$) | `[-0.031, -0.185, -0.722, -0.698]` | Step drop $= -0.536\,\text{m}$ | Steepest downward plunge across an $11\,\text{km}$ cell span. |
| **1B. Steepest Rise ($dE$)** | Row 45, Col 38 ($dE$) | `[-0.921, -0.668, -0.215, +0.038]` | Step jump $= +0.453\,\text{m}$ | Steepest horizontal Easting shift change across an $11\,\text{km}$ cell span. |
| **1C. Flat Bending Sag** | Row 45, Col 35 ($dN$) | `[+0.733, +0.390, +0.388, +0.650]` | Bowing $= 38.0\,\text{mm}$, $\Delta g = 1.5\,\text{mm}$ | Bounding nodes appear almost flat, yet incoming/outgoing stencil gradients force a $38\,\text{mm}$ interior sag. |
| **2A. True S-Shape** | Row 23, Col 36 ($dN$) | `[-0.722, -0.698, -0.010, +0.166]` | $u_{\text{inf}} = 0.48, \text{contrast} = 0.883$ | Sigmoidal inflection connecting a lower plateau to an upper shelf, now clearly discernible at magnified scale. |
| **2B. Strongest Bowl** | Row 23, Col 35 ($dN$) | `[-0.185, -0.722, -0.698, -0.010]` | Sagitta $= 74.7\,\text{mm}$ at $u_v = 0.40$ | Dips down to $-0.786\,\text{m}$, creating a deep trough before rising back to the shared border. |
| **2C. Strongest Dome** | Row 3, Col 34 ($dN$) | `[+0.556, +0.911, +0.928, +0.615]` | Crown $= 41.1\,\text{mm}$ at $u_v = 0.57$ | Arches upward to crest higher than both bounding nodes before descending. |
| **2D. Worst-Case Kink** | Row 10, Col 16 ($dN$) | `[-0.009, -0.537, -0.883, -0.488]` | $\Delta\text{slope} = 0.788\,\text{m/step}$ | Largest tangent angle break across an $11\,\text{km}$ border in active Romanian territory. |

---

## 4. Visual Diagnostics
- **198 km Multi-Cell Extension**: [`plots/bicubic_cell_boundary_extension.png`](plots/bicubic_cell_boundary_extension.png)
- **Group 1: Gradient & Linear Extremes (3-Panel)**: [`plots/extreme_scenarios_gradient_and_linear.png`](plots/extreme_scenarios_gradient_and_linear.png)
- **Group 2: Curvature & Kinks Extremes (4-Panel)**: [`plots/extreme_scenarios_curvature_and_kinks.png`](plots/extreme_scenarios_curvature_and_kinks.png)

---

## 5. Takeaways & Verdict
- **Specification Invariant**: The $C^1$ slope kinks at $11\,\text{km}$ boundaries are a direct, deterministic consequence of the official `TransDatRO` one-sided 3-point finite-difference stencil shifting across cells.
- **Hidden Internal Bending**: Cells with virtually flat endpoints ($< 2\,\text{mm}$ difference) can still undergo up to $38\,\text{mm}$ of internal sag or bowing due to steep outer stencil nodes.
- **Vertical Scale Magnification**: By constraining plot limits to the actual range of the true curve rather than allowing the unconstrained isolated polynomial extrapolation to expand the axis, subtle curvature shapes (such as the S-bend) are now rendered at the same high-resolution vertical scale as bowls and domes.
