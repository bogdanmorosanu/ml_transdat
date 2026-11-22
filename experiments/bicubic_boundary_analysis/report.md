# Bicubic Cell Boundary Extension Analysis (`bicubic_boundary_analysis`)

## 1. Objective & Hypothesis
- **Objective**: Diagnose why the `pytransdat` ground-truth shift curve exhibits visible slope changes ("kinks") at $11\,\text{km}$ SPG cell boundaries in 1D transect plots (`plot_transect_66km`), and verify whether `pytransdat`'s `Grid.BiInterp` implementation matches the official published `TransDatRO` mathematical specification.
- **Hypothesis**: The slope kinks arise because `Grid.BiInterp` computes each cell's local cubic Hermite polynomial using **one-sided 3-point finite-difference** corner slopes rather than symmetric central differences, causing adjacent cells to disagree on the tangent at shared $11\,\text{km}$ boundary nodes.

---

## 2. Experimental Setup / Methodology
1. **198 km (18-Cell) Easting Transect**: Extended the exact `plot_transect_66km` transect (`row 24`, `cols 18–24`) by 6 cells before and 6 cells after (`cols 12–30`), overlaying each cell's `Grid.BiInterp` polynomial evaluated over $u \in [-1, 2]$ in distinct dashed colors against the true `pytransdat` curve and exact `.spg` nodal values.
2. **Isolated Single-Cell Zoom (`row 24, cols 20–21`)**: Plotted the isolated `Grid.BiInterp` polynomial, its left ($u=0$, `cols 20–22`) and right ($u=1$, `cols 19–21`) Hermite tangent lines, and the extended `pytransdat` true curve across $u \in [-1, 2]$.
3. **Full-Grid Stress Test**: Scanned all valid interior $4 \times 4$ stencils in `rom_grid3d_25.09.spg` to locate the strongest S-shape (`row 23, col 36, dN`) and strongest bowl/trough (`row 23, col 35, dN`), comparing `pytransdat` against standard central-difference Catmull-Rom interpolation.
4. **Primary-Source Literature Verification**: Cross-checked `pytransdat/trans_grid.py` against **Avramiuc, Dragomir, & Rus (2009)** (*RevCAD*, p. 112) and **Ilie et al. (2022)** (*ISPRS Annals*).

---

## 3. Quantitative Results / Analysis
- **Exact Specification Match**: Page 8 (p. 112) of Avramiuc, Dragomir, & Rus (2009) publishes the exact one-sided 3-point finite-difference formulas ($f_x, f_y$) and 4-corner cross-derivative ($f_{xy}$) implemented in `pytransdat/trans_grid.py`.
- **Extreme S-Shape (`row 23, cols 36–37, dN`)**: Nodes `[-0.722, -0.698, -0.010, +0.166] m` (`0.6874 m` jump across one $11\,\text{km}$ cell). Even in this worst-case cell, `pytransdat`'s interior curve ($u \in [0, 1]$) remains visually very close to standard Catmull-Rom.
- **Extreme Bowl (`row 23, cols 35–36, dN`)**: Nodes `[-0.185, -0.722, -0.698, -0.010] m` (left tangent `-0.308`, right tangent `+0.304`), clearly illustrating the $C^1$ slope kink at cell borders where adjacent cells switch one-sided stencils.

---

## 4. Visual Diagnostics
- **198 km Multi-Cell Extension**: [`plots/bicubic_cell_boundary_extension.png`](plots/bicubic_cell_boundary_extension.png)
- **Isolated Single-Cell Zoom (`cols 20–21`)**: [`plots/single_cell_zoom_col20.png`](plots/single_cell_zoom_col20.png)
- **Extreme S-Shape & Bowl Stress Test**: [`plots/extreme_scenarios_s_and_bowl.png`](plots/extreme_scenarios_s_and_bowl.png)

---

## 5. Takeaways & Verdict
- **Verdict**: `pytransdat` is a faithful, term-for-term port of the official `TransDatRO` algorithm. The $C^1$ slope kinks at $11\,\text{km}$ borders are inherent to `TransDatRO`'s one-sided 3-point finite-difference Hermite formulation, while interior cell curves remain well-behaved and within a few millimeters to centimeters of standard Catmull-Rom even on the most extreme cells in Romania.
