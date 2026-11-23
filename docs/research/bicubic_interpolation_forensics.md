# Bicubic Interpolation Forensics & Alternative 11 km Grid Schemes

- **Status**: `Validated`
- **Spawned From**: [`bicubic_boundary_analysis`](bicubic_boundary_analysis.md)
- **Linked Sources**: [`experiments/bicubic_interpolation_forensics/`](../../experiments/bicubic_interpolation_forensics/report.md), [`literature/transdatro_mathematical_model.md`](../../literature/transdatro_mathematical_model.md), [`src/models/grid_bicubic_variants.py`](../../src/models/grid_bicubic_variants.py)

---

## Current Verdict
`pytransdat` is a 100% exact clone of ANCPI's official `TransDatRO` specification ([`Help_TransDatRO_code_source_EN.pdf`](../../articles/Help_TransDatRO_code_source_EN.pdf), max discrepancy $2.22 \times 10^{-15}\,\text{m}$), confirming there is no porting bug. However, the official `TransDatRO` algorithm is **not** a smooth $C^1/C^2$ bicubic spline—it is a two-part hybrid combining English Wikipedia's 16-equation bicubic matrix with textbook one-sided 3-point edge finite differences (`-3, +4, -1`) applied to every interior cell wall, imposing a **5.6× to 11.5× learnability error penalty** on smooth ML/surface approximators.

---

## Premise & Rationale
Following [`bicubic_boundary_analysis`](bicubic_boundary_analysis.md), we conducted a deep forensic investigation to determine:
1. Whether all four published ANCPI/Romanian papers agree with `pytransdat` (resolving any "two rabbits" risk).
2. Where the non-standard one-sided derivative stencil originated and whether it matches any geodetic or image-processing standard.
3. Which alternative interpolation schemes perform best on the fixed $11\,\text{km}$ SPG lattice, and how much `TransDatRO`'s $C^1$ border kinks penalize smooth ML models.

---

## Key Strategic Findings
- **Master Spec & Symbolic Proof**: [`Help_TransDatRO_code_source_EN.pdf`](../../articles/Help_TransDatRO_code_source_EN.pdf) is the unabridged master specification (fixing clipped equations and omitted backward stencils in Avramiuc et al., 2009). SymPy confirms all 16 coefficients in `Grid.BiInterp` match $M^{-1}\mathbf{df}$ identically.
- **Wikipedia + Edge-Stencil Origin**: `TransDatRO` assembled Wikipedia's lexicographical 16-equation $A^{-1}$ matrix (evidenced by pasted `texvc` screenshots in Avramiuc et al., 2009) with textbook 3-point forward/backward boundary stencils (`-3, +4, -1`). Because `TransDatRO` already requires all 16 nodes of the $4\times 4$ box (`NoDataGridErr`), edge-only stencils were never necessary; using them produced the "Swapped Parabola" blend $(1-x)Q_{\text{right}}(x)+xQ_{\text{left}}(x)$ with boundary slope breaks $-\frac{1}{2}\Delta^4 g_{k-2}$ (up to `1.071 m/step`, mean `0.135 m/step`).
- **Top 3 Alternative $11\,\text{km}$ Schemes** (implemented in [`src/models/grid_bicubic_variants.py`](../../src/models/grid_bicubic_variants.py)):
  1. **#1 Catmull-Rom (`catmull_rom`, Local $C^1$)**: Minimal 1-line fix (symmetric central differences); eliminates every border kink (`0.0005 m/step` max) at `16.75 mm` RMS vs. `TransDatRO`.
  2. **#2 PCHIP (`pchip`, Monotone $C^1$)**: Eliminates both border kinks and internal cubic bowing (`0.37 mm` vs. `38.03 mm` in Scenario 1C) at `21.80 mm` RMS vs. `TransDatRO`.
  3. **#3 Natural Cubic Spline (`natural_spline_c2`, Global $C^2$)**: Seamless $C^2$ curvature across all $11\,\text{km}$ borders at `18.02 mm` RMS vs. `TransDatRO`.
- **Quantified ML Learnability Penalty**: Across a $10\times 10$ cell active region, a smooth Thin-Plate Spline RBF trained on $3\times 3$ sub-cell samples achieves **`0.667 mm` RMSE** on `Natural Spline C²` and **`1.364 mm` RMSE** on `Catmull-Rom C¹`, versus **`7.674 mm` RMSE** (`35.498 mm` max) on `TransDatRO`'s kinked surface—proving that sub-millimeter ML replication of `TransDatRO` requires cell-aware or sub-cell $(u, v)$ architectures.
