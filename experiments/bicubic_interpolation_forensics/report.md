# Bicubic Interpolation Forensics & Alternative 11 km Grid Schemes (`bicubic_interpolation_forensics`)

## 1. Objective, Hypothesis & Executive Forensic Conclusion

### Objective & Hypothesis
* **Objective**: Conduct a complete forensic mathematical, historical, and empirical audit of the 2D bicubic grid-shift interpolation implemented in [`pytransdat/trans_grid.py`](../../pytransdat/trans_grid.py) (replicating ANCPI's official `TransDatRO` / `RomGEO` engine), determine why it exhibits $C^1$ slope discontinuities ("kinks") at $11\,\text{km}$ grid boundaries, trace its origins, evaluate smooth $11\,\text{km}$ alternatives, and quantify the learnability penalty imposed on smooth Machine Learning models.
* **Hypothesis**: The $C^1$ border kinks are an authentic property of the official `TransDatRO` specification rather than a Python porting bug, and those kinks directly degrade smooth function approximators (MLPs, RBFs) attempting to learn the transformation without cell-aware features.

### Executive Forensic Conclusion
1. **`pytransdat` is 100% Authentic**: The bicubic interpolation implemented in [`pytransdat/trans_grid.py`](../../pytransdat/trans_grid.py) matches ANCPI's official master specification ([`articles/Help_TransDatRO_code_source_EN.pdf`](../../articles/Help_TransDatRO_code_source_EN.pdf)) down to machine epsilon ($2.22 \times 10^{-15}\,\text{m}$). There is no coding bug in `pytransdat`.
2. **The Official Algorithm Violates Smoothness Standards**: Although ANCPI's documentation calls it a *"bicubic-spline interpolation whose main feature is the use of smooth interpolation surfaces"*, the implemented algorithm is **not** a $C^1$ or $C^2$ bicubic spline and does **not** match any international geodetic standard (`NTv2`, `OSTN15`, `NADCON 5`, or `OGP Guidance Note 7-2`).
3. **Root Cause Identified**: The algorithm is a custom two-part assembly combining English Wikipedia's 16-equation bicubic matrix with textbook **3-point one-sided edge finite-difference stencils** (`-3, +4, -1`) applied to every interior cell wall. Because each node's slope changes depending on which side of a cell wall it sits on, the surface exhibits sharp **$C^1$ slope kinks** (up to `1.071 m/step`, mean `0.135 m/step`) at every $11\,\text{km}$ grid line.
4. **ML Learnability Penalty Confirmed**: An empirical smooth-surface test (Thin-Plate Spline RBF) proves that `TransDatRO`'s artificial border kinks impose a **5.6× to 11.5× learnability error penalty** (`7.674 mm` RMSE on `TransDatRO` vs. `1.364 mm` on `Catmull-Rom C¹` and `0.667 mm` on `Natural Spline C²`).

---

## 2. Stage 1: Literature Corpus Audit & Code Implementation Concordance

Four documents in `articles/` were audited against [`pytransdat/trans_grid.py`](../../pytransdat/trans_grid.py):

| Key | File Path | Bibliographic Provenance | Role in Investigation |
| :--- | :--- | :--- | :--- |
| **Doc A** | [`articles/Help_TransDatRO_code_source_EN.pdf`](../../articles/Help_TransDatRO_code_source_EN.pdf) | ANCPI. *TransDatRO_code_source: Direct and reverse transformation algorithm...* (13 pp.). | **Authoritative Master Specification**: Distributed with `TransDatRO_code_source.jar`. Contains pristine Appendix 3 (pp. 10–12) with all 16 condition equations and both forward/backward derivative operators. |
| **Doc B** | [`articles/18_143_Paper10_RevCAD09_2009.pdf`](../../articles/18_143_Paper10_RevCAD09_2009.pdf) | Avramiuc, N., Dragomir, P. I., & Rus, T. (2009). *RevCAD – Journal of Geodesy and Cadastre*, pp. 105–114. | **First Academic Paper**: Contains journal typesetting truncations (clipped Eqs. 4, 8, 12, 16), omitted backward difference operators, and a "9 nodes" Fig. 3 caption typo. |
| **Doc C** | [`articles/Algorithm_for_direct_and_inverse_coordin.pdf`](../../articles/Algorithm_for_direct_and_inverse_coordin.pdf) | Avramiuc, N., Dragomir, P. I., & Rus, T. (2009). Preprint/reprint. | **Duplicate Reprint**: Verbatim identical 10 pages to Doc B. |
| **Doc D** | [`articles/isprs-annals-V-4-2022-181-2022.pdf`](../../articles/isprs-annals-V-4-2022-181-2022.pdf) | Ilie, D., Balotă, O. L., Iordan, D., & Nicoară, P. S. (2022). *ISPRS Annals*, V-4-2022, 181–188. | **Modern Benchmark**: Wraps `TransDatRO` v1.04 Java core in Python (`RoTLAS APP v1.62`) and validates against 306 NGN Class B points (2D RMSE $0.8\,\text{mm}$). |

### Cross-Document Concordance & Defect Summary
* **Fig. 3 Stencil Caption**: Doc B/C reads *"Grid cell (with 9 nodes)"*; Doc A corrected this to *"Composed cell (with 16 nodes)"*, matching `sg_size = 4` in code.
* **First-Derivative Stencils**: Doc B/C omitted the backward stencils ($f_x^i, f_y^i$) completely; Doc A prints both forward ($f_x^d, f_y^d$) and backward ($f_x^i, f_y^i$) formulas matching `df[4..11]` in `pytransdat`.
* **Cross-Derivative Typo**: Both Doc A and Doc B print $p(i+2, j+2)$ in $f_{xy}^m$; Fig. 4 spans only $(i\pm 1, j\pm 1)$ and $(i+2, j+2)$ indexes out of bounds on a $4\times 4$ subgrid. Both Java and `pytransdat` evaluate $(i+1, j+1)$.
* **16×16 Matrix Inversion (`Proof 1` in `run_unit.py`)**: SymPy computer algebra confirms all 16 closed-form expressions `self.__a[0..15]` in `Grid.BiInterp` are the exact symbolic inverse of Doc A's 16 condition equations.

---

## 3. Stage 2: Historical Origins, Developer Rationale & Mathematical Forensics

### 3.1 A Two-Part Assembly (2006–2008)
1. **Part A — Core 16-Equation Solver (Sourced Verbatim from English Wikipedia)**:
   - Pages 111–112 of Doc B contain **pasted bitmap screenshots** of MediaWiki's `texvc` PNG renders (including a red vertical cursor line).
   - `TransDatRO` uses Wikipedia's lexicographical corner ordering $(0,0), (1,0), (0,1), (1,1)$ and $A^{-1}$ matrix (distinguishing it from *Numerical Recipes* `bcucof`'s counter-clockwise ordering).
2. **Part B — Custom Finite-Difference Stencils**:
   - Wikipedia's *"Finding derivatives from function values"* subsection stated only in prose that unknown derivatives must be estimated via finite differences.
   - Looking at a single cell in isolation, the developer rejected 2-point central differences `(right - left) / 2` (which omit the corner value itself) and instead selected textbook 3-point one-sided forward differences (`(-g[7] + 4*g[6] - 3*g[5]) / 2`) on the left wall and backward differences (`(3*g[6] - 4*g[5] + g[4]) / 2`) on the right wall.

### 3.2 Audit of Section 7 References & Industry Comparison
* None of the 17 references in Section 7 of Avramiuc et al. (2009) contain the one-sided bicubic formula: `[1]`–`[4], [7]` are general Romanian geodesy textbooks; `[8]` is Avramiuc's 2006 PhD thesis; `[14]` (Blais 1990) covers **Least Squares Prediction** for *populating* the grid nodes; `[17]` (**OGP Guidance Note 7, Part 2**, 2006) explicitly specifies **bilinear** interpolation.
* In image processing (Keys 1981; OpenCV) and numerical computing, 3-point one-sided differences (`-3, +4, -1`) are strictly emergency outer-edge formulas. Because `TransDatRO` requires all 16 nodes of the $4\times 4$ box to be valid (`NoDataGridErr`), every valid cell corner already had both neighbors in memory—edge-only stencils were never necessary.

### 3.3 The "Swapped Parabola" Identity & Fourth-Difference Slope Break (`Proofs 2–4`)
Across 4 collinear nodes $\{g_4, g_5, g_6, g_7\}$ at $x \in \{-1, 0, 1, 2\}$, let $Q_{\text{left}}(x)$ pass through $(-1, 0, 1)$ and $Q_{\text{right}}(x)$ pass through $(0, 1, 2)$:
* **Standard $C^1$ Catmull-Rom**: $P_{\text{Catmull-Rom}}(x) = (1 - x) Q_{\text{left}}(x) + x Q_{\text{right}}(x)$.
* **TransDatRO 1D Cubic**: $P_{\text{TransDatRO}}(x) = (1 - x) Q_{\text{right}}(x) + x Q_{\text{left}}(x)$ (swapped blend).
* **Exact Boundary Law**: Positional jump across node $k$ is identically $0$ ($C^0$ continuous), while the tangent slope break is:
  $$\Delta \text{slope}_k = f_{x,\text{right cell}}^d(0) - f_{x,\text{left cell}}^i(1) = \frac{-g_{k+2} + 4g_{k+1} - 6g_k + 4g_{k-1} - g_{k-2}}{2} = -\frac{1}{2} \Delta^4 g_{k-2}$$
* Why it passed ANCPI validation: Discrete validation on 894 scattered points ($\sigma \approx 4.2\text{–}4.6\,\text{cm}$) cannot detect $C^1$ kinks along $11\,\text{km}$ grid lines, and the developer did not overfit the stencil to those sparse points.

---

## 4. Stage 3: Alternative 11 km Grid Interpolators & Learnability Scorecard

Keeping the $11\,\text{km}$ SPG grid fixed (as generating a denser $1\,\text{km}$ grid requires ANCPI's 894 control points and LSP covariance parameters), we implemented 6 alternatives in [`src/models/grid_bicubic_variants.py`](../../src/models/grid_bicubic_variants.py) (`ParameterizedBicubicGridShiftModel`) plus `CellBilinearShiftModel`.

Evaluated across `36,752` interior points (`2,230` interior borders) and a `10×10` cell smooth TPS-RBF learnability test (`results/stage3_interpolator_comparison.json`):

| Method | Continuity | RMS Diff vs. `TransDatRO` (mm) | Max Border Kink (m/step) | Mean Border Kink (m/step) | Scen. 1C Bowing (mm) | TPS-RBF All RMSE (mm) | TPS-RBF Border RMSE (mm) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`TransDatRO` (`pytransdat`)** | $C^0$ | `0.00` | `1.0709` | `0.1353` | `38.03` | **`7.674`** | **`8.025`** |
| **Catmull-Rom (Central Diff)** | $C^1$ | `16.75` | **`0.0005`** | **`0.0001`** | `38.03` | `1.364` | `1.434` |
| **Natural Cubic Spline** | $C^2$ | `18.02` | **`0.0004`** | **`0.0001`** | `51.27` | **`0.667`** | **`0.693`** |
| **PCHIP (Monotone)** | $C^1$ | `21.80` | **`0.0006`** | **`0.0001`** | **`0.37`** | `2.681` | `2.848` |
| **Akima (Local 5-Point)** | $C^1$ | `19.71` | **`0.0005`** | **`0.0001`** | `56.03` | `2.304` | `2.455` |
| **Lagrange 16-Node** | $C^0$ | `11.16` | `0.3573` | `0.0451` | `37.89` | `2.449` | `2.515` |
| **Bilinear (NTv2 Standard)** | $C^0$ | `24.41` | `0.7413` | `0.1157` | `0.00` | `6.071` | `6.322` |

*(Note: A Global $C^2$ Natural Cubic Spline does not fit a single high-degree polynomial across a 72-node row; it fits piecewise degree-3 cubics per cell with tridiagonal-matched $C^2$ curvature across borders.)*

---

## 5. Stage 4: Top 3 Candidate Selection, Rationale & Difference Analysis

Ranked Top 3 candidates for a fixed $11\,\text{km}$ grid (`results/stage4_top3_candidates_summary.json`):

1. **#1 Catmull-Rom (`catmull_rom`, Local $C^1$)**: Drop-in 1-line fix to `TransDatRO` inside the same $4\times 4$ box. Eliminates every kink while staying closest to `TransDatRO` (`P50 = 10.5 mm`, `P95 = 30.6 mm`, `P99 = 45.0 mm`—within ANCPI's $\pm 45.6\,\text{mm}$ network residual std. dev.).
2. **#2 PCHIP (`pchip`, Local $C^1$ Monotone)**: Fritsch–Carlson (1980) harmonic-mean slopes eliminate *both* border kinks and internal cubic overshoot/sag (`0.37 mm` bowing in Scenario 1C; `P50 = 14.3 mm`, `P95 = 39.4 mm` vs. `TransDatRO`).
3. **#3 Natural Cubic Spline (`natural_spline_c2`, Global $C^2$)**: Seamless $C^2$ curvature everywhere and the cleanest target for smooth ML / surface models (`0.667 mm` TPS-RBF RMSE, **11.5× lower** than `TransDatRO`; `P50 = 12.1 mm`, `P95 = 32.5 mm`).

---

## 6. Visual Diagnostics

### Full 7-Method Comparisons (99 km / 9-Cell Windows & 198 km Transect)
* [`plots/extreme_scenarios_gradient_and_linear_9cells.png`](plots/extreme_scenarios_gradient_and_linear_9cells.png)
* [`plots/extreme_scenarios_curvature_and_kinks_9cells.png`](plots/extreme_scenarios_curvature_and_kinks_9cells.png)
* [`plots/transect_198km_methods_comparison.png`](plots/transect_198km_methods_comparison.png)

### Focused Top-3 Candidate Comparisons & National Diagnostics
![Top 3 Group 1 Extreme Scenarios](plots/top3_extreme_scenarios_gradient_and_linear_9cells.png)
![Top 3 Group 2 Extreme Scenarios](plots/top3_extreme_scenarios_curvature_and_kinks_9cells.png)
![Top 3 198 km Transect & Differences](plots/top3_transect_and_differences_198km.png)
![Top 3 National Analysis](plots/top3_national_difference_and_smoothness_analysis.png)

---

## 7. Takeaways & Architectural Verdict for `ml_transdat`

1. **Exact Target Reality**: Any ML model trained to replicate `pytransdat` / `TransDatRO` to sub-millimeter precision is learning a **piecewise-bicubic surface with deterministic $C^1$ slope kinks along every $11\,\text{km}$ grid line**, not a globally smooth physical surface.
2. **ML Architecture Mandate**: Smooth global approximators (standard MLPs, RBFs) incur a `5.6×–11.5×` penalty when fitting `TransDatRO` (`7.674 mm` vs. `0.667–1.364 mm` on smooth targets). Sub-millimeter replication of `TransDatRO` requires **cell-aware** or **sub-cell $(u, v)$** representations.
3. **Reusable Benchmarking**: [`src/models/grid_bicubic_variants.py`](../../src/models/grid_bicubic_variants.py) (`ParameterizedBicubicGridShiftModel`) enables future ML experiments to benchmark directly against smooth `Catmull-Rom`, `PCHIP`, or `Natural Spline C²` targets.

---

## 8. Primary & Methodological Bibliography
* **ANCPI** — *TransDatRO_code_source* ([`articles/Help_TransDatRO_code_source_EN.pdf`](../../articles/Help_TransDatRO_code_source_EN.pdf)).
* **Avramiuc, N. (2006)** — *Contribuţii privind transformările de coordonate în sisteme geodezice eterogene* (PhD Thesis, UTCB).
* **Avramiuc, N., Dragomir, P. I., & Rus, T. (2009)** — *RevCAD*, pp. 105–114 ([`articles/18_143_Paper10_RevCAD09_2009.pdf`](../../articles/18_143_Paper10_RevCAD09_2009.pdf)).
* **Ilie, D., et al. (2022)** — *ISPRS Annals*, Vol. V-4-2022, pp. 181–188 ([`articles/isprs-annals-V-4-2022-181-2022.pdf`](../../articles/isprs-annals-V-4-2022-181-2022.pdf)).
* **OGP (2006)** — *Geomatics Guidance Note 7, Part 2*; **NRCan (1995)** — *NTv2*; **NOAA (2017, 2020)** — *NADCON 5*; **Ordnance Survey (2016)** — *OSTN15*.
* **Keys (1981)** (*IEEE Trans. ASSP*); **Fritsch & Carlson (1980)** (*SIAM J. Numer. Anal.*); **Akima (1970)** (*JACM*); **Press et al. (1992)** (*Numerical Recipes*, Ch. 3); **Wikipedia** (*"Bicubic interpolation"*).
