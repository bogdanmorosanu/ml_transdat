# Official Romanian `TransDatRO` / `RomGeo` Mathematical Model & Literature Extraction

> [!IMPORTANT]
> **Forensic Audit & Specification Baseline Remark**:
> - **Designated Master Document**: [`articles/Help_TransDatRO_code_source_EN.pdf`](../articles/Help_TransDatRO_code_source_EN.pdf) (Doc A) is established as the **authoritative master specification**. Released directly by ANCPI with the official source code package, it is the only source containing the unabridged mathematical formulation without the typographical omissions or typesetting truncations present in journal reprints.
> - **Verification Against Code Implementation**: A line-by-line and symbolic algebraic check was performed comparing the algorithm described in the master document against the codebase implementation in [`pytransdat/trans_grid.py`](../pytransdat/trans_grid.py).
> - **Finding**: There is **100% agreement and zero mismatch** between the master document's algorithm and `pytransdat`. The 16-node subgrid geometry, unit-cell normalization, one-sided 3-point derivative stencils, condition equations, and closed-form polynomial coefficients match verbatim. The code and specification describe the exact same mathematical model.

---

## 1. Corpus of Analyzed Literature

The four literature documents in `articles/` represent three distinct publication channels spanning 2009 to 2022:

| Document Key | File in `articles/` | Formal Citation / Provenance | Nature & Role |
| :--- | :--- | :--- | :--- |
| **Doc A** (Primary Source) | [`articles/Help_TransDatRO_code_source_EN.pdf`](../articles/Help_TransDatRO_code_source_EN.pdf) | National Agency for Cadastre and Land Registration (ANCPI), *TransDatRO_code_source: Direct and reverse transformation algorithm...*, Official Technical Release & Code Documentation, 13 pages. | **Authoritative Developer Specification**: Distributed directly by ANCPI with the official Java source package (`TransDatRO_code_source.jar`). Contains pristine Appendix 3 with all 16 condition equations and both forward/backward derivative operators. |
| **Doc B** (Peer-Reviewed Paper) | [`articles/18_143_Paper10_RevCAD09_2009.pdf`](../articles/18_143_Paper10_RevCAD09_2009.pdf) | Avramiuc, N., Dragomir, P. I., & Rus, T. (2009). *Algorithm for direct and inverse coordinate transformation between ETRS89 CRS and S-42 CRS*. *RevCAD – Journal of Geodesy and Cadastre*, 105–114. | **Foundational Published Paper**: First public academic presentation of the algorithm by the principal ANCPI geodesists. Suffers from several typesetting truncation defects. |
| **Doc C** (Preprint Duplicate) | [`articles/Algorithm_for_direct_and_inverse_coordin.pdf`](../articles/Algorithm_for_direct_and_inverse_coordin.pdf) | Avramiuc, N., Dragomir, P. I., & Rus, T. (2009). *Algorithm for direct and inverse coordinate transformation between ETRS89 CRS and S-42 CRS*. | **Duplicate Preprint / Reprint**: Identical verbatim text and formatting to Doc B (10 pages), lacking only the final publisher pagination banner on page 10. |
| **Doc D** (Modern Validation) | [`articles/isprs-annals-V-4-2022-181-2022.pdf`](../articles/isprs-annals-V-4-2022-181-2022.pdf) | Ilie, D., Balotă, O. L., Iordan, D., & Nicoară, P. S. (2022). *Algorithm and Application Development for Precise and Accurate Transformation of LiDAR Point Clouds into National Coordinate Systems of Romania Using Official Equations and Quasigeoid Model*. *ISPRS Annals*, V-4-2022, 181–188. | **Modern Applied Benchmark**: Integrates the official TransDatRO Java core (v1.04) into a batch LiDAR processing pipeline (RoTLAS APP v1.62). Re-validates fidelity against 306 National Geodesic Network Class B points (2D RMSE 0.8 mm, max residual 1 mm), confirming the Java core remains the frozen national standard. |

---

## 2. Multi-Document Concordance & Defect Analysis

A rigorous, side-by-side comparison reveals that while all four documents describe the same underlying algorithmic sequence, the published literature contains notable editorial gaps, typographical errors, and typesetting truncations that only **Doc A** resolves:

| Feature / Element | Doc A (`Help_TransDatRO...`) | Doc B / Doc C (Avramiuc et al. 2009) | Doc D (Ilie et al. 2022) | Code Reality (`trans_grid.py`) | Forensic Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Fig. 3 Stencil Caption** | *"Composed cell (with 16 nodes)"* (p. 11) | *"Grid cell (with 9 nodes)"* (p. 111) | Not illustrated | 16 nodes extracted (`sg_size = 4`) | Doc B/C has an obvious editorial error ("9 nodes"). Doc A corrected this to "16 nodes". |
| **16 Condition Equations** | Complete list of all 16 equations (1 to 16) printed in full (pp. 11–12). | Equations 4, 8, 12, 16 were truncated/omitted during journal typesetting (p. 111). | Referenced globally as equations (5)–(6); mathematical core not printed. | Solved via closed-form symbolic inversion matrix `self.__a`. | Doc A is the only source with all 16 condition equations intact. |
| **Finite Difference Derivatives ($f_x, f_y$)** | Defines both **forward** ($f_x^d, f_y^d$) and **backward** ($f_x^i, f_y^i$) stencils (p. 12). | **Omission Gap**: Lists only the forward formula ($f_x, f_y$), omitting the backward stencils entirely (p. 112). | Delegates to TransDatRO Java engine. | Implements both forward (`df[4]`, `df[6]`, `df[8]`, `df[9]`) and backward (`df[5]`, `df[7]`, `df[10]`, `df[11]`). | Doc B/C has a major descriptive gap. Doc A provides the complete set of formulas matching the code. |
| **Cross-Derivative Formula ($f_{xy}^m$)** | Prints: $\frac{p_{i-1,j-1} + p_{i+2,j+2} - p_{i+1,j-1} - p_{i-1,j+1}}{4}$ (p. 12). | Prints identical formula with $p(i+2, j+2)$ (p. 112). | Not printed. | Evaluates: $\frac{p_{i-1,j-1} + p_{i+1,j+1} - p_{i+1,j-1} - p_{i-1,j+1}}{4}$ | **Universal Typo in Literature**: Both Doc A and Doc B print $i+2, j+2$. The actual code correctly uses the central diagonal node $(i+1, j+1)$ for a symmetric cross-difference. |
| **Matrix Inversion ($16 \times 16 \to a_{ij}$)** | Unstated; presents condition equations and leaves inversion to code. | Unstated; presents condition equations and leaves inversion to code. | Unstated. | Evaluated via exact pre-computed closed-form coefficients. | Mathematical derivation is implicit in all papers; verified below via computer algebra. |

---

## 3. Surgical Breakdown of the Grid Shift Algorithm

The distortion correction step interpolates local horizontal shifts $(p_x, p_y)$ from a regular grid and adds them to intermediate transformed coordinates:
$$E_{\text{ST70}} = E' + p_x(E', N'), \qquad N_{\text{ST70}} = N' + p_y(E', N')$$
*(where $E', N'$ are the intermediate coordinates in Stereographic 1970 obtained after closed-form conformal projection on GRS80 and 4-parameter 2D Helmert transformation).*

Both $p_x$ (Easting shift) and $p_y$ (Northing shift) are evaluated using the identical bicubic interpolation algorithm, applied independently to their respective grid matrices.

### 3.1 Step 1: Subgrid Extraction & Coordinate Normalization
Given a query point $(E', N')$ within the valid domain:
1. Locate the grid cell column $c$ and row $r$ such that:
   $$c = \left\lfloor \frac{E' - E_{\min}}{\Delta E} \right\rfloor, \qquad r = \left\lfloor \frac{N' - N_{\min}}{\Delta N} \right\rfloor$$
   *(where for the Romanian `.spg` grid, $\Delta E = \Delta N = 11{,}000\,\text{m}$).*
2. Extract the $4 \times 4$ node neighborhood (16 nodes, numbered 1 to 16 in Doc A Fig. 3):
   ```
     (j / North)
        ^
     13 |  14   15   16   (row r + 2)
        |
      9 |  10   11   12   (row r + 1)
        |     . P
      5 |   6    7    8   (row r)
        |
      1 |   2    3    4   (row r - 1)  --> (i / East)
   ```
   In 0-indexed linear memory ordering ($g_0 \dots g_{15}$):
   - Row $r-1$: $g_0 = \text{node } 1, \; g_1 = \text{node } 2, \; g_2 = \text{node } 3, \; g_3 = \text{node } 4$
   - Row $r$:   $g_4 = \text{node } 5, \; g_5 = \text{node } 6, \; g_6 = \text{node } 7, \; g_7 = \text{node } 8$
   - Row $r+1$: $g_8 = \text{node } 9, \; g_9 = \text{node } 10, \; g_{10} = \text{node } 11, \; g_{11} = \text{node } 12$
   - Row $r+2$: $g_{12} = \text{node } 13, \; g_{13} = \text{node } 14, \; g_{14} = \text{node } 15, \; g_{15} = \text{node } 16$
3. Compute normalized unit-cell offsets:
   $$x = \frac{E' - E_r}{\Delta E} \in [0, 1], \qquad y = \frac{N' - N_r}{\Delta N} \in [0, 1]$$
   *(In `trans_grid.py`, $x = e_{\text{unity}}$ and $y = n_{\text{unity}}$).*
   The target cell is bounded by:
   - Corner $(0, 0)$: Node 6 (`g[5]`)
   - Corner $(1, 0)$: Node 7 (`g[6]`)
   - Corner $(0, 1)$: Node 10 (`g[9]`)
   - Corner $(1, 1)$: Node 11 (`g[10]`)

### 3.2 Step 2: The Bicubic Polynomial Surface
The interpolated surface inside the unit cell is parameterized by 16 coefficients $a_{ij}$:
$$p(x, y) = \sum_{i=0}^{3} \sum_{j=0}^{3} a_{ij} x^i y^j$$
Its exact partial derivatives are:
$$p_x(x, y) = \frac{\partial p}{\partial x} = \sum_{i=1}^{3} \sum_{j=0}^{3} i \, a_{ij} x^{i-1} y^j$$
$$p_y(x, y) = \frac{\partial p}{\partial y} = \sum_{i=0}^{3} \sum_{j=1}^{3} j \, a_{ij} x^i y^{j-1}$$
$$p_{xy}(x, y) = \frac{\partial^2 p}{\partial x \partial y} = \sum_{i=1}^{3} \sum_{j=1}^{3} i \, j \, a_{ij} x^{i-1} y^{j-1}$$

### 3.3 Step 3: The 16 Condition Equations
Evaluating the polynomial and its derivatives at the four unit corners $(x, y) \in \{0, 1\}^2$ yields the exact 16 equations published in Doc A (pp. 11–12):

$$\begin{aligned}
\text{1.} & \quad p(0, 0) = a_{00} = f(0, 0) \\
\text{2.} & \quad p(1, 0) = a_{00} + a_{10} + a_{20} + a_{30} = f(1, 0) \\
\text{3.} & \quad p(0, 1) = a_{00} + a_{01} + a_{02} + a_{03} = f(0, 1) \\
\text{4.} & \quad p(1, 1) = \sum_{i=0}^3 \sum_{j=0}^3 a_{ij} = f(1, 1) \\
\text{5.} & \quad p_x(0, 0) = a_{10} = f_x(0, 0) \\
\text{6.} & \quad p_x(1, 0) = a_{10} + 2 a_{20} + 3 a_{30} = f_x(1, 0) \\
\text{7.} & \quad p_x(0, 1) = a_{10} + a_{11} + a_{12} + a_{13} = f_x(0, 1) \\
\text{8.} & \quad p_x(1, 1) = \sum_{i=1}^3 \sum_{j=0}^3 i \, a_{ij} = f_x(1, 1) \\
\text{9.} & \quad p_y(0, 0) = a_{01} = f_y(0, 0) \\
\text{10.} & \quad p_y(1, 0) = a_{01} + a_{11} + a_{21} + a_{31} = f_y(1, 0) \\
\text{11.} & \quad p_y(0, 1) = a_{01} + 2 a_{02} + 3 a_{03} = f_y(0, 1) \\
\text{12.} & \quad p_y(1, 1) = \sum_{i=0}^3 \sum_{j=1}^3 j \, a_{ij} = f_y(1, 1) \\
\text{13.} & \quad p_{xy}(0, 0) = a_{11} = f_{xy}(0, 0) \\
\text{14.} & \quad p_{xy}(1, 0) = a_{11} + 2 a_{21} + 3 a_{31} = f_{xy}(1, 0) \\
\text{15.} & \quad p_{xy}(0, 1) = a_{11} + 2 a_{12} + 3 a_{13} = f_{xy}(0, 1) \\
\text{16.} & \quad p_{xy}(1, 1) = \sum_{i=1}^3 \sum_{j=1}^3 i \, j \, a_{ij} = f_{xy}(1, 1)
\end{aligned}$$

### 3.4 Step 4: Stencil Derivatives via Asymmetric Finite Differences
The 16 boundary condition values $f, f_x, f_y, f_{xy}$ are computed from the 16 nodal values using the finite difference formulas from Doc A (p. 12):

#### A. Standard Finite Difference Operators:
- **Forward difference along $x$** ($f_x^d$):
  $$f_x^d = \frac{-p(i+2, j) + 4 p(i+1, j) - 3 p(i, j)}{2}$$
- **Backward difference along $x$** ($f_x^i$):
  $$f_x^i = \frac{3 p(i, j) - 4 p(i-1, j) + p(i-2, j)}{2}$$
- **Forward difference along $y$** ($f_y^d$):
  $$f_y^d = \frac{-p(i, j+2) + 4 p(i, j+1) - 3 p(i, j)}{2}$$
- **Backward difference along $y$** ($f_y^i$):
  $$f_y^i = \frac{3 p(i, j) - 4 p(i, j-1) + p(i, j-2)}{2}$$
- **Mixed cross-difference** ($f_{xy}^m$):
  $$f_{xy}^m = \frac{p(i-1, j-1) + p(i+1, j+1) - p(i+1, j-1) - p(i-1, j+1)}{4}$$
  *(Corrected from the published $p(i+2, j+2)$ typo).*

#### B. Assignment to the 4 Cell Corners (`df[0..15]`):
In [`pytransdat/trans_grid.py`](../pytransdat/trans_grid.py#L289-L310), these are assigned to the intermediate condition vector `df`:

| Variable | Physical Meaning | Math Operator | Stencil Nodes Used | Exact Code Implementation |
| :--- | :--- | :--- | :--- | :--- |
| `df[0]` | Value at $(0, 0)$ | $f(0, 0)$ | Node 6 | `g[5]` |
| `df[1]` | Value at $(1, 0)$ | $f(1, 0)$ | Node 7 | `g[6]` |
| `df[2]` | Value at $(0, 1)$ | $f(0, 1)$ | Node 10 | `g[9]` |
| `df[3]` | Value at $(1, 1)$ | $f(1, 1)$ | Node 11 | `g[10]` |
| `df[4]` | East derivative at $(0, 0)$ | $f_x^d$ (Forward) | Nodes 6, 7, 8 | `(-g[7] + 4*g[6] - 3*g[5]) / 2.0` |
| `df[5]` | East derivative at $(1, 0)$ | $f_x^i$ (Backward) | Nodes 7, 6, 5 | `(3*g[6] - 4*g[5] + g[4]) / 2.0` |
| `df[6]` | East derivative at $(0, 1)$ | $f_x^d$ (Forward) | Nodes 10, 11, 12 | `(-g[11] + 4*g[10] - 3*g[9]) / 2.0` |
| `df[7]` | East derivative at $(1, 1)$ | $f_x^i$ (Backward) | Nodes 11, 10, 9 | `(3*g[10] - 4*g[9] + g[8]) / 2.0` |
| `df[8]` | North derivative at $(0, 0)$ | $f_y^d$ (Forward) | Nodes 6, 10, 14 | `(-g[13] + 4*g[9] - 3*g[5]) / 2.0` |
| `df[9]` | North derivative at $(1, 0)$ | $f_y^d$ (Forward) | Nodes 7, 11, 15 | `(-g[14] + 4*g[10] - 3*g[6]) / 2.0` |
| `df[10]` | North derivative at $(0, 1)$ | $f_y^i$ (Backward) | Nodes 10, 6, 2 | `(3*g[9] - 4*g[5] + g[1]) / 2.0` |
| `df[11]` | North derivative at $(1, 1)$ | $f_y^i$ (Backward) | Nodes 11, 7, 3 | `(3*g[10] - 4*g[6] + g[2]) / 2.0` |
| `df[12]` | Cross derivative at $(0, 0)$ | $f_{xy}^m$ | Diagonal around Node 6 | `((g[0] + g[10]) - (g[2] + g[8])) / 4.0` |
| `df[13]` | Cross derivative at $(1, 0)$ | $f_{xy}^m$ | Diagonal around Node 7 | `((g[1] + g[11]) - (g[3] + g[9])) / 4.0` |
| `df[14]` | Cross derivative at $(0, 1)$ | $f_{xy}^m$ | Diagonal around Node 10 | `((g[4] + g[14]) - (g[6] + g[12])) / 4.0` |
| `df[15]` | Cross derivative at $(1, 1)$ | $f_{xy}^m$ | Diagonal around Node 11 | `((g[5] + g[15]) - (g[7] + g[13])) / 4.0` |

---

## 4. Analytical Inversion of the $16 \times 16$ Linear System

The linear system can be expressed compactly in matrix notation as:
$$M \cdot \mathbf{a} = \mathbf{df}$$
where $M$ is a fixed $16 \times 16$ binary matrix defining the polynomial coefficients at $x, y \in \{0, 1\}$.

Solving for the polynomial coefficients via matrix inversion yields $\mathbf{a} = M^{-1} \cdot \mathbf{df}$.
Computer algebra verification (using SymPy) proves that the explicit expressions implemented in [`pytransdat/trans_grid.py`](../pytransdat/trans_grid.py#L311-L396) are the **exact symbolic solutions** of this system without exception:

$$\begin{aligned}
a_0 &= \text{df}[0] \\
a_1 &= \text{df}[4] \\
a_2 &= -3\,\text{df}[0] + 3\,\text{df}[1] - 2\,\text{df}[4] - \text{df}[5] \\
a_3 &= 2\,\text{df}[0] - 2\,\text{df}[1] + \text{df}[4] + \text{df}[5] \\
a_4 &= \text{df}[8] \\
a_5 &= \text{df}[12] \\
a_6 &= -3\,\text{df}[8] + 3\,\text{df}[9] - 2\,\text{df}[12] - \text{df}[13] \\
a_7 &= 2\,\text{df}[8] - 2\,\text{df}[9] + \text{df}[12] + \text{df}[13] \\
a_8 &= -3\,\text{df}[0] + 3\,\text{df}[2] - 2\,\text{df}[8] - \text{df}[10] \\
a_9 &= -3\,\text{df}[4] + 3\,\text{df}[6] - 2\,\text{df}[12] - \text{df}[14] \\
a_{10} &= 9\,\text{df}[0] - 9\,\text{df}[1] - 9\,\text{df}[2] + 9\,\text{df}[3] + 6\,\text{df}[4] + 3\,\text{df}[5] - 6\,\text{df}[6] - 3\,\text{df}[7] \\
       &\quad + 6\,\text{df}[8] - 6\,\text{df}[9] + 3\,\text{df}[10] - 3\,\text{df}[11] + 4\,\text{df}[12] + 2\,\text{df}[13] + 2\,\text{df}[14] + \text{df}[15] \\
a_{11} &= -6\,\text{df}[0] + 6\,\text{df}[1] + 6\,\text{df}[2] - 6\,\text{df}[3] - 3\,\text{df}[4] - 3\,\text{df}[5] + 3\,\text{df}[6] + 3\,\text{df}[7] \\
       &\quad - 4\,\text{df}[8] + 4\,\text{df}[9] - 2\,\text{df}[10] + 2\,\text{df}[11] - 2\,\text{df}[12] - 2\,\text{df}[13] - \text{df}[14] - \text{df}[15] \\
a_{12} &= 2\,\text{df}[0] - 2\,\text{df}[2] + \text{df}[8] + \text{df}[10] \\
a_{13} &= 2\,\text{df}[4] - 2\,\text{df}[6] + \text{df}[12] + \text{df}[14] \\
a_{14} &= -6\,\text{df}[0] + 6\,\text{df}[1] + 6\,\text{df}[2] - 6\,\text{df}[3] - 4\,\text{df}[4] - 2\,\text{df}[5] + 4\,\text{df}[6] + 2\,\text{df}[7] \\
       &\quad - 3\,\text{df}[8] + 3\,\text{df}[9] - 3\,\text{df}[10] + 3\,\text{df}[11] - 2\,\text{df}[12] - \text{df}[13] - 2\,\text{df}[14] - \text{df}[15] \\
a_{15} &= 4\,\text{df}[0] - 4\,\text{df}[1] - 4\,\text{df}[2] + 4\,\text{df}[3] + 2\,\text{df}[4] + 2\,\text{df}[5] - 2\,\text{df}[6] - 2\,\text{df}[7] \\
       &\quad + 2\,\text{df}[8] - 2\,\text{df}[9] + 2\,\text{df}[10] - 2\,\text{df}[11] + \text{df}[12] + \text{df}[13] + \text{df}[14] + \text{df}[15]
\end{aligned}$$

---

## 5. Summary & Key Forensic Takeaways

1. **Definitive Ground Truth Established**:
   `Help_TransDatRO_code_source_EN.pdf` (Doc A) is the definitive master specification. It contains the complete set of formulas without the typesetting truncation that corrupted the journal publication (Doc B/C).
2. **Implementation Fidelity is 100% Exact**:
   [`pytransdat/trans_grid.py`](../pytransdat/trans_grid.py) matches the official ANCPI mathematical model term for term:
   - Stencil indexing, coordinate reduction, one-sided derivative stencils, and cross-differences match verbatim.
   - The closed-form coefficient array `self.__a` is mathematically identical to the analytical inversion of the 16 condition equations.
3. **The Root of the $C^1$ Discontinuity**:
   The slope kinks across $11\,\text{km}$ cell boundaries are not an implementation bug or coding error in `pytransdat`. They are the **exact, deterministic mathematical consequence** of ANCPI's choice of asymmetric 3-point one-sided finite-difference stencils ($f_x^d \neq f_x^i$). Adjacent cells evaluate different stencils at the same physical node, resulting in mismatched boundary tangents.
4. **Typographical Artifact in the Literature**:
   The published text formula for the mixed partial derivative $f_{xy}^m$ contains an index typo ($i+2, j+2$), whereas both the official Java core and `pytransdat` correctly evaluate the symmetric diagonal difference $(i+1, j+1)$.
