# Official Romanian `TransDatRO` / `RomGeo` Mathematical Model & Published Sources

> [!NOTE]
> **Purpose**: Documents the official mathematical formulation, algorithmic rationale, and empirical validation of Romania's coordinate transformation (`ETRS89` $\leftrightarrow$ `S-42 / Stereo 70` + `Black Sea 1975`), extracted directly from the primary published papers in `articles/` and cross-verified against [`pytransdat/trans_grid.py`](../pytransdat/trans_grid.py).

---

## 1. Primary Published Sources

1. **Avramiuc, N., Dragomir, P. I., & Rus, T. (2009)** — *"Algorithm for direct and inverse coordinate transformation between ETRS89 CRS and S-42 CRS"*, *RevCAD – Journal of Geodesy and Cadastre*, University "1 Decembrie 1918" Alba Iulia, pp. 105–114. ([`articles/18_143_Paper10_RevCAD09_2009.pdf`](../articles/18_143_Paper10_RevCAD09_2009.pdf))
2. **Ilie, D., Balotă, O. L., Iordan, D., & Nicoară, P. S. (2022)** — *"Algorithm and Application Development for Precise and Accurate Transformation of LiDAR Point Clouds into National Coordinate Systems of Romania Using Official Equations and Quasigeoid Model"*, *ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences*, Vol. V-4-2022, pp. 181–188. ([`articles/isprs-annals-V-4-2022-181-2022.pdf`](../articles/isprs-annals-V-4-2022-181-2022.pdf))

---

## 2. Historical & Geodetic Rationale

Romania's legacy national CRS **S-42** (Krasovsky 1940 ellipsoid, Oblique Stereographic 1970 projection, Black Sea 1975 normal heights) is a non-geocentric astro-geodetic datum with large internal triangulation network distortions. In contrast, **ETRS89** (GRS80 ellipsoid, realized via ROMPOS at ETRF2000 epoch 2000.0) is geocentric and distortion-free.

Across **894 common geodetic control points**, a rigid 4-parameter 2D Helmert transformation leaves large non-linear residuals (Avramiuc et al., 2009, Table 1):

| Component | Mean (m) | Std. Dev. (m) | Min (m) | Max (m) |
| :--- | :---: | :---: | :---: | :---: |
| **East** | 0.0000 | 0.2648 | -0.8632 | +0.8466 |
| **North** | -0.0000 | 0.3756 | -1.1928 | +1.3288 |

To bridge the two systems without destroying local spatial/cadastral relationships, ANCPI / NCC adopted a **hybrid two-step strategy** modeled via **Least Squares Prediction (LSP)** on a regular distortion grid.

---

## 3. Complete Direct & Inverse Transformation Pipeline

### 3.1 Direct Transformation ($\text{ETRS89} \to \text{Stereo 70} + \text{MN75}$)
1. **Closed-Form Oblique Stereographic Projection on GRS80** $(B, L)_{\text{ETRS89}} \to (N_G, E_G)$:
   Uses closed-form conformal sphere mapping (avoiding 1–2 cm Taylor series truncation errors) with projection pole $\varphi_0 = 46^\circ$, $\lambda_0 = 25^\circ$, scale factor $k_0 = 0.99975$, and false coordinates $F_N = F_E = 500{,}000\,\text{m}$ (GRS80: $a = 6{,}378{,}137\,\text{m}$, $f = 1 / 298.257222101$).
2. **2D 4-Parameter Helmert Transformation** $(N_G, E_G) \to (E', N')$:
   Bridges the GRS80 oblique stereographic plane to approximate Stereo 70 coordinates via translations $(X_0, Y_0)$, scale $m$, and rotation $R_z$.
3. **Bicubic Distortion Grid Correction** $(E', N') \to (E_{\text{ST70}}, N_{\text{ST70}})$:
   Interpolates local shifts $p_x(E', N')$, $p_y(E', N')$ from the official distortion grid (`ETRS89_KRASOVSKI42.GRT` / `ETRS89_KRASOVSKI42_2D.GRT`; early 15 km grid had 2,106 nodes, refined to 11 km with 3,816 nodes and later `.spg` releases) and adds them:
   $$N_{\text{ST70}} = N' + p_y(E', N'), \qquad E_{\text{ST70}} = E' + p_x(E', N')$$
4. **Quasigeoid Height Conversion** $h_{\text{el}} \to H_{\text{MN75}}$:
   Interpolates quasigeoid undulation $\zeta(\varphi, \lambda)$ from `EGG97_QGRJ.GRT` and subtracts: $H_{\text{MN75}} = h_{\text{el}} - \zeta(\varphi, \lambda)$.

### 3.2 Inverse Transformation ($\text{Stereo 70} + \text{MN75} \to \text{ETRS89}$)
Reverses the four stages in order:
- Subtract interpolated grid shifts from $(E_{\text{ST70}}, N_{\text{ST70}})$ to recover $(E', N')$,
- Apply inverse 4-parameter 2D Helmert,
- Apply inverse closed-form Oblique Stereographic formulas on GRS80 to recover $(B, L)$,
- Add interpolated $\zeta(\varphi, \lambda)$ to $H_{\text{MN75}}$ to recover $h_{\text{el}}$.

---

## 4. Exact Verification of the Bicubic Interpolation Formula

> [!IMPORTANT]
> **Mystery Resolved**: Page 8 (p. 112) of Avramiuc, Dragomir, & Rus (2009) publishes the **exact one-sided 3-point finite-difference equations** implemented in [`pytransdat/trans_grid.py`](../pytransdat/trans_grid.py#L268-L412). `pytransdat` is a faithful, term-for-term match of the official published `TransDatRO` algorithm.

For a unit cell $(x, y) \in [0, 1]^2$ surrounded by a $4 \times 4$ stencil of 16 nodes (Fig. 3, p. 111—note the caption says "9 nodes" while the diagram shows all 16 nodes $1\dots 16$ with inner cell $6, 7, 11, 10$), the surface is:
$$p(x, y) = \sum_{i=0}^{3} \sum_{j=0}^{3} a_{ij} x^i y^j$$

The 16 coefficients $a_{ij}$ are solved from the 4 corner values $f$, first partials $f_x, f_y$, and mixed partial $f_{xy}$ at $(0,0), (1,0), (0,1), (1,1)$. In a local node-centered coordinate system $(i, j)$ (Avramiuc et al., 2009, Fig. 4 & p. 112), the published derivative formulas are:

$$f_x = \frac{-p(i+2, j) + 4\,p(i+1, j) - 3\,p(i, j)}{2}$$
$$f_y = \frac{-p(i, j+2) + 4\,p(i, j+1) - 3\,p(i, j)}{2}$$
$$f_{xy} = \frac{p(i-1, j-1) + p(i+2, j+2) - p(i+1, j-1) - p(i-1, j+1)}{4}$$

### Implications for `ml_transdat`
1. **Why Slope Kinks Occur at 11 km Borders**: Because $f_x$ and $f_y$ use **one-sided 3-point forward/backward differences** rather than symmetric central differences $\frac{p(i+1, j) - p(i-1, j)}{2}$, adjacent cells compute **different** tangent slopes at their shared boundary node.
2. **Why `pytransdat` Matches**: In [`Grid.BiInterp`](../pytransdat/trans_grid.py#L295-L310), `df[4] = (-g[7] + 4*g[6] - 3*g[5]) / 2.0` (forward at $u=0$) and `df[5] = (3*g[6] - 4*g[5] + g[4]) / 2.0` (backward at $u=1$) are the exact algebraic equivalents of the published $f_x$ formula, and `df[12..15]` match $f_{xy}$ verbatim.

---

## 5. Empirical Accuracy & Software Benchmarks

### 5.1 Control-Point Residuals After 11 km Distortion Grid (Avramiuc et al., 2009, Table 2)
Evaluated across 894 common points on the 11,000 m grid (3,816 nodes):

| Component | Mean (m) | Std. Dev. (m) | Min (m) | Max (m) | Within $\pm 3\sigma$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **East** | +0.0001 | 0.0415 | -0.1729 | +0.1750 | 98.32% |
| **North** | -0.0000 | 0.0456 | -0.2022 | +0.1644 | 97.99% |

### 5.2 LiDAR / `RoTLAS APP` Validation vs. Commercial GIS (Ilie et al., 2022)
- **Official `TransDatRo` Baseline**: Open-source Java core (v1.04 / desktop v4.06), stable and unchanged in mathematical equations; limited to text files ($\le 1\text{M}$ points) and lacks native UTM 34N/35N $\to$ ETRS89 conversion.
- **RoTLAS APP v1.62** (Python + `pyproj` + adapted `TransDatRo` v1.04 Java core) validated against `TransDatRo` v4.06 across **96 points (UTM 34N)** and **210 points (UTM 35N, National Geodesic Network Class B)**:
  - **2D RMSE**: `0.8 mm` (`0.0008 m`)
  - **3D RMSE**: `0.9 mm` (`0.0009 m`), max per-coordinate residual `1 mm`.
- **Commercial Software Failure Modes** (210-point UTM 35N benchmark):
  - **Global Mapper v23.0**: Lacks official planimetric distortion grid $\to$ 3D std. dev. `0.181 m` (max 3D error `1.110 m`, `dH` errors up to `58 mm` from applying quasigeoid at shifted 2D positions).
  - **ArcGIS Pro v2.9**: Missing proper Stereo 70 / Helmert definition $\to$ planimetric RMSE `~122 m` (unusable).
  - **Quasigeoid Accuracy Tiers**: $\pm 8.5\,\text{cm}$ (Bucharest), $\pm 10\text{–}12\,\text{cm}$ (12 updated counties), $\pm 15\text{–}50\,\text{cm}$ (remaining 29 counties).
