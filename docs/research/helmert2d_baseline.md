# 2D 4-Parameter Helmert Baseline (No Grid)

- **Status**: `Validated`
- **Spawned From**: [`spg_grid_analysis`](spg_grid_analysis.md)
- **Linked Sources**: [`experiments/helmert2d_baseline/`](../../experiments/helmert2d_baseline/), [`src/models/helmert2d.py`](../../src/models/helmert2d.py)
- **Core Question**: What is the best possible accuracy of a grid-free 2-step 2D conformal transformation (`StereoProj` + 4-parameter `Helmert2D`) when refitted directly to the valid nodes of `rom_grid3d_25.09.spg`?

---

## 1. Premise & Motivation
In `pytransdat`, the embedded `Helmert2D` parameters (`os_st70` / `st70_os`) remained fixed across grid releases while the `geodetic_shifts` grid was updated to `25.09`. Refitting those 4 parameters ($t_N, t_E, dm, R_z$) via closed-form OLS on the 2,222 valid SPG grid nodes finds the optimal average 2D conformal alignment for the current grid, without 3D Cartesian conversions or Krasovsky 1940 projections.

---

## 2. Quantitative Discoveries

- **Fitted vs. Embedded (`st70_to_os`)**: Scale shifts by $-0.2918\,\text{ppm}$ ($-0.1156 \to -0.4074\,\text{ppm}$) and rotation by $-0.0082''$ ($+0.2274'' \to +0.2192''$), absorbing a slight global radial/rotational trend in `25.09`.
- **Performance on `ro_grid_4x4.parquet` (36,435 points, `etrs89_to_st70`)**:
  - **Refitted `Helmert2D`**: $\text{Mean} = 0.3743\,\text{m}$, $\text{RMSE} = 0.4429\,\text{m}$, $\text{p95} = 0.8515\,\text{m}$, $\text{Max} = 2.6811\,\text{m}$, $< 1\,\text{m} = 98.1\%$.
  - **Embedded `Helmert2D` (no grid)**: $\text{Mean} = 0.3716\,\text{m}$, $\text{RMSE} = 0.4472\,\text{m}$, $\text{Max} = 2.7570\,\text{m}$, $< 1\,\text{m} = 97.8\%$.
  - **Exact Bidirectional Round-Trip**: Max $5.24 \times 10^{-9}\,\text{m}$.

---

## 3. Architectural & Modeling Verdict
- **Validated**: A simple 2D 4-parameter conformal Helmert on `StereoProj` outperforms PyProj's 7-parameter 3D Helmert on tail metrics ($\text{Max} = 2.68\,\text{m}$ vs. $2.82\,\text{m}$, $< 1\,\text{m} = 98.1\%$ vs. $98.0\%$) with exact analytical invertibility.
- **Confirmed Floor**: The $\sim 0.44\,\text{m}$ RMSE / $\sim 2.68\,\text{m}$ max error floor is intrinsic to any global linear/conformal model without non-linear local distortion modeling.
