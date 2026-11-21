# Research Unit 07: 2D Polynomial Surface Regressor (AI Briefing)

## 1. Unit Metadata
* **Unit ID**: `07`
* **Slug**: `polynomial_surface_regressor`
* **Target Domain**: 2D Geodetic Distortion Shift Modeling ($dN, dE$) in meters.
* **Input Features**: Normalized centered Stereo 70 coordinates ($u, v) \in [-1, 1]$.
* **Training Sets**:
  - `SPG_NODES_VALID` (2,689 valid anchor nodes at ~11 km resolution).
  - `RO_GRID_4X4` (36,435 densified subgrid points at ~2.7 km resolution).
* **Evaluation Sets**:
  1. `RO_TEST_UNIFORM_4X4` (36,752 nationwide stratified continuous points).
  2. `RO_TEST_ADAPTIVE_3MM_MID` (176,350 quadtree midpoints concentrated in high-curvature zones).

## 2. Core Scientific Findings
1. **Progression Beyond the Flat Helmert Floor**:
   - **Degree 1 (Flat Tilted Plane, 3 terms)**: Yields nationwide uniform $\text{RMSE} = 42.8\,\text{cm}$ and adaptive $\text{RMSE} = 47.8\,\text{cm}$ ($\text{Max} = 2.60\,\text{m}$), exactly matching the linear conformal baseline floor ($44.3\,\text{cm}$).
   - **Degree 2 (Quadratic Bowl / Saddle, 6 terms)**: Bending the flat plane drops uniform RMSE to **$33.0\,\text{cm}$** (a $23\%$ error reduction).
   - **Degree 3 (Classical Cubic Surface, 10 terms)**: Introducing inflection / S-curves cuts uniform RMSE to **$29.5\,\text{cm}$** and adaptive RMSE to **$37.3\,\text{cm}$**.
   - **Degree 5 (Classical Geodetic Quintic, 21 terms)**: Drops uniform RMSE to **$27.0\,\text{cm}$** and max error from $2.60\,\text{m} \to 1.84\,\text{m}$ ($29\%$ reduction in worst-case error).
   - **Degree 8 (45 terms)**: Uniform RMSE drops to **$21.6\,\text{cm}$** ($\text{Max} = 1.11\,\text{m}$).
   - **Degree 12 (91 terms)**: Pushes uniform RMSE to **$19.2\,\text{cm}$**, and max error drops below $1\,\text{m}$ ($0.89\,\text{m}$).
2. **Asymptotic Limits of Global Polynomial Surfaces**:
   - Even when pushing degree to $D=14\dots 20$ or training on dense 4x4 subgrids ($36.4\text{k}$ points), a global polynomial surface hits an asymptotic capacity ceiling around **$\sim 17 - 18\,\text{cm}$**.
   - At high degrees ($D \ge 16$), border oscillations (Runge's phenomenon) begin to emerge, causing adaptive stress test maximum error to increase ($1.34\,\text{m} \to 2.53\,\text{m}$).
3. **Strategic Takeaway**:
   - Bivariate polynomial surfaces successfully bend the flat Helmert plane to absorb nationwide macroscopic curvature, cutting error by more than half ($44.3\,\text{cm} \to 19.2\,\text{cm}$) with only 91 coefficients per axis.
   - However, because polynomials are **strictly global** (changing one coefficient alters the surface across the entire country), they cannot resolve steep localized domes (like Banat) without causing ripple oscillations elsewhere.
   - This provides the definitive mathematical motivation for **Level 3: Radial Basis Functions (RBF) & Thin Plate Splines (TPS)**, which combine a polynomial trend with localized elastic basis functions.

## 3. Directory Layout
* [`run_unit.py`](run_unit.py): Unified execution pipeline running degree sweeps, benchmarks, and transect/scaling plots.
* [`report.md`](report.md): Narrative report with scorecard and embedded diagnostic figures.
* [`unit_brief.json`](unit_brief.json): Machine-readable scorecard JSON.
* [`results/`](results/): Detailed scorecards (`scorecard.csv`, `summary_table.md`, `scorecard_summary.json`).
* [`plots/`](plots/): Diagnostic figures (`cell_transect_66km_polynomial.png`, `polynomial_degree_scaling.png`, error maps).
