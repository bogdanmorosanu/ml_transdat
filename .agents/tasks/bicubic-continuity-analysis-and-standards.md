# Task: Bicubic Continuity Analysis & Geodetic Standards Survey

**Date**: 2026-11-19  
**Branch**: `exp/knn_spatial_regressor`  
**Status**: Completed  
**Scope**: Documentation & Geodetic Literature Research  

## Objective
Investigate and document the mathematical continuity properties of Romania's official 2D `.spg` grid transformation (RomGEO / `pytransdat`), verify whether the slope discontinuity ("kink") observed at $11\,\text{km}$ cell boundaries is an implementation error or historical geodetic practice, validate behavior against the official RomGEO desktop application, and document findings across living docs and literature.

---

## Key Activities & Implementation

1. **Numerical Differentiation Across Boundary Nodes**:
   - Sampled fine 1D cross-sections across position $44\,\text{km}$ ($E = 450{,}783.04\,\text{m}$, $N = 450{,}783.04\,\text{m}$).
   - Numerically proved that the first derivative changes abruptly across the boundary:
     - Left slope: $+2.877 \times 10^{-5}\,\text{m/m}$ (rising)
     - Right slope: $-1.333 \times 10^{-5}\,\text{m/m}$ (falling)
     - $\Delta\text{slope} = 4.210 \times 10^{-5}\,\text{m/m}$

2. **Official RomGEO Desktop Validation**:
   - Generated a 31-point test sequence ranging from $\pm 5\,\text{km}$ down to $\pm 1\,\text{cm}$ and directly on the boundary node.
   - Evaluated coordinates in official RomGEO and compared with `pytransdat`.
   - **Result**: Confirmed $100.0\%$ fidelity down to $< 0.357\,\text{mm}$ (matching RomGEO's 5-decimal-place arcsecond rounding resolution). Confirmed the slope kink originates from the official CNC/RomGEO binary grid implementation.

3. **International Geodetic Literature Survey (`literature/geodetic_grid_interpolation_standards.md`)**:
   - Surveyed global national transformation standards (Canada NTv2, US NADCON 5, Great Britain OSTN15, PROJ).
   - Documented why national agencies historically mandated bilinear ($C^0$) or biquadratic rather than bicubic splines (preventing oscillations/overshoots).
   - Identified the mathematical source of RomGEO's local $4 \times 4$ stencil in classic *Numerical Recipes in C* Chapter 3.6 (`bcucof`).
   - Analyzed the inductive bias mismatch between legacy piecewise $C^0$ grids and smooth $C^\infty$ ML models.

4. **Living Documentation & Index Synchronization**:
   - Updated [`docs/grid.md`](../../docs/grid.md) (Section 2.2) documenting the piecewise local bicubic formulation and continuity characteristics.
   - Updated [`literature/README.md`](../../literature/README.md), [`literature/references.md`](../../literature/references.md) (Category 3: R3.1 to R3.6), and [`literature/search_log.md`](../../literature/search_log.md).

---

## Verification
- RomGEO vs. `pytransdat` max discrepancy: $0.357\,\text{mm}$ ($0.000357\,\text{m}$).
- Living docs and literature cross-references verified.
