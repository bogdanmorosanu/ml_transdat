# Task: Level 3 2D RBF & Thin Plate Spline Surface Regressor Experiment

**Date**: 2026-11-21  
**Branch**: `exp/rbf_surface_regressor`  
**Status**: Completed  
**Scope**: Model Architecture | Spatial Modeling | Benchmarking | Documentation  

---

## Objective
Implement and benchmark **Level 3** of our mathematical coordinate transformation hierarchy: **localized continuous surface deformation** ("elastic wavy sheet" modeling) applied to Romania's 2D grid shift distortion field ($(st70_n, st70_e) \to (\Delta N, \Delta E)$). Overcome the non-local monomial boundary oscillations (Runge's ceiling at ~18 cm) of Level 2 bivariate polynomials and achieve sub-centimeter nationwide accuracy with zero discrete lookup tables ($0\,\text{MB}$ grid data).

---

## Key Activities & Implementation

1. **Model Architecture (`src/models/rbf_surface.py`)**:
   - Implemented `RBFSurfaceRegressor` inheriting from `BaseGridShiftModel` with coordinate normalization to centered $[-1, 1]$.
   - Supports Thin Plate Spline ($r^2 \ln r$), Multiquadric, Inverse Multiquadric, Gaussian, Linear, and Cubic kernels.
   - Natively integrates simultaneous low-degree polynomial trends ($D=1$ Affine, $D=2$ Quadratic, $D=3$ Cubic) and nugget regularized smoothing ($\lambda \ge 0.0$).
   - Exported in `src/models/__init__.py`.

2. **Automated Unit Tests (`tests/unit/test_rbf_surface.py`)**:
   - Developed 6 unit tests covering parameter validation, exact interpolation ($< 10^{-10}\,\text{m}$ error at nodes), trend degrees, kernel types, and pickle serialization. All passed in $0.18\,\text{s}$.

3. **Research Unit 08 (`experiments/rbf_surface_regressor/`)**:
   - Built automated experiment orchestrator (`run_unit.py`) evaluating 8 model configurations on 2,222 base anchors and 36,435 densified subgrid anchors against dual canonical test sets: `RO_TEST_UNIFORM_4X4` (36,752 off-lattice pts) and `RO_TEST_ADAPTIVE_3MM_MID` (176,350 stress test pts).
   - Generated full diagnostics: 3D perspective surface deformation, 2D plan-view contours, 66 km multi-cell transects, and comparative bar charts.
   - Documented empirical findings in `report.md` and machine-readable `unit_brief.json`.

4. **Axis Decoupling Grid Sweep**:
   - Swept smoothing penalties $\lambda_N$ (Northing) and $\lambda_E$ (Easting) independently from $0.0$ to $10^{-2}$.
   - Proved that both axes share the exact same optimum ($\lambda_N = \lambda_E = 2.0 \times 10^{-4}$), yielding $8.63\,\text{mm}$ nationwide 2D RMSE.

5. **Cross-Model Benchmark Integration (`benchmarks/grid_shift_models/`)**:
   - Categorized Level 3 models into the high-precision Millimeter-to-Centimeter tier in `benchmarks/grid_shift_models/`.
   - Integrated `Thin Plate Spline (11km)` and `Thin Plate Spline (4x4, 2.7km)` into `run_benchmark.py` and regenerated all scorecards, plots, and json summaries.

6. **Living Documentation Synchronization**:
   - Created reference-first unit [`docs/research/rbf_surface_regressor.md`](../../docs/research/rbf_surface_regressor.md).
   - Updated DAG topology in [`docs/research/roadmap.md`](../../docs/research/roadmap.md) and causal evolution in [`docs/research/narrative.md`](../../docs/research/narrative.md).

---

## Verification & Key Empirical Results

* **Sub-Centimeter Accuracy ($0\,\text{MB}$ Lookup)**: Thin Plate Spline on 2,222 base anchors with optimal smoothing ($\lambda = 2 \times 10^{-4}$) achieves **$8.63\,\text{mm}$ nationwide RMSE** ($80.6\% < 1\,\text{cm}$, $100.0\% < 5\,\text{cm}$), outperforming Level 2 Polynomials by **$22.2\times$** ($191.8\,\text{mm} \to 8.63\,\text{mm}$) and Level 1 Helmert by **$51.3\times$** ($442.9\,\text{mm} \to 8.63\,\text{mm}$).
* **Corridor Benchmark Validation**: Evaluated on `ro_corridor_snw` (36,435 pts), Base TPS scores **$8.05\,\text{mm}$ RMSE**, outperforming Fourier MLP ($22.11\,\text{mm}$) and KNN ($28.54\,\text{mm}$). Densified TPS scores **$4.27\,\text{mm}$ RMSE** ($3.49\,\text{mm}$ mean, $97.5\% < 1\,\text{cm}$).
* **The Smoothing Dividend**: Smoothing ($\lambda = 2 \times 10^{-4}$) improved accuracy by $35\%$ over exact pinning ($13.25\,\text{mm} \to 8.63\,\text{mm}$) by relaxing the elastic sheet over RomGEO's legacy $C^0/C^1$ finite-difference stencil kinks.
* **The Legacy Grid Wall**: Densifying training data to 36k points brings $96.7\%$ of Romania within $< 1\,\text{cm}$, but nationwide RMSE plateaus near $\sim 5.7\text{--}6.9\,\text{mm}$ due to discrete cell slope breaks in the ground truth grid.
