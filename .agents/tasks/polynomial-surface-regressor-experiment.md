# Task: 2D Polynomial Surface Regressor Experiment

**Date**: 2026-11-21
**Branch**: `exp/polynomial_surface_regressor`
**Status**: Completed
**Scope**: Model Architecture, Degree Sweeps, Benchmarking & Documentation

## Objective
Implement, evaluate, and benchmark a 2D Bivariate Polynomial Surface Regressor (`PolynomialSurfaceRegressor` in `src/models/polynomial_surface.py`) to explore Level 2 mathematical models that upgrade the flat rigid plane of the 2D Helmert baseline into a smooth, curved mathematical surface across degrees $D=1 \dots 14$.

## Key Activities & Implementation
1. **Model Architecture (`src/models/polynomial_surface.py`)**:
   - Implemented `PolynomialSurfaceRegressor(BaseGridShiftModel)` with centered, aspect-ratio preserved coordinate normalization to $[-1, 1]$.
   - Integrated `PolynomialFeatures` with `Ridge` regression ($\alpha=10^{-8}$) to ensure well-conditioned closed-form solutions.
   - Added analytical formula export (`get_formula(axis)`), term ranking, and parameter count properties.
   - Exported in `src/models/__init__.py`.
2. **Unit Tests (`tests/test_polynomial_surface.py`)**:
   - Added unit test suite covering fit/predict shapes, term progression formula $(D+1)(D+2)/2$, parameter validation, and exception handling.
3. **Experiment Pipeline (`experiments/polynomial_surface_regressor/run_unit.py`)**:
   - Implemented `PolynomialSurfaceRegressorUnit` (Unit 07) evaluating 10 model configurations (including sparse Lasso models).
   - Evaluated across both `RO_TEST_UNIFORM_4X4` (36,752 pts) and `RO_TEST_ADAPTIVE_3MM_MID` (176,350 pts).
   - Generated multi-model 66 km multi-cell transect plot (`plots/cell_transect_66km_polynomial.png`), degree scaling curve (`plots/polynomial_degree_scaling.png`), 3D perspective surface grid (`plots/polynomial_3d_surfaces_comparison.png`), and 2D contour heatmap grid (`plots/polynomial_2d_heatmaps_comparison.png`).
4. **Benchmark Suite Taxonomy & Restructuring (`benchmarks/`)**:
   - Created dedicated suite `benchmarks/global_analytical_models/` (evaluating Helmert 2D, Polynomials Deg 1–12, and Sparse Lasso on standard canonical workloads).
   - Updated `benchmarks/grid_shift_models/` to include the neural network (`Fourier MLP (4x4, 2.7km)` achieving $22.1\,\text{mm}$ RMSE on corridor).
   - Rewrote `benchmarks/README.md` defining the 3-criterion taxonomy (Locality, Architecture, Precision Scale) to keep benchmarks clean and readable for future sessions.
5. **Documentation & Registries**:
   - Created `experiments/polynomial_surface_regressor/report.md`, `README_AI.md`, and structured `unit_brief.json`.
   - Created research unit `docs/research/polynomial_surface_regressor.md`.
   - Updated `docs/research/roadmap.md`, `docs/research/narrative.md`, and `experiments/README.md`.

## Verification & Results
- **Degree Scaling & Accuracy Progression**:
  - `poly_deg1_11km` ($D=1$, 3 terms): Uniform RMSE = $42.8\,\text{cm}$, Adaptive RMSE = $47.8\,\text{cm}$, Max = $2.55\,\text{m}$ (matches Helmert 2D baseline floor).
  - `poly_deg2_11km` ($D=2$, 6 terms): Uniform RMSE = $33.0\,\text{cm}$ ($23\%$ error reduction), Adaptive RMSE = $42.1\,\text{cm}$.
  - `poly_deg3_11km` ($D=3$, 10 terms): Uniform RMSE = $29.5\,\text{cm}$, Adaptive RMSE = $37.3\,\text{cm}$.
  - `poly_deg5_11km` ($D=5$, 21 terms): Uniform RMSE = $27.0\,\text{cm}$, Adaptive RMSE = $34.9\,\text{cm}$, Max = $1.84\,\text{m}$ ($29\%$ reduction in worst-case error).
  - `poly_deg8_11km` ($D=8$, 45 terms): Uniform RMSE = $21.6\,\text{cm}$, Adaptive RMSE = $31.6\,\text{cm}$, Max = $1.11\,\text{m}$.
  - `poly_deg12_11km` ($D=12$, 91 terms): Uniform RMSE = $19.2\,\text{cm}$, Adaptive RMSE = $26.9\,\text{cm}$, Max = $0.89\,\text{m}$ ($< 1\,\text{m}$ nationwide).
  - `poly_deg12_4x4` ($D=12$, 91 terms on 4x4 grid): Uniform RMSE = $17.1\,\text{cm}$, Adaptive RMSE = $27.4\,\text{cm}$.
- **Test Suite & Code Quality**:
  - Unit tests passed: `tests/test_polynomial_surface.py` (6/6 tests).
  - Full test suite passed: `python -m unittest discover tests` (39/39 tests).
