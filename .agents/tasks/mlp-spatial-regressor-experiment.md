# Task: Multilayer Perceptron (MLP) Spatial Regressor Experiment

**Date**: 2026-11-20
**Branch**: `exp/mlp-spatial-regressor`
**Status**: Completed
**Scope**: Model Architecture, Training, Continuous Evaluation & Documentation

## Objective
Implement, train, benchmark, and document a parametric continuous Multilayer Perceptron (MLP) neural network regressor (`src/models/mlp.py`) to learn the 2D planar distortion shift field ($dN, dE$) across Romania from official SPG nodes and densified subgrids, evaluated against the new nationwide continuous benchmarks (`RO_TEST_UNIFORM_4X4` and `RO_TEST_ADAPTIVE_3MM_MID`).

## Key Activities & Implementation
1. **Model Architecture (`src/models/mlp.py`)**:
   - Implemented `MLPGridShiftModel(BaseGridShiftModel)` with internal coordinate min-max normalization to $[-1, 1]$.
   - Integrated Random Fourier Features (RFF) positional encoding to overcome spectral bias.
   - Configurable optimization support for quasi-Newton L-BFGS (for small node sets) and mini-batch Adam with tight convergence tolerance controls (`tol`, `n_iter_no_change`, `batch_size`).
2. **Visualizer Robustness (`src/viz/plots.py`)**:
   - Added guard condition in `plot_subcell_heatmap` to handle continuous float $(u, v)$ coordinates without attempting to allocate unmanageable $36\text{k}\times 36\text{k}$ mesh arrays.
3. **Experiment Pipeline (`experiments/mlp_spatial_regressor/run_unit.py`)**:
   - Implemented `MLPSpatialRegressorUnit` (Unit 06) evaluating 4 model configurations:
     - `mlp_raw_11km` (11 km nodes, no Fourier features)
     - `mlp_fourier_11km` (11 km nodes, Fourier scale 3.0)
     - `mlp_fourier_4x4` (4x4 subgrid, 36.4k pts, Fourier scale 3.0, Adam)
     - `mlp_fourier_8x8` (8x8 subgrid, 146.2k pts, Fourier scale 3.0, Adam)
   - Benchmarked across both `RO_TEST_UNIFORM_4X4` (36,752 pts) and `RO_TEST_ADAPTIVE_3MM_MID` (176,350 pts).
   - Generated multi-model 66 km multi-cell transect plot (`plots/cell_transect_66km_mlp.png`), spatial error maps, error distributions, and scorecards.
4. **Documentation & Registries**:
   - Created `experiments/mlp_spatial_regressor/report.md`, `README_AI.md`, and structured `unit_brief.json`.
   - Updated `experiments/README.md` and research unit index.

## Verification & Results
- **Training Progression & Accuracy**:
  - `mlp_raw_11km`: RMSE = $182.3\,\text{mm}$ (Uniform), $257.9\,\text{mm}$ (Adaptive), Max = $1.52\,\text{m}$ (exhibits spectral bias).
  - `mlp_fourier_11km`: RMSE = $49.5\,\text{mm}$ (Uniform), $46.0\,\text{mm}$ (Adaptive), Max = $41.6\,\text{cm}$ (3.7x reduction).
  - `mlp_fourier_4x4`: RMSE = $13.0\,\text{mm}$ (Uniform), $32.5\,\text{mm}$ (Adaptive), P95 = $23.1\,\text{mm}$, $99.8\% < 5\,\text{cm}$.
  - `mlp_fourier_8x8`: RMSE = $14.6\,\text{mm}$ (Uniform), $30.1\,\text{mm}$ (Adaptive), P95 = $44.7\,\text{mm}$, Max = $15.2\,\text{cm}$ nationwide.
- **Test Suite & Code Hygiene**:
  - All 33 repository unit tests passed in $14.1\,\text{s}$ (`Ran 33 tests in 14.075s OK`).
  - `ruff check` and `ruff format` passed with 0 errors across `src/`, `experiments/`, and `tests/`.
