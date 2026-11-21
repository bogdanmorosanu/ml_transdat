# Research Unit 06: Multilayer Perceptron (MLP) Spatial Regressor (AI Briefing)

## 1. Unit Metadata
* **Unit ID**: `06`
* **Slug**: `mlp_spatial_regressor`
* **Target Domain**: 2D Geodetic Distortion Shift Modeling ($dN, dE$) in meters.
* **Input Features**: Planar Stereo 70 coordinates ($N, E$).
* **Training Set**: `SPG_NODES_VALID` (2,689 valid anchor nodes at ~11 km resolution).
* **Evaluation Sets**:
  1. `RO_TEST_UNIFORM_4X4` (36,752 nationwide stratified continuous points).
  2. `RO_TEST_ADAPTIVE_3MM_MID` (176,350 quadtree midpoints concentrated in high-curvature zones).

## 2. Core Scientific Finding: Spectral Bias & Positional Encodings
1. **Raw Coordinate Regression Failure**:
   * Feedforward MLPs mapping $(N, E) \to (dN, dE)$ fail due to spectral bias, achieving only $\text{RMSE} \approx 182\,\text{mm}$ (Uniform) and $258\,\text{mm}$ (Adaptive), with max errors reaching $1.52\,\text{m}$.
   * Network depth does not solve this: increasing capacity to 3 hidden layers yields negligible gain ($176\,\text{mm}$).
2. **Fourier Positional Encoding Resolution**:
   * Augmenting $(N, E)$ with Random Fourier Features ($\sigma=3.0, 32$ frequencies) enables the MLP to capture higher spatial frequencies, slashing RMSE to **$49.5\,\text{mm}$** (Uniform) and **$46.0\,\text{mm}$** (Adaptive).
   * Unlike raw MLPs or decision trees, Fourier MLPs generate $C^\infty$ smooth interpolating curves that generalize well on the adaptive stress test.
3. **Bandwidth Limits**:
   * Higher frequencies ($\sigma=5.0$) over-oscillate (Runge-like phenomenon) due to the $11\,\text{km}$ Nyquist sampling limit of the training nodes.

## 3. Directory Layout
* [`run_unit.py`](run_unit.py): Runnable pipeline executing training, inference, metrics, and transect plotting.
* [`report.md`](report.md): Narrative report with scorecard and embedded diagnostic plots.
* [`unit_brief.json`](unit_brief.json): Machine-readable scorecard JSON.
* [`results/`](results/): Detailed scorecards (`scorecard.csv`, `summary_table.md`).
* [`plots/`](plots/): Diagnostic figures (`cell_transect_66km_mlp.png`, spatial error maps, error distributions).
