# Experiments Registry & Research Unit Guide

Welcome to the **Research Units Registry** for `ml_transdat`.

This directory contains modular, self-contained scientific experiments evaluating coordinate transformations between **ETRS89** and **Stereo 70**.

---

## 1. Active Research Units Index

| Directory Slug | Title & Hypothesis | Status | Key Metrics / Verdict |
|:---|:---|:---:|:---|
| [`spg_grid_analysis/`](spg_grid_analysis/) | **SPG 2D Distortion Grid Analysis**: Physical profiling of official CNC bicubic distortion field ($dN, dE$) and vector orientations. | `Validated` | $\text{Mean}=0.372\,\text{m}$, $\text{Max}=2.757\,\text{m}$. Identifies major distortion nodes in Banat/West, Maramureș, and Dobrogea. |
| [`pyproj_baseline/`](pyproj_baseline/) | **Standard PROJ Baseline**: Benchmarking `pyproj` (`EPSG:3844`) against ground-truth `pytransdat` on nationwide `ro_grid_4x4`. | `Validated` | $\text{Mean 2D}=0.3601\,\text{m}$, $\text{RMSE}=0.4348\,\text{m}$, $\text{Max}=2.82\,\text{m}$. Validates baseline without local grid shifts. |
| [`helmert2d_baseline/`](helmert2d_baseline/) | **2D 4-Parameter Helmert Baseline**: Closed-form OLS refit of 4-parameter 2D Helmert on valid SPG nodes (`StereoProj` + `Helmert2D`, no grid). | `Validated` | $\text{Mean 2D}=0.3743\,\text{m}$, $\text{RMSE}=0.4429\,\text{m}$, $\text{Max}=2.6811\,\text{m}$, $<1\,\text{m}=98.1\%$. Optimal 2D conformal floor. |
| [`knn_spatial_regressor/`](knn_spatial_regressor/) | **KNN 2D Spatial Regressor**: Evaluates K-Nearest Neighbors ($K \in \{1, 4, 8, 16\}$ with Uniform vs. IDW) on learning 2D SPG shifts directly from control nodes. | `Validated` | $\text{RMSE}=0.0353\,\text{m}$ ($3.5\,\text{cm}$) on `ro_grid_4x4` ($92.1\%$ reduction over floor), $\text{Max}=0.4783\,\text{m}$ with $K=4$ IDW. Proves $K=4$ cell-corner sweet spot. |
| [`decision_tree_regressor/`](decision_tree_regressor/) | **Decision Tree 2D Spatial Regressor**: Evaluates orthogonal step-plateau trees across 11 km, `4x4`, `8x8`, and `16x16` grids plus impurity pruning (`1e-8`). | `Validated` | $\text{RMSE}=0.0817\,\text{m}$ (11 km on `8x8`) $\to$ $0.0040\,\text{m}$ (`16x16` on corridor, $89.4\,\text{MB}$); `1e-8` pruning compresses to $2.81\,\text{MB}$ at $0.0188\,\text{m}$. |
| [`adaptive_quadtree_grid/`](adaptive_quadtree_grid/) | **Adaptive Quadtree Grid**: Curvature-driven 2D space partitioning using pure `pytransdat` grid shifts across 4 tolerance tiers (`2cm`, `1cm`, `3mm`, `1mm`). | `Validated` | $\text{Mean 2D}=0.143\,\text{cm}$ ($3\,\text{mm}$ tier, 176k cells) and $\text{Mean 2D}=0.050\,\text{cm}$ ($1\,\text{mm}$ tier, 564k cells, $\text{P95}=1.0\,\text{mm}$). $>97\%$ point reduction over uniform grids. |
| [`mlp_spatial_regressor/`](mlp_spatial_regressor/) | **Multilayer Perceptron (MLP) Spatial Regressor**: Evaluates continuous neural networks on valid SPG nodes tested against new uniform (`4x4`) and adaptive (`3mm`) national benchmarks. Identifies spectral bias and resolves it via Random Fourier Features. | `Validated` | Raw MLP $\text{RMSE}=182\,\text{mm}$; Fourier MLP ($\sigma=3.0$) achieves $\text{RMSE}=49.5\,\text{mm}$ (Uniform 4x4) and $\text{RMSE}=46.0\,\text{mm}$ (Adaptive 3mm) with $C^\infty$ smoothness and 0 grid lookup. |
| [`polynomial_surface_regressor/`](polynomial_surface_regressor/) | **2D Polynomial Surface Regressor**: Evaluates bivariate polynomial surfaces ($D=1\dots 12$) on normalized coordinates, upgrading the flat Helmert plane to a curved mathematical surface. | `Validated` | $\text{RMSE}=42.8\,\text{cm}$ ($D=1$, 3 terms) $\to$ $27.0\,\text{cm}$ ($D=5$, 21 terms) $\to$ $19.2\,\text{cm}$ ($D=12$, 91 terms); cuts max error from $2.60\,\text{m} \to 0.89\,\text{m}$. |
| [`rbf_surface_regressor/`](rbf_surface_regressor/) | **Thin Plate Spline (TPS) Regressor**: Localized 2D elastic sheet surface deformation with nugget smoothing ($\lambda=10^{-4}$), smoothing legacy grid slope breaks. | `Validated` | $\text{RMSE}=\mathbf{9.3\,\text{mm}}$ nationwide ($<1\,\text{cm}$), $\text{Max}=21.9\,\text{cm}$ ($0\,\text{MB}$ grid lookup); first continuous sub-centimeter model. |
| [`cell_planar_features/`](cell_planar_features/) | **Grid Cell Curvature & Shape Classification**: Functional geometric classification of $11\times 11\,\text{km}$ cells into a 3-class polynomial order taxonomy (Linear, Dome/Bowl, S-Shape). | `Validated` | Proves $53.2\%$ Dome vs. $46.8\%$ S-Shape at $5\,\text{mm}$, and $92.5\%$ Dome at $10\,\text{mm}$; delivers high-speed $O(1)$ spatial feature extractor. |

---

## 2. Anatomy of a Research Unit

Each folder in `experiments/<slug>/` is an independent, self-contained experiment adhering to this standard blueprint:

```text
experiments/<slug>/
├── run_unit.py          # [Execution Engine] Standalone Python script running pipeline end-to-end
├── report.md            # [Human Walkthrough] Narrative, objectives, results tables & embedded figures
├── README_AI.md         # [Agent Briefing] High-density summary of assumptions, metrics, and file index
├── unit_brief.json      # [Machine JSON] Structured scorecard metrics for automated evaluation
├── results/             # [Outputs] Metric JSON summaries, evaluations, benchmarks
├── plots/               # [Outputs] Saved visual figures (spatial maps, distributions, quivers)
└── data/                # [Optional] Unit-specific datasets or intermediate splits (if needed)
```

---

## 3. How to Run an Experiment

All research units are runnable from workspace root using the project conda environment:

```powershell
# Run SPG Grid Analysis
D:\anaconda3\envs\ml_transdat\python.exe -m experiments.spg_grid_analysis.run_unit

# Run PyProj Baseline
D:\anaconda3\envs\ml_transdat\python.exe -m experiments.pyproj_baseline.run_unit

# Run 2D 4-Parameter Helmert Baseline
D:\anaconda3\envs\ml_transdat\python.exe -m experiments.helmert2d_baseline.run_unit

# Run KNN 2D Spatial Regressor
D:\anaconda3\envs\ml_transdat\python.exe -m experiments.knn_spatial_regressor.run_unit

# Run Decision Tree 2D Spatial Regressor
D:\anaconda3\envs\ml_transdat\python.exe -m experiments.decision_tree_regressor.run_unit

# Run Adaptive Quadtree Grid
D:\anaconda3\envs\ml_transdat\python.exe -m experiments.adaptive_quadtree_grid.run_unit

# Run Fourier MLP Spatial Regressor
D:\anaconda3\envs\ml_transdat\python.exe -m experiments.mlp_spatial_regressor.run_unit

# Run 2D Polynomial Surface Regressor
D:\anaconda3\envs\ml_transdat\python.exe -m experiments.polynomial_surface_regressor.run_unit
```
