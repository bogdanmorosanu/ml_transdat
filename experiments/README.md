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
```
