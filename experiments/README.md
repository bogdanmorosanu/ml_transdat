# Experiments Registry & Research Unit Guide

Welcome to the **Research Units Registry** for `ml_transdat`.

This directory contains modular, self-contained scientific experiments evaluating coordinate transformations between **ETRS89** and **Stereo 70**.

---

## 1. Active Research Units Index

| ID | Directory Slug | Title & Hypothesis | Status | Key Metrics / Verdict |
|:---|:---|:---|:---:|:---|
| **01** | [`01_spg_grid_analysis/`](01_spg_grid_analysis/) | **SPG 2D Distortion Grid Analysis**: Physical profiling of official CNC bicubic distortion field ($dN, dE$) and vector orientations. | `Validated` | $\text{Mean}=0.372\,\text{m}$, $\text{Max}=2.757\,\text{m}$. Identifies major distortion nodes in Banat/West, Maramureș, and Dobrogea. |
| **02** | [`02_pyproj_baseline/`](02_pyproj_baseline/) | **Standard PROJ Baseline**: Benchmarking `pyproj` (`EPSG:3844`) against ground-truth `pytransdat` on nationwide $3 \times 3$ grid. | `Validated` | $\text{Mean 2D}=0.3601\,\text{m}$, $\text{RMSE}=0.4348\,\text{m}$, $\text{Max}=2.82\,\text{m}$. Validates baseline without local grid shifts. |
| **03** | [`03_helmert2d_baseline/`](03_helmert2d_baseline/) | **2D 4-Parameter Helmert Baseline**: Closed-form OLS refit of 4-parameter 2D Helmert on valid SPG nodes (`StereoProj` + `Helmert2D`, no grid). | `Validated` | $\text{Mean 2D}=0.3743\,\text{m}$, $\text{RMSE}=0.4429\,\text{m}$, $\text{Max}=2.6811\,\text{m}$, $<1\,\text{m}=98.1\%$. Optimal 2D conformal floor. |

---

## 2. Anatomy of a Research Unit

Each folder in `experiments/<XX_slug>/` is an independent, self-contained experiment adhering to this standard blueprint:

```text
experiments/<XX_slug>/
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
# Run Unit 01: SPG Grid Analysis
D:\anaconda3\envs\ml_transdat\python.exe -m experiments.01_spg_grid_analysis.run_unit

# Run Unit 02: PyProj Baseline
D:\anaconda3\envs\ml_transdat\python.exe -m experiments.02_pyproj_baseline.run_unit

# Run Unit 03: 2D 4-Parameter Helmert Baseline
D:\anaconda3\envs\ml_transdat\python.exe -m experiments.03_helmert2d_baseline.run_unit
```
