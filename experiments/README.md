# Experiments Registry & Research Unit Guide

Welcome to the **Research Units Registry** for `ml_transdat`.

This directory contains modular, self-contained scientific experiments evaluating coordinate transformations between **ETRS89** and **Stereo 70**.

---

## 1. Active Research Units Index

| ID | Directory Slug | Title & Hypothesis | Status | Key Metrics / Verdict |
|:---|:---|:---|:---:|:---|
| **01** | [`01_spg_grid_analysis/`](01_spg_grid_analysis/) | **SPG 2D Distortion Grid Analysis**: Physical profiling of official CNC bicubic distortion field ($dN, dE$) and vector orientations. | `Validated` | $\text{Mean}=0.372\,\text{m}$, $\text{Max}=2.757\,\text{m}$. Identifies major distortion nodes in Banat/West, Maramureș, and Dobrogea. |
| **02** | [`02_pyproj_baseline/`](02_pyproj_baseline/) | **Standard PROJ Baseline**: Benchmarking `pyproj` (`EPSG:3844`) against ground-truth `pytransdat` on nationwide $3 \times 3$ grid. | `Validated` | $\text{Mean 2D}=0.3601\,\text{m}$, $\text{RMSE}=0.4348\,\text{m}$, $\text{Max}=2.82\,\text{m}$. Validates baseline without local grid shifts. |

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
```

---

## 4. Guidelines for Adding a New Research Unit

When initiating a new experiment:
1. **Assign Sequential Prefix**: Pick the next unused 2-digit number (e.g., `03_neural_mlp_direct/`).
2. **Subclass `ResearchUnit`**: Implement `run_unit.py` using `src.core` metrics, samplers, and visualizers.
3. **Generate Dual Documentation**:
   - `report.md`: For human inspection and review.
   - `README_AI.md` & `unit_brief.json`: For fast token-efficient agent AI priming.
4. **Update Registry & Roadmap**: Register the unit here in `experiments/README.md` and link the branch in `docs/research/roadmap.md`.
