# Experiments Registry & Research Unit Guide

Welcome to the **Research Units Registry** for `ml_transdat`.

This directory contains modular, self-contained scientific experiments evaluating coordinate transformations between **ETRS89** and **Stereo 70**.

---

## 1. Active Research Units Index

| Directory Slug | Type | Title & Hypothesis | Status | Key Metrics / Verdict |
|:---|:---:|:---|:---:|:---|
| [`spg_grid_analysis/`](spg_grid_analysis/) | `analysis` | **SPG 2D Distortion Grid Analysis**: Physical profiling of official CNC bicubic distortion field ($dN, dE$) and vector orientations. | `Validated` | $\text{Mean}=0.372\,\text{m}$, $\text{Max}=2.757\,\text{m}$. Identifies major distortion nodes in Banat/West, Maramureș, and Dobrogea. |
| [`pyproj_baseline/`](pyproj_baseline/) | `transformation` | **Standard PROJ Baseline**: Benchmarking `pyproj` (`EPSG:3844`) against ground-truth `pytransdat` on nationwide `ro_grid_4x4`. | `Validated` | $\text{Mean 2D}=0.3601\,\text{m}$, $\text{RMSE}=0.4348\,\text{m}$, $\text{Max}=2.82\,\text{m}$. Validates baseline without local grid shifts. |
| [`helmert2d_baseline/`](helmert2d_baseline/) | `transformation` | **2D 4-Parameter Helmert Baseline**: Closed-form OLS refit of 4-parameter 2D Helmert on valid SPG nodes (`StereoProj` + `Helmert2D`, no grid). | `Validated` | $\text{Mean 2D}=0.3743\,\text{m}$, $\text{RMSE}=0.4429\,\text{m}$, $\text{Max}=2.6811\,\text{m}$, $<1\,\text{m}=98.1\%$. Optimal 2D conformal floor. |
| [`knn_spatial_regressor/`](knn_spatial_regressor/) | `grid_shift` | **KNN 2D Spatial Regressor**: Evaluates K-Nearest Neighbors ($K \in \{1, 4, 8, 16\}$ with Uniform vs. IDW) on learning 2D SPG shifts directly from control nodes. | `Validated` | $\text{RMSE}=0.0353\,\text{m}$ ($3.5\,\text{cm}$) on `ro_grid_4x4` ($92.1\%$ reduction over floor), $\text{Max}=0.4783\,\text{m}$ with $K=4$ IDW. Proves $K=4$ cell-corner sweet spot. |
| [`decision_tree_regressor/`](decision_tree_regressor/) | `grid_shift` | **Decision Tree 2D Spatial Regressor**: Evaluates orthogonal step-plateau trees across 11 km, `4x4`, `8x8`, and `16x16` grids plus impurity pruning (`1e-8`). | `Validated` | $\text{RMSE}=0.0817\,\text{m}$ (11 km on `8x8`) $\to$ $0.0040\,\text{m}$ (`16x16` on corridor, $89.4\,\text{MB}$); `1e-8` pruning compresses to $2.81\,\text{MB}$ at $0.0188\,\text{m}$. |
| [`adaptive_quadtree_grid/`](adaptive_quadtree_grid/) | `grid_shift` | **Adaptive Quadtree Grid**: Curvature-driven 2D space partitioning using pure `pytransdat` grid shifts across 4 tolerance tiers (`2cm`, `1cm`, `3mm`, `1mm`). | `Validated` | $\text{Mean 2D}=0.142\,\text{cm}$ ($3\,\text{mm}$ tier, 176k cells) and $\text{Mean 2D}=0.050\,\text{cm}$ ($1\,\text{mm}$ tier, 564k cells, $\text{P95}=1.0\,\text{mm}$). $>97\%$ point reduction over uniform grids. |
| [`mlp_spatial_regressor/`](mlp_spatial_regressor/) | `grid_shift` | **Multilayer Perceptron (MLP) Spatial Regressor**: Evaluates continuous neural networks on valid SPG nodes tested against uniform (`4x4`) and adaptive (`3mm`) benchmarks. Resolves spectral bias via Random Fourier Features. | `Validated` | Raw MLP $\text{RMSE}=182\,\text{mm}$; Fourier MLP ($\sigma=3.0$) achieves $\text{RMSE}=13.0\,\text{mm}$ (4x4 trained, Uniform 4x4 test) and $C^\infty$ smoothness with $0\,\text{MB}$ grid lookup. |
| [`polynomial_surface_regressor/`](polynomial_surface_regressor/) | `grid_shift` | **2D Polynomial Surface Regressor**: Evaluates bivariate polynomial surfaces ($D=1\dots 12$) on normalized coordinates, upgrading the flat Helmert plane to a curved mathematical surface. | `Validated` | $\text{RMSE}=42.8\,\text{cm}$ ($D=1$, 3 terms) $\to$ $27.0\,\text{cm}$ ($D=5$, 21 terms) $\to$ $19.2\,\text{cm}$ ($D=12$, 91 terms); cuts max error from $2.60\,\text{m} \to 0.89\,\text{m}$. |
| [`rbf_surface_regressor/`](rbf_surface_regressor/) | `grid_shift` | **Thin Plate Spline (TPS) Regressor**: Localized 2D elastic sheet surface deformation with nugget smoothing ($\lambda=2\times 10^{-4}$), smoothing legacy grid slope breaks. | `Validated` | $\text{RMSE}=\mathbf{8.6\,\text{mm}}$ nationwide ($<1\,\text{cm}$), $\text{Max}=24.7\,\text{cm}$ ($0\,\text{MB}$ grid lookup); first continuous sub-centimeter model from 11 km nodes. |
| [`cell_planar_features/`](cell_planar_features/) | `analysis` | **Grid Cell Curvature & Shape Classification**: Functional geometric classification of $11\times 11\,\text{km}$ cells into a 3-class polynomial order taxonomy (Linear, Dome/Bowl, S-Shape). | `Validated` | Proves $53.2\%$ Dome vs. $46.8\%$ S-Shape at $5\,\text{mm}$, and $92.5\%$ Dome at $10\,\text{mm}$; exports shared features to `data/features/` for $O(1)$ extraction. |
| [`mlp_shape_features/`](mlp_shape_features/) | `grid_shift` | **Fourier MLP with Cell Shape Features**: Controlled ablation testing 3/5/10 mm per-axis shape categories (`cat_dn`, `cat_de`) and S-Shape sample weighting (`2x`, `3x`) on `mlp_fourier_4x4`. | `Dead End` | Discrete shape inputs degrade RMSE (`13.0` $\to$ `14.4 mm` uniform, `32.5` $\to$ `40.8 mm` adaptive); `2x`/`3x` S-Shape weighting trims `Max` outliers (`374.7` $\to$ `179.1 mm`) without improving adaptive RMSE. |
| [`cell_neighbor_features/`](cell_neighbor_features/) | `analysis` | **3×3 Grid Cell Neighbor Feature Extraction**: Extracts 40 continuous one-sided corner slopes and diagonal twists from the 3×3 `.spg` neighborhood (`11 km` grid). | `Validated` | Extracts 2,297 interior cells (`0` NaNs) to `data/features/cell_neighbor_features.parquet`; provides $O(1)$ `CellNeighborFeatureExtractor` for downstream models. |
| [`mlp_neighbor_features/`](mlp_neighbor_features/) | `grid_shift` | **Fourier MLP with 3×3 Neighbor Features & Center Distance**: Controlled ablation testing 32 slopes, 40 slopes+twists, and continuous `[0, 1]` midpoint distance on `mlp_fourier_4x4`. | `Dead End` | Adding neighbor slopes/twists or symmetric midpoint distance degrades RMSE (`13.0` $\to$ `13.7–15.5 mm` uniform, `32.5` $\to$ `42.0–47.3 mm` adaptive); `center_dist` helps 40-feature models (`51.4` $\to$ `47.0 mm`) but does not beat baseline. |
| [`torch_mlp_regressor/`](torch_mlp_regressor/) | `grid_shift` | **Deep PyTorch Residual Neural Network**: Evaluates PyTorch ResNets (`ReLU`, `SiLU`, `GELU`), smooth cyclic `(u, v)` cell position, and raw/Hermite-weighted `3×3` neighbor features on `RO_GRID_4X4`. | `Validated` | PyTorch ResNet (`ReLU`) cuts uniform RMSE from `13.0` $\to$ `7.2 mm`; adding cyclic `(u, v)` improves to **`6.8 mm`** (`95.6% < 1 cm`, `71.3% < 1 cm` adaptive). Neighbor slopes/twists (even Hermite-weighted) hurt adaptive RMSE/Max and are not promoted. |
| [`submillimeter_exploration/`](submillimeter_exploration/) | `grid_shift` | **Multi-Track Sub-Millimeter Exploration**: Screens 7 improvement tracks (two-stage residuals, adaptive datasets, harmonics, Huber/L-BFGS, learnable 2D feature grids, SIREN/Matérn/HistGBR, and bidirectional verification). | `Validated` | `T7_bilinear_plus_featgrid_8x8` reaches **`0.77 mm`** uniform mean / **`0.84 mm`** corridor RMSE (`78.3% < 1 mm`, `0.66 MB`); zero-lookup `T4_data_adaptive_hybrid` cuts adaptive RMSE by 94% (`44.18` $\to$ `2.59 mm`). |
| [`bicubic_boundary_analysis/`](bicubic_boundary_analysis/) | `analysis` | **Bicubic Cell Boundary Extension Analysis**: Visualizes `pytransdat` `Grid.BiInterp` polynomials extended across neighboring cells (`[-1, 2]`), Hermite tangents, and extreme S-shape/bowl stress tests, cross-verified against Avramiuc et al. (2009). | `Validated` | Confirmed `Grid.BiInterp` matches published `TransDatRO` one-sided 3-point finite-difference formulas verbatim; $C^1$ slope kinks at 11 km borders are inherent to the specification. |

---

## 2. Anatomy of a Research Unit

Each folder in `experiments/<slug>/` is an independent, self-contained experiment adhering to this standard blueprint:

```text
experiments/<slug>/
├── run_unit.py          # [Execution Engine] Standalone Python script subclassing GridShift/Transformation/AnalysisExperimentUnit
├── report.md            # [Human Walkthrough] Narrative, objectives, results tables, embedded figures, and Benchmark Candidates
├── unit_brief.json      # [Machine JSON] Standardized Schema v2 scorecard, artifacts, and benchmark_candidates
├── results/             # [Outputs] Metric JSON summaries, evaluations, scorecards
├── plots/               # [Outputs] Saved visual figures (spatial maps, distributions, quivers)
└── data/                # [Optional] Unit-specific intermediate splits (shared datasets go to data/features/)
```

---

## 3. How to Run an Experiment

All research units are runnable from workspace root using the project conda environment:

```powershell
# 1. SPG Grid Analysis (analysis)
D:\anaconda3\envs\ml_transdat\python.exe -m experiments.spg_grid_analysis.run_unit

# 2. PyProj Baseline (transformation)
D:\anaconda3\envs\ml_transdat\python.exe -m experiments.pyproj_baseline.run_unit

# 3. 2D 4-Parameter Helmert Baseline (transformation)
D:\anaconda3\envs\ml_transdat\python.exe -m experiments.helmert2d_baseline.run_unit

# 4. KNN 2D Spatial Regressor (grid_shift)
D:\anaconda3\envs\ml_transdat\python.exe -m experiments.knn_spatial_regressor.run_unit

# 5. Decision Tree 2D Spatial Regressor (grid_shift)
D:\anaconda3\envs\ml_transdat\python.exe -m experiments.decision_tree_regressor.run_unit

# 6. Adaptive Quadtree Grid (grid_shift)
D:\anaconda3\envs\ml_transdat\python.exe -m experiments.adaptive_quadtree_grid.run_unit

# 7. Fourier MLP Spatial Regressor (grid_shift)
D:\anaconda3\envs\ml_transdat\python.exe -m experiments.mlp_spatial_regressor.run_unit

# 8. 2D Polynomial Surface Regressor (grid_shift)
D:\anaconda3\envs\ml_transdat\python.exe -m experiments.polynomial_surface_regressor.run_unit

# 9. RBF & Thin Plate Spline Surface Regressor (grid_shift)
D:\anaconda3\envs\ml_transdat\python.exe -m experiments.rbf_surface_regressor.run_unit

# 10. Grid Cell Planar & Shape Classification (analysis)
D:\anaconda3\envs\ml_transdat\python.exe -m experiments.cell_planar_features.run_unit

# 11. Fourier MLP with Cell Shape Features (grid_shift)
D:\anaconda3\envs\ml_transdat\python.exe -m experiments.mlp_shape_features.run_unit

# 12. 3x3 Grid Cell Neighbor Feature Extraction (analysis)
D:\anaconda3\envs\ml_transdat\python.exe -m experiments.cell_neighbor_features.run_unit

# 13. Fourier MLP with 3x3 Cell Neighbor Features (grid_shift)
D:\anaconda3\envs\ml_transdat\python.exe -m experiments.mlp_neighbor_features.run_unit

# 14. Deep PyTorch Residual Neural Network Spatial Regressor (grid_shift)
D:\anaconda3\envs\ml_transdat\python.exe -m experiments.torch_mlp_regressor.run_unit

# 15. Multi-Track Sub-Millimeter Exploration (grid_shift)
D:\anaconda3\envs\ml_transdat\python.exe -m experiments.submillimeter_exploration.run_unit
# 15. Bicubic Cell Boundary Extension Analysis (analysis)
D:\anaconda3\envs\ml_transdat\python.exe -m experiments.bicubic_boundary_analysis.run_unit
```
