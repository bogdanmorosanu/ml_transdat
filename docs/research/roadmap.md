# Research Roadmap & Knowledge Graph

Master map of research units, annotated relationships, and current scientific state for **ETRS89 ↔ Stereo 70** coordinate transformation modeling.

> **Current State TL;DR**: All three foundational baselines (`spg_grid_analysis`, `pyproj_baseline`, `helmert2d_baseline`) are **Validated**. They establish a $\sim 0.44\,\text{m}$ RMSE / $2.68\,\text{m}$ max-error floor for global linear/conformal models and confirm that sub-centimeter ($< 0.01\,\text{m}$) accuracy requires learning the non-linear 2D bicubic `.spg` distortion field via residual ML models.

---

## 1. Research Topology DAG

```mermaid
flowchart TD
    SPG["spg_grid_analysis<br><b>Status: Validated</b><br>Profiles physical shifts (mean=0.37m, max=2.76m)"]
    PROJ["pyproj_baseline<br><b>Status: Validated</b><br>Evaluates EPSG:3844 (RMSE=0.43m, max=2.82m)"]
    H2D["helmert2d_baseline<br><b>Status: Validated</b><br>Refitted 2D conformal floor (RMSE=0.44m, max=2.68m)"]
    KNN["knn_spatial_regressor<br><b>Status: Validated</b><br>1st ML baseline: 3.5cm RMSE via K=4 IDW (92% reduction)"]
    DT["decision_tree_regressor<br><b>Status: Validated</b><br>Step-lookup scaling: 8.2cm (11km) -> 0.4cm (16x16, 89MB)"]
    ML["Next: Continuous & Neural Surface Models<br><b>Status: Exploring</b><br>Goal: Sub-centimeter precision (<0.01m) via RBF / MLP"]

    SPG -->|"motivates"| PROJ
    SPG -->|"motivates"| H2D
    PROJ -->|"establishes floor for"| ML
    H2D -->|"provides analytical base for"| KNN
    KNN -->|"establishes local baseline for"| ML
    KNN -->|"contrasted with"| DT
    DT -->|"motivates smooth interpolation"| ML
```

---

## 2. Annotated Relationships

| From | To | Relation | Meaning / Rationale |
| :--- | :--- | :--- | :--- |
| `spg_grid_analysis` | `pyproj_baseline` | `motivates` | Profiling `.spg` shifts showed localized $2.76\,\text{m}$ distortions; we needed to verify whether standard GIS (`EPSG:3844`) already models them (it does not). |
| `spg_grid_analysis` | `helmert2d_baseline` | `motivates` | Knowing the `.spg` field is smooth and dominated by 2D shifts motivated fitting an optimal grid-free 4-parameter 2D conformal floor. |
| `pyproj_baseline` | `Residual ML Models` | `establishes floor` | Sets the external GIS industry benchmark ($\text{RMSE}=0.435\,\text{m}$, $\text{Max}=2.82\,\text{m}$) that any ML model must decisively beat. |
| `helmert2d_baseline` | `Residual ML Models` | `provides base` | Proves `StereoProj + Helmert2D` is analytically invertible ($5.24 \times 10^{-9}\,\text{m}$ round-trip) and leaves clean, zero-bias 2D residuals for ML learning. |
| `helmert2d_baseline` | `knn_spatial_regressor` | `provides base` | Residuals from `StereoProj + Helmert2D` define the target 2D distortion field for spatial learning. |
| `knn_spatial_regressor` | `Continuous ML Models` | `establishes local baseline` | Sets the first spatial ML baseline ($0.0353\,\text{m}$ RMSE) and reveals official grid $C^0/C^1$ boundary slope kinks. |
| `knn_spatial_regressor` | `decision_tree_regressor` | `contrasted with` | Contrasts smooth local distance-weighted blending ($3.5\,\text{cm}$) against orthogonal axis-aligned step plateaus ($8.2\,\text{cm}$ on base nodes). |
| `decision_tree_regressor` | `Continuous ML Models` | `motivates smooth interpolation` | Proves step-lookup models require massive densification ($586\text{k}$ pts / $89\,\text{MB}$) to reach $4\,\text{mm}$, motivating smooth interpolators. |

---

## 3. Research Units Index

| Slug | Status | Strategic Summary | Primary Sources |
| :--- | :---: | :--- | :--- |
| [`spg_grid_analysis`](spg_grid_analysis.md) | `Validated` | Physical profiling of `.spg` 2D shifts ($\text{Mean}=0.372\,\text{m}$, $\text{Max}=2.757\,\text{m}$); identifies localized peaks in Banat/West, Maramureș, and Dobrogea. | [`experiments/spg_grid_analysis/`](../../experiments/spg_grid_analysis/README_AI.md) |
| [`pyproj_baseline`](pyproj_baseline.md) | `Validated` | Standard `EPSG:3844` vs. ground truth ($\text{RMSE}=0.4348\,\text{m}$, $\text{Max}=2.82\,\text{m}$); proves standard GIS lacks local distortion modeling. | [`experiments/pyproj_baseline/`](../../experiments/pyproj_baseline/README_AI.md) |
| [`helmert2d_baseline`](helmert2d_baseline.md) | `Validated` | Refitted 4-param 2D conformal Helmert on 2,222 valid nodes ($\text{RMSE}=0.4429\,\text{m}$, $\text{Max}=2.6811\,\text{m}$); optimal linear/conformal floor. | [`experiments/helmert2d_baseline/`](../../experiments/helmert2d_baseline/README_AI.md) |
| [`knn_spatial_regressor`](knn_spatial_regressor.md) | `Validated` | First non-linear ML baseline ($K=4$ IDW achieves $\text{RMSE}=0.0353\,\text{m}$, $92.1\%$ reduction over floor); discovers piecewise $C^0/C^1$ grid kinks. | [`experiments/knn_spatial_regressor/`](../../experiments/knn_spatial_regressor/README_AI.md) |
| [`decision_tree_regressor`](decision_tree_regressor.md) | `Validated` | Orthogonal step-plateau scaling ($8.17\,\text{cm}$ on 11 km nodes $\to$ $0.40\,\text{cm}$ on `16x16` at $89.4\,\text{MB}$); impurity pruning compresses $87\%$ ($2.8\,\text{MB}$ at $1.88\,\text{cm}$). | [`experiments/decision_tree_regressor/`](../../experiments/decision_tree_regressor/report.md) | [`experiments/knn_spatial_regressor/`](../../experiments/knn_spatial_regressor/README_AI.md) |
