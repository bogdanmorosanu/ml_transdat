# PyProj (EPSG:3844) Baseline Evaluation

- **Status**: `Validated`
- **Spawned From**: [`spg_grid_analysis`](spg_grid_analysis.md)
- **Linked Sources**: [`experiments/pyproj_baseline/`](../../experiments/pyproj_baseline/)
- **Core Question**: What is the performance ceiling of standard open-source GIS transformation pipelines (PROJ / PyProj) that lack official Romanian distortion grid files?

---

## 1. Premise & Motivation
Most geospatial applications (e.g. QGIS, GDAL, GeoPandas) perform transformations between ETRS89 (`EPSG:4258`) and Stereo 70 via `EPSG:3844` (Pulkovo 1942(58) / Stereo70). We must quantify the exact error of this standard industry baseline across Romania against the golden standard `pytransdat`.

---

## 2. Experimental Setup
- **Evaluation Dataset**: `data/grids/ro_grid_4x4.parquet` (36,435 points).
- **Execution Script**: `python -m experiments.pyproj_baseline.run_unit`.
- **Target Systems**: `EPSG:4258` (ETRS89) $\leftrightarrow$ `EPSG:3844` (Stereo 70).

---

## 3. Quantitative Discoveries

| Direction | Mean 2D Error | 2D RMSE | Median (p50) | p95 | Max Error | $<1\,\text{m}$ Pass |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **`etrs89_to_st70`** | $0.3601\,\text{m}$ | $0.4348\,\text{m}$ | $0.3058\,\text{m}$ | $0.8460\,\text{m}$ | $2.8207\,\text{m}$ | $98.0\%$ |
| **`st70_to_etrs89`** | $0.3601\,\text{m}$ | $0.4347\,\text{m}$ | $0.3058\,\text{m}$ | $0.8459\,\text{m}$ | $2.8189\,\text{m}$ | $98.0\%$ |

---

## 4. Architectural & Modeling Verdict
- **Validated**: The standard PROJ pipeline successfully applies 7-parameter Helmert parameters, eliminating global translation offsets (mean bias is negligible: $+0.009\,\text{m}$ in $N$, $-0.016\,\text{m}$ in $E$).
- **The Gap**: Residual errors directly mirror the unmodeled SPG distortion field profiled in Branch 01 ($0.36\,\text{m}$ mean, $2.82\,\text{m}$ max).
- **Target for Future Research**: Any candidate ML architecture or hybrid engine must significantly outperform this $0.36\,\text{m}$ baseline, aiming for sub-centimeter survey-grade accuracy ($< 0.01\,\text{m}$).
