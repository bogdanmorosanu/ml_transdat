# Research Unit 02: PyProj (EPSG:3844) Baseline Evaluation

## Executive Summary
This unit establishes the baseline accuracy of standard GIS transformation pipelines using `pyproj` (`3.8.0` / PROJ `9.7.1`) between **ETRS89** (`EPSG:4258`) and **Stereo 70** (`EPSG:3844`, Pulkovo 1942(58) / Stereo70). Evaluated against golden-standard `pytransdat` outputs on the nationwide 36,435-point grid `ro_grid_4x4.parquet`.

---

## Bidirectional Scorecard

| Metric | Inverse: `ETRS89 -> Stereo70` | Forward: `Stereo70 -> ETRS89` |
|:---|:---:|:---:|
| **Sample Count** | $36,435$ | $36,435$ |
| **2D Mean Error** | $0.3601\,\text{m}$ | $0.3601\,\text{m}$ |
| **2D Std Dev** | $0.2437\,\text{m}$ | $0.2436\,\text{m}$ |
| **2D RMSE** | $0.4348\,\text{m}$ | $0.4347\,\text{m}$ |
| **Median (p50)** | $0.3058\,\text{m}$ | $0.3058\,\text{m}$ |
| **95th Percentile (p95)** | $0.8460\,\text{m}$ | $0.8459\,\text{m}$ |
| **99th Percentile (p99)** | $1.1367\,\text{m}$ | $1.1362\,\text{m}$ |
| **Maximum 2D Error** | $2.8207\,\text{m}$ | $2.8189\,\text{m}$ |
| **Signed Bias ($dN, dE$)** | $+0.0087\,\text{m}, -0.0160\,\text{m}$ | $-0.0077\,\text{m}, +0.0179\,\text{m}$ |
| **Pass Rate $< 10\,\text{cm}$** | $8.4\%$ | $8.3\%$ |
| **Pass Rate $< 1\,\text{m}$** | $98.0\%$ | $98.0\%$ |
| **Vectorized Throughput** | $> 3.5\times 10^6\,\text{pts/s}$ | $> 1.8\times 10^6\,\text{pts/s}$ |

---

## Visual Gallery

### 1. Spatial Error Heatmaps (2D Error in Meters)
The spatial distribution of `pyproj` error directly mirrors the SPG distortion grid profiled in Research Unit 01.

| ETRS89 $\rightarrow$ Stereo 70 | Stereo 70 $\rightarrow$ ETRS89 |
|:---:|:---:|
| ![Spatial Inv](plots/pyproj_epsg3844_etrs89_to_st70_spatial_error.png) | ![Spatial Fwd](plots/pyproj_epsg3844_st70_to_etrs89_spatial_error.png) |

---

### 2. Error Distributions (Histogram & ECDF)

| ETRS89 $\rightarrow$ Stereo 70 | Stereo 70 $\rightarrow$ ETRS89 |
|:---:|:---:|
| ![Dist Inv](plots/pyproj_epsg3844_etrs89_to_st70_hist_ecdf.png) | ![Dist Fwd](plots/pyproj_epsg3844_st70_to_etrs89_hist_ecdf.png) |

---

### 3. Signed Residual Scatter & Sub-Cell Profiles

| Signed Residuals ($dE$ vs $dN$) | Sub-Cell $(u, v)$ Error Profile |
|:---:|:---:|
| ![Scatter](plots/pyproj_epsg3844_etrs89_to_st70_signed_scatter.png) | ![Subcell](plots/pyproj_epsg3844_etrs89_to_st70_subcell_heatmap.png) |

---

## Scientific Takeaways & Next Steps
- **The Performance Ceiling of Pure PROJ**: `EPSG:3844` applies the 7-parameter Helmert transformation between Pulkovo 1942(58) and ETRS89, achieving sub-meter accuracy across 98% of Romania.
- **The Remaining Residuals**: Because standard PROJ does not load the Romanian `.spg` grid, residual errors range up to $2.82\,\text{m}$.
- **Target for ML / Hybrid Models**: Candidate models must push the mean error below $1\,\text{cm}$ ($0.01\,\text{m}$) and max error below $5\,\text{cm}$ ($0.05\,\text{m}$) to be considered survey-grade replacements.
