# Branch 01: SPG 2D Distortion Grid Analysis

- **Status**: `Validated`
- **Related Unit**: [`experiments/01_spg_grid_analysis/`](../../experiments/01_spg_grid_analysis/)
- **Core Question**: What is the physical spatial distribution, magnitude, and directional orientation of the official Romanian cartographic distortion field?

---

## 1. Premise & Motivation
Before designing machine learning models to replicate the Romanian coordinate transformation, we must profile the target distortion function itself. The official Centrul Național de Cartografie (CNC) grid (`rom_grid3d_25.09.spg`) applies 16-node bicubic spline shifts ($dN, dE$) in Stereo 70 meters across an $11\,\text{km} \times 11\,\text{km}$ primary cell grid.

---

## 2. Experimental Setup
- **Evaluation Dataset**: `data/benchmarks/ro_grid_3x3.parquet` (36,435 bidirectionally valid points, 2,297 grid cells, $2.75\,\text{km}$ sub-grid resolution).
- **Execution Script**: `python -m experiments.01_spg_grid_analysis.run_unit`.
- **Outputs**: Component heatmaps ($dN, dE$, magnitude), vector field quiver plot, and distribution diagnostics.

---

## 3. Quantitative Discoveries
- **Mean Horizontal Shift**: $0.3716\,\text{m}$ (Median: $0.2642\,\text{m}$)
- **95th / 99th Percentile**: $1.1420\,\text{m}$ / $1.6841\,\text{m}$
- **Maximum Shift**: $2.7570\,\text{m}$
- **Northing Extremes ($dN$)**: Min $=-2.7364\,\text{m}$, Max $=+1.2293\,\text{m}$
- **Easting Extremes ($dE$)**: Min $=-0.9421\,\text{m}$, Max $=+1.1808\,\text{m}$

---

## 4. Architectural & Modeling Verdict
- **Validated**: The distortion field is spatially continuous and coherent, with prominent regional vectors (south-westward in Banat, northward in Maramureș).
- **Implication for ML**: Models cannot rely solely on low-order polynomial terms; local spatial features (coordinate encodings, splines, or neural embeddings) are mandatory to capture the non-linear border fringes.
