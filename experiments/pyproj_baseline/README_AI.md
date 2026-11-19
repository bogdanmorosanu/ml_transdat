# Research Unit 02 Briefing: PyProj Baseline Evaluation

## Purpose
Establishes the standard GIS benchmark accuracy using `pyproj` (`EPSG:3844` $\leftrightarrow$ `EPSG:4258`) against golden-standard `pytransdat`.

## Core Metadata
- **Unit ID**: `02`
- **Unit Slug**: `pyproj_baseline`
- **Library Stack**: `pyproj 3.8.0` / `PROJ 9.7.1`
- **CRS Mapping**: Source/Target `EPSG:4258` (ETRS89) $\leftrightarrow$ `EPSG:3844` (Pulkovo 1942(58) / Stereo70)
- **Pipeline Description**: Conformal Oblique Stereographic + 7-parameter Helmert (`Pulkovo 1942(58) to ETRS89 (4)`)

## Quantitative Performance
- **Mean 2D Horizontal Error**: $0.3601\,\text{m}$ (both directions)
- **RMSE 2D**: $0.4348\,\text{m}$
- **Median Error (p50)**: $0.3058\,\text{m}$
- **95th Percentile (p95)**: $0.8460\,\text{m}$
- **99th Percentile (p99)**: $1.1367\,\text{m}$
- **Max Error**: $2.8207\,\text{m}$
- **Systematic Bias ($dN, dE$)**: $+0.0087\,\text{m}, -0.0160\,\text{m}$ (negligible global shift)
- **Pass Rates**: $< 10\,\text{cm}$: $8.4\%$ | $< 1\,\text{m}$: $98.0\%$

## Key Failure Modes
- PROJ lacks the national `.spg` bicubic distortion grid; residuals directly reflect uncompensated local tectonic/network distortions.
- Maximum errors concentrated along border fringes (West, North, Dobrogea).

## Artifact Index
- `run_unit.py`: Standalone execution script.
- `report.md`: Visual and narrative summary.
- `unit_brief.json`: Machine-readable metrics payload.
- `results/`: Per-direction scorecard JSON files.
- `plots/`: Diagnostic heatmaps, hist/ecdf, signed scatter, and subcell plots.
