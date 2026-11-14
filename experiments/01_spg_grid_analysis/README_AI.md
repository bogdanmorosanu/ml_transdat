# Research Unit 01 Briefing: SPG 2D Distortion Grid Analysis

## Purpose
Physical profile of the ground-truth official Romanian 2D distortion grid `rom_grid3d_25.09.spg` (`Grid2D.interp` bicubic spline).

## Core Metadata
- **Unit ID**: `01`
- **Unit Slug**: `spg_grid_analysis`
- **Benchmark Source**: `data/benchmarks/ro_grid_3x3.parquet` (36,435 points, 2,297 grid cells)
- **Grid File**: `pytransdat/grids/rom_grid3d_25.09.spg`
- **Grid Resolution**: 11 km x 11 km primary cell step, sampled at 2.75 km sub-grid resolution

## Quantitative Summary
- **Mean Shift Magnitude**: $0.3716\,\text{m}$
- **Median Magnitude (p50)**: $0.2642\,\text{m}$
- **95th Percentile (p95)**: $1.1420\,\text{m}$
- **99th Percentile (p99)**: $1.6841\,\text{m}$
- **Max Shift Magnitude**: $2.7570\,\text{m}$
- **Northing Shift ($dN$) Bounds**: $[-2.7364, +1.2293]\,\text{m}$ (Mean: $-0.0043\,\text{m}$)
- **Easting Shift ($dE$) Bounds**: $[-0.9421, +1.1808]\,\text{m}$ (Mean: $+0.0120\,\text{m}$)

## High-Risk Geographic Hotspots
- **West / Banat / Timiș**: Peak horizontal displacement up to $2.76\,\text{m}$ oriented southwest.
- **North / Maramureș**: Strong positive northing shifts ($> +1.0\,\text{m}$).
- **South-West / Danube**: Pronounced localized northing dip.

## Artifact Index
- `run_unit.py`: Executable reproduction script.
- `report.md`: Human visual and narrative report.
- `unit_brief.json`: Machine-readable scorecard.
- `plots/spg_shift_quiver.png`: Vector field orientation plot.
- `plots/spg_shift_mag.png`: Combined magnitude spatial heatmap.
- `plots/spg_shift_dn.png`: Signed northing shift heatmap.
- `plots/spg_shift_de.png`: Signed easting shift heatmap.
