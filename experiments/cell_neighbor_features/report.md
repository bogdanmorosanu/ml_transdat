# 3×3 Grid Cell Neighbor Feature Extraction (`cell_neighbor_features`)

## Objective
Extract continuous $3 \times 3$ neighborhood geometry features for every valid $11 \times 11\,\text{km}$ `.spg` grid cell, capturing the exact mathematical ingredients used by bicubic interpolation without lossy categorical labels or central-difference slope cancellation.

## Engineered Features (40 Total)
For each of the 4 cell corners (`bl`, `br`, `tl`, `tr`) and both shift channels (`dn`, `de`):
- **`left_slope`** (`center - left`), **`right_slope`** (`right - center`), **`bottom_slope`** (`center - bottom`), **`top_slope`** (`top - center`)
- **`twist`** (`(tr_diag - br_diag - tl_diag + bl_diag) / 4.0`)

## Key Results
- **Valid interior cells**: `2,297` (`0` NaNs across all `40` features)
- **Max / Mean |slope|**: `2080.796 mm` / `94.559 mm` per `11 km` step
- **Max / Mean |twist|**: `660.115 mm` / `43.341 mm`
- **Runtime**: `0.70 s`

## Usage in Downstream Experiments
```python
from src.geometry.neighbor_features import CellNeighborFeatureExtractor

extractor = CellNeighborFeatureExtractor("data/features/cell_neighbor_features.parquet")
features = extractor.lookup_points(n_coords, e_coords)  # shape: (N, 40)
```

## Plots
- `plots/national_neighbor_slope_max_map.png`
- `plots/national_neighbor_twist_max_map.png`
