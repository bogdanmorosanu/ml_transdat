# Research Unit: 3×3 Grid Cell Neighbor Features (`cell_neighbor_features`)

- **Status**: `Validated`
- **Primary Sources**: [`experiments/cell_neighbor_features/report.md`](../../experiments/cell_neighbor_features/report.md) | [`src/geometry/neighbor_features.py`](../../src/geometry/neighbor_features.py)

## Hypothesis & Rationale
Discrete cell-shape categories (`cell_planar_features` $\to$ `mlp_shape_features`) degraded Fourier MLP accuracy because hard categorical boundaries introduce step discontinuities without quantitative geometry. Since `pytransdat` uses bicubic interpolation over a $3 \times 3$ cell neighborhood ($4 \times 4 = 16$ nodes), extracting continuous **one-sided corner slopes** (`left`, `right`, `bottom`, `top`—preventing central-difference peak/valley cancellation) and **diagonal twists** ($f_{xy}$) at all 4 cell corners provides the exact mathematical ingredients of the local surface.

## Key Findings
- Extracted **40** continuous features across **2,297** valid interior `11 km` `.spg` cells (`0` NaNs) to `data/features/cell_neighbor_features.parquet`.
- Delivered $O(1)$ `CellNeighborFeatureExtractor` for downstream ML models.
