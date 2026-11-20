# Grids Directory Layout (`data/grids/`)

This directory houses spatial grid datasets for training, benchmarking, and evaluating coordinate transformation models between ETRS89 and Stereo70.

Uniform multi-resolution grids are **generated dynamically on the fly** via [`GridManager`](../../src/data/catalog.py), while curvature-driven quadtree grids are stored statically below.

---

## 1. Dynamic Uniform Grids (Generated On the Fly via `GridManager`)

Accessed transparently via `GridId` (and locally disk-cached under gitignored `.cache/grids/`):
- `GridId.SPG_NODES_VALID`: Valid 2D nodes extracted from `pytransdat/grids/rom_grid3d_25.09.spg` (`2,689` land nodes; `2,259` bidirectionally valid).
- `GridId.RO_GRID_4X4`: Uniform 4x4 subgrid (`2.75 km` step, `36,435` bidirectionally valid points).
- `GridId.RO_GRID_8X8`: Uniform 8x8 subgrid (`1.375 km` step, `146,258` points).
- `GridId.RO_GRID_16X16`: Uniform 16x16 subgrid (`687.5 m` step, `586,063` points).
- `GridId.SINGLE_CELL_10K`: Localized `10,000`-point workload sampled inside one `11 x 11 km` SPG cell.

Optional static materialization helper: [`scripts/generate_grids.py`](../../scripts/generate_grids.py).

---

## 2. `adaptive/` (Static Curvature-Driven Quadtree Grids)

Non-uniform, quadtree-partitioned grids stored statically as Parquet tables due to their multi-minute recursive generation time:
- `adaptive_<tol>_dn.parquet`: Adaptive leaf cells for the Northing shift component ($\Delta N$).
- `adaptive_<tol>_de.parquet`: Adaptive leaf cells for the Easting shift component ($\Delta E$).

Available tolerances:
- `adaptive_2cm_dn.parquet` / `adaptive_2cm_de.parquet`: 2 cm tolerance (~26k / 24k leaves).
- `adaptive_1cm_dn.parquet` / `adaptive_1cm_de.parquet`: 1 cm tolerance (~50k / 45k leaves).
- `adaptive_3mm_dn.parquet` / `adaptive_3mm_de.parquet`: 3 mm tolerance (~176k leaves).
- `adaptive_1mm_dn.parquet` / `adaptive_1mm_de.parquet`: 1 mm tolerance (~564k leaves).

Generator: [`scripts/generate_adaptive_grid.py`](../../scripts/generate_adaptive_grid.py).
