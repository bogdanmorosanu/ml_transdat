# Grids Directory Layout (`data/grids/`)

This directory houses spatial grid datasets generated for training, benchmarking, and evaluating coordinate transformation models between ETRS89 and Stereo70.

The datasets are partitioned into two distinct paradigms:

## 1. `densified/` (Uniform Resolution Grids)
Uniformly densified multi-resolution grid nodes derived directly by recursively subdividing the original binary `.spg` distortion grid cells at fixed step sizes:
- `spg_nodes_valid.parquet`: Baseline valid grid nodes extracted from `pytransdat/grids/rom_grid3d_25.09.spg` (~2.5 km resolution, 26,056 valid nodes).
- `ro_grid_4x4.parquet`: 4x4 uniform densification per base cell (~625 m spacing).
- `ro_grid_8x8.parquet`: 8x8 uniform densification per base cell (~312.5 m spacing).
- `ro_grid_16x16.parquet`: 16x16 uniform densification per base cell (~156 m spacing).

**Characteristics**:
- Strict regular grid topology.
- Constant cell dimension across the entire country regardless of local terrain or distortion slope.
- Generator: [`scripts/generate_grids.py`](../../scripts/generate_grids.py).

---

## 2. `adaptive/` (Curvature-Driven Quadtree Grids)
Non-uniform, quadtree-partitioned grids generated dynamically based on interpolation-error tolerance:
- `adaptive_<tol>_dn.parquet`: Adaptive leaf cells for the Northing shift component ($\Delta N$).
- `adaptive_<tol>_de.parquet`: Adaptive leaf cells for the Easting shift component ($\Delta E$).

Available tolerances:
- `adaptive_2cm_dn.parquet` / `adaptive_2cm_de.parquet`: 2 cm tolerance (~26k / 24k leaves).
- `adaptive_1cm_dn.parquet` / `adaptive_1cm_de.parquet`: 1 cm tolerance (~50k / 45k leaves).
- `adaptive_3mm_dn.parquet` / `adaptive_3mm_de.parquet`: 3 mm tolerance (official standard, ~210k / 195k leaves).
- `adaptive_1mm_dn.parquet` / `adaptive_1mm_de.parquet`: 1 mm tolerance (aspirational target, ~700k leaves).

**Characteristics**:
- Variable cell dimensions: large cells (up to 35+ km) in smooth areas, fine cells (down to ~97 m) concentrated strictly on steep distortion ridges.
- Evaluates 5 interior test points (cell center + 4 sub-quadrant centers) to ensure error remains strictly bounded within the target tolerance.
- Leaves store bounding boxes (`n_min, n_max, e_min, e_max`) and 4-corner ground-truth shifts (`v00, v01, v10, v11`) for $O(\log N)$ point containment and bilinear interpolation.
- Generator: [`scripts/generate_adaptive_grid.py`](../../scripts/generate_adaptive_grid.py).
