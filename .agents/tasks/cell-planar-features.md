# Task: Grid Cell Planar Decomposition & Shape Classification

**Date**: 2026-11-21  
**Branch**: `exp/cell-planar-features`  
**Status**: Completed  
**Scope**: Feature Engineering | Data Architecture | Spatial Classification  

---

## Objective
Investigate local geometric patch decomposition and feature engineering for Romania's $11 \times 11\,\text{km}$ official distortion grid (`rom_grid3d_25.09.spg`), transitioning from global $(N, E)$ coordinates to local zero-reduced coordinates $(u, v) \in [0, 1]$ and classifying cells into a strict 3-class polynomial curvature taxonomy (`Linear`, `Dome/Bowl`, `S-Shape`). Deliver an $O(1)$ persistent spatial feature extractor for downstream ML models.

---

## Key Activities & Implementation

1. **Reusable Geometry Module (`src/geometry/`)**:
   - Implemented [`src/geometry/plane.py`](../../src/geometry/plane.py): `CellPlane` and `CellPlane2D` dataclasses for fitting 3D planes, base zero-offsets ($z_{00}$), slopes ($s_N, s_E$), gradients, and bilinear surfaces on 4-corner cell patches.
   - Implemented [`src/geometry/shape_classifier.py`](../../src/geometry/shape_classifier.py): Progressive polynomial order fitting (Order 1 Linear, Order 2 Parabolic, Order 3 Cubic) evaluated on 25 interior test points ($5 \times 5$ grid) per cell against `pytransdat`.
   - Clamped residual Category 4 anomalies into Category 3, formalizing a permanent **3-class taxonomy**: `1: Linear`, `2: Dome/Bowl`, `3: S-Shape`.
   - Externalized [`CellShapeFeatureExtractor`](../../src/geometry/shape_classifier.py): High-speed $O(1)$ matrix-indexed spatial lookup mapping arbitrary $(N, E)$ query points (scalar or vectorized) to $(u, v)$ fractional coordinates, base shifts, and shape categories.

2. **Automated Unit Testing (`tests/`)**:
   - Created [`tests/test_cell_plane.py`](../../tests/test_cell_plane.py): Verifies exact planar recovery, saddle twist, and full `.spg` extraction (100% passing).
   - Created [`tests/test_shape_classifier.py`](../../tests/test_shape_classifier.py): Verifies synthetic polynomial order recovery (Linear, Dome, S-Shape) and single-point/vectorized lookups with `CellShapeFeatureExtractor` (100% passing).

3. **Experiment Unit & Reporting (`experiments/cell_planar_features/`)**:
   - Implemented [`classify_cell_shapes.py`](../../experiments/cell_planar_features/classify_cell_shapes.py) running full-grid classification sweeps on all 2,297 interior cells across tolerances ($2\,\text{mm}, 3\,\text{mm}, 5\,\text{mm}, 10\,\text{mm}, 15\,\text{mm}$).
   - Generated tabular datasets: [`results/cell_shape_metrics.parquet`](../../experiments/cell_planar_features/results/cell_shape_metrics.parquet) and `cell_shape_metrics.csv`.
   - Generated national spatial maps: [`plots/national_shape_map_5mm.png`](../../experiments/cell_planar_features/plots/national_shape_map_5mm.png) and [`plots/national_shape_map_10mm.png`](../../experiments/cell_planar_features/plots/national_shape_map_10mm.png).
   - Documented in [`report.md`](../../experiments/cell_planar_features/report.md) and [`README.md`](../../experiments/cell_planar_features/README.md).

4. **Living Documentation Synchronization**:
   - Created research unit [`docs/research/cell_planar_features.md`](../../docs/research/cell_planar_features.md).
   - Updated [`docs/research/roadmap.md`](../../docs/research/roadmap.md), [`docs/research/narrative.md`](../../docs/research/narrative.md), and [`experiments/README.md`](../../experiments/README.md).

---

## Verification & Empirical Results

- **Workspace Test Suite**: 46/46 unit tests passing across the repository in 14.8s. Ruff clean (0 errors).
- **Tolerance Sweep ($N = 2,297$ Interior Cells)**:
  - **At $5.0\,\text{mm}$ (Primary Benchmark)**:
    - $\Delta N$: $73.3\%$ Dome/Bowl, $26.0\%$ S-Shape, $0.7\%$ Linear.
    - $\Delta E$: $76.1\%$ Dome/Bowl, $21.4\%$ S-Shape, $2.5\%$ Linear.
    - **Combined 2D**: **$53.2\%$ Dome/Bowl**, **$46.8\%$ S-Shape**, **$0.0\%$ Linear**. Category 4 drops to $< 0.5\%$.
  - **At $10.0\,\text{mm}$ ($1\,\text{cm}$)**:
    - **Combined 2D**: **$92.5\%$ Dome/Bowl**, **$4.8\%$ S-Shape**, **$2.7\%$ Linear** (proving that within $1\,\text{cm}$, over $92\%$ of Romania behaves as simple 2nd-degree parabolas).
- **Axis Concordance**:
  - In **$68.7\%$** of cells at $5\,\text{mm}$ (and **$69.6\%$** at $10\,\text{mm}$), $\Delta N$ and $\Delta E$ share the identical shape category. Mismatch across two levels (Linear on one axis, S-Shape on the other) is $< 0.5\%$ nationwide.
