# Task: Adaptive Quadtree Grid Experiment & Multi-Resolution Restructuring

- **Date**: 2026-11-20
- **Branch**: `exp/adaptive-quadtree-grid`
- **Execution Type**: Research, Data Engineering & Baseline Experimentation

---

## 1. Intent & Context
Investigate whether non-uniform, curvature-driven space partitioning (Quadtree) can replace brute-force uniform grid densification (`16x16`, `32x32`), capturing official Romanian `.spg` 2D grid shifts to millimeter precision with drastically fewer nodes.

---

## 2. Changes Implemented

### A. Data Directory Restructuring (`data/grids/`)
- Separated grids into two functional categories:
  - `data/grids/densified/`: Fixed uniform step grids (`spg_nodes_valid.*`, `ro_grid_4x4.*`, `ro_grid_8x8.*`, `ro_grid_16x16.*`).
  - `data/grids/adaptive/`: Non-uniform curvature-driven Quadtree grids across 4 tolerance tiers (`2cm`, `1cm`, `3mm`, `1mm`).
- Added comprehensive documentation in [`data/grids/README.md`](../../data/grids/README.md).

### B. Pure PyTransDat Grid Generator (`scripts/generate_adaptive_grid.py`)
- Removed PyProj and spherical coordinate projection roundtrips.
- Formulated as a pure one-way shift lookup directly from `pytransdat`'s 2D engine: $(N, E) \to (\text{corr}_N, \text{corr}_E)$ via `tr._t_gr2d.interp(N, E)`.
- Implemented **5-point interior verification check** (cell center + 4 sub-quadrant centers at $25\%$ and $75\%$).
- Generated full-Romania coverage across 4 tolerance tiers:
  - `2cm`: 26,066 leaves ($\Delta N$), 24,317 leaves ($\Delta E$)
  - `1cm`: 55,091 leaves ($\Delta N$), 51,356 leaves ($\Delta E$)
  - `3mm`: 176,350 leaves ($\Delta N$), 159,046 leaves ($\Delta E$)
  - `1mm`: 564,325 leaves ($\Delta N$), 526,360 leaves ($\Delta E$)

### C. Experiment Unit (`experiments/adaptive_quadtree_grid/`)
- Built [`run.py`](../../experiments/adaptive_quadtree_grid/run.py) evaluating all 4 tiers against 36,435 corridor benchmark points.
- Produced [`results.json`](../../experiments/adaptive_quadtree_grid/results.json) and comparative figures ([`tier_comparison.png`](../../experiments/adaptive_quadtree_grid/plots/tier_comparison.png), [`quadtree_100km_tolerances.png`](../../experiments/adaptive_quadtree_grid/plots/quadtree_100km_tolerances.png)).
- Documented findings in [`README.md`](../../experiments/adaptive_quadtree_grid/README.md).

### D. Living Research Documentation Synchronization
- Created dedicated research unit: [`docs/research/adaptive_quadtree_grid.md`](../../docs/research/adaptive_quadtree_grid.md).
- Updated DAG topology, annotated edges, and research units table in [`docs/research/roadmap.md`](../../docs/research/roadmap.md).
- Added Phase 2 milestone #4 to [`docs/research/narrative.md`](../../docs/research/narrative.md).
- Registered unit in [`experiments/README.md`](../../experiments/README.md).

---

## 3. Benchmark Verification

Corridor Benchmark Results (`ro_corridor_snw.parquet`, 36,435 points):

| Tier | Cells ($\Delta N$ / $\Delta E$) | 2D Mean | 2D Median | 2D P95 | 2D P99 | 2D Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`2 cm`** | `26,066 / 24,317` | $0.927\,\text{cm}$ | $0.808\,\text{cm}$ | $2.051\,\text{cm}$ | $2.771\,\text{cm}$ | $5.921\,\text{cm}$ |
| **`1 cm`** | `55,091 / 51,356` | $0.497\,\text{cm}$ | $0.419\,\text{cm}$ | $1.104\,\text{cm}$ | $1.918\,\text{cm}$ | $5.921\,\text{cm}$ |
| **`3 mm`** *(official)* | `176,350 / 159,046` | **$0.143\,\text{cm}$ ($1.4\,\text{mm}$)** | **$0.125\,\text{cm}$** | **$0.308\,\text{cm}$ ($3.1\,\text{mm}$)** | $0.449\,\text{cm}$ | $2.632\,\text{cm}$ |
| **`1 mm`** *(aspirational)* | `564,325 / 526,360` | **$0.050\,\text{cm}$ ($0.5\,\text{mm}$)** | **$0.044\,\text{cm}$** | **$0.105\,\text{cm}$ ($1.0\,\text{mm}$)** | $0.159\,\text{cm}$ | **$0.577\,\text{cm}$ ($5.8\,\text{mm}$)** |
