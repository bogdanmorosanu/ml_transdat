# Experiment Report: Adaptive Quadtree Grid (`adaptive_quadtree_grid`)

## 1. Objective & Hypothesis
Generate compact, per-component ($\Delta N$, $\Delta E$) adaptive Quadtree grids in planimetric Stereo 70 $(N, E)$ coordinates across 4 tolerance levels (`2cm`, `1cm`, `3mm`, `1mm`), and evaluate their accuracy as a pure geometric lookup/interpolation baseline using direct `pytransdat` 2D grid shifts (`AdaptiveQuadtreeRegressor`).

---

## 2. Multi-Tier Corridor Benchmark Results (`ro_corridor_snw`, 36,435 pts)

| Tier | Cells ($\Delta N$ / $\Delta E$) | 2D Mean | 2D Median | 2D P95 | 2D P99 | 2D Max |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`2 cm`** | `26,066 / 24,317` | **`0.927 cm`** | `0.808 cm` | **`2.051 cm`** | `2.770 cm` | `5.920 cm` |
| **`1 cm`** | `55,091 / 51,356` | **`0.497 cm`** | `0.443 cm` | **`1.104 cm`** | `1.472 cm` | `5.920 cm` |
| **`3 mm`** *(official)* | `176,350 / 159,046` | **`0.142 cm`** (`1.4 mm`) | `0.125 cm` | **`0.308 cm`** (`3.1 mm`) | `0.428 cm` | `2.632 cm` |
| **`1 mm`** *(aspirational)* | `564,325 / 526,360` | **`0.050 cm`** (`0.5 mm`) | `0.043 cm` | **`0.105 cm`** (`1.0 mm`) | `0.147 cm` | `0.577 cm` (`5.8 mm`) |

---

## 3. Visual Diagnostics & Artifacts
- [`plots/tier_comparison.png`](plots/tier_comparison.png): Comparative log-scale error across all 4 tiers.
- [`plots/quadtree_100km_tolerances.png`](plots/quadtree_100km_tolerances.png): Disaggregation visual on 100 km block.
- [`plots/quadtree_1mm_ro_corridor_snw_grid_shift_2d_spatial_error.png`](plots/quadtree_1mm_ro_corridor_snw_grid_shift_2d_spatial_error.png) & [`plots/quadtree_1mm_ro_corridor_snw_grid_shift_2d_hist_ecdf.png`](plots/quadtree_1mm_ro_corridor_snw_grid_shift_2d_hist_ecdf.png).
- Adaptive Parquet Grids: [`data/grids/adaptive/`](../../data/grids/adaptive/) (`2cm`, `1cm`, `3mm`, `1mm`).
- Scorecards: [`results/scorecard.csv`](results/scorecard.csv), [`results/summary_table.md`](results/summary_table.md).

---

## 4. Key Takeaways
1. **Clear Geometric Scaling**: Moving from `2cm` to `1mm` scales error down predictably by a factor of ~18x while scaling cell count by ~21x.
2. **The 3 mm Official Standard**: Achieves **1.4 mm mean 2D error** and **3.1 mm P95 error** across the corridor with ~176k cells, proving that official sub-centimeter geodetic tolerance is achievable with a compact representation.
3. **The 1 mm Aspirational Target**: Delivers **0.5 mm mean 2D error** and **1.0 mm P95 error** across the entire 36,435-point benchmark corridor.
4. **Purity**: Zero PyProj overhead; pure official `.spg` grid shifts extracted directly via `pytransdat._t_gr2d`.

---

## 5. Benchmark Candidates
Promoted to `benchmarks/grid_shift_models/` via `specs_from_unit(AdaptiveQuadtreeGridUnit, suite="grid_shift_models")`:

| Variant | Benchmark Label | Train Grid | Exact Configuration |
| :--- | :--- | :--- | :--- |
| `quadtree_3mm` | `Adaptive Quadtree (3mm tier)` | `None` (pre-built lookup) | `AdaptiveQuadtreeRegressor(dn_grid, de_grid)` from `3mm` adaptive pair |
| `quadtree_1mm` | `Adaptive Quadtree (1mm tier)` | `None` (pre-built lookup) | `AdaptiveQuadtreeRegressor(dn_grid, de_grid)` from `1mm` adaptive pair |
