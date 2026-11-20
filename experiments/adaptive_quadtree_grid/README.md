# Experiment: Adaptive Quadtree Grid (`adaptive-quadtree-grid`)

## Objective
Generate compact, per-component ($\Delta N$, $\Delta E$) adaptive Quadtree grids in planimetric Stereo70 $(N, E)$ coordinates across 4 tolerance levels (`2cm`, `1cm`, `3mm`, `1mm`), and evaluate their accuracy as a pure geometric lookup/interpolation baseline (no ML model) using direct `pytransdat` 2D grid shifts.

## Multi-Tier Corridor Benchmark Results (`data/benchmarks/ro_corridor_snw.parquet`, 36,435 pts)
Directly modeling official `.spg` 2D grid shifts $((N, E) \to (\text{corr}_N, \text{corr}_E))$ via pure `pytransdat`:

| Tier | Cells ($\Delta N$ / $\Delta E$) | 2D Mean | 2D Median | 2D P95 | 2D P99 | 2D Max |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`2 cm`** | `26,066 / 24,317` | **`0.927 cm`** | `0.808 cm` | **`2.051 cm`** | `2.770 cm` | `5.920 cm` |
| **`1 cm`** | `55,091 / 51,356` | **`0.497 cm`** | `0.443 cm` | **`1.104 cm`** | `1.472 cm` | `5.920 cm` |
| **`3 mm`** *(official)* | `176,350 / 159,046` | **`0.142 cm`** (`1.4 mm`) | `0.125 cm` | **`0.308 cm`** (`3.1 mm`) | `0.428 cm` | `2.632 cm` |
| **`1 mm`** *(aspirational)* | `564,325 / 526,360` | **`0.050 cm`** (`0.5 mm`) | `0.043 cm` | **`0.105 cm`** (`1.0 mm`) | `0.147 cm` | `0.577 cm` (`5.8 mm`) |

## Artifacts
- Generator script: [`scripts/generate_adaptive_grid.py`](../../scripts/generate_adaptive_grid.py)
- Multi-tier evaluation script: [`run.py`](run.py)
- Results JSON: [`results.json`](results.json)
- Plots:
  - [`plots/tier_comparison.png`](plots/tier_comparison.png) (Comparative log-scale error across all 4 tiers)
  - [`plots/quadtree_100km_tolerances.png`](plots/quadtree_100km_tolerances.png) (Disaggregation visual on 100km block)
  - [`plots/error_distribution.png`](plots/error_distribution.png)
- Generated grids:
  - [`data/grids/adaptive/`](../../data/grids/adaptive/) (`2cm`, `1cm`, `3mm`, `1mm` parquet files)

## Key Takeaways
1. **Clear Geometric Scaling**: Moving from `2cm` to `1mm` scales error down predictably by a factor of ~18× while scaling cell count by ~21×.
2. **The 3 mm Official Standard**: Achieves **1.4 mm mean 2D error** and **3.1 mm P95 error** across the corridor with only ~176k cells, proving that official sub-centimeter geodetic tolerance is achievable with a compact representation.
3. **The 1 mm Aspirational Target**: Delivers **0.5 mm mean 2D error** and **1.0 mm P95 error** across the entire 36,435-point benchmark corridor.
4. **Purity**: Zero PyProj overhead; pure official `.spg` grid shifts extracted directly via `pytransdat._t_gr2d`.
