# Hybrid Stage-1 Surface + Stage-2 Grid-Free ResNet (`hybrid_surface_resnet`)

- **Status**: `Validated`
- **Spawned From**: [`optimal_surface_regressor`](optimal_surface_regressor.md), [`submillimeter_exploration`](submillimeter_exploration.md)
- **Linked Sources**: [`experiments/hybrid_surface_resnet/`](../../experiments/hybrid_surface_resnet/report.md), [`src/models/composite_shift.py`](../../src/models/composite_shift.py), [`src/models/torch_mlp.py`](../../src/models/torch_mlp.py), [`src/models/_torch_trainer.py`](../../src/models/_torch_trainer.py), [`src/data/catalog.py`](../../src/data/catalog.py)

---

## Current Verdict
Pairing `optimal_surface_regressor`'s #3 Stage-1 base (`lpoly_deg2_k16_tric`, fitted on `RO_GRID_4X4`) with a **compact, grid-free Stage-2 PyTorch ResNet (`TorchMLPGridShiftModel`, `0 MB` spatial lookup grid)** (`hidden_dim=256`, `num_res_blocks=3`, `SiLU`, `Huber` loss `beta=5 mm`, `cyclic_cell_harmonics=2`, multi-scale RFF `(1.0, 3.0, 10.0)` with `64` frequencies) trained on `RO_GRID_32X32` achieves **`0.31 mm` mean (`98.39% < 1 mm`)** on Full-Romania `16x16` and **`1.66 mm` mean (`79.31% < 1 mm`)** on `Adapt 1mm` (`4.082 MB` total). It decisively beats standalone `TorchMLP_32x32_only` (`1.22 mm` `16x16` mean / `50.89% < 1 mm`) and closes nearly the entire gap to lookup-based `hybrid_surface_featgrid` (`0.24 mm`, `99.3% < 1 mm`) without storing any spatial feature grid.

---

## Premise & Rationale
In `hybrid_surface_featgrid`, Stage 2 relied on two learned spatial lookup grids (`53×72` and `105×143`). Here we tested the parallel **zero-lookup-grid** branch: whether removing the macro trend via the Top-3 `4x4` smooth Stage-1 surfaces (`cubic_k25_s0`, `tps_deg2_k25_s1e-5`, `lpoly_deg2_k16_tric`) allows a pure `TorchMLPGridShiftModel` ResNet (`0 MB` lookup) to reach sub-millimeter precision under the exact same two-phase protocol.

---

## Key Strategic Findings
- **Stage-1 Ranking Reverses for Grid-Free ResNets**: In the 56-run Option-B matrix on `box10_interior`, `lpoly_deg2_k16_tric` beat `cubic_k25_s0` and `tps_deg2_k25_s1e-5` on unseen `Adapt 1mm` (`1.577 mm` vs `2.973 mm` vs `3.160 mm` RMSE), because without local lookup tables a shared global ResNet benefits most from `lpoly`'s lower standalone residual amplitude ([`subgrid_matrix.md`](../../experiments/hybrid_surface_resnet/results/subgrid_matrix.md)).
- **`SiLU` + `Huber` Dominance**: `SiLU` outperformed `ReLU` across all surfaces and densities (`1.577 mm` vs `3.240 mm` `A1` RMSE at `32x32 h128`), and `Huber` beat `MSE` by ~2× (`2.973 mm` vs `5.054 mm` `A1` RMSE on `cubic_k25_s0`).
- **Country-Scale Capacity & Multi-Scale RFF Scaling**: While `h128` peaked on `box10_interior` (`100` cells), Full Romania (`~23×` larger) was capacity- and wavelength-limited at `h128`: scaling to `h256` + multi-scale RFF `(1.0, 3.0, 10.0)` cut `16x16` mean from `0.57 mm` → **`0.31 mm`** (`89.71%` → **`98.39% < 1 mm`**) and `Adapt 1mm` `< 1 mm` from `62.13%` → **`79.31%`**, while `h512` plateaued at ~3× the runtime ([`report.md`](../../experiments/hybrid_surface_resnet/report.md)).
- **Shared GPU Speedup**: Adding GPU-resident tensor batching and once-per-epoch synchronization in [`src/models/_torch_trainer.py`](../../src/models/_torch_trainer.py) cut Full-Romania (`2.35M` pts) training time to `~4 min` (`h128`) and `~7 min` (`h256`) for all future PyTorch experiments.
