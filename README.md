# ML TransDat (`ml_transdat`)

A Machine Learning research and experimentation project aimed at modeling and replicating the Romanian official geodetic coordinate transformation algorithm between:
- **ETRS89** (Geographic 3D/2D: Latitude, Longitude, Ellipsoidal Height)
- **Stereo70** (Projected: Northing X, Easting Y, Normal/Orthometric Elevation Z)

---

## 1. Project Overview

The official Romanian coordinate transformation relies on two tightly coupled mathematical components:
1. **Mathematical Projection**: Oblique Stereographic 1970 projection on the Krasovsky 1940 reference ellipsoid (along with datum shift transformations).
2. **Grid Interpolation (Distortion Adjustment)**: Local shift corrections applied using the official Romanian distortion grid (`.spg`), interpolating local shifts ($\Delta N, \Delta E, \Delta Z$) across the territory.

### Objective
Explore, design, train, and evaluate machine learning architectures (e.g., residual neural networks, Fourier feature MLPs, RBF/kernel models, tree ensembles) to approximate this transformation in both:
- **Direct direction**: $ETRS89 \to Stereo70$
- **Inverse direction**: $Stereo70 \to ETRS89$

Models are benchmarked against `pytransdat` on accuracy (**$\le 1\text{ cm}$** aspirational goal), inference latency, memory footprint, and parameter/model size.

---

## 2. Repository Structure

- [`pytransdat/`](pytransdat/): Immutable pure-Python reference implementation incorporating `pytransdat/grids/rom_grid3d_25.09.spg` (see [`docs/core_engine.md`](docs/core_engine.md) for API usage examples).
- [`src/`](src/): Reusable ML & evaluation pipeline (`src/core/` for samplers, metrics, visualization; `src/models/` for candidate model implementations).
- [`experiments/`](experiments/): Self-contained slug-based research units (`experiments/<slug>/`), indexed in [`experiments/README.md`](experiments/README.md).
- [`data/`](data/): Canonical spatial grids (`data/grids/spg_nodes_valid.parquet`, `ro_grid_4x4.parquet`, `ro_grid_8x8.parquet`) and directional trajectory benchmarks (`data/benchmarks/ro_corridor_snw.parquet`).
- [`docs/`](docs/): Living mathematical documentation ([`core_engine.md`](docs/core_engine.md), [`grid.md`](docs/grid.md)) and scientific research graph ([`docs/research/roadmap.md`](docs/research/roadmap.md), [`docs/research/narrative.md`](docs/research/narrative.md)).
- [`literature/`](literature/): Geodetic ML literature survey, verified references, and search log.
- [`tests/`](tests/): Unit and benchmark invariant tests (`python -m unittest discover tests`).

---

## 3. Environment & Execution

The project uses a dedicated conda environment:
- Path: `D:\anaconda3\envs\ml_transdat`
- Python Executable: `D:\anaconda3\envs\ml_transdat\python.exe`

Run tests or any experiment unit from the workspace root:
```powershell
D:\anaconda3\envs\ml_transdat\python.exe -m unittest discover tests
D:\anaconda3\envs\ml_transdat\python.exe -m experiments.<slug>.run_unit
```
