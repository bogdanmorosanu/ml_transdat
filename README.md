# ML TransDat (ml_transdat)

A Machine Learning research and experimentation project aimed at modeling and replicating the Romanian official geodetic coordinate transformation algorithm between:
- **ETRS89** (Geographic 3D/2D: Latitude, Longitude, Ellipsoidal Height)
- **Stereo70** (Projected: Northing X, Easting Y, Normal/Orthometric Elevation Z)

---

## 1. Project Overview

The official Romanian coordinate transformation relies on two tightly coupled mathematical components:
1. **Mathematical Projection**: Oblique Stereographic 1970 projection on the Krasovsky 1940 reference ellipsoid (along with datum shift transformations).
2. **Grid Interpolation (Distortion Adjustment)**: Local shift corrections applied using the official Romanian distortion grid (`.spg`), interpolating local shifts ($\Delta N, \Delta E, \Delta Z$) across the territory.

### Objective
Through empirical trial-and-error, explore, design, train, and evaluate machine learning architectures (such as deep neural networks, residual networks, and polynomial-hybrid models) to approximate this transformation in both:
- **Direct direction**: $ETRS89 \to Stereo70$
- **Inverse direction**: $Stereo70 \to ETRS89$

The models will be benchmarked on accuracy (with **$\le 1\text{ cm}$** as an ideal aspirational goal across Northing, Easting, and Elevation), computational efficiency, and explainability against the analytical ground-truth baseline.

---

## 2. Ground Truth Benchmark (`pytransdat/`)

The pure-Python reference implementation in `pytransdat/` (incorporating `pytransdat/grids/rom_grid3d_25.09.spg`) serves as the immutable ground-truth baseline for:
- Generating synthetic spatial training and evaluation datasets across Romania.
- Benchmarking the numerical accuracy and latency of candidate ML models.

To run a quick sanity check of the transformation engine:
```bash
python example_transform.py
```

---

## 3. Living Documentation

Detailed mathematical background and grid specifications are maintained under [`docs/`](docs/):
- [`docs/core_engine.md`](docs/core_engine.md): Formulas for the Krasovsky 1940 ellipsoid, conformal sphere mapping, Oblique Stereographic projection, and 3D height conversions.
- [`docs/grid.md`](docs/grid.md): Specification of the `.spg` binary shift grid and bilinear interpolation boundaries.
- [`docs/research/roadmap.md`](docs/research/roadmap.md): Living research map tracking branches, hypotheses, rationales, and discoveries.

---

## 4. Development Workflow & Agent Tracking

All experiments, trial-and-error sessions, architectural choices, and performance comparisons are tracked chronologically under [`.agents/tasks/`](.agents/tasks/).

### Environment
The project uses a dedicated conda environment:
- Path: `D:\anaconda3\envs\ml_transdat`
- Python Executable: `D:\anaconda3\envs\ml_transdat\python.exe`
