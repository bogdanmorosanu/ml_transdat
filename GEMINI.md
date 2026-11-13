# ML TransDat Project Invariants & Agent Workflow

## Project Objective
Develop, experiment with, and evaluate Machine Learning (ML) models capable of replicating the coordinate transformation algorithm between:
- **ETRS89** (Geographic 3D / 2D coordinates: Latitude, Longitude, Ellipsoidal Height)
- **Stereo70** (Projected coordinates: Northing X, Easting Y, Normal/Orthometric Height Z)

The transformation consists of two primary steps:
1. **Mathematical projection**: Oblique Stereographic 1970 projection on Krasovsky 1940 ellipsoid (and datum transformation).
2. **Grid interpolation / distortion adjustment**: Local shift corrections using the official Romanian distortion grid (`.spg`).

Both direct ($ETRS89 \to Stereo70$) and inverse ($Stereo70 \to ETRS89$) directions will be studied. Multiple ML model architectures, feature formulations, and training strategies will be explored through trial and error.

---

## Core Architecture & Reference Invariants
- **Ground Truth Benchmark (`pytransdat/`)**:
  - The `pytransdat` library and its grid file (`pytransdat/grids/rom_grid3d_25.09.spg`) are the ground truth baseline.
  - Used to generate synthetic training/evaluation datasets and to benchmark ML model accuracy and inference speed.
  - **Do NOT modify** the content or loading mechanism of the binary grid files in `pytransdat/grids/`.
- **Python Environment & Package Management**:
  - Always use the dedicated conda environment: `D:\anaconda3\envs\ml_transdat` (Python executable: `D:\anaconda3\envs\ml_transdat\python.exe`).
  - **Strict Package Installation Invariant**: The agent must **NEVER** install, update, or remove Python packages (e.g., via `pip install`, `conda install`, etc.). All package installations and environment changes must be performed by the user. If an experiment or script requires a library that is not currently installed, the agent must inform the user and request that the user install it.
- **Machine Learning & Scientific Libraries**:
  - Machine learning code, dataset generation, and evaluation pipelines may utilize scientific libraries (`numpy`, `scipy`, `pandas`, `scikit-learn`, `torch`, etc.) installed in the `ml_transdat` environment.
- **Relative File Paths & Portability**:
  - Always use workspace-relative paths in code, task documentation, and markdown links (e.g., `docs/core_engine.md`, `pytransdat/trans_ro.py`).
  - Never hardcode absolute file paths or machine-specific URLs into committed project files.

---

## Living Documentation (`docs/`)
The following living documentation files are maintained in the repository:
- [`docs/core_engine.md`](docs/core_engine.md): Mathematical formulations, projection equations, coordinate systems, and precision standards.
- [`docs/grid.md`](docs/grid.md): Specification of the `.spg` binary grid file, bilineary interpolation mechanics, and distortion field coverage.
- [`docs/research/roadmap.md`](docs/research/roadmap.md): Living research map tracking active branches, hypotheses, rationales, and discoveries.

---

## Development Workflow & Task Discipline
- **Task Records**: Document all research, architecture decisions, and trial-and-error experiment results under `.agents/tasks/`.
- **Code Quality**: Keep code well-typed, modular, and with clear docstrings explaining data shapes and mathematical/ML assumptions.
