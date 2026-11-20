# ML TransDat Project Invariants & Agent Workflow

## Project Objective
Develop, experiment with, and evaluate Machine Learning (ML) models capable of replicating the coordinate transformation algorithm between:
- **ETRS89** (Geographic 3D / 2D coordinates: Latitude, Longitude, Ellipsoidal Height)
- **Stereo70** (Projected coordinates: Northing X, Easting Y, Normal/Orthometric Elevation Z)

The transformation consists of two primary steps:
1. **Mathematical projection**: Oblique Stereographic 1970 projection on Krasovsky 1940 ellipsoid (and datum transformation).
2. **Grid interpolation / distortion adjustment**: Local shift corrections using the official Romanian distortion grid (`.spg`).

Both direct ($ETRS89 \to Stereo70$) and inverse ($Stereo70 \to ETRS89$) directions are studied.

---

## Agent Personas & Progressive Discovery
To keep context lean while providing deep domain expertise, select and read only the relevant persona file(s) from `.agents/personas/` based on user intent:

| Intent / Prompt Type | Persona File to Read |
| :--- | :--- |
| ML concepts, model families, features, loss, error diagnosis | [`.agents/personas/ml-researcher.md`](.agents/personas/ml-researcher.md) |
| Prior art, papers, geodetic ML benchmarks, external inspiration | [`.agents/personas/lit-researcher.md`](.agents/personas/lit-researcher.md) |
| Code design, refactoring, `src/` architecture, unified benchmarking | [`.agents/personas/ml-architect.md`](.agents/personas/ml-architect.md) |
| Admin, git, `task-init`, `task-record`, `task-commit`, quick lookups | *(Default / General mode — no persona file loaded)* |

- **Blending Rule**: When a prompt spans multiple domains (e.g., ML formulation + code structure), read all relevant persona files and synthesize a single cohesive response.

---

## Core Architecture & Reference Invariants
- **Ground Truth Benchmark (`pytransdat/`)**:
  - The `pytransdat` library and its grid file (`pytransdat/grids/rom_grid3d_25.09.spg`) are the immutable ground-truth baseline.
  - **Do NOT modify** the content or loading mechanism of the binary grid files in `pytransdat/grids/`.
- **Python Environment & Package Management**:
  - Always use the dedicated conda environment: `D:\anaconda3\envs\ml_transdat` (Python executable: `D:\anaconda3\envs\ml_transdat\python.exe`).
  - **Strict Package Installation Invariant**: The agent must **NEVER** install, update, or remove Python packages (e.g., via `pip install`, `conda install`, etc.). Inform the user if a required library is missing.
- **Machine Learning & Scientific Libraries**:
  - ML code, dataset generation, and evaluation pipelines may use libraries installed in `ml_transdat` (`numpy`, `scipy`, `pandas`, `scikit-learn`, `torch`, etc.).
- **Relative File Paths & Portability**:
  - Always use workspace-relative paths in code, task documentation, and markdown links (e.g., `docs/core_engine.md`, `pytransdat/trans_ro.py`).
  - Never hardcode absolute file paths or machine-specific URLs into committed project files.

---

## Workspace Layout & Living Documentation
- [`docs/architecture.md`](docs/architecture.md): Canonical code architecture, `src/` module layering, model/data/eval/viz contracts, and extension checklist.
- [`docs/core_engine.md`](docs/core_engine.md): Mathematical formulations, `TransRO` API usage, projection equations, and precision standards.
- [`docs/grid.md`](docs/grid.md): `.spg` binary grid specification, interpolation mechanics, and distortion coverage.
- [`docs/research/GEMINI.md`](docs/research/GEMINI.md), [`docs/research/roadmap.md`](docs/research/roadmap.md), [`docs/research/narrative.md`](docs/research/narrative.md): Decoupled, reference-first research units, knowledge graph, and causal scientific evolution.
- [`experiments/GEMINI.md`](experiments/GEMINI.md), [`experiments/README.md`](experiments/README.md): Slug-based, self-contained experiment units (`experiments/<slug>/`) with reusable code isolated in [`src/`](src/).
- [`literature/README.md`](literature/README.md): Curated geodetic ML literature survey, verified references, and search log.
- [`data/`](data/): Canonical spatial grids (`data/grids/`) and trajectory benchmarks (`data/benchmarks/`).
- [`.agents/tasks/GEMINI.md`](.agents/tasks/GEMINI.md), [`.agents/tasks/master_tasks.md`](.agents/tasks/master_tasks.md): Chronological engineering/execution task log.

---

## Multi-Branch Workflow & Task Discipline
- **Branching**:
  - `main`: Validated baselines, shared `src/` and `data/`, and synced central indexes (`roadmap.md`, `narrative.md`, `experiments/README.md`, `master_tasks.md`).
  - `exp/<slug>`: Isolated branch for candidate experiments or mid-conversation baseline pivots.
  - `infra/<slug>`: Shared `src/`, `data/`, or tooling changes.
- **Task vs. Research Separation**:
  - Engineering/execution outcomes belong in `.agents/tasks/` (`task-record` skill).
  - Scientific hypotheses, rationale, and verdicts belong in `docs/research/` (`research-sync` skill).
- **Code Quality**: Keep code well-typed, modular, and compliant with [`.agents/rules/python-style.md`](.agents/rules/python-style.md).
