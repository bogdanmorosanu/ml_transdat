# Python Code Style & Conventions

These conventions apply to all Python code in `ml_transdat`, including dataset generation, machine learning models, training scripts, evaluation benchmarks, and core utilities.

---

## 1. Naming Conventions
- **Modules & Scripts**: `snake_case` (e.g., `trans_ro.py`, `train_baseline.py`, `eval_metrics.py`).
- **Classes**: `PascalCase` (e.g., `TransRO`, `CoordinateDataset`, `MLPCoordinateModel`).
  - Internal helper classes use a single leading underscore (e.g., `_Ellipsoid`, `_ConfSph`).
  - Custom exceptions use `PascalCase` ending in `Err` or `Error` (e.g., `OutOfGridErr`).
- **Functions & Methods**: `snake_case` (e.g., `etrs89_to_st70`, `generate_grid_samples`, `compute_coordinate_loss`).
  - Protected or internal helper methods use a leading underscore (e.g., `_build_features`, `_normalize_coords`).
- **Variables**: Concise, domain-specific `snake_case`. Coordinate variables standardly use:
  - `lat` / `lon` / `h`: Latitude (rad or deg), Longitude (rad or deg), Ellipsoidal Height (meters).
  - `x` / `y` / `z` (or `n` / `e` / `z`): Northing (meters), Easting (meters), Orthometric Elevation (meters).
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `DEG_TO_RAD_FACTOR`, `KRASOVSKY_A`, `FALSE_NORTHING`).

---

## 2. Type Annotations & Data Structures
- **Universal Typing**: All function and method signatures across ML scripts, utilities, and models must include Python type annotations (e.g., `tuple[float, float]`, `Optional[float]`, `dict[str, Any]`, `np.ndarray`, `torch.Tensor`).
- **Coordinate Types**: Explicitly define or alias coordinate shapes (e.g., `Tuple2D = tuple[float, float]`, `Tuple3D = tuple[float, float, float]`) for readability and static analysis.
- **Tensors & Arrays**: Document expected tensor shapes in docstrings or inline comments (e.g., `coords: (N, 3) [lat_rad, lon_rad, h_m]`).

---

## 3. Docstrings & Documentation
- **Format**: Follow reStructuredText (reST) / Sphinx conventions:
  - Document `:param <name>:` and `:type <name>:`.
  - Document `:return:` and `:rtype:`.
  - Document `:raises <Error>:` for all explicitly raised exceptions.
- **Formulas & Theory**: Where coordinate projections or ML loss formulations are defined, include the mathematical formula or a reference to `docs/core_engine.md` or `docs/grid.md`.

---

## 4. Exception Handling & Logging
- **No Bare Pass**: Never use bare `except:` or `except Exception: pass`. Catch specific exceptions (e.g., `ValueError`, `KeyError`, `FileNotFoundError`) and provide informative logs or re-raise domain-specific errors.
- **Diagnostic Logging**: Use `logging.getLogger(__name__)` or structured printouts for training progress, metric summaries, and experiment diagnostics.

---

## 5. Library Boundaries & Invariants
- **Core Baseline (`pytransdat/`)**:
  - The ground-truth reference library remains pure Python (standard library only, with `numpy` strictly for grid deserialization).
  - Never modify files in `pytransdat/grids/`.
- **Machine Learning & Pipeline Code**:
  - Training scripts, data loaders, feature transforms, and evaluation code may freely use scientific and ML libraries (`numpy`, `scipy`, `pandas`, `scikit-learn`, `torch`, etc.) installed in `D:\anaconda3\envs\ml_transdat`.
- **Relative Paths**:
  - Always use workspace-relative paths in code, task records, and doc links. Never hardcode absolute file paths.
