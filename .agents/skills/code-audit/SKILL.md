---
name: code-audit
description: Audits code quality, typing, formatting, and src/ vs. experiments/ architectural boundaries across Python scripts, utilities, and ML pipelines. Use before finishing significant development sessions.
---

# Code Audit Skill (`code-audit`)

## Purpose
Executes static analysis, formatting checks, architectural boundary checks, and code hygiene audits across `ml_transdat`.

---

## Audit Workflow

### 1. Python Code Hygiene & Ruff Lint
Run Ruff lint and automated architecture invariant tests from the designated environment:
```powershell
D:\anaconda3\envs\ml_transdat\python.exe -m ruff check src benchmarks experiments tests scripts
D:\anaconda3\envs\ml_transdat\python.exe -m unittest tests/test_architecture_invariants.py -v
```
*(Config is pinned in `pyproject.toml`.)*

### 2. Architectural Boundary Check (`experiments/GEMINI.md`)
- Verify that `experiments/<slug>/` contains **only** orchestration (`run_unit.py`), reports, outputs (`results/`, `plots/`), and gitignored `models/`—no reusable model classes, fitting algorithms, samplers, metrics, or visualizers duplicated outside `src/`.
- Verify every `experiments/<slug>/run_unit.py` subclasses `GridShiftExperimentUnit`, `TransformationExperimentUnit`, or `AnalysisExperimentUnit`, and `after_run()` never re-runs `.predict()` on datasets already in `results`.
- Verify `plots/` uses unified 1-row-per-model figures (`{dataset_id}_{direction}_{plot_type}.png`) with zero legacy per-model diagnostic PNGs.
- Verify that all Python code follows [`.agents/rules/python-style.md`](../../rules/python-style.md) (type annotations, reST docstrings, no bare `except:`).

### 3. File Path & Invariant Safeguards
- Verify zero machine-specific absolute file paths in committed scripts/docs (all links and paths must be workspace-relative).
- Verify ground-truth grid files in `pytransdat/grids/` have not been modified.
