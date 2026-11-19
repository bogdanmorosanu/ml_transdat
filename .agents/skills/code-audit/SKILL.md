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
Run Ruff lint and format checks from the designated environment:
```powershell
D:\anaconda3\envs\ml_transdat\python.exe -m ruff check .
D:\anaconda3\envs\ml_transdat\python.exe -m ruff format --check .
```
*(If Ruff is not installed in the environment, skip external linting and inspect manually).*

### 2. Architectural Boundary Check (`experiments/GEMINI.md`)
- Verify that `experiments/<slug>/` contains **only** orchestration (`run_unit.py`), reports, and outputs—no reusable model classes, fitting algorithms, samplers, metrics, or visualizers duplicated outside `src/`.
- Verify that all Python code follows [`.agents/rules/python-style.md`](../../rules/python-style.md) (type annotations, reST docstrings, no bare `except:`).

### 3. File Path & Invariant Safeguards
- Verify zero machine-specific absolute file paths in committed scripts/docs (all links and paths must be workspace-relative).
- Verify ground-truth grid files in `pytransdat/grids/` have not been modified.
