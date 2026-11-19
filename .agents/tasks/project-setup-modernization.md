# Task: Project Setup & Research Workflow Modernization

**Date**: 2026-11-19
**Branch**: `main`
**Status**: Completed
**Scope**: Documentation | Workflow | Refactoring

## Objective
Reorganize `ml_transdat` project setup to support progressive-discovery personas, decoupled reference-first research units, slug-only naming across `docs/research/`, `experiments/`, and `.agents/tasks/`, a chronological tasks master log, and multi-branch-aware skills.

## Key Activities & Implementation
- **Progressive-Discovery Personas (`.agents/personas/`, `GEMINI.md`)**:
  - Created [`ml-researcher.md`](../personas/ml-researcher.md), [`lit-researcher.md`](../personas/lit-researcher.md), and [`ml-architect.md`](../personas/ml-architect.md).
  - Updated [`GEMINI.md`](../../GEMINI.md) with a persona routing table, blending rule, Default/General fallback, and multi-branch workflow summary.
- **Research Knowledge Graph & Narrative (`docs/research/`)**:
  - Created [`docs/research/GEMINI.md`](../../docs/research/GEMINI.md) and [`docs/research/narrative.md`](../../docs/research/narrative.md).
  - Updated [`docs/research/roadmap.md`](../../docs/research/roadmap.md) with TL;DR state, Mermaid DAG, and annotated relationships table.
  - Renamed `01_`–`03_` research units to slug-only files (`spg_grid_analysis.md`, `pyproj_baseline.md`, `helmert2d_baseline.md`).
- **Tasks Master Log & Slug Migration (`.agents/tasks/`)**:
  - Created [`.agents/tasks/GEMINI.md`](GEMINI.md) and [`.agents/tasks/master_tasks.md`](master_tasks.md).
  - Renamed `001-`–`005-` task files to slug-only names.
- **Experiments & Cleanup (`experiments/`, `README.md`)**:
  - Renamed `experiments/01_*`–`03_*` directories to `spg_grid_analysis/`, `pyproj_baseline/`, and `helmert2d_baseline/`, updating internal paths, [`experiments/GEMINI.md`](../../experiments/GEMINI.md), and [`experiments/README.md`](../../experiments/README.md).
  - Removed obsolete `example_transform.py` and updated [`README.md`](../../README.md).
- **Skills Modernization (`.agents/skills/`)**:
  - Updated `task-init`, `task-record`, `task-commit`, and `code-audit`.
  - Replaced `research-roadmap` with [`research-sync`](../skills/research-sync/SKILL.md).

## Verification & Results
- `python -m unittest discover tests`: **8/8 tests passed** in 5.615s.
- Executed all renamed experiment units (`experiments.spg_grid_analysis.run_unit`, `experiments.pyproj_baseline.run_unit`, `experiments.helmert2d_baseline.run_unit`): all exited `0` with identical baseline scorecards.
- Automated markdown link & reference check across 34 files: zero stale `example_transform.py` or `PyTransDatRO v` references, zero `file:///` absolute links, and all active docs/research/experiment links resolved.
