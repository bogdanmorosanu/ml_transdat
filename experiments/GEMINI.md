# Experiments Workflow & Architectural Rules

Rules for any agent creating, modifying, or running experiments in `experiments/`.

---

## 1. Reusable Code Isolation (`src/`)
- **Never duplicate reusable logic inside `experiments/<XX_slug>/`.**
- Any reusable model architecture, fitting algorithm, mathematical helper, sampler, metric, or visualizer must live in `src/` (e.g., `src/models/`, `src/core/`) and be imported by `run_unit.py`.
- Experiment folders should contain only orchestration (`run_unit.py`), generated reports (`report.md`, `README_AI.md`, `unit_brief.json`), and outputs (`results/`, `plots/`, optional `data/`).

---

## 2. Standard `ResearchUnit` Blueprint
When creating a new experiment:
1. **Sequential Prefix**: Assign the next unused 2-digit number (e.g., `04_<slug>/`).
2. **Subclass `ResearchUnit`**: Implement `run_unit.py` subclassing `src.core.base_unit.ResearchUnit`, using `src.core.metrics` and `src.core.viz`.
3. **Required Outputs**:
   - `unit_brief.json`: Machine-readable metrics and metadata payload.
   - `report.md`: Human-facing narrative, tables, and embedded figures.
   - `README_AI.md`: High-density agent briefing.
   - `results/` and `plots/`: Saved JSON scorecards and diagnostic figures.

---

## 3. Automatic Registry & Roadmap Sync
Whenever a new experiment is created or its results/status change, **automatically update**:
- `experiments/README.md`: Add/update the row in **Active Research Units Index** and the command in **How to Run an Experiment**.
- `docs/research/roadmap.md`: Update the Research Topology DAG and Research Branches Index, and create/update `docs/research/<XX_slug>.md`.

---

## 4. Execution & Portability
- Always run units from the workspace root using the dedicated conda environment:
  ```powershell
  D:\anaconda3\envs\ml_transdat\python.exe -m experiments.<XX_slug>.run_unit
  ```
- Use workspace-relative paths only; never hardcode absolute machine-specific paths.
