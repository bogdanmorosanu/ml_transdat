# Experiments Workflow & Architectural Rules

Rules for any agent creating, modifying, or running experiments in `experiments/`.

---

## 1. Reusable Code Isolation (`src/`)
- **Never duplicate reusable logic inside `experiments/<slug>/`.**
- Any reusable model architecture, fitting algorithm, mathematical helper, sampler, metric, or visualizer must live in `src/` (see [`docs/architecture.md`](../docs/architecture.md)) and be imported by `run_unit.py`.
- Experiment folders should contain only orchestration (`run_unit.py`), generated reports (`report.md`, `README_AI.md`, `unit_brief.json`), and outputs (`results/`, `plots/`, optional `data/`).

---

## 2. Standard Research Unit Blueprint
When creating a new experiment:
1. **Slug-Only Directory**: Use a concise, descriptive `snake_case` slug with **no numeric prefix** (e.g., `experiments/fourier_mlp_residuals/`) so parallel git branches never collide.
2. **Subclass `GridShiftExperimentUnit` or `TransformationExperimentUnit`**: Implement `run_unit.py` subclassing `src.core.base_unit.GridShiftExperimentUnit` (for 2D residual models) or `TransformationExperimentUnit` (for full CRS transformers), using `src.data.GridId`, `src.evaluation`, and `src.viz` (`PlotGroup` / `PlotType`).
3. **Required Outputs**:
   - `unit_brief.json`: Machine-readable metrics and metadata payload.
   - `report.md`: Human-facing narrative, tables, and embedded figures.
   - `README_AI.md`: High-density agent briefing.
   - `results/` and `plots/`: Saved JSON/CSV/Markdown scorecards and diagnostic figures.

---

## 3. Automatic Registry Sync
Whenever a new experiment is finalized or merged to `main`, update:
- `experiments/README.md`: Add/update the row in **Active Research Units Index** and the command in **How to Run an Experiment**.
- If the experiment changes scientific conclusions or spawns/resolves a hypothesis, run `research-sync` to update `docs/research/`.

---

## 4. Execution & Portability
- Always run units from the workspace root using the dedicated conda environment:
  ```powershell
  D:\anaconda3\envs\ml_transdat\python.exe -m experiments.<slug>.run_unit
  ```
- Use workspace-relative paths only; never hardcode absolute machine-specific paths.
