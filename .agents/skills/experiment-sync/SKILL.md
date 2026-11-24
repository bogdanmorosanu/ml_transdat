---
name: experiment-sync
description: Audits, validates, and synchronizes an experiment unit in experiments/<slug>/ (code isolation in src/, shared datasets in data/, execution of run_unit.py, standardized unit_brief.json v2, human-readable report.md, and experiments/README.md). Use whenever checkpointing or finalizing an experiment.
---

# Experiment Sync Skill (`experiment-sync`)

## Purpose
Keeps every experiment in `experiments/<slug>/` modular, reproducible, and consistently documented—whether it is a modeling experiment (`grid_shift`, `transformation`) or an exploratory/feature-engineering study (`analysis`).

---

## Execution Workflow

### Step 1: Classify & Audit Boundaries (`experiments/GEMINI.md`)
1. Identify the experiment's `ExperimentType`:
   - `grid_shift`: `GridShiftExperimentUnit` (2D residual shift models).
   - `transformation`: `TransformationExperimentUnit` (full bidirectional CRS transformers).
   - `analysis`: `AnalysisExperimentUnit` (exploratory profiling, geometry, or feature engineering).
2. **Code Isolation Check**:
   - Inspect `experiments/<slug>/`. It must contain **only** `run_unit.py`, `report.md`, `unit_brief.json`, `results/`, `plots/`, and gitignored `models/` (plus optional local `data/`).
   - Confirm `run_unit.py` subclasses `GridShiftExperimentUnit`, `TransformationExperimentUnit`, or `AnalysisExperimentUnit` (never a standalone `main()` script), and `after_run()` never re-runs `.predict()` on datasets already in `results`.
   - Any reusable model class, feature extractor, sampler, metric, or plotting helper must be extracted to `src/` and imported by `run_unit.py`.
3. **Shared Data Externalization Check**:
   - If the experiment produces datasets meant to be consumed by `src/` or other experiments (e.g., `cell_shape_metrics.parquet`), verify they are written to `data/features/` (or `data/grids/`), never read from `experiments/<slug>/results/` by `src/`.

### Step 2: Execute & Validate Machine Output (`unit_brief.json` v2)
Run the unit from the workspace root:
```powershell
D:\anaconda3\envs\ml_transdat\python.exe -m experiments.<slug>.run_unit
```
Verify:
- `unit_brief.json` exists and conforms to **Schema v2** (`schema_version == "2.0"`, `slug`, `experiment_type`, `status`, `runtime_s`, `key_metrics`, `artifacts`, `benchmark_candidates`, `extra`), including the surface learnability fields (`crmse_2d_mm`, `iqr_2d_mm`, `roughness_grad_rms_mm_km`, `local_jitter_ratio`, `learnability_score_mm`).
- `results/` and `plots/` contain the expected scorecards, **unified multi-model row diagnostic figures** (`{dataset_id}_{direction}_{plot_type}.png`, 1 row per model—no per-model diagnostic PNGs), and—for `grid_shift` units with `keep_in_memory=True` models—the canonical 9-cell extreme scenario plots (`extreme_scenarios_gradient_and_linear_9cells.png` and `extreme_scenarios_curvature_and_kinks_9cells.png`).
- `models/*.pkl` checkpoints exist for trained models (`persist_model=True`) and are ignored by Git.

### Step 3: Synchronize Human Report (`report.md`)
- Ensure `experiments/<slug>/report.md` exists and follows the category baseline template from `experiments/GEMINI.md`:
  - **Objective & Hypothesis**
  - **Experimental Setup / Methodology**
  - **Quantitative Results / Analysis**
  - **Visual Diagnostics** (embedded `plots/*.png`)
  - **Takeaways & Verdict**
  - **Benchmark Candidates** *(required for `grid_shift` / `transformation` units with promoted variants; must match `benchmark_candidates` in `unit_brief.json` and `ExperimentModelSpec` in `run_unit.py`)*
- **Preserve & Expand Particularities**: Never strip experiment-specific sections (e.g., mathematical derivations, failure-mode deep dives, Nyquist/smoothing analyses); expand `report.md` to capture any new findings from the conversation.
- Ensure no stale `README.md` or `README_AI.md` files remain inside `experiments/<slug>/`.

### Step 4: Synchronize Central Registries
- Update `experiments/README.md`:
  - Add or update the row in **Active Research Units Index** (including `Type` and `Key Metrics / Verdict`).
  - Ensure `python -m experiments.<slug>.run_unit` is listed in **How to Run an Experiment**.
- If any `src/` modules or `data/` layouts changed, update `docs/architecture.md`.
- Remind the user if `research-sync` (for `docs/research/`) or `benchmark-sync` (for `benchmarks/`) should be run next.
