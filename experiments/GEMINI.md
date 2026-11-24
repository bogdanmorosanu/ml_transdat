# Experiments Workflow & Architectural Rules

Rules for any agent creating, modifying, or running experiments in `experiments/`.

---

## 1. Reusable Code & Data Isolation (`src/` & `data/`)
- **Never duplicate reusable logic inside `experiments/<slug>/`.**
- Any reusable model architecture, fitting algorithm, mathematical helper, sampler, metric, or visualizer must live in `src/` (see [`docs/architecture.md`](../docs/architecture.md)) and be imported by `run_unit.py`.
- Any reusable dataset produced by an experiment for downstream models or `src/` modules (e.g., `cell_shape_metrics.parquet`) **must** be exported to `data/features/` (or `data/grids/`)—`src/` must never read from `experiments/`.
- Experiment folders should contain only orchestration (`run_unit.py`), the human report (`report.md`), machine brief (`unit_brief.json`), outputs (`results/`, `plots/`), gitignored model checkpoints (`models/*.pkl`), and optional unit-local `data/`.
- **Unified Multi-Model Row Plot Rule**: Diagnostic plots (`PlotGroup`) are always generated via `PlotDispatcher.render_grouped` as **one unified PNG per `(dataset_id, direction, plot_type)`** (`{dataset_id}_{direction}_{plot_type}.png`) with **1 row per model**—never one diagnostic PNG file per model.
- **Standardized 9-Cell Extreme Scenario Transects**: `GridShiftExperimentUnit` automatically renders `plots/extreme_scenarios_gradient_and_linear_9cells.png` and `plots/extreme_scenarios_curvature_and_kinks_9cells.png` (using the canonical 7 extreme scenarios over 99 km windows) for models with `keep_in_memory=True` (override `get_extreme_scenario_models(fitted_models)` to customize labels or reference models). Do not use the legacy `plot_transect_66km`.
- **Stage-2 ML Surface Learnability Metrics & Visuals**: Every evaluation automatically computes `cRMSE`, `IQR`, `Roughness Grad RMS/P95 (mm/km)`, `Local Jitter Ratio`, and `Learnability Score (mm)` in scorecards and `unit_brief.json`. Use `MetricGroup.SURFACE_LEARNABILITY`, `PlotGroup.SURFACE_DIAGNOSTIC` (`RESIDUAL_ROUGHNESS`), `ModelEvaluator.rank_surface_candidates(results)`, and `plot_surface_learnability_ranking(...)` when evaluating Stage-1 base surfaces.
- **Fingerprint-Synced Model Persistence**: Trained models default to `persist_model=True` and are saved to gitignored `experiments/<slug>/models/<spec_name>.pkl` with a SHA-256 fingerprint of model code, hyperparameters, and `train_grid`. Both `run_unit.py` and `BenchmarkSuite` automatically reuse valid checkpoints or retrain if code/parameters change.
- **Zero Re-Inference in `after_run()`**: Never call `model.predict()` in `after_run()` on datasets already present in `results`; reuse `results` via `ModelEvaluator.tabulate_delta_vs_baseline(...)` or `plot_multi_dataset_comparison(...)`.

---

## 2. Three-Category Experiment Taxonomy
Every unit in `experiments/<slug>/` belongs to one of three canonical `ExperimentType`s:

| Category (`ExperimentType`) | Base Class (`src/core/base_unit.py`) | Purpose | Benchmark-Eligible? |
| :--- | :--- | :--- | :---: |
| `grid_shift` | `GridShiftExperimentUnit` | Trains/evaluates `BaseGridShiftModel` variants (`(N, E) -> (dN, dE)`) | Yes |
| `transformation` | `TransformationExperimentUnit` | Evaluates bidirectional `BaseCoordinateTransformer` (`ETRS89 <-> Stereo70`) | Yes |
| `analysis` | `AnalysisExperimentUnit` | Exploratory profiling, geometric analysis, or feature engineering | No |

---

## 3. Standard Experiment Blueprint
When creating a new experiment:
1. **Slug-Only Directory**: Use a concise, descriptive `snake_case` slug with **no numeric prefix** (e.g., `experiments/fourier_mlp_residuals/`) so parallel git branches never collide.
2. **Single Entry Point (`run_unit.py`)**: Subclass `GridShiftExperimentUnit`, `TransformationExperimentUnit`, or `AnalysisExperimentUnit` implementing `slug` (no numeric `unit_id`).
3. **Required Outputs**:
   - `unit_brief.json`: Standardized **Schema v2** JSON (`schema_version`, `slug`, `experiment_type`, `status`, `runtime_s`, `key_metrics`, `artifacts`, `benchmark_candidates`, `extra`), written automatically by `run_unit.py`.
   - `report.md`: Human-facing narrative following the baseline structure:
     1. Objective & Hypothesis
     2. Experimental Setup / Methodology
     3. Quantitative Results / Analysis
     4. Visual Diagnostics (embedded `plots/*.png`)
     5. Takeaways & Verdict
     6. **Benchmark Candidates** *(for `grid_shift` / `transformation` units: exact promoted variants, training grids, and hyperparameters matching `ExperimentModelSpec(..., benchmark_candidate=True)`)*
     - *Expand freely* with extra sections for experiment-specific derivations, failure analyses, or particularities.
   - `results/` and `plots/`: Saved CSV/JSON/Markdown scorecards and diagnostic figures.
   - Do **not** create duplicate `README.md` or `README_AI.md` files inside `experiments/<slug>/`.

---

## 4. Drift-Free Benchmark Promotion & Sync Skills
- Mark champion model variants directly in `run_unit.py` via `ExperimentModelSpec(..., benchmark_candidate=True, benchmark_suite="<suite>", benchmark_label="<label>")` so `benchmarks/<suite>/run_benchmark.py` pulls exact validated configs via `specs_from_unit(...)`.
- Use **`experiment-sync`** (`.agents/skills/experiment-sync/SKILL.md`) whenever checkpointing or finalizing an experiment.
- Use **`benchmark-sync`** (`.agents/skills/benchmark-sync/SKILL.md`) whenever promoting or updating validated champions in `benchmarks/`.
- Use **`research-sync`** (`.agents/skills/research-sync/SKILL.md`) to update `docs/research/` when scientific conclusions change.

---

## 5. Execution & Portability
- Always run units from the workspace root using the dedicated conda environment:
  ```powershell
  D:\anaconda3\envs\ml_transdat\python.exe -m experiments.<slug>.run_unit
  ```
- Use workspace-relative paths only; never hardcode absolute machine-specific paths.
