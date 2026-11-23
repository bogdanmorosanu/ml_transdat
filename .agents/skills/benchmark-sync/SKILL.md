---
name: benchmark-sync
description: Promotes validated champion model variants from a model-bearing experiment (grid_shift or transformation) into a cross-model benchmark suite in benchmarks/<suite>/ without hyperparameter drift, re-runs the suite, and updates benchmarks/README.md.
---

# Benchmark Sync Skill (`benchmark-sync`)

## Purpose
Integrates or updates validated champion models from an experiment into `benchmarks/<suite>/` using `specs_from_unit(...)` so benchmark evaluations always match the exact methodology and hyperparameters validated in the source experiment.

---

## Execution Workflow

### Step 1: Verify Eligibility & Read Provenance
1. Read `experiments/<slug>/unit_brief.json` and `experiments/<slug>/run_unit.py`.
2. **Eligibility Gate**:
   - Confirm `experiment_type` is `grid_shift` or `transformation`.
   - If `experiment_type == "analysis"`, **stop**—exploratory/feature units are not benchmark model suites.
3. Inspect `benchmark_candidates` in `unit_brief.json` and the **Benchmark Candidates** section of `experiments/<slug>/report.md`.
   - If no `ExperimentModelSpec` has `benchmark_candidate=True` yet, help the user select the champion variant(s), set `benchmark_candidate=True`, `benchmark_suite="<suite>"`, and `benchmark_label="<Display Label>"` in `run_unit.py`, and re-sync the experiment's `report.md` / `unit_brief.json`.

### Step 2: Select Target Benchmark Suite
Map the candidate(s) using the splitting criteria in `benchmarks/README.md`:
- `global_analytical_models`: Closed-form global equations with $0\,\text{MB}$ grid lookup (decimeter tier).
- `grid_shift_models`: Local spatial partitioning, neural fields, and RBF/TPS models (millimeter-to-centimeter tier).
- Or propose a new `BenchmarkSuite` under `benchmarks/<new_suite>/` if a new modeling paradigm or precision tier is introduced.

### Step 3: Wire Drift-Free Promotion & Execute
1. In `benchmarks/<suite>/run_benchmark.py` (which subclasses `src.core.BenchmarkSuite`), pull candidates directly via `specs_from_unit(<UnitClass>, suite="<suite>", unit_dir=Path("experiments/<slug>"))`.
   - **Never copy-paste hyperparameters manually** into `run_benchmark.py`.
   - `specs_from_unit(..., unit_dir=...)` automatically links each candidate to `experiments/<slug>/models/<spec_name>.pkl` and verifies its code+config SHA-256 fingerprint, reusing synced checkpoints or retraining automatically if out of sync.
2. Run the benchmark suite from workspace root:
   ```powershell
   D:\anaconda3\envs\ml_transdat\python.exe -m benchmarks.<suite>.run_benchmark
   ```
3. Verify all 3 standardized views were updated:
   - `benchmarks/<suite>/summary_scorecard.md`
   - `benchmarks/<suite>/plots/` (`01_precision_comparison.png`, `02_performance_speed.png`, `03_memory_breakdown_stacked.png`)
   - `benchmarks/<suite>/results/benchmark_summary.json` (confirm `source_experiment` provenance is recorded).

### Step 4: Synchronize `benchmarks/README.md`
- Update the **Active Benchmark Suites** table in `benchmarks/README.md` with any newly added/updated models and target accuracy ranges.
- Present a concise comparison summary of how the promoted model(s) performed relative to the existing suite.
