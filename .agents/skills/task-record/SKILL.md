---
name: task-record
description: Unified documentation orchestrator that records session outcomes, ML experiments, and architectural decisions in .agents/tasks/, synchronizes living documentation in docs/, and verifies workspace cleanliness. Use whenever concluding or checkpointing a task.
---

# Unified Task Record Skill (`task-record`)

## Purpose
This skill serves as the **documentation orchestrator** for the `ml_transdat` repository. It records chronological session logs in `.agents/tasks/`, captures ML experiment results (architectures, loss/accuracy metrics, inference latencies), and keeps authoritative technical documentation in `docs/` synchronized with codebase changes.

---

## Operating Principles

1. **Single Entrypoint for Task Wrap-up**:
   - Executed at the conclusion of every development session, experiment run, or significant milestone.

2. **Outcome-Oriented Technical Truth**:
   - Documents tangible outcomes: data generated, model architectures tested, performance against `pytransdat` ground-truth benchmark, and clear next steps.

3. **No Web App / API Artifacts**:
   - Focuses on ML models, training pipelines, evaluation metrics, and geodetic algorithms.

---

## Execution Workflow

### Step 1: Discover Session Changes & Artifacts
Inspect modified files, generated data files, model checkpoints, and task history:
- `git status -s`
- Check any generated scripts, notebooks, or evaluation artifacts.

### Step 2: Determine Next Sequential Task File
- Inspect `.agents/tasks/`.
- Identify the highest numbered task file: `YYYY-MM-DD-task-N-<slug>.md`.
- Increment `N` or create a new entry for the current session.

### Step 3: Write Task Record
Create the task record file in `.agents/tasks/` using the standard structure:
```markdown
# Task: <Title>

**Date**: YYYY-MM-DD
**Status**: Completed / In Progress
**Scope**: <Data Gen | Model Architecture | Training | Evaluation | Documentation>

## Objective
<Concise summary of the task goals>

## Key Activities & Implementation
- <Activity 1>
- <Activity 2>

## ML Experiment / Evaluation Results (if applicable)
- **Model Architecture**: <e.g., MLP, Residual Network, Polynomial + Neural Hybrid>
- **Transformation Direction**: <ETRS89 -> Stereo70 or Stereo70 -> ETRS89>
- **Dataset Size / Distribution**: <Points count, bounding box coverage>
- **Accuracy vs. Baseline**:
  - Max Error: <ΔX, ΔY, ΔZ in meters or millimeters>
  - Mean Absolute Error (MAE): <...>
- **Inference Speed**: <latency per point or batch throughput>

## Next Steps
- <Immediate next experiment or action item>
```

### Step 4: Synchronize Living Documentation (`docs/`)
If the task modified core assumptions, formulas, or added permanent utilities:
- Update `docs/core_engine.md` or `docs/grid.md` if relevant.
- Ensure all markdown links remain relative and valid.

### Step 5: Final Summary
Present a concise summary to the user with a clickable link to the new task file.
