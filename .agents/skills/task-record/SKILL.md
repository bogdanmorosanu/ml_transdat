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

2. **Outcome-Oriented Technical Truth & Scoped Context**:
   - Documents tangible, completed outcomes: actions taken, code/data generated, architectures tested, performance metrics, and decisions made.
   - Accurately captures only the unrecorded work from the current conversation context since the previous task record.
   - Does **not** hypothesize or prescribe future "Next Steps"—focuses strictly on what has already been accomplished.

3. **No Web App / API Artifacts**:
   - Focuses on ML models, training pipelines, evaluation metrics, and geodetic algorithms.

---

## Execution Workflow

### Step 1: Discover Unrecorded Context & Artifacts
Analyze conversation history and workspace changes:
- Determine what activities and milestones were completed since the last recorded task.
- Inspect modified/untracked files via `git status -s`.
- Check any generated scripts, notebooks, or evaluation artifacts.

### Step 2: Determine Next Sequential Task File
- Inspect `.agents/tasks/` using filesystem commands (`Get-ChildItem -Path .agents/tasks` in PowerShell or `python -c "import pathlib; print([p.name for p in pathlib.Path('.agents/tasks').glob('*.md')])"`).
  > [!CAUTION]
  > Do **not** rely on IDE/MCP tree-listing tools for dot-prefixed directories like `.agents/` as they frequently omit hidden/dot folders or their contents, leading to false-empty results. Always verify directly via shell or Python `pathlib`.
- Parse all existing files matching `[0-9]{3}-*.md` and identify the current highest integer prefix.
- Increment the highest number by 1 (e.g. if `002-...` is highest, use `003-<descriptive-slug>.md`). Only start at `001` if `.agents/tasks/` contains zero matching markdown files after verified shell/pathlib inspection.

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
```

### Step 4: Synchronize Living Documentation (`docs/`)
If the task modified core assumptions, formulas, or added permanent utilities:
- Update `docs/core_engine.md` or `docs/grid.md` if relevant.
- Ensure all markdown links remain relative and valid.

### Step 5: Final Summary
Present a concise summary to the user with a clickable link to the new task file.
