---
name: task-record
description: Unified documentation orchestrator that records session outcomes, ML experiments, and architectural decisions in .agents/tasks/ using slug-based naming, appends to master_tasks.md in execution order, and synchronizes living documentation. Use whenever concluding or checkpointing a task.
---

# Unified Task Record Skill (`task-record`)

## Purpose
Records chronological engineering and execution outcomes in `.agents/tasks/` following [`.agents/tasks/GEMINI.md`](../../tasks/GEMINI.md), keeping `.agents/tasks/master_tasks.md` synchronized in execution order.

*(Note: Scientific hypotheses, rationale, and research verdicts belong in `docs/research/` via `research-sync`, not duplicated here.)*

---

## Execution Workflow

### Step 1: Discover Unrecorded Context & Branch
- Inspect modified/untracked files via `git status -s` (or `pycharm` MCP `git_status`) and note the current git branch.
- Determine what tangible milestones were completed since the last entry in `.agents/tasks/master_tasks.md`.

### Step 2: Determine Slug-Based Task Filename
- Follow `.agents/tasks/GEMINI.md`: **never use numeric prefixes** (`001-`, `002-`, ...).
- Name the file `<slug>.md` (or `YYYY-MM-DD_<slug>.md` if revisiting a previously recorded feature).

### Step 3: Write Task Record
Create `.agents/tasks/<slug>.md` using this structure:
```markdown
# Task: <Title>

**Date**: YYYY-MM-DD
**Branch**: <current-branch>
**Status**: Completed / In Progress
**Scope**: <Data Gen | Model Architecture | Training | Evaluation | Refactoring | Documentation>

## Objective
<Concise summary of the task goals>

## Key Activities & Implementation
- <Activity 1>
- <Activity 2>

## Verification & Results (if applicable)
- <Tests run, experiment metrics, or file checks>
```

### Step 4: Append to `.agents/tasks/master_tasks.md`
- Append a single row to the bottom of the table in `.agents/tasks/master_tasks.md` in exact execution order:
  `| YYYY-MM-DD | `<branch>` | [`<slug>`](<slug>.md) | <1-sentence summary of what changed> |`

### Step 5: Synchronize Living Docs & Summarize
- Update `docs/core_engine.md`, `docs/grid.md`, or `experiments/README.md` if permanent utilities or units were added/modified.
- Present a concise summary with clickable links to the new task file and `master_tasks.md`.
