---
name: task-init
description: Initializes any ML TransDat development task or chat session. Discovers user intent, checks active git branch and status, dynamically indexes living architecture/research docs and task history (.agents/tasks/master_tasks.md), verifies Python environment readiness, and outputs a 4-bullet readiness briefing. Use at the start of any new session or task.
---

# Task Initialization Skill (`task-init`)

## Purpose
This skill serves as the **universal entrypoint** for sessions in `ml_transdat`. It primes context with living documentation, research state, task history, git branch status, and Python environment readiness.

---

## Execution Workflow

### Step 1: Discover User Intent & Task Context
Analyze the user's initial prompt to identify:
- **Category**: Data Generation, Baseline Benchmarking, Feature Engineering, Model Training, Architecture/Refactoring, Literature Research, Evaluation, or Documentation.
- **Direction**: Direct ($ETRS89 \to Stereo70$), Inverse ($Stereo70 \to ETRS89$), or Bidirectional.
- **Persona(s)**: Select the appropriate persona file(s) per `GEMINI.md` (or Default/General mode for admin/git tasks).

### Step 2: Check Git Branch & Index Living Documentation
1. **Git State**: Check `git branch --show-current` and `git status -s` (via `pycharm` MCP `git_status` or shell).
2. **Task History**: Read `.agents/tasks/master_tasks.md` to see recent engineering/execution history in order.
3. **Research & Experiment State**:
   - `docs/research/roadmap.md` and `docs/research/narrative.md` (current scientific state & causal evolution)
   - `experiments/README.md` (active experiment units)
   - `literature/README.md` (when relevant to modeling or prior art)
   - `docs/core_engine.md` and `docs/grid.md` (when relevant to projection/grid math)

### Step 3: Verify Python Environment
Verify that the active interpreter matches the invariant:
- Executable: `D:\anaconda3\envs\ml_transdat\python.exe`
- Verify that `numpy` and `pytransdat` import correctly.

### Step 4: Output 4-Bullet Readiness Briefing
Conclude initialization with a concise 4-bullet briefing:
- **Intent & Branch**: Clarified user objective, scope, and active git branch (`<branch>`).
- **Context Primed**: Relevant docs, research state, and recent tasks indexed.
- **Environment**: Python environment and core imports verified.
- **Next Action**: Concrete immediate next step to begin the task.
