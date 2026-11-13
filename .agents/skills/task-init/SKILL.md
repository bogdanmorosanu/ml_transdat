---
name: task-init
description: Initializes any ML TransDat development task or chat session. Discovers user intent, dynamically indexes living architecture docs (docs/) and task history (.agents/tasks/), verifies Python environment readiness, and outputs a 4-bullet readiness briefing. Use at the start of any new session or task.
---

# Task Initialization Skill (`task-init`)

## Purpose
This skill serves as the **universal entrypoint** for all development, research, data generation, and model training sessions in the `ml_transdat` repository. It dynamically primes the agent context with living documentation and task history, verifies the dedicated Python environment (`D:\anaconda3\envs\ml_transdat`), and outputs a high-signal readiness briefing.

---

## Operating Principles

1. **Universal Applicability**:
   - Adapts to any task domain: data generation, baseline transformation benchmarks, ML feature engineering, model architecture prototyping, evaluation, and documentation.

2. **Dynamic Context Priming (Zero Guesswork)**:
   - Discovers authoritative documentation in `docs/` and historical task records in `.agents/tasks/`.

3. **No Web Server Checks**:
   - Focuses strictly on Python environment integrity, GPU/CPU availability (if PyTorch/ML libraries are used), and workspace cleanliness.

---

## Execution Workflow

Execute these steps sequentially when initialized:

### Step 1: Discover User Intent & Task Context
Analyze the user's initial prompt to identify:
- **Category**: Data Generation, Baseline Benchmarking, Feature Engineering, Model Training, Evaluation, or Documentation.
- **Direction**: Direct ($ETRS89 \to Stereo70$), Inverse ($Stereo70 \to ETRS89$), or Bidirectional.
- **Reference**: Ground truth baseline (`pytransdat`) vs. candidate ML model.

### Step 2: Index Living Documentation & Task History
Scan the workspace for context:
1. **Core Documentation**:
   - `docs/core_engine.md` (projection math, Krasovsky ellipsoid, datum parameters)
   - `docs/grid.md` (grid interpolation, bounding box, shift mechanics)
2. **Recent Tasks**:
   - Check `.agents/tasks/` for the latest task file to understand ongoing experiments or recent progress.

### Step 3: Verify Python Environment
Verify that the active interpreter matches the invariant:
- Executable: `D:\anaconda3\envs\ml_transdat\python.exe`
- Verify that `numpy` and `pytransdat` import correctly.

### Step 4: Output 4-Bullet Readiness Briefing
Conclude initialization with a concise 4-bullet briefing:
- **Intent**: Clarified user objective and scope.
- **Context Primed**: Relevant documents and recent task history indexed.
- **Environment**: Python environment and core imports verified.
- **Next Action**: Concrete immediate next step to begin the task.
