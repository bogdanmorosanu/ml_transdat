---
name: research-roadmap
description: Manages the branching scientific research roadmap in docs/research/. Creates numbered branch files/folders (01, 02, ...), records hypotheses, rationales, and outcomes, and updates docs/research/roadmap.md to map connections and summaries across branches.
---

# Research Roadmap Skill (`research-roadmap`)

## Purpose
This skill maintains the **scientific journey** in `docs/research/`. Whenever a new research premise, modeling hypothesis, or supporting infrastructure path is worth pursuing, this skill creates its dedicated branch documentation and updates the master map in `docs/research/roadmap.md`.

---

## Execution Workflow

### Step 1: Determine Next Branch Number
- Inspect `docs/research/` for existing numbered branches (`01_*`, `02_*`, ...).
- Assign the next sequential 2-digit prefix (`01`, `02`, `03`, ...) with a concise, descriptive slug (e.g., `01_pure_mlp_baseline/` or `01_pure_mlp_baseline.md`).

### Step 2: Create Branch Documentation
Document the branch's specific scientific journey:
- **Premise / Hypothesis**: What we are testing and why.
- **Parent / Connections**: Which earlier branch(es) or findings led here, or if it merges multiple paths.
- **Methodology & Experiments**: Key setups, datasets, and model configurations tested.
- **Outcome & Verdict**: `Active`, `Promising`, `Validated`, `Dead End`, or `Merged`, with quantitative takeaways (accuracy vs. `pytransdat`, latency, parameter efficiency).

### Step 3: Update Master Map (`docs/research/roadmap.md`)
- Add or update the branch entry in `docs/research/roadmap.md` with:
  - A **brief description** of the branch.
  - **Status** and **connections** to other branches (splits, merges, dead ends).
  - Updated Mermaid tree/DAG diagram if the topology changed.
- Keep `docs/research/roadmap.md` high-level and scannable; keep deep details inside the branch file/folder.
