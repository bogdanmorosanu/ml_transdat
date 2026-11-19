---
name: research-sync
description: Captures scientific hypotheses, rationale, pivots, and verdicts from conversations into reference-first research units (docs/research/<slug>.md), and keeps the knowledge graph (roadmap.md) and causal evolution (narrative.md) synchronized. Use on demand during or after brainstorming or experiments.
---

# Research Sync Skill (`research-sync`)

## Purpose
Maintains the scientific knowledge graph and causal narrative in `docs/research/` following [`docs/research/GEMINI.md`](../../../docs/research/GEMINI.md).

---

## Execution Workflow

### Step 1: Read Current Research State
Read:
- `docs/research/GEMINI.md`
- `docs/research/roadmap.md`
- `docs/research/narrative.md`
- Current git branch (`git branch --show-current` or `pycharm` MCP `git_status`).

### Step 2: Extract Scientific Delta from Conversation
Identify any new or updated:
- Hypotheses, design rationales, or alternatives considered/rejected.
- Pivots or spawned sub-questions (remember: 1 conversation can yield multiple research units; 1 research unit can synthesize multiple experiments).
- Quantitative verdicts (`Exploring`, `Active`, `Validated`, `Dead End`, `Superseded`).

### Step 3: Create / Update Research Units (`docs/research/<slug>.md`)
- Use **slug-only filenames** (never numeric prefixes).
- Keep each unit **concise and reference-first**: link to `experiments/<slug>/`, `literature/`, or `.agents/tasks/<slug>.md` instead of duplicating full tables, plots, or code details.

### Step 4: Update Graph & Causal Narrative (Branch-Safe)
- **On `main`** (or when finalizing/merging a branch):
  - Update `docs/research/roadmap.md` (TL;DR, Mermaid DAG, **Annotated Relationships Table** with plain-English meaning, and Research Units Index).
  - Append the causal step(s) to `docs/research/narrative.md` in logical order.
- **On a feature/experiment branch (`exp/<slug>`)**:
  - Update `docs/research/<slug>.md` freely; defer `roadmap.md` and `narrative.md` updates until merge to `main` (unless explicitly requested) to avoid parallel-branch merge conflicts.
