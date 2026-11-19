# Tasks Tracking Rules (`.agents/tasks/`)

Rules governing engineering and execution task records in `.agents/tasks/`.

---

## 1. Purpose & Boundary
- `.agents/tasks/` tracks **what was built, modified, tested, and verified** (code, data, scripts, refactoring, setup).
- Scientific hypotheses, rationale, and research verdicts belong in [`docs/research/`](../../docs/research/GEMINI.md) (managed via `research-sync`).

---

## 2. Naming & Structure
- **Slug-Only Naming**: Never use sequential numeric prefixes (`001-`, `002-`, ...). Name task files `<slug>.md` (or `YYYY-MM-DD_<slug>.md` when revisiting a previously touched feature).
- **Required Header**: Every task file must include `**Date**`, `**Branch**`, `**Status**`, and `**Scope**`.

---

## 3. Master Chronological Log (`master_tasks.md`)
- Every time a task record is created, **append** a row to [`master_tasks.md`](master_tasks.md) in the exact order of execution:
  `| YYYY-MM-DD | <branch> | [<slug>](<slug>.md) | <1-sentence summary of what changed> |`
- This ensures agents (`task-init`) and humans can read the full project history in execution order from a single file without scanning every task document.
