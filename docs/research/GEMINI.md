# Research Knowledge Graph Rules (`docs/research/`)

Rules governing scientific research units, causal narrative, and relationship tracking in `docs/research/`.

---

## 1. Core Principles
- **Slug-Only Naming**: Never use numeric prefixes (`01_`, `02_`, ...). Name research units by topic slug: `docs/research/<slug>.md`.
- **Many-to-Many Decoupling**: Research units are independent of `experiments/` and git branches. One conversation or experiment may spawn multiple research units; one research unit may synthesize multiple experiments or branches.
- **Reference-First / No Duplication**: Keep each `<slug>.md` concise and strategic (core question, rationale, key strategic metrics/verdicts, and clickable links to primary sources in `experiments/`, `literature/`, or `.agents/tasks/`). Never duplicate full experiment tables, plots, or code details.

---

## 2. Research Unit Template (`docs/research/<slug>.md`)
```markdown
# <Topic Title>

- **Status**: `Exploring` | `Active` | `Validated` | `Dead End` | `Superseded`
- **Spawned From**: [<parent-slug>](<parent-slug>.md) *(or Initial Baseline)*
- **Linked Sources**: [`experiments/<slug>/`](../../experiments/<slug>/), [`literature/`](../../literature/README.md), [task](../../.agents/tasks/<slug>.md)

## Current Verdict
<2–3 sentences summarizing the current conclusion and strategic takeaway.>

## Premise & Rationale
<Why we opened this question, alternatives considered, and key reasoning.>

## Key Strategic Findings
- <High-level quantitative/qualitative takeaway with link to detailed source>
```

---

## 3. Master Files & Parallel-Branch Safety
- **`roadmap.md`**: Master knowledge graph (current state TL;DR, Mermaid DAG, **Annotated Relationships Table** explaining *why* nodes connect, and Research Units Index).
- **`narrative.md`**: Ordered **causal evolution** grouped by Phases (not calendar timestamps)—forms the linear catch-up spine for AI brainstorming and future paper writing.
- **Parallel-Branch Rule**: On an `exp/<slug>` branch, create/update `docs/research/<slug>.md` freely. Update `roadmap.md` and `narrative.md` only when merging to `main` (or resolve simple markdown appends on merge) to avoid conflicts across parallel chats.
