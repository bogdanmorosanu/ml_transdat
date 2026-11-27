# Exploration & Strategic Assessments (`docs/exploration/`)

This directory stores **forward-looking exploratory reports, strategic assessments, and cross-cutting roadmaps** for `ml_transdat`.

## Purpose & Relationship to Other Docs

- **`docs/exploration/` (Brainstorming & Strategy)**: Captures big-picture reviews of what we have learned across multiple experiments, diagnoses fundamental bottlenecks, and ranks unexplored leads across datasets, features, training objectives, and model families before we commit to building new experiments.
- **[`docs/research/`](../research/roadmap.md) (Scientific Knowledge Graph)**: Tracks concrete, hypothesis-driven research units (`Exploring`, `Active`, `Validated`, `Dead End`, `Superseded`) and their causal relationships.
- **[`experiments/`](../../experiments/README.md) (Execution Units)**: Houses self-contained, reproducible code runs, scorecards, and plots for individual hypotheses.

---

## Exploratory Reports Index

| Report | Focus | Status |
| :--- | :--- | :---: |
| [`submillimeter_strategy_report.md`](submillimeter_strategy_report.md) | Comprehensive plain-English assessment of all tested models, why continuous models plateau at ~4–7 mm, and a 7-step prioritized roadmap toward **1 mm precision**. | `Active` |
| [`submillimeter_v2_model_landscape.md`](submillimeter_v2_model_landscape.md) | Exhaustive pre-finalization audit of all tested paradigms (`[x]`) and 12 untested neural INR, structured hybrid, non-neural ML, and geostatistical candidates (`[ ]`) for `submillimeter_exploration_v2`. | `Active` |
