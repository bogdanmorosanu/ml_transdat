# `src/` Architectural Invariants

- All reusable data loaders, models, evaluators, metrics, and visualizers live here — never inside `experiments/` or `benchmarks/`.
- Follow the contracts and layering defined in [`docs/architecture.md`](../docs/architecture.md).
- **Documentation Sync Rule**: Whenever you add, rename, remove, or refactor a module, base class, registry enum (`GridId`, `MetricType`, `PlotType`), or package boundary in `src/`, **you MUST update [`docs/architecture.md`](../docs/architecture.md)** in the same task so the architecture blueprint stays accurate.
