# Scratch Space

This directory is designated for **ephemeral prototypes**, exploratory scripts, and temporary visual experiments.

### Rules of Engagement
1. **Disposable by Design**: Files here can be modified or deleted without breaking production benchmarks or research units.
2. **Promotion Path**: Once an experiment or utility proves stable, formalize it into a new Research Unit (`experiments/XX_slug/`) or promote it to the shared core (`src/core/`).
3. **No External Dependents**: Production tests in `tests/` and research units in `experiments/` must **never** import from `scratch/`.
