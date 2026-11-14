# Task: Geodetic ML Literature Survey & Reference Cataloging

**Date**: 2026-11-14
**Status**: Completed
**Scope**: Literature Survey | Scientific Research | Documentation

## Objective
Conduct a broad, structured literature survey of machine learning techniques applied to geodetic coordinate transformations, stereographic projections, and distortion/shift grids. Provide clear conceptual summaries, catalogue verified references, and identify actionable modeling paradigms applicable to the Romanian ETRS89 ↔ Stereo 70 transformation challenge.

## Key Activities & Implementation
- **Tooling Clarification & Strategy Alignment**:
  - Clarified autonomous web search capabilities and sub-agent research workflows within Antigravity.
  - Defined a 2-tier categorization strategy:
    - *Category 1*: Closely related transformations (stereographic projections, distortion/shift grids, and hybrid residual modeling).
    - *Category 2*: Broader ML coordinate transformation literature (MLP/BPANN, RBFNN/TPS, ELM, LS-SVM, Gaussian Processes, PINNs, and Tree-Based ensembles).
- **Literature Search & Logging**:
  - Executed targeted academic searches across scientific publishers (MDPI, Springer, Elsevier/ScienceDirect, ResearchGate, arXiv).
  - Maintained complete search queries and logged negative results in [`literature/search_log.md`](../../literature/search_log.md).
- **Synthesis & Reference Catalog**:
  - Authored [`literature/references.md`](../../literature/references.md) with verified, clickable URLs categorized by model class.
  - Authored [`literature/README.md`](../../literature/README.md) structuring findings into methodology groups and knowledge gaps.
  - Expanded detailed reference summaries in [`literature/category1_deep_dive.md`](../../literature/category1_deep_dive.md) with plain-English breakdowns of ML mechanics (MLP, RBFNN, RBNC, Fourier Feature Encodings, spectral bias).
  - Extended research into non-neural classical ML models (**XGBoost**, **Random Forest**, **RFSI**), evaluating trade-offs (fast tabular training vs. piecewise-constant step artifacts).

## Key Research Findings & Takeaways
1. **Absence of Existing Stereo 70 ML Work**:
   - Confirmed no published papers or technical reports exist for ML-based ETRS89 ↔ Stereo 70 coordinate transformations or the Romanian `.spg` distortion grid.
2. **Primary Modeling Paradigm — Residual-Based Neural Network Correction (RBNC)**:
   - Transforming coordinates end-to-end with black-box neural networks introduces instability and high extrapolation error.
   - The established best practice in geodetic literature is a two-stage hybrid approach: apply a deterministic baseline (e.g., `StereoProj + Helmert2D`) and train ML models strictly on the 2D spatial distortion residuals ($dN, dE$).
3. **Candidate Architectures for Distortion Grids**:
   - *Radial Basis Function Networks (RBFNN) / Thin Plate Splines (TPS)*: Strong local response suited for spatially clustered distortion patterns.
   - *Fourier Feature Networks (FFN)*: Mitigate spectral bias in MLPs to resolve high-frequency local distortion pockets.
   - *Tree-Based Ensembles (XGBoost / Random Forest)*: Viable fast baseline, though piecewise-constant nature requires scrutiny against geodetic smoothness criteria.
