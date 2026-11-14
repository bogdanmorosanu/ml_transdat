# Category 1 Deep Dive: Stereographic Projections, Shift Grids & Residual Modeling

This document provides a detailed, accessible breakdown of each **Category 1** reference (`R1.1`–`R1.7` in [references.md](references.md)). Because ML terminology can be dense, every technical concept is paired with a **plain-English explanation** of what happens behind the scenes. The focus throughout is strictly on **coordinate transformation**—how the models are set up, what problems they solve, what issues were encountered, and what we can apply to `ml_transdat`.

---

## 1. Reference-by-Reference Detailed Summaries

### R1.1 — Conformal / Stereographic Projection & Neural Network Coordinate Transformation (MDPI)
- **Source**: [mdpi.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEjXkHlj7I1oQCSutVEsmro-nv-tjq2ULvjFs-Q_ZjiU5PnZF6oxmBtoIVZg7WZ5Bt9OWWthoyqt_bl_hBrzO1b2VRFcdL4mZDBuj9iaa_Oedg9wGOJorbahPqYog==)
- **The Problem Being Solved**:
  - Converting geographic coordinates (latitude $\varphi$, longitude $\lambda$, height $h$) into a flat map projection (like Stereographic) normally requires heavy, nonlinear trigonometry. When you also have to account for real-world local distortions (e.g., old survey errors or tectonic shifts), rigid mathematical formulas leave noticeable errors.
- **How ML Was Used**:
  - Researchers trained **Multi-Layer Perceptrons (MLPs)** using **Backpropagation (BPANN)** directly on pairs of known control points (input: source coordinates; output: target projected coordinates).
  - **What happens behind the scenes (MLP & Backpropagation)**: Think of an MLP as a stack of adjustable mathematical filters (layers of "neurons"). You feed in $(\varphi, \lambda)$, it makes a guess at $(N, E)$, compares its guess to the true $(N, E)$, and then **backpropagation** nudges every internal knob slightly to reduce the error. Repeating this thousands of times lets the network bend and curve to match the data.
- **Issues Encountered & Key Lessons**:
  - **Coordinate Wrapping / Scale Issues**: Feeding raw latitude/longitude degrees directly into a neural network can confuse it because angles wrap around (e.g., $-180^\circ$ meets $+180^\circ$) and degrees of latitude vs. longitude represent different physical distances. Converting inputs to local projected coordinates or using **trigonometric encoding** ($\sin/\cos$) fixes this.
  - **Extrapolation Risk**: Pure end-to-end MLPs can behave unpredictably outside the geographic boundary of the training points.

---

### R1.2 — Grid Shift & Distortion Grid Approximation Using BPANN and RBFNN (MDPI)
- **Source**: [mdpi.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQH415HJgbn9bWJjwT_JehJLWBmGcsBw5E5LjgSpZJl6Fy45ZGz_XmcJqoD_z0JgYVK_TSF-h90nTW5GOAQJze7eKuc8gPTrpQlegVmxOp398hXcV_nF9pwQmgphaj8=)
- **The Problem Being Solved**:
  - Traditional datum shifts use discrete **shift-grid files** (like NTv2 `.gsb` files) that store $(\Delta\varphi, \Delta\lambda)$ at fixed grid intersections and use bilinear interpolation in between. The paper tests whether neural networks can act as a continuous, data-driven replacement for these grids.
- **What Was Tested & Compared**:
  - **BPANN (Backpropagation MLP)** vs. **RBFNN (Radial Basis Function Neural Network)** vs. classical **polynomial fitting** and **multiple linear regression**.
  - **What happens behind the scenes (RBFNN)**: Unlike an MLP—which tries to fit one big global mathematical surface across the whole country—an RBFNN places "bell curves" (radial basis functions, usually Gaussian) centered on specific control points. To predict the shift at a new point, it mostly listens to the bell curves *closest* to that point and blends them together.
- **Key Findings**:
  - **RBFNN consistently outperformed BPANN** in accuracy and stability for spatial coordinate-shift approximation. Because distortion grids have localized pockets of error (e.g., one region shifted southwest, another shifted north), RBFNN's *local* response handles regional quirks much better than a global MLP, and it trains much faster.

---

### R1.3 — NTv2 / 7-Parameter Helmert vs. Localized Distortion Grid Modeling (ResearchGate)
- **Source**: [researchgate.net](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGeua5rdwHcDL_NfOTqRr8KFdtY3_GZpf6zb7RxuBGog7S8LNGIOxeaQVHXq5707QBQ89zn5wo4eZw58L86NVylpbaXlVWBBTc2KofgtjAJbRBqdpW9qyXMVL1ozWwn01WMAZ6qeVdyy2S_T7HIjGlbT4KFTDttSrQxnoABK9PiiIVJgoSJefcR_uBA_vxYXTT8AM6Ad3Jenbjhugq7PHld4jxE5UQXikB-ypXMKfCTTPo7nn3DAYBSFylitpzSfsvHHKM7fUohjWQOh__gHf1gGxboFfMR-cGErjFmq2j3-7E2G65GPymoSj9KHumoPAjtB3l1an1lR7XstMmria2LwxBYOrDEYJa3tsXlFcZoLQgg2d2PaUoAYtwp_uTPGZtEiJQ96MlUb6ua7g==)
- **The Problem Being Solved**:
  - Explains *why* a single 7-parameter Helmert transformation (which only allows uniform translation, rotation, and scale across an entire country) fails when transforming legacy national datums.
- **Key Takeaways for Coordinate Transformation**:
  - Legacy triangulation networks contain **non-uniform internal strain** and historical surveying errors that vary from region to region.
  - NTv2 grids solve this by storing a dense mesh of local shifts. Any ML model attempting to replace or replicate such a grid must be able to represent **spatially varying, non-linear local distortions** rather than just a global rigid-body fit.

---

### R1.4 & R1.5 — Residual-Based Neural Network Correction (RBNC) (*Geodesy and Geodynamics* / arXiv / ResearchGate)
- **Sources**: [R1.4 (arXiv)](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQF9WeZe8zmjthB6Rcz4J0tqFYDVNCEdmCvjNULMZgXDWh7CQu3mn9ClYzoORtwN5tgvWGwSqiBjbf2l5l3ICIUTS5F4aXbA5UwaQmDeFu1A2-BJ5RQp) | [R1.5 (ResearchGate)](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQE5uUhadGhMw8wDHOAoh8v0ynPvbPRcbUMf4D8dRSgvUsnwyCLQkgbYE98yHVezvqd9pEIPHpj6j-1ZlxYlqHyKXM0XjoeW4aEKJCSrMq8Z6z54r6mz5UFB1cnsrr4InXKM5HJE_lJ82Pbed0CVUNGyEOnW9qfzLcoar0obhtXdQpey3FEzyQx244Wp0FbIJH_a4LmZQy5pB6_6cG1XEzdXSWXMXlL0kuj3OPtPUnTQC48mfG5FOkVj5cwV1-fIyTgU)
- **The Problem Being Solved**:
  - Direct ("end-to-end") neural networks that try to learn $(\varphi, \lambda) \to (N, E)$ from scratch struggle with huge numbers (e.g., hundreds of thousands of meters), can overfit when control points are sparse, and lack physical predictability.
- **How RBNC Works (The Two-Stage Hybrid)**:
  1. **Stage 1 (Classical Math)**: Run a trusted, deterministic geometric model (like Conformal, Affine, or Helmert) to get a solid baseline coordinate $\mathbf{X}_{\text{model}}$.
  2. **Stage 2 (ML Residual)**: Train a lightweight neural network $f_{\text{ANN}}$ *only* on the leftover error (**residual**) between the baseline and the true coordinate:
     $$\mathbf{X}_{\text{final}} = \mathbf{X}_{\text{model}} + f_{\text{ANN}}(\mathbf{X}_{\text{source}})$$
  - **What happens behind the scenes**: Instead of asking the neural network to learn the entire Earth's curvature and projection math, you let classical geodesy do 99.9% of the work. The neural network only has to learn a small, smooth "correction map" (e.g., $\pm 2.7\,\text{m}$).
- **Issues Solved & Results**:
  - Much smaller networks can be used, training is far more stable, and it achieves **lower RMSE** than both standalone classical models and direct end-to-end neural networks—especially when data is unevenly distributed.

---

### R1.6 & R1.7 — Conformal Mapping Coordinates PINNs (`CoCo-PINNs`) (OpenReview / arXiv)
- **Sources**: [R1.6 (OpenReview)](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHzrICQy6JEH0nzLvYl-as2DQKznlpj1pMF9Xl9p8xZodF_vLa9cYxmJSznRN_DBCDsyinZaXthNK3ZSSD2s58nk1eXJAwNFAjxpVEfnZyzKeJM1LVgv9dO-jTC41DCvuE=) | [R1.7 (arXiv)](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQESQ6Ck8fGBN13fOND9KJMpcueFXNC3K4VlGA3eBRiFOhyH0dfZyo_58TvbOsE2SYohj4IW1cB_veKKmp0tgBVUv_yuS-YKAqjlpai-XI74e1ExPPZskQ==)
- **Scope Note**: These papers focus on solving physics equations (PDEs) in materials science, *not* geodetic datum shifts. However, they introduce two mathematical/ML tricks directly relevant to coordinate transformations:
  1. **Conformal Coordinate Mapping**: Transforming an irregular physical shape into a clean, canonical coordinate space (like a unit circle) before the neural network processes it.
  2. **Fourier Feature Encodings**:
     - **What happens behind the scenes**: Standard neural networks suffer from **"spectral bias"**—they are naturally good at learning broad, smooth trends, but terrible at learning sharp, fine-grained local bumps (like a sudden $2\,\text{m}$ distortion pocket in one county). **Fourier Feature Encoding** fixes this by passing the input coordinates $(x, y)$ through sine and cosine waves at multiple frequencies ($\sin(2^k x), \cos(2^k x)$) *before* entering the network. This gives the network "rulers" of different sizes so it can easily see both nationwide trends and local bumps.

---

## 2. Overall Summary: What We Should Use & Test in `ml_transdat`

Combining these findings with our existing baselines in [`docs/research/roadmap.md`](../docs/research/roadmap.md) (where `StereoProj + Helmert2D` already achieves $\text{RMSE}=0.4429\,\text{m}$), here is a concrete, prioritized roadmap of what to test:

1. **Use the RBNC Hybrid Setup (Do Not Train End-to-End)**:
   - Keep `StereoProj + Helmert2D` (or `EPSG:3844`) as Stage 1.
   - Train ML models **strictly on the 2D distortion residuals** $(dN, dE)$ from the `.spg` grid. This keeps targets bounded ($\text{mean}=0.37\,\text{m}, \text{max}=2.76\,\text{m}$) and guarantees physical stability.
2. **Test Local-Kernel Models First (`R1.2`)**:
   - **RBFNN / Thin Plate Splines (TPS)**: Since distortion grids are spatially localized, radial-kernel models are the strongest classical ML baseline to compare against a standard MLP.
3. **Upgrade MLPs with Fourier Feature Encodings (`R1.6`)**:
   - Standard MLPs will likely blur out sharp local pockets (like Banat/West or Maramureș). Adding **Fourier Feature Encodings** to normalized $(N, E)$ inputs is the most promising way for a neural network to hit sub-centimeter ($< 0.01\,\text{m}$) RMSE.
4. **Validate Spatially, Not Just Randomly**:
   - Use **spatial block cross-validation** (holding out entire regions/grid cells) alongside random splits, so we know the model generalizes to unseen areas rather than just memorizing nearby grid nodes.
