# Scientific Narrative & Causal Evolution

An ordered, causal record of how our understanding, hypotheses, and modeling decisions have evolved—serving as a fast catch-up spine for AI brainstorming and future paper writing.

---

## Phase 1: Understanding the Physical Distortion Field & Linear Baselines

1. **Profiling the Ground-Truth Distortion Field** ([`spg_grid_analysis`](spg_grid_analysis.md))
   - *Question*: What does the Romanian `.spg` distortion field look like spatially, and where are its largest non-linearities?
   - *Finding*: 2D shifts average $0.372\,\text{m}$ but peak at $2.757\,\text{m}$ in Banat/West, with strong directional gradients in Maramureș and Dobrogea.
   - *Impact*: Confirmed that a global rigid transform cannot achieve sub-centimeter accuracy and motivated benchmarking standard GIS and optimal 2D conformal floors.

2. **Benchmarking Standard GIS (`EPSG:3844`)** ([`pyproj_baseline`](pyproj_baseline.md))
   - *Question*: Does modern `pyproj` (`EPSG:3844`) already capture the local Romanian distortion field?
   - *Finding*: No—its nationwide error ($\text{RMSE}=0.4348\,\text{m}$, $\text{Max}=2.82\,\text{m}$) mirrors the raw `.spg` distortion magnitude.
   - *Impact*: Ruled out standard 7-parameter 3D Bursa-Wolf as sufficient and established the industry baseline to beat.

3. **Establishing the Optimal 2D Conformal Floor** ([`helmert2d_baseline`](helmert2d_baseline.md))
   - *Question*: What is the best accuracy achievable by a grid-free 2-step 2D conformal model (`StereoProj + Helmert2D`) refitted to the `25.09` grid?
   - *Finding*: Refitting on 2,222 valid SPG nodes achieves $\text{RMSE}=0.4429\,\text{m}$, $\text{Max}=2.6811\,\text{m}$, and exact analytical invertibility ($5.24 \times 10^{-9}\,\text{m}$ round-trip).
   - *Impact*: Locked in `StereoProj + Helmert2D` as the deterministic geometric base stage, isolating the remaining residuals for non-linear ML models.

4. **Literature Survey & Multi-Resolution Evaluation Grids** ([`literature/`](../../literature/README.md), [`grid-restructuring-and-multiresolution-generation`](../../.agents/tasks/grid-restructuring-and-multiresolution-generation.md))
   - *Question*: What ML paradigms best model geodetic shift grids, and what data splits prevent spatial overfitting?
   - *Finding*: No published ML work exists for `Stereo 70`; geodetic literature favors **Residual-Based Neural Network Correction (RBNC)**, **RBFNN/TPS**, and **Fourier Feature Networks** to overcome spectral bias. Separated `spg_nodes_valid.parquet` (2,689 training anchors) from `ro_grid_4x4.parquet` (36,435 pts) and `ro_grid_8x8.parquet` (146,258 pts).
   - *Impact*: Set the stage for testing candidate residual ML models against the $< 0.01\,\text{m}$ goal.

---

## Phase 2: Spatial Partition Models & Discovery of Grid Continuity Limits

5. **Benchmarking the First ML Baseline: K-Nearest Neighbors** ([`knn_spatial_regressor`](knn_spatial_regressor.md))
   - *Question*: How effectively do local spatial partition models (KNN) interpolate 2D distortion shifts inside $11\,\text{km}$ grid cells when trained on control anchors?
   - *Finding*: $K=4$ Inverse Distance Weighting achieves $\text{RMSE} = 0.0353\,\text{m}$ ($3.5\,\text{cm}$) on `ro_grid_4x4`—a **$92.1\%$ error reduction** over the baseline floor. Increasing $K$ to 8 or 16 degrades performance ($6.0\,\text{cm}$ and $9.2\,\text{cm}$) because positive-only IDW weights dilute local gradients.
   - *Impact*: Established our first non-linear baseline.

6. **Discovery of Official Grid Piecewise $C^0/C^1$ Kinks** ([`geodetic_grid_interpolation_standards.md`](../../literature/geodetic_grid_interpolation_standards.md))
   - *Question*: Why do 1D transects of official RomGEO bicubic splines exhibit slope breaks at $11\,\text{km}$ cell boundaries, and how does this compare to global practice?
   - *Finding*: Verified against official RomGEO desktop software ($< 0.36\,\text{mm}$ agreement) that the Romanian `.spg` engine uses a local $4 \times 4$ patch formula (*Numerical Recipes* `bcucof`) with one-sided finite differences. At cell borders, stencil shifts create a first-derivative slope break ($\Delta\text{slope} \approx 4.2 \times 10^{-5}\,\text{m/m}$). Globally, agencies (Canada NTv2, US NADCON 5, UK OSTN15) universally accept $C^0$ boundary kinks to prevent polynomial overshooting.
   - *Impact*: Identified a key inductive bias mismatch: smooth ML models (RBF, Neural Networks) will naturally smooth out legacy grid kinks unless supplied with cell-interior coordinates $(u, v)$.
