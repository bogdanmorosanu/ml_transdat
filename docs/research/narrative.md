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
