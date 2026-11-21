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

3. **Decision Tree Spatial Regressor & Density Scaling (`decision_tree_regressor`)**:
   - *Finding*: Unconstrained trees memorize $100\%$ of training nodes ($0.0000\,\text{m}$), but orthogonal step-plateaus yield $8.17\,\text{cm}$ error on `8x8` when trained on 11 km nodes. Densifying to `4x4` ($2.46\,\text{cm}$), `8x8` ($0.80\,\text{cm}$), and `16x16` ($0.40\,\text{cm}$) halves error per halving of spacing while quadrupling model size ($0.41\,\text{MB} \to 89.4\,\text{MB}$). Impurity pruning (`1e-8`) compresses `8x8`/`16x16` to $2.8\,\text{MB}$ at $1.88\,\text{cm}$ RMSE.
   - *Impact*: Establishes the step-lookup scaling law and motivates smooth/ensemble models that can interpolate between the 2,689 base nodes without massive grid densification.

4. **Curvature-Driven Adaptive Quadtree Space Partitioning (`adaptive_quadtree_grid`)**:
   - *Question*: Can we avoid brute-force uniform densification by dynamically splitting cells only where the distortion field bends away from local bilinear planarity?
   - *Finding*: Checking 5 interior points (cell center + 4 sub-quadrant centers) against bilinear interpolation allows generating full-Romania grids down to $1\,\text{mm}$ tolerance with $564\,\text{k}$ cells ($45\,\text{MB}$ Parquet)—over $97\%$ smaller than a uniform 100 m grid ($27+$ million points). Evaluated on the national corridor benchmark, the $3\,\text{mm}$ official tier achieves $1.4\,\text{mm}$ mean 2D error ($3.1\,\text{mm}$ P95), and the $1\,\text{mm}$ aspirational tier achieves $0.5\,\text{mm}$ mean 2D error ($1.0\,\text{mm}$ P95).
   - *Impact*: Establishes an ultra-accurate pure geometric lookup baseline and provides the blueprint for curvature-stratified test benchmark generation.

---

## Phase 3: Continuous Neural Surface Models & Spectral Bias Resolution

5. **Continuous Neural Network Spatial Regressor (`mlp_spatial_regressor`)**:
   - *Question*: Can a compact, pure parametric neural network (Multilayer Perceptron) learn the continuous 2D distortion field of Romania without storing any discrete lookup grid, and how does it perform on off-lattice points?
   - *Finding*: Standard feedforward networks on raw coordinates suffer from **spectral bias**, yielding $18.2\,\text{cm}$ uniform RMSE and failing on local domes ($>1.5\,\text{m}$ max error in Banat). Augmenting inputs with **Random Fourier Features (RFF)** ($\sigma=3.0$) resolved spectral bias, cutting error by $3.7\times$ down to $4.6\,\text{cm}$ RMSE on the 2,689 official anchor nodes.
   - *Density Scaling*: Scaling training data to the $4\times 4$ densified subgrid (36,435 pts) dropped nationwide uniform RMSE to **$1.3\,\text{cm}$** ($99.8\% < 5\,\text{cm}$), with a median error of **$8.8\,\text{mm}$**. Scaling further to the $8\times 8$ subgrid (146,258 pts) regularized edge-case anomalies (cutting max error from $37.5\,\text{cm} \to 15.2\,\text{cm}$), but reached a natural capacity floor around $\sim 3\,\text{cm}$ on the curvature-weighted adaptive stress test.
   - *Impact*: Proves that a ~200 KB continuous neural network can achieve near-centimeter accuracy nationwide with zero external lookup table, while revealing that pushing to sub-millimeter precision requires hybrid decomposition (e.g., Helmert + local residual models).

---

## Phase 4: Classical Mathematical & Analytical Surface Models

6. **2D Bivariate Polynomial Surface Regressors (`polynomial_surface_regressor`)**:
   - *Question*: How much accuracy gain is achieved by upgrading the rigid flat plane of the 2D Helmert baseline into a smooth curved 2D polynomial surface of degree $D$, and at what degree does Runge's phenomenon emerge?
   - *Finding*:
     - Moving from a flat plane ($D=1$, $42.8\,\text{cm}$ uniform RMSE) to a quadratic bowl/saddle ($D=2$, $33.0\,\text{cm}$) reduces error by **$23\%$** with just 6 parameters per axis.
     - Odd-degree surfaces ($D=3, 5$) introduce inflection S-curves, reducing uniform RMSE to **$27.0\,\text{cm}$** and slashing worst-case max error by **$76\,\text{cm}$** ($2.60\,\text{m} \to 1.84\,\text{m}$).
     - Pushing to Degree 12 achieves **$19.2\,\text{cm}$ uniform RMSE** and cuts maximum nationwide error below 1 meter (**$89.3\,\text{cm}$**).
     - However, global polynomials hit an asymptotic ceiling near $\sim 17\text{--}18\,\text{cm}$; at higher degrees ($D \ge 16$), border ripple oscillations (Runge's phenomenon) begin to emerge.
   - *Impact*: Validates historical geodetic practice (stopping at degrees 3 to 5) and establishes the curved macroscopic foundation, providing the direct mathematical motivation for Level 3: Radial Basis Functions (RBF) and Thin Plate Splines (TPS).
