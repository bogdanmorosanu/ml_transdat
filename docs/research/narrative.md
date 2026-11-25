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


7. **Localized Continuous Surface Deformation via Thin Plate Splines (`rbf_surface_regressor`)**:
   - *Question*: Can an elastic continuous sheet model (Thin Plate Spline $r^2 \ln r$) overcome polynomial boundary oscillations and achieve sub-centimeter accuracy when trained on 2,222 base grid anchors?
   - *Finding*:
     - **Sub-Centimeter Accuracy**: Combining Thin Plate Spline with gentle nugget smoothing ($\lambda = 10^{-4}$) achieves **$9.3\,\text{mm}$ nationwide RMSE** on `RO_TEST_UNIFORM_4X4` ($77.4\% < 1\,\text{cm}$, $99.9\% < 5\,\text{cm}$), a **$20.6\times$ error reduction** over Level 2 polynomials.
     - **Grid Kink Filtering**: Relaxing exact interpolation improves nationwide accuracy by $30\%$ ($13.2\,\text{mm} \to 9.3\,\text{mm}$) and cuts maximum nationwide error by $12.4\,\text{cm}$ ($34.3\,\text{cm} \to 21.9\,\text{cm}$) by smoothing over the legacy $C^0/C^1$ boundary slope breaks of Romania's official `.spg` grid.
     - **Foundation Invariance**: Testing Affine (Deg 1) vs. Quadratic (Deg 2) vs. Cubic (Deg 3) underlying polynomial trends yielded virtually identical performance ($13.25\,\text{mm}$ exact / $9.30\,\text{mm}$ smooth), proving that localized radial bending completely absorbs macroscopic curvature.
   - *Impact*: Delivers our first continuous, closed-form mathematical model achieving **sub-centimeter nationwide accuracy** with zero lookup tables ($0\,\text{MB}$ grid data).

---

## Phase 5: Local Patch Re-Parameterization & Functional Curvature Taxonomy

8. **Grid Cell Curvature & Shape Classification (`cell_planar_features`)**:
   - *Question*: Instead of forcing continuous models to learn nationwide coordinates $(N, E)$ across $800\,\text{km}$, can we re-parameterize the domain into normalized local cell coordinates $(u, v) \in [0, 1]$ and classify cells into discrete polynomial order archetypes?
   - *Finding*:
     - **Curvature Trumps Slope**: Planar steepness and compass aspect do not govern transformation complexity; a steep ramp is just as linear as a flat floor. Mathematical difficulty is strictly determined by internal non-linear curvature.
     - **3-Class Taxonomy**: Evaluating 25 interior test points per cell across polynomial orders proves that under a $5\,\text{mm}$ geodetic tolerance, Category 4 (high-order bicubic anomalies) vanishes ($< 0.5\%$), yielding a clean **$53.2\%$ Parabolic Dome/Bowl** vs. **$46.8\%$ Cubic S-Shape** split. Within $1\,\text{cm}$ ($10\,\text{mm}$), **$92.5\%$ of Romania** is captured by a simple 2nd-degree parabola.
     - **High Axis Concordance**: In $68.7\%$ of cells at $5\,\text{mm}$ (and $69.6\%$ at $10\,\text{mm}$), Northing and Easting shifts share the exact same shape category.
   - *Impact*: Delivers `CellShapeFeatureExtractor` providing $O(1)$ spatial feature lookups, providing downstream neural networks and hybrid models with localized scale-free $(u, v)$ coordinates, zero-offset base shifts, and functional shape priors.

---

## Phase 6: Algorithmic Boundary Continuity & Extreme Scenario Verification

9. **Bicubic Cell Boundary & Extreme Scenarios Analysis (icubic_boundary_analysis)**:
   - *Question*: Are the visible slope kinks along \,\text{km}$ cell boundaries in 1D transect plots an implementation bug in pytransdat, or are they mathematically inherent to official TransDatRO? Furthermore, how does the true bicubic curve behave under extreme geometric conditions (S-curves, bowls, domes, flat-sag bridges, and steepest gradient shifts)?
   - *Finding*:
     - **Verification of Official Specification**: Comparing pytransdat/trans_grid.py against Avramiuc, Dragomir, & Rus (2009, p. 112) confirms that Grid.BiInterp matches official TransDatRO verbatim. Boundary partial derivatives (, f_E$) are calculated using **one-sided 3-point finite-difference stencils** (forward at 0, backward at 1). Because adjacent cells evaluate different boundary tangents, ^0$ coordinate continuity is exact, but ^1$ first-derivative slope kinks are **inherent to the specification**.
     - **Mathematical S-Shape Discovery**: Formulating the S-shape criterion via second-derivative zero-crossing (''(u_{\text{inf}}) = 0$ with {\text{inf}} \in [0.2, 0.8]$ and opposite curvature signs at endpoints) isolated a genuine sigmoidal curve (
ow 23, col 36, dN, {\text{inf}} = 0.48$).
     - **True Vertical-Scale Magnification**: Unconstrained polynomial extrapolation over 5 cells ( \in [-2, 3]$) blows up cubic terms to $> 20\,\text{m}$, which naively squashes the true \,\text{m}$ ground-truth curve. Clamping the vertical axis to true curve and node values (\%$ margin) magnifies the internal S-bend, revealing its true sigmoidal geometry.
     - **Grouped Extreme Scenarios Taxonomy**: Established a 7-scenario suite evaluated across 5-cell extended windows, split into **Group 1 (Gradient & Linear Extremes)** (Steepest Fall of $-0.536\,\text{m}$, Steepest Easting Rise of $+0.453\,\text{m}$, and Flat-Sag Bowing of .0\,\text{mm}$ between identical nodes) and **Group 2 (Curvature & Kinks Extremes)** (True S-Shape, .7\,\text{mm}$ Bowl Trough, .1\,\text{mm}$ Dome Crest, and .788\,\text{m/step}$ worst-case ^1$ boundary crease).
   - *Impact*: Confirms ground-truth reference fidelity and proves that continuous global ML models will naturally smooth over Romania's official \,\text{km}$ slope creases unless equipped with localized cell priors or micro-grid feature representations.

10. **Bicubic Interpolation Forensics, Origins & Alternative 11 km Schemes ([`bicubic_interpolation_forensics`](bicubic_interpolation_forensics.md))**:
    - *Question*: Where did `TransDatRO`'s non-standard one-sided finite-difference bicubic formula come from, which alternative $11\,\text{km}$ schemes eliminate the $C^1$ kinks, and how severely do those kinks penalize smooth ML models?
    - *Finding*:
      - **Master Spec & Origins**: [`Help_TransDatRO_code_source_EN.pdf`](../../articles/Help_TransDatRO_code_source_EN.pdf) is the unabridged master spec (`2.22e-15 m` parity with `pytransdat`). Forensic comparison shows `TransDatRO` combined English Wikipedia's 16-equation bicubic matrix (pasted `texvc` screenshots in Avramiuc et al., 2009) with textbook 3-point edge finite differences (`-3, +4, -1`) on every interior cell wall—producing the "Swapped Parabola" blend and fourth-difference border slope breaks up to `1.071 m/step`.
      - **Top 3 Alternative $11\,\text{km}$ Schemes**: Implemented in [`src/models/grid_bicubic_variants.py`](../../src/models/grid_bicubic_variants.py): **#1 Catmull-Rom ($C^1$, `16.75 mm` RMS vs. `TransDatRO`)**, **#2 PCHIP ($C^1$ monotone, `0.37 mm` bowing)**, and **#3 Natural Cubic Spline ($C^2$, `18.02 mm` RMS)**.
      - **Quantified ML Learnability Penalty**: Smooth Thin-Plate Spline RBF interpolation achieves `0.667 mm` RMSE on `Natural Spline C²` and `1.364 mm` on `Catmull-Rom C¹`, versus `7.674 mm` (`35.498 mm` max) on `TransDatRO`—a **5.6× to 11.5× penalty**.
    - *Impact*: Establishes why smooth global ML models hit a millimeter-to-centimeter wall on `TransDatRO`, validates cell-aware / `(u, v)` architectures, and provides reusable smooth $11\,\text{km}$ targets for future ML benchmarking.

11. **Exhaustive 10-Family Stage-1 Surface Learnability Benchmark ([`optimal_surface_regressor`](optimal_surface_regressor.md))**:
    - *Question*: Which smooth Stage-1 base surface family trained on `RO_GRID_4X4` best removes the bulk `.spg` distortion shift without memorizing legacy 11 km boundary kinks—leaving low-amplitude, low-roughness residuals for Stage-2 ML models?
    - *Finding*:
      - **Learnability-First Ranking**: Evaluating 10 families via Stage-0 subgrid calibration + Stage-1 screening (176 configs) + Stage-2 Top-3 Full-Romania promotion revealed that **Polyharmonic `cubic_k25_s0`** (`16.015 mm` Overall), **Thin Plate Spline `tps_deg2_k25_s1e-5`** (`16.531 mm`, #1 on `16x16`), and **Local Quadratic LOESS `lpoly_deg2_k16_tric`** (`16.696 mm`, 3× faster inference) form the Top-3 Stage-1 base surfaces.
      - **Methodological Rules**: Always fit on Full `RO_GRID_4X4` (`0.006s`) to avoid coordinate-normalization scale shifts, and use contiguous `CellRegion` subgrids rather than point-thinning to preserve `Roughness Grad RMS`.
    - *Impact*: Locks in the Top-3 Stage-1 base surfaces for Stage-2 residual ML training.

---

## Phase 7: Two-Stage Surface + Neural Residual Hybrids

12. **Stage-1 Smooth Surface + Stage-2 Neural Feature Grid ([`hybrid_surface_featgrid`](hybrid_surface_featgrid.md))**
    - *Question*: When we pair the Top-3 `RO_GRID_4X4` Stage-1 surfaces (`cubic_k25_s0`, `tps_deg2_k25_s1e-5`, `lpoly_deg2_k16_tric`) with `NeuralFeatureGridShiftModel`, what training density, MLP width, loss function, and harmonic order maximize sub-millimeter generalization—and how does the winner behave across Full Romania?
    - *Finding*:
      - **Subgrid Matrix (`box10_interior`)**: Training Stage 2 on `4x4` fails (`7.6–11.2 mm`) because Stage 1 already interpolates `4x4`. Scaling to `RO_GRID_32X32` with `huber` loss (`beta=5 mm`) unlocks `hidden_dim=128` (`0.412 mm` vs. `0.936 mm` under `mse`), while `cyclic_cell_harmonics=2` generalizes better than `4` (`0.412 mm` vs. `0.676 mm`). `cubic_k25_s0` ranked #1 (`0.351 mm` mean / `0.412 mm` RMSE / `98.87% < 1 mm` on unseen `Adapt 1mm`), vindicating `optimal_surface_regressor`'s roughness-based ranking.
      - **Full-Romania Interior vs. Border Trade-Off**: `Hybrid_cubic_k25_s0_plus_featgrid_32x32` delivers **`0.24 mm` mean (`99.3% < 1 mm`, `P99 = 0.9 mm`)** on `16x16` and `0.1–0.6 mm` across 9-cell extreme scenarios, **but** `cubic_k25_s0`'s `~0.65–0.69 m` concave-border extrapolation spikes—down-weighted by `huber` loss—inflate Full-Romania `Adapt 1mm` `RMSE` (`14.84 mm` vs. `5.42 mm` for standalone `FeatGrid_32x32_only`).
    - *Impact*: Establishes the `32x32` + `Huber` + `h128` recipe for sub-millimeter interior accuracy while identifying Stage-1 border extrapolation as the key challenge for future hybrid architectures.
