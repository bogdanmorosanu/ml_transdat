# Task: KNN Spatial Regressor Experiment

**Date**: 2026-11-19  
**Branch**: `exp/knn_spatial_regressor`  
**Status**: Completed  
**Scope**: Model Architecture, Training & Benchmark Evaluation  

## Objective
Develop, verify, and benchmark our first Machine Learning baseline model—the **K-Nearest Neighbors (KNN) Spatial Regressor**—specifically designed to learn the 2D local distortion shifts ($(N, E) \mapsto (\delta N, \delta E)$) directly from the official Romanian grid control nodes (`spg_nodes_valid.parquet`, 2,689 anchors) rather than performing an end-to-end geographic-to-grid transformation.

---

## Key Activities & Implementation

1. **Reusable Spatial Estimator (`src/models/knn_regressor.py`)**:
   - Implemented `KNNSpatialRegressor`, a modular, typed wrapper around `sklearn.neighbors.KNeighborsRegressor` using a 2D Euclidean KDTree.
   - Supports configurable neighborhood sizes ($K$) and weighting schemes (`'uniform'` vs. `'distance'`).
   - Integrated and exported in `src/models/__init__.py`.

2. **Automated Unit Testing (`tests/test_knn_regressor.py`)**:
   - Built a 5-test suite verifying exact nodal interpolation for distance-weighted lookups ($\text{residual} = 0.0\,\text{m}$), boundary guards, input validation, and symmetrical cell-center averaging.
   - All 5 tests passed in $0.005\,\text{s}$.

3. **SPG Binary Band Order Bugfix (`scripts/generate_grids.py`)**:
   - Discovered that in `rom_grid3d_25.09.spg`, Band 0 stores Easting shifts ($dE$) and Band 1 stores Northing shifts ($dN$).
   - Corrected the inverted column assignment in `generate_spg_nodes()` and regenerated `data/grids/spg_nodes_valid.parquet` and `data/grids/spg_nodes_valid.meta.json`, bringing canonical control nodes into exact numerical sync with `Grid2D`.

4. **Experiment Unit Execution (`experiments/knn_spatial_regressor/`)**:
   - Subclassed `src.core.base_unit.ResearchUnit` (Unit ID: `04`).
   - Evaluated candidate configurations across $K \in \{1, 4, 8, 16\}$ under both Uniform and Inverse Distance Weighting (IDW).
   - Tested across three canonical multi-resolution tiers:
     - **Tier 1 (Nodes / Train)**: `spg_nodes_valid.parquet` ($2,689$ points).
     - **Tier 2 (Val 4x4 / Interior)**: `ro_grid_4x4.parquet` ($36,435$ points, $2.75\,\text{km}$ cell spacing).
     - **Tier 3 (Test 8x8 / Generalization)**: `ro_grid_8x8.parquet` ($146,258$ points, $1.375\,\text{km}$ cell spacing).
   - Generated complete diagnostic suites: 66 km 6-cell transect across 7 nodes, $2 \times 3$ comparison matrix, spatial error heatmaps, ECDFs, quiver vectors, and subcell error profiles.

---

## Verification & Results

### Multi-Resolution Performance Scorecard

| Model Configuration | Training Nodes RMSE | Val 4x4 RMSE ($2.75\,\text{km}$) | Val 4x4 Max Error | Val 4x4 $P_{95}$ | Val 4x4 $<1\,\text{cm}$ | Test 8x8 RMSE ($1.375\,\text{km}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| *Baseline Floor (Uncorrected)* | $0.4472\,\text{m}$ | $0.4472\,\text{m}$ | $2.7570\,\text{m}$ | $0.9840\,\text{m}$ | $0.0\%$ | $0.4472\,\text{m}$ |
| **K=1 (Uniform / Voronoi)** | **$0.0000\,\text{m}$** | $0.0838\,\text{m}$ | $1.0370\,\text{m}$ | $0.1624\,\text{m}$ | $9.1\%$ | $0.0815\,\text{m}$ |
| **K=4 (Uniform Averaging)** | $0.1199\,\text{m}$ | $0.0599\,\text{m}$ | $0.8193\,\text{m}$ | $0.1137\,\text{m}$ | $6.0\%$ | $0.0592\,\text{m}$ |
| **K=4 (Distance / IDW) 🏆** | **$0.0000\,\text{m}$** | **$0.0353\,\text{m}$** | **$0.4783\,\text{m}$** | **$0.0679\,\text{m}$** | **$19.3\%$** | **$0.0346\,\text{m}$** |
| **K=8 (Uniform Averaging)** | $0.1794\,\text{m}$ | $0.0875\,\text{m}$ | $0.8551\,\text{m}$ | $0.1673\,\text{m}$ | $2.3\%$ | $0.0865\,\text{m}$ |
| **K=8 (Distance / IDW)** | **$0.0000\,\text{m}$** | $0.0601\,\text{m}$ | $0.6758\,\text{m}$ | $0.1171\,\text{m}$ | $10.4\%$ | $0.0596\,\text{m}$ |
| **K=16 (Uniform Averaging)** | $0.2573\,\text{m}$ | $0.1319\,\text{m}$ | $0.9785\,\text{m}$ | $0.2562\,\text{m}$ | $1.2\%$ | $0.1308\,\text{m}$ |
| **K=16 (Distance / IDW)** | **$0.0000\,\text{m}$** | $0.0921\,\text{m}$ | $0.8097\,\text{m}$ | $0.1815\,\text{m}$ | $8.1\%$ | $0.0922\,\text{m}$ |

### Key Findings
1. **Massive Floor Beating**: $K=4$ Distance achieves **$0.0353\,\text{m}$ ($3.5\,\text{cm}$)** RMSE, delivering a **$92.1\%$ error reduction** over the baseline floor ($0.4472\,\text{m}$).
2. **Optimal Stencil Size ($K=4$)**: Increasing $K$ to 8 or 16 degrades performance because positive-only IDW weights pull interior predictions toward distant nodes ($15\text{--}30\,\text{km}$ away), over-smoothing local gradients.
3. **Regression Suite**: Full test suite passes cleanly ($13/13$ tests OK).
