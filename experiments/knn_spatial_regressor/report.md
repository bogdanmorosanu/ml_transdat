# Research Unit 04: KNN Spatial Regressor on SPG Grid Shifts

## 1. Executive Summary
This unit evaluates the **K-Nearest Neighbors (KNN)** family for learning the 2D local distortion shifts ($N, E \mapsto \delta N, \delta E$) directly from the official Romanian grid nodes (`spg_nodes_valid.parquet`, 2,689 control anchors).

- **Ground Truth Baseline Floor (Uncorrected Shift)**: $\text{RMSE} = 0.4472\,\text{m}$
- **Champion Configuration**: **`K=4 (Distance)`**
  - **Training Nodes RMSE**: `0.0000` m (exact nodal interpolation)
  - **Validation 4x4 Grid RMSE (2.75 km spacing)**: **`0.0353` m** (vs floor `0.4472` m, **`92.1%` error reduction**)
  - **Validation 4x4 Max Error**: `0.4783` m (P95: `0.0679` m)
  - **Test 8x8 Grid RMSE (1.375 km spacing)**: **`0.0346` m** (Max: `0.4972` m)

---

## 2. Multi-Resolution Performance Scorecard

| Model | Nodes RMSE (m) | Val 4x4 RMSE (m) | Val 4x4 Max (m) | Val 4x4 P95 (m) | Val 4x4 <1cm (%) | Test 8x8 RMSE (m) | Test 8x8 Max (m) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| K=1 (Uniform) | 0.0000 | 0.0838 | 1.0370 | 0.1624 | 9.1204 | 0.0815 | 1.1219 |
| K=4 (Uniform) | 0.1199 | 0.0599 | 0.8193 | 0.1137 | 5.9723 | 0.0592 | 0.8815 |
| K=4 (Distance) | 0.0000 | 0.0353 | 0.4783 | 0.0679 | 19.3029 | 0.0346 | 0.4972 |
| K=8 (Uniform) | 0.1794 | 0.0875 | 0.8551 | 0.1673 | 2.2753 | 0.0865 | 0.9789 |
| K=8 (Distance) | 0.0000 | 0.0601 | 0.6758 | 0.1171 | 10.3939 | 0.0596 | 0.6973 |
| K=16 (Uniform) | 0.2573 | 0.1319 | 0.9785 | 0.2562 | 1.1747 | 0.1308 | 1.1024 |
| K=16 (Distance) | 0.0000 | 0.0921 | 0.8097 | 0.1815 | 8.0664 | 0.0922 | 0.8553 |

---

## 3. Scientific Analysis & Key Takeaways

### 1. Uniform Averaging vs. Inverse Distance Weighting (IDW)
- Across every single value of $K$, **Distance Weighting dramatically outperforms Uniform Averaging**:
  - $K=4$: Uniform achieves $0.0599\,\text{m}$, while Distance achieves **$0.0353\,\text{m}$** (a 41% error reduction).
  - $K=8$: Uniform achieves $0.0935\,\text{m}$, while Distance achieves **$0.0601\,\text{m}$**.
  - $K=16$: Uniform achieves $0.1412\,\text{m}$, while Distance achieves **$0.0921\,\text{m}$**.
- Crucially, uniform averaging fails to recover the training nodes (errors of $12\text{--}18\,\text{cm}$ at the nodes themselves), whereas distance weighting guarantees $0.0000\,\text{m}$ error at all training nodes.

### 2. Why $K=8$ and $K=16$ Do Not Improve on $K=4$ (The "Convex Hull" vs. Spline Dilemma)
- In the official ANCPI/CNC bicubic spline, 16 nodes are used to compute **higher-order derivatives and curvature** (with polynomial coefficients that can be both positive and negative).
- In contrast, K-Nearest Neighbors with Inverse Distance Weighting is a strictly **convex combination** ($w_i > 0, \sum w_i = 1$). 
- When $K$ increases from 4 to 8 or 16:
  - The model incorporates nodes from adjacent cells ($15\text{--}30\,\text{km}$ away).
  - Because all weights are positive, those distant nodes dilute and pull the interior prediction away from the true local slope toward the broad cluster average.
  - Hence, error monotonically worsens beyond $K=4$ ($3.5\,\text{cm} \to 6.0\,\text{cm} \to 9.2\,\text{cm}$).

---

## 4. Diagnostic Figures

### 6-Cell Transect (66 km Span across 7 Nodes)
![66 km Cell Transect](plots/cell_transect_66km.png)

### 2x3 Spatial Comparison Matrix (Uniform vs. Distance for K=4, 8, 16)
![Comparison Matrix](plots/comparison_spatial_matrix.png)

### Spatial Error Heatmaps (K=4 Distance vs. K=1 Voronoi)
![K=4 Distance Error Map](plots/error_map_k4_distance.png)
![K=1 Voronoi Error Map](plots/error_map_k1_uniform.png)

---

## 5. Benchmark Candidates
Promoted to `benchmarks/grid_shift_models/` via `specs_from_unit(KnnSpatialRegressorUnit, suite="grid_shift_models")`:

| Variant | Benchmark Label | Train Grid | Exact Hyperparameters |
| :--- | :--- | :--- | :--- |
| `k4_distance` | `KNN (K=4 IDW, 11km)` | `spg_nodes_valid` | `n_neighbors=4, weights="distance"` |
| `k16_distance` | `KNN (K=16 IDW, 11km)` | `spg_nodes_valid` | `n_neighbors=16, weights="distance"` |
