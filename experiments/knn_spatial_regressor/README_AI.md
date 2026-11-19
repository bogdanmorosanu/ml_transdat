# AI Summary: Research Unit 04 (knn_spatial_regressor)

- **Slug**: `knn_spatial_regressor`
- **Family**: Spatial Partition / K-Nearest Neighbors (KNN) Regressor
- **Target**: 2D Grid Shift Transformation $(N, E) \mapsto (\delta N, \delta E)$
- **Status**: Completed / Validated
- **Key Metrics (K=4 Distance)**:
  - **Train Nodes RMSE**: 0.000000 m
  - **Val 4x4 RMSE**: 0.0353 m
  - **Val 4x4 Max**: 0.4783 m
  - **Val 4x4 P95**: 0.0679 m
  - **Test 8x8 RMSE**: 0.0346 m
  - **Pass <1cm**: 19.3%
- **Findings**:
  1. $K=4$ Distance-Weighted is the empirical sweet spot (3.5 cm RMSE on 4x4, 3.4 cm on 8x8).
  2. Distance weighting decisively beats uniform averaging across all K values (e.g. 3.5 cm vs 6.0 cm at K=4).
  3. K=8 and K=16 distance-weighted underperform K=4 because positive-only IDW weights over-smooth local gradients across multiple cells.
