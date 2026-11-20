# Unit 05: Decision Tree Spatial Regressor

## 1. Objective & Hypothesis
Evaluate how multi-output `DecisionTreeRegressor` (`(N, E) -> (dN, dE)`) models the 2D Romanian distortion grid across increasing spatial sampling densities (11 km base nodes, 2.75 km 4x4 grid, 1.375 km 8x8 grid) and impurity-based compression.

## 2. Benchmark Scorecard

| Model | Leaves | Depth | Size (MB) | Train RMSE (m) | Grid 8x8 RMSE (m) | Corridor RMSE (m) | Corridor Max (m) | Fit Time (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| dt_base_11km | 2689 | 22 | 0.4100 | 0.0000 | 0.0817 | 0.0678 | 0.2946 | 0.0070 |
| dt_4x4_2.75km | 36435 | 30 | 5.5600 | 0.0000 | 0.0246 | 0.0161 | 0.0910 | 0.0600 |
| dt_8x8_1.375km | 146258 | 36 | 22.3200 | 0.0000 | 0.0000 | 0.0080 | 0.0465 | 0.3420 |
| dt_8x8_pruned_1e8 | 18424 | 27 | 2.8100 | 0.0176 | 0.0176 | 0.0188 | 0.0851 | 0.2090 |

## 3. Key Findings
1. **Exact Nodal Memorization**: Unconstrained decision trees achieve $0.0000\,\text{m}$ error on every training dataset, creating exactly 1 leaf per unique floating-point training point.
2. **Step-Plateau Interior Error**: On the 11 km base grid, orthogonal axis-aligned cuts create flat rectangular plateaus with $0.0844\,\text{m}$ ($8.44\,\text{cm}$) interior error on 4x4 and $0.0817\,\text{m}$ on 8x8—more than double KNN $K=4$ IDW ($0.0353\,\text{m}$).
3. **Densification Scaling**: Training on 4x4 ($36,435$ pts) drops unseen 8x8 error to $0.0246\,\text{m}$ ($2.46\,\text{cm}$). Training on 8x8 ($146,258$ pts) achieves **$0.0080\,\text{m}$ ($0.80\,\text{cm}$)** on the continuous corridor, at the cost of $22.32\,\text{MB}$ model size.
4. **Impurity Pruning Compression**: Setting `min_impurity_decrease=1e-8` compresses the 8x8 tree from $146,258$ leaves ($22.32\,\text{MB}$) down to $18,424$ leaves ($2.81\,\text{MB}$ — an $87.4\%$ reduction) while maintaining $0.0188\,\text{m}$ ($1.88\,\text{cm}$) corridor RMSE.
