# Unit 05: Decision Tree Spatial Regressor

## Hypothesis
An unconstrained multi-output `DecisionTreeRegressor` will achieve `0.0000 m` error on training nodes (`spg_nodes_valid.parquet`) by creating 1 leaf per unique floating-point node, while producing flat rectangular step-plateaus inside grid cells. Densifying the training grid (`4x4`, `8x8`, `16x16`) will halve the error per halving of grid spacing at the cost of $4\times$ model size.

## Results
- **Base 11 km (2,689 pts)**: `0.0000 m` train, `0.0817 m` on `8x8`, `0.0678 m` on corridor (`0.41 MB`).
- **4x4 (36,435 pts)**: `0.0246 m` on `8x8`, `0.0161 m` on corridor (`5.56 MB`).
- **8x8 (146,258 pts)**: `0.0080 m` on corridor (`22.32 MB`).
- **16x16 (586,063 pts)**: `0.0040 m` (`4.01 mm`) on corridor (`89.43 MB`).
- **Impurity Pruning (`1e-8`)**: Compresses `8x8` and `16x16` to `~2.81 MB` (~18,400 leaves) with `0.0188 m` corridor RMSE.

## Verdict
**Validated as Scaling Baseline / Step-Lookup Upper Bound.** Decision trees cannot interpolate smoothly between nodes; they act as orthogonal lookup tables. Achieving sub-centimeter accuracy requires $\ge 146\text{k}$ points ($22\text{--}89\text{ MB}$), motivating ensembles (Random Forest / GBDT) or smooth interpolators (RBF / Neural Networks) next.
