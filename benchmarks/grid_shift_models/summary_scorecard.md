# Cross-Model Grid Shift Benchmark Summary

Precision evaluated on `ro_corridor_snw` (36,435 points). Compares validated champions across KNN, Decision Trees, Adaptive Quadtrees, Fourier MLP, PyTorch ResNet, and Thin Plate Splines with exact source-experiment hyperparameters. All error metrics are reported in **millimeters (mm)**.

## 1. Comprehensive Precision Scorecard (mm)

| Model | Mean 2D Error (mm) | Std Dev 2D (mm) | RMSE 2D Error (mm) | Min 2D Error (mm) | P50 2D Error (mm) | P90 2D Error (mm) | P95 2D Error (mm) | P99 2D Error (mm) | Max 2D Error (mm) | Signed Bias dN (mm) | Signed Bias dE (mm) | Pass < 1 mm (%) | Pass < 1 cm (%) | Pass < 5 cm (%) | Pass < 10 cm (%) | Pass < 1 m (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| KNN (K=4 IDW, 11km) | 24.03 | 15.40 | 28.54 | 0.03 | 21.11 | 44.17 | 53.36 | 73.41 | 153.57 | -1.19 | -1.33 | 0.33 | 16.70 | 93.47 | 99.88 | 100.00 |
| KNN (K=16 IDW, 11km) | 65.72 | 40.61 | 77.25 | 0.10 | 58.20 | 119.77 | 142.50 | 192.31 | 282.43 | -4.45 | -4.37 | 0.01 | 2.67 | 40.88 | 82.13 | 100.00 |
| Decision Tree (11km) | 56.01 | 38.23 | 67.81 | 0.14 | 47.80 | 108.06 | 128.93 | 177.96 | 294.62 | -0.15 | 0.24 | 0.07 | 4.98 | 52.26 | 87.30 | 100.00 |
| Decision Tree (8x8, pruned 1e-8) | 16.12 | 9.72 | 18.82 | 0.06 | 14.33 | 29.33 | 34.56 | 45.77 | 85.14 | 0.00 | -0.01 | 0.43 | 30.32 | 99.46 | 100.00 | 100.00 |
| Adaptive Quadtree (3mm tier) | 1.43 | 1.05 | 1.77 | 0.00 | 1.25 | 2.64 | 3.08 | 4.49 | 26.32 | -0.02 | -0.03 | 39.17 | 99.91 | 100.00 | 100.00 | 100.00 |
| Adaptive Quadtree (1mm tier) | 0.50 | 0.34 | 0.60 | 0.00 | 0.44 | 0.90 | 1.05 | 1.59 | 5.77 | -0.00 | -0.01 | 93.76 | 100.00 | 100.00 | 100.00 | 100.00 |
| Fourier MLP (4x4, 2.7km) | 8.83 | 5.39 | 10.35 | 0.04 | 7.73 | 16.04 | 19.48 | 26.09 | 36.19 | -2.45 | 1.19 | 1.16 | 66.52 | 100.00 | 100.00 | 100.00 |
| Thin Plate Spline (11km) | 6.74 | 4.41 | 8.05 | 0.05 | 5.83 | 12.47 | 15.36 | 21.53 | 33.75 | 0.09 | -0.14 | 2.20 | 81.45 | 100.00 | 100.00 | 100.00 |
| Thin Plate Spline (4x4, 2.7km) | 3.49 | 2.46 | 4.27 | 0.01 | 2.88 | 6.74 | 8.35 | 11.81 | 21.76 | 0.04 | 0.09 | 10.01 | 97.54 | 100.00 | 100.00 | 100.00 |
| PyTorch ResNet + Cyclic CellPos (4x4) | 3.91 | 2.62 | 4.70 | 0.00 | 3.34 | 7.27 | 8.93 | 12.90 | 24.95 | 0.03 | 0.13 | 6.71 | 96.84 | 100.00 | 100.00 | 100.00 |

## 2. Performance Speed & Throughput Across Named Workloads

### Workload: `corridor_36k` (`ro_corridor_snw`, 36,435 pts)

| Model | Evaluation Time (s) | Throughput (pts/s) | Latency (µs/pt) | Fit Time (s) |
| :--- | :---: | :---: | :---: | :---: |
| KNN (K=4 IDW, 11km) | 0.0524 | 695,491.7 | 1.4378 | 0.0046 |
| KNN (K=16 IDW, 11km) | 0.0761 | 478,506.3 | 2.0898 | 0.0015 |
| Decision Tree (11km) | 0.0018 | 20,738,232.2 | 0.0482 | 0.0082 |
| Decision Tree (8x8, pruned 1e-8) | 0.0025 | 14,300,011.8 | 0.0699 | 0.2365 |
| Adaptive Quadtree (3mm tier) | 0.0143 | 2,547,884.3 | 0.3925 | 0.3159 |
| Adaptive Quadtree (1mm tier) | 0.0181 | 2,018,246.5 | 0.4955 | 1.1031 |
| Fourier MLP (4x4, 2.7km) | 0.1320 | 275,978.4 | 3.6235 | 106.6137 |
| Thin Plate Spline (11km) | 1.5079 | 24,163.5 | 41.3848 | 1.3156 |
| Thin Plate Spline (4x4, 2.7km) | 2.3076 | 15,789.1 | 63.3349 | 0.0092 |
| PyTorch ResNet + Cyclic CellPos (4x4) | 0.2787 | 130,723.1 | 7.6498 | 115.0567 |

### Workload: `uniform_4x4` (`ro_grid_4x4`, 36,435 pts)

| Model | Evaluation Time (s) | Throughput (pts/s) | Latency (µs/pt) | Fit Time (s) |
| :--- | :---: | :---: | :---: | :---: |
| KNN (K=4 IDW, 11km) | 0.0410 | 888,999.0 | 1.1249 | 0.0046 |
| KNN (K=16 IDW, 11km) | 0.0794 | 459,049.1 | 2.1784 | 0.0015 |
| Decision Tree (11km) | 0.0026 | 13,881,586.4 | 0.0720 | 0.0082 |
| Decision Tree (8x8, pruned 1e-8) | 0.0034 | 10,861,853.1 | 0.0921 | 0.2365 |
| Adaptive Quadtree (3mm tier) | 0.0171 | 2,128,983.6 | 0.4722 | 0.3159 |
| Adaptive Quadtree (1mm tier) | 0.0207 | 1,761,958.0 | 0.5706 | 1.1031 |
| Fourier MLP (4x4, 2.7km) | 0.1329 | 274,063.0 | 3.6488 | 106.6137 |
| Thin Plate Spline (11km) | 1.5712 | 23,189.2 | 43.1235 | 1.3156 |
| Thin Plate Spline (4x4, 2.7km) | 2.6996 | 13,496.4 | 74.0937 | 0.0092 |
| PyTorch ResNet + Cyclic CellPos (4x4) | 0.2702 | 134,855.5 | 7.4153 | 115.0567 |

### Workload: `single_cell_10k` (`single_cell_10k`, 10,000 pts)

| Model | Evaluation Time (s) | Throughput (pts/s) | Latency (µs/pt) | Fit Time (s) |
| :--- | :---: | :---: | :---: | :---: |
| KNN (K=4 IDW, 11km) | 0.0215 | 465,077.3 | 2.1502 | 0.0046 |
| KNN (K=16 IDW, 11km) | 0.0348 | 287,001.7 | 3.4843 | 0.0015 |
| Decision Tree (11km) | 0.0007 | 14,781,966.0 | 0.0676 | 0.0082 |
| Decision Tree (8x8, pruned 1e-8) | 0.0007 | 14,725,372.0 | 0.0679 | 0.2365 |
| Adaptive Quadtree (3mm tier) | 0.0046 | 2,196,933.1 | 0.4552 | 0.3159 |
| Adaptive Quadtree (1mm tier) | 0.0054 | 1,836,176.3 | 0.5446 | 1.1031 |
| Fourier MLP (4x4, 2.7km) | 0.0352 | 284,431.9 | 3.5158 | 106.6137 |
| Thin Plate Spline (11km) | 0.4068 | 24,582.7 | 40.6791 | 1.3156 |
| Thin Plate Spline (4x4, 2.7km) | 0.2373 | 42,134.3 | 23.7336 | 0.0092 |
| PyTorch ResNet + Cyclic CellPos (4x4) | 0.0250 | 400,623.4 | 2.4961 | 115.0567 |

## 3. Storage & Memory Breakdown (4 Components)

Components:
- **Code Size**: Model implementation file on disk.
- **Library Dependencies**: Installed package footprint of external runtime dependencies.
- **Model Instance / Weights**: In-memory parameters, split trees, or index metadata.
- **Grid Lookup Data**: Retained control nodes or indexed cell tables.

| Model | Code Size (bytes) | Dependencies Size (bytes) | Model Weights/Instance Size (bytes) | Grid Lookup Size (bytes) | Total Footprint (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| KNN (K=4 IDW, 11km) | 3,075 | 41,029,584 | 30,809 | 86,048 | 39.2432 |
| KNN (K=16 IDW, 11km) | 3,075 | 41,029,584 | 30,811 | 86,048 | 39.2432 |
| Decision Tree (11km) | 4,184 | 41,029,584 | 431,362 | 0 | 39.5442 |
| Decision Tree (8x8, pruned 1e-8) | 4,184 | 41,029,584 | 2,948,962 | 0 | 41.9452 |
| Adaptive Quadtree (3mm tier) | 12,444 | 0 | 9,040 | 24,148,512 | 23.0503 |
| Adaptive Quadtree (1mm tier) | 12,444 | 0 | 9,814 | 78,529,320 | 74.9126 |
| Fourier MLP (4x4, 2.7km) | 12,550 | 73,428,937 | 2,148,272 | 0 | 72.0880 |
| Thin Plate Spline (11km) | 6,111 | 123,489,234 | 151,515 | 0 | 117.9188 |
| Thin Plate Spline (4x4, 2.7km) | 6,111 | 123,489,234 | 3,032,257 | 0 | 120.6661 |
| PyTorch ResNet + Cyclic CellPos (4x4) | 17,815 | 4,280,113,426 | 2,321,091 | 0 | 4084.0648 |
