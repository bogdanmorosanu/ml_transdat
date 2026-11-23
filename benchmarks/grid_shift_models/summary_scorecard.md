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
| PyTorch ResNet + Cyclic CellPos (Adaptive Hybrid) | 2.57 | 1.70 | 3.08 | 0.01 | 2.24 | 4.68 | 5.70 | 8.42 | 17.96 | 0.05 | 0.03 | 14.55 | 99.61 | 100.00 | 100.00 | 100.00 |
| Bilinear + Neural Feature Grid (8x8) | 0.71 | 0.46 | 0.84 | 0.00 | 0.61 | 1.32 | 1.59 | 2.20 | 4.44 | -0.04 | 0.01 | 78.30 | 100.00 | 100.00 | 100.00 | 100.00 |

## 2. Performance Speed & Throughput Across Named Workloads

### Workload: `corridor_36k` (`ro_corridor_snw`, 36,435 pts)

| Model | Evaluation Time (s) | Throughput (pts/s) | Latency (µs/pt) | Fit Time (s) |
| :--- | :---: | :---: | :---: | :---: |
| KNN (K=4 IDW, 11km) | 0.0371 | 981,623.0 | 1.0187 | 0.0011 |
| KNN (K=16 IDW, 11km) | 0.0775 | 470,249.8 | 2.1265 | 0.0011 |
| Decision Tree (11km) | 0.0046 | 7,958,715.6 | 0.1256 | 0.0078 |
| Decision Tree (8x8, pruned 1e-8) | 0.0027 | 13,668,592.5 | 0.0732 | 0.1999 |
| Adaptive Quadtree (3mm tier) | 0.0133 | 2,745,480.7 | 0.3642 | 0.3625 |
| Adaptive Quadtree (1mm tier) | 0.0176 | 2,064,446.3 | 0.4844 | 1.1004 |
| Fourier MLP (4x4, 2.7km) | 0.1824 | 199,739.4 | 5.0065 | 103.2459 |
| Thin Plate Spline (11km) | 1.9109 | 19,066.7 | 52.4475 | 1.2529 |
| Thin Plate Spline (4x4, 2.7km) | 3.2135 | 11,338.1 | 88.1984 | 0.0086 |
| PyTorch ResNet + Cyclic CellPos (4x4) | 0.3975 | 91,665.0 | 10.9093 | 118.1399 |
| PyTorch ResNet + Cyclic CellPos (Adaptive Hybrid) | 0.2831 | 128,698.9 | 7.7701 | 343.2767 |
| Bilinear + Neural Feature Grid (8x8) | 0.0371 | 981,073.2 | 1.0193 | 310.9969 |

### Workload: `uniform_4x4` (`ro_grid_4x4`, 36,435 pts)

| Model | Evaluation Time (s) | Throughput (pts/s) | Latency (µs/pt) | Fit Time (s) |
| :--- | :---: | :---: | :---: | :---: |
| KNN (K=4 IDW, 11km) | 0.0705 | 517,141.5 | 1.9337 | 0.0011 |
| KNN (K=16 IDW, 11km) | 0.1138 | 320,123.4 | 3.1238 | 0.0011 |
| Decision Tree (11km) | 0.0028 | 13,124,527.2 | 0.0762 | 0.0078 |
| Decision Tree (8x8, pruned 1e-8) | 0.0039 | 9,419,596.7 | 0.1062 | 0.1999 |
| Adaptive Quadtree (3mm tier) | 0.0293 | 1,241,820.0 | 0.8095 | 0.3625 |
| Adaptive Quadtree (1mm tier) | 0.0208 | 1,752,921.5 | 0.5735 | 1.1004 |
| Fourier MLP (4x4, 2.7km) | 0.1607 | 226,724.6 | 4.4106 | 103.2459 |
| Thin Plate Spline (11km) | 1.8442 | 19,756.2 | 50.6170 | 1.2529 |
| Thin Plate Spline (4x4, 2.7km) | 3.6233 | 10,055.7 | 99.4465 | 0.0086 |
| PyTorch ResNet + Cyclic CellPos (4x4) | 0.2423 | 150,365.2 | 6.6505 | 118.1399 |
| PyTorch ResNet + Cyclic CellPos (Adaptive Hybrid) | 0.2778 | 131,136.3 | 7.6257 | 343.2767 |
| Bilinear + Neural Feature Grid (8x8) | 0.0118 | 3,088,025.9 | 0.3238 | 310.9969 |

### Workload: `single_cell_10k` (`single_cell_10k`, 10,000 pts)

| Model | Evaluation Time (s) | Throughput (pts/s) | Latency (µs/pt) | Fit Time (s) |
| :--- | :---: | :---: | :---: | :---: |
| KNN (K=4 IDW, 11km) | 0.0227 | 441,027.9 | 2.2674 | 0.0011 |
| KNN (K=16 IDW, 11km) | 0.0404 | 247,578.1 | 4.0391 | 0.0011 |
| Decision Tree (11km) | 0.0007 | 14,291,839.3 | 0.0700 | 0.0078 |
| Decision Tree (8x8, pruned 1e-8) | 0.0007 | 14,210,601.0 | 0.0704 | 0.1999 |
| Adaptive Quadtree (3mm tier) | 0.0043 | 2,312,620.0 | 0.4324 | 0.3625 |
| Adaptive Quadtree (1mm tier) | 0.0047 | 2,121,745.8 | 0.4713 | 1.1004 |
| Fourier MLP (4x4, 2.7km) | 0.0407 | 245,423.5 | 4.0746 | 103.2459 |
| Thin Plate Spline (11km) | 0.5060 | 19,761.0 | 50.6047 | 1.2529 |
| Thin Plate Spline (4x4, 2.7km) | 0.2576 | 38,816.1 | 25.7625 | 0.0086 |
| PyTorch ResNet + Cyclic CellPos (4x4) | 0.0283 | 353,739.4 | 2.8269 | 118.1399 |
| PyTorch ResNet + Cyclic CellPos (Adaptive Hybrid) | 0.0293 | 341,218.9 | 2.9307 | 343.2767 |
| Bilinear + Neural Feature Grid (8x8) | 0.0031 | 3,179,852.4 | 0.3145 | 310.9969 |

## 3. Storage & Memory Breakdown (4 Components)

Components:
- **Code Size**: Model implementation file on disk.
- **Library Dependencies**: Installed package footprint of external runtime dependencies.
- **Model Instance / Weights**: In-memory parameters, split trees, or index metadata.
- **Grid Lookup Data**: Retained control nodes or indexed cell tables.

| Model | Code Size (bytes) | Dependencies Size (bytes) | Model Weights/Instance Size (bytes) | Grid Lookup Size (bytes) | Total Footprint (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| KNN (K=4 IDW, 11km) | 3,075 | 41,029,584 | 30,843 | 86,048 | 39.2433 |
| KNN (K=16 IDW, 11km) | 3,075 | 41,029,584 | 30,845 | 86,048 | 39.2433 |
| Decision Tree (11km) | 4,184 | 41,029,584 | 431,452 | 0 | 39.5443 |
| Decision Tree (8x8, pruned 1e-8) | 4,184 | 41,029,584 | 2,949,052 | 0 | 41.9453 |
| Adaptive Quadtree (3mm tier) | 12,444 | 0 | 9,040 | 24,148,512 | 23.0503 |
| Adaptive Quadtree (1mm tier) | 12,444 | 0 | 9,814 | 78,529,320 | 74.9126 |
| Fourier MLP (4x4, 2.7km) | 12,746 | 73,428,937 | 2,148,306 | 0 | 72.0882 |
| Thin Plate Spline (11km) | 8,477 | 123,489,234 | 151,553 | 0 | 117.9211 |
| Thin Plate Spline (4x4, 2.7km) | 8,477 | 123,489,234 | 3,032,295 | 0 | 120.6684 |
| PyTorch ResNet + Cyclic CellPos (4x4) | 19,652 | 4,280,113,426 | 1,215,638 | 0 | 4083.0123 |
| PyTorch ResNet + Cyclic CellPos (Adaptive Hybrid) | 19,652 | 4,280,113,426 | 1,215,638 | 0 | 4083.0123 |
| Bilinear + Neural Feature Grid (8x8) | 5,089 | 4,280,113,426 | 29,441 | 663,648 | 4082.5001 |
