# Cross-Model Grid Shift Benchmark Summary

Precision evaluated on canonical corridor benchmark (`ro_corridor_snw`, 36,435 points). All distance/error metrics are reported in **millimeters (mm)**.

## 1. Comprehensive Precision Scorecard (mm)

| Model | Mean 2D Error (mm) | Std Dev 2D (mm) | RMSE 2D Error (mm) | Min 2D Error (mm) | P50 2D Error (mm) | P90 2D Error (mm) | P95 2D Error (mm) | P99 2D Error (mm) | Max 2D Error (mm) | Signed Bias dN (mm) | Signed Bias dE (mm) | Pass < 1 mm (%) | Pass < 1 cm (%) | Pass < 5 cm (%) | Pass < 10 cm (%) | Pass < 1 m (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| KNN (K=4 IDW, 11km) | 24.03 | 15.40 | 28.54 | 0.03 | 21.11 | 44.17 | 53.36 | 73.41 | 153.57 | -1.19 | -1.33 | 0.33 | 16.70 | 93.47 | 99.88 | 100.00 |
| KNN (K=16 IDW, 11km) | 65.72 | 40.61 | 77.25 | 0.10 | 58.20 | 119.77 | 142.50 | 192.31 | 282.43 | -4.45 | -4.37 | 0.01 | 2.67 | 40.88 | 82.13 | 100.00 |
| Decision Tree (11km) | 56.01 | 38.23 | 67.81 | 0.14 | 47.80 | 108.06 | 128.93 | 177.96 | 294.62 | -0.15 | 0.24 | 0.07 | 4.98 | 52.26 | 87.30 | 100.00 |
| Decision Tree (8x8, pruned 1e-8) | 16.12 | 9.72 | 18.82 | 0.06 | 14.33 | 29.33 | 34.56 | 45.77 | 85.14 | 0.00 | -0.01 | 0.43 | 30.32 | 99.46 | 100.00 | 100.00 |
| Adaptive Quadtree (3mm tier) | 1.43 | 1.05 | 1.77 | 0.00 | 1.25 | 2.64 | 3.08 | 4.49 | 26.32 | -0.02 | -0.03 | 39.17 | 99.91 | 100.00 | 100.00 | 100.00 |
| Adaptive Quadtree (1mm tier) | 0.50 | 0.34 | 0.60 | 0.00 | 0.44 | 0.90 | 1.05 | 1.59 | 5.77 | -0.00 | -0.01 | 93.76 | 100.00 | 100.00 | 100.00 | 100.00 |
| Fourier MLP (4x4, 2.7km) | 19.63 | 10.17 | 22.11 | 0.10 | 18.51 | 33.47 | 38.12 | 46.87 | 63.25 | 3.56 | 8.05 | 0.23 | 18.35 | 99.53 | 100.00 | 100.00 |

## 2. Performance Speed & Throughput Across Named Workloads

### Workload: `corridor_36k` (`ro_corridor_snw`, 36,435 pts)

| Model | Evaluation Time (s) | Throughput (pts/s) | Latency (µs/pt) | Fit Time (s) |
| :--- | :---: | :---: | :---: | :---: |
| KNN (K=4 IDW, 11km) | 0.0571 | 637,557.4 | 1.5685 | 0.0013 |
| KNN (K=16 IDW, 11km) | 0.1006 | 362,169.4 | 2.7611 | 0.0010 |
| Decision Tree (11km) | 0.0022 | 16,275,797.4 | 0.0614 | 0.0037 |
| Decision Tree (8x8, pruned 1e-8) | 0.0026 | 13,849,399.4 | 0.0722 | 0.2102 |
| Adaptive Quadtree (3mm tier) | 0.0180 | 2,027,083.4 | 0.4933 | 0.2661 |
| Adaptive Quadtree (1mm tier) | 0.0269 | 1,354,476.1 | 0.7383 | 1.0473 |
| Fourier MLP (4x4, 2.7km) | 0.0941 | 387,004.5 | 2.5839 | 3.5109 |

### Workload: `uniform_4x4` (`ro_grid_4x4`, 36,435 pts)

| Model | Evaluation Time (s) | Throughput (pts/s) | Latency (µs/pt) | Fit Time (s) |
| :--- | :---: | :---: | :---: | :---: |
| KNN (K=4 IDW, 11km) | 0.0532 | 684,907.0 | 1.4601 | 0.0013 |
| KNN (K=16 IDW, 11km) | 0.1084 | 336,164.9 | 2.9747 | 0.0010 |
| Decision Tree (11km) | 0.0030 | 12,186,841.5 | 0.0821 | 0.0037 |
| Decision Tree (8x8, pruned 1e-8) | 0.0037 | 9,738,073.0 | 0.1027 | 0.2102 |
| Adaptive Quadtree (3mm tier) | 0.0227 | 1,606,743.6 | 0.6257 | 0.2661 |
| Adaptive Quadtree (1mm tier) | 0.0237 | 1,538,919.7 | 0.6532 | 1.0473 |
| Fourier MLP (4x4, 2.7km) | 0.0767 | 474,902.0 | 2.1057 | 3.5109 |

### Workload: `single_cell_10k` (`single_cell_10k`, 10,000 pts)

| Model | Evaluation Time (s) | Throughput (pts/s) | Latency (µs/pt) | Fit Time (s) |
| :--- | :---: | :---: | :---: | :---: |
| KNN (K=4 IDW, 11km) | 0.0258 | 387,359.7 | 2.5816 | 0.0013 |
| KNN (K=16 IDW, 11km) | 0.0434 | 230,215.3 | 4.3438 | 0.0010 |
| Decision Tree (11km) | 0.0008 | 12,419,274.7 | 0.0805 | 0.0037 |
| Decision Tree (8x8, pruned 1e-8) | 0.0008 | 13,159,626.3 | 0.0760 | 0.2102 |
| Adaptive Quadtree (3mm tier) | 0.0058 | 1,724,524.5 | 0.5799 | 0.2661 |
| Adaptive Quadtree (1mm tier) | 0.0054 | 1,835,232.8 | 0.5449 | 1.0473 |
| Fourier MLP (4x4, 2.7km) | 0.0209 | 479,074.0 | 2.0874 | 3.5109 |

## 3. Storage & Memory Breakdown (4 Components)

Components:
- **Code Size**: Model implementation file on disk.
- **Library Dependencies**: Installed package footprint of external runtime dependencies.
- **Model Instance / Weights**: In-memory parameters, split trees, or index metadata.
- **Grid Lookup Data**: Retained control nodes (KNN) or indexed quadtree cell tables.

| Model | Code Size (bytes) | Dependencies Size (bytes) | Model Weights/Instance Size (bytes) | Grid Lookup Size (bytes) | Total Footprint (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| KNN (K=4 IDW, 11km) | 3,075 | 41,029,584 | 30,809 | 86,048 | 39.2432 |
| KNN (K=16 IDW, 11km) | 3,075 | 41,029,584 | 30,811 | 86,048 | 39.2432 |
| Decision Tree (11km) | 4,184 | 41,029,584 | 431,362 | 0 | 39.5442 |
| Decision Tree (8x8, pruned 1e-8) | 4,184 | 41,029,584 | 2,948,962 | 0 | 41.9452 |
| Adaptive Quadtree (3mm tier) | 12,444 | 0 | 9,040 | 24,148,512 | 23.0503 |
| Adaptive Quadtree (1mm tier) | 12,444 | 0 | 9,814 | 78,529,320 | 74.9126 |
| Fourier MLP (4x4, 2.7km) | 6,332 | 73,428,937 | 819,706 | 0 | 70.8151 |
