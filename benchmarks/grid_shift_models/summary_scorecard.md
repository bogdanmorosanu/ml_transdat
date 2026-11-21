# Cross-Model Grid Shift Benchmark Summary

Precision evaluated on `ro_corridor_snw` (36,435 points). Compares validated champions across KNN, Decision Trees, Adaptive Quadtrees, Fourier MLP, and Thin Plate Splines with exact source-experiment hyperparameters. All error metrics are reported in **millimeters (mm)**.

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

## 2. Performance Speed & Throughput Across Named Workloads

### Workload: `corridor_36k` (`ro_corridor_snw`, 36,435 pts)

| Model | Evaluation Time (s) | Throughput (pts/s) | Latency (µs/pt) | Fit Time (s) |
| :--- | :---: | :---: | :---: | :---: |
| KNN (K=4 IDW, 11km) | 0.0381 | 955,411.5 | 1.0467 | 0.0014 |
| KNN (K=16 IDW, 11km) | 0.0702 | 518,904.0 | 1.9271 | 0.0012 |
| Decision Tree (11km) | 0.0018 | 20,648,909.0 | 0.0484 | 0.0042 |
| Decision Tree (8x8, pruned 1e-8) | 0.0027 | 13,335,895.5 | 0.0750 | 0.2099 |
| Adaptive Quadtree (3mm tier) | 0.0140 | 2,610,779.9 | 0.3830 | 0.2663 |
| Adaptive Quadtree (1mm tier) | 0.0188 | 1,941,253.6 | 0.5151 | 1.0214 |
| Fourier MLP (4x4, 2.7km) | 0.1337 | 272,491.9 | 3.6698 | 106.2382 |
| Thin Plate Spline (11km) | 1.4777 | 24,657.0 | 40.5564 | 1.3353 |
| Thin Plate Spline (4x4, 2.7km) | 2.2490 | 16,200.5 | 61.7264 | 0.0069 |

### Workload: `uniform_4x4` (`ro_grid_4x4`, 36,435 pts)

| Model | Evaluation Time (s) | Throughput (pts/s) | Latency (µs/pt) | Fit Time (s) |
| :--- | :---: | :---: | :---: | :---: |
| KNN (K=4 IDW, 11km) | 0.0378 | 964,358.3 | 1.0370 | 0.0014 |
| KNN (K=16 IDW, 11km) | 0.0802 | 454,133.0 | 2.2020 | 0.0012 |
| Decision Tree (11km) | 0.0024 | 15,098,835.5 | 0.0662 | 0.0042 |
| Decision Tree (8x8, pruned 1e-8) | 0.0033 | 10,898,567.2 | 0.0918 | 0.2099 |
| Adaptive Quadtree (3mm tier) | 0.0179 | 2,035,975.3 | 0.4938 | 0.2663 |
| Adaptive Quadtree (1mm tier) | 0.0218 | 1,671,759.7 | 0.6013 | 1.0214 |
| Fourier MLP (4x4, 2.7km) | 0.1345 | 270,962.3 | 3.6906 | 106.2382 |
| Thin Plate Spline (11km) | 1.4747 | 24,707.4 | 40.4738 | 1.3353 |
| Thin Plate Spline (4x4, 2.7km) | 2.5607 | 14,228.4 | 70.2821 | 0.0069 |

### Workload: `single_cell_10k` (`single_cell_10k`, 10,000 pts)

| Model | Evaluation Time (s) | Throughput (pts/s) | Latency (µs/pt) | Fit Time (s) |
| :--- | :---: | :---: | :---: | :---: |
| KNN (K=4 IDW, 11km) | 0.0197 | 507,472.5 | 1.9705 | 0.0014 |
| KNN (K=16 IDW, 11km) | 0.0320 | 312,958.7 | 3.1953 | 0.0012 |
| Decision Tree (11km) | 0.0007 | 15,153,811.2 | 0.0660 | 0.0042 |
| Decision Tree (8x8, pruned 1e-8) | 0.0006 | 15,410,695.0 | 0.0649 | 0.2099 |
| Adaptive Quadtree (3mm tier) | 0.0043 | 2,339,783.3 | 0.4274 | 0.2663 |
| Adaptive Quadtree (1mm tier) | 0.0050 | 1,989,455.9 | 0.5026 | 1.0214 |
| Fourier MLP (4x4, 2.7km) | 0.0365 | 274,007.1 | 3.6495 | 106.2382 |
| Thin Plate Spline (11km) | 0.3969 | 25,195.6 | 39.6895 | 1.3353 |
| Thin Plate Spline (4x4, 2.7km) | 0.2098 | 47,658.5 | 20.9826 | 0.0069 |

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
| Fourier MLP (4x4, 2.7km) | 6,332 | 73,428,937 | 2,147,922 | 0 | 72.0817 |
| Thin Plate Spline (11km) | 6,111 | 123,489,234 | 151,515 | 0 | 117.9188 |
| Thin Plate Spline (4x4, 2.7km) | 6,111 | 123,489,234 | 3,032,257 | 0 | 120.6661 |
