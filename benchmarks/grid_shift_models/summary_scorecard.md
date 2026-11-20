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

## 2. Performance Speed & Throughput Across Named Workloads

### Workload: `corridor_36k` (`ro_corridor_snw`, 36,435 pts)

| Model | Evaluation Time (s) | Throughput (pts/s) | Latency (µs/pt) | Fit Time (s) |
| :--- | :---: | :---: | :---: | :---: |
| KNN (K=4 IDW, 11km) | 0.0388 | 937,849.9 | 1.0663 | 0.0013 |
| KNN (K=16 IDW, 11km) | 0.0761 | 478,805.0 | 2.0885 | 0.0010 |
| Decision Tree (11km) | 0.0018 | 20,203,504.5 | 0.0495 | 0.0037 |
| Decision Tree (8x8, pruned 1e-8) | 0.0025 | 14,469,817.5 | 0.0691 | 0.2111 |
| Adaptive Quadtree (3mm tier) | 0.0135 | 2,698,069.5 | 0.3706 | 0.2781 |
| Adaptive Quadtree (1mm tier) | 0.0195 | 1,867,551.7 | 0.5355 | 0.9925 |

### Workload: `uniform_4x4` (`ro_grid_4x4`, 36,435 pts)

| Model | Evaluation Time (s) | Throughput (pts/s) | Latency (µs/pt) | Fit Time (s) |
| :--- | :---: | :---: | :---: | :---: |
| KNN (K=4 IDW, 11km) | 0.0445 | 818,339.2 | 1.2220 | 0.0013 |
| KNN (K=16 IDW, 11km) | 0.0919 | 396,434.6 | 2.5225 | 0.0010 |
| Decision Tree (11km) | 0.0025 | 14,425,703.6 | 0.0693 | 0.0037 |
| Decision Tree (8x8, pruned 1e-8) | 0.0032 | 11,248,147.7 | 0.0889 | 0.2111 |
| Adaptive Quadtree (3mm tier) | 0.0182 | 1,998,409.4 | 0.5030 | 0.2781 |
| Adaptive Quadtree (1mm tier) | 0.0216 | 1,687,149.2 | 0.5959 | 0.9925 |

### Workload: `single_cell_10k` (`single_cell_10k`, 10,000 pts)

| Model | Evaluation Time (s) | Throughput (pts/s) | Latency (µs/pt) | Fit Time (s) |
| :--- | :---: | :---: | :---: | :---: |
| KNN (K=4 IDW, 11km) | 0.0206 | 485,057.8 | 2.0616 | 0.0013 |
| KNN (K=16 IDW, 11km) | 0.0324 | 308,976.7 | 3.2365 | 0.0010 |
| Decision Tree (11km) | 0.0007 | 14,664,907.9 | 0.0682 | 0.0037 |
| Decision Tree (8x8, pruned 1e-8) | 0.0007 | 14,760,149.3 | 0.0677 | 0.2111 |
| Adaptive Quadtree (3mm tier) | 0.0046 | 2,175,710.3 | 0.4596 | 0.2781 |
| Adaptive Quadtree (1mm tier) | 0.0053 | 1,873,501.2 | 0.5338 | 0.9925 |

## 3. Storage & Memory Breakdown (4 Components)

Components:
- **Code Size**: Model implementation file on disk.
- **Library Dependencies**: Installed package footprint of external runtime dependencies.
- **Model Instance / Weights**: In-memory parameters, split trees, or index metadata.
- **Grid Lookup Data**: Retained control nodes (KNN) or indexed quadtree cell tables.

| Model | Code Size (bytes) | Dependencies Size (bytes) | Model Weights/Instance Size (bytes) | Grid Lookup Size (bytes) | Total Footprint (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| KNN (K=4 IDW, 11km) | 2,992 | 41,029,584 | 30,809 | 86,048 | 39.2432 |
| KNN (K=16 IDW, 11km) | 2,992 | 41,029,584 | 30,811 | 86,048 | 39.2432 |
| Decision Tree (11km) | 4,079 | 41,029,584 | 431,362 | 0 | 39.5441 |
| Decision Tree (8x8, pruned 1e-8) | 4,079 | 41,029,584 | 2,948,962 | 0 | 41.9451 |
| Adaptive Quadtree (3mm tier) | 12,092 | 0 | 9,040 | 24,148,512 | 23.0500 |
| Adaptive Quadtree (1mm tier) | 12,092 | 0 | 9,814 | 78,529,320 | 74.9123 |
