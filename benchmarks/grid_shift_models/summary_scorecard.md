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
| KNN (K=4 IDW, 11km) | 0.0519 | 702,486.0 | 1.4235 | 0.0176 |
| KNN (K=16 IDW, 11km) | 0.0732 | 497,430.6 | 2.0103 | 0.0010 |
| Decision Tree (11km) | 0.0027 | 13,331,503.8 | 0.0750 | 0.0085 |
| Decision Tree (8x8, pruned 1e-8) | 0.0033 | 11,031,215.0 | 0.0907 | 0.1914 |
| Adaptive Quadtree (3mm tier) | 0.0144 | 2,524,510.7 | 0.3961 | 0.2403 |
| Adaptive Quadtree (1mm tier) | 0.0208 | 1,750,908.3 | 0.5711 | 0.8741 |
| Fourier MLP (4x4, 2.7km) | 0.1304 | 279,312.7 | 3.5802 | 101.1036 |
| Thin Plate Spline (11km) | 1.4363 | 25,366.7 | 39.4218 | 1.2652 |
| Thin Plate Spline (4x4, 2.7km) | 2.2785 | 15,990.6 | 62.5367 | 0.0232 |
| PyTorch ResNet + Cyclic CellPos (4x4) | 0.2483 | 146,756.6 | 6.8140 | 126.0570 |
| PyTorch ResNet + Cyclic CellPos (Adaptive Hybrid) | 0.2219 | 164,199.1 | 6.0902 | 340.2506 |
| Bilinear + Neural Feature Grid (8x8) | 0.0250 | 1,458,718.7 | 0.6855 | 294.5727 |

### Workload: `uniform_4x4` (`ro_grid_4x4`, 36,435 pts)

| Model | Evaluation Time (s) | Throughput (pts/s) | Latency (µs/pt) | Fit Time (s) |
| :--- | :---: | :---: | :---: | :---: |
| KNN (K=4 IDW, 11km) | 0.0408 | 892,714.9 | 1.1202 | 0.0176 |
| KNN (K=16 IDW, 11km) | 0.0822 | 443,445.6 | 2.2551 | 0.0010 |
| Decision Tree (11km) | 0.0025 | 14,303,941.6 | 0.0699 | 0.0085 |
| Decision Tree (8x8, pruned 1e-8) | 0.0036 | 10,109,881.0 | 0.0989 | 0.1914 |
| Adaptive Quadtree (3mm tier) | 0.0173 | 2,108,262.9 | 0.4768 | 0.2403 |
| Adaptive Quadtree (1mm tier) | 0.0210 | 1,733,093.6 | 0.5801 | 0.8741 |
| Fourier MLP (4x4, 2.7km) | 0.1477 | 246,630.2 | 4.0547 | 101.1036 |
| Thin Plate Spline (11km) | 1.4995 | 24,298.2 | 41.1554 | 1.2652 |
| Thin Plate Spline (4x4, 2.7km) | 2.6934 | 13,527.5 | 73.9234 | 0.0232 |
| PyTorch ResNet + Cyclic CellPos (4x4) | 0.2275 | 160,177.9 | 6.2431 | 126.0570 |
| PyTorch ResNet + Cyclic CellPos (Adaptive Hybrid) | 0.2314 | 157,458.5 | 6.3509 | 340.2506 |
| Bilinear + Neural Feature Grid (8x8) | 0.0112 | 3,243,452.6 | 0.3083 | 294.5727 |

### Workload: `single_cell_10k` (`single_cell_10k`, 10,000 pts)

| Model | Evaluation Time (s) | Throughput (pts/s) | Latency (µs/pt) | Fit Time (s) |
| :--- | :---: | :---: | :---: | :---: |
| KNN (K=4 IDW, 11km) | 0.0204 | 489,524.2 | 2.0428 | 0.0176 |
| KNN (K=16 IDW, 11km) | 0.0348 | 287,188.0 | 3.4820 | 0.0010 |
| Decision Tree (11km) | 0.0009 | 11,621,150.4 | 0.0861 | 0.0085 |
| Decision Tree (8x8, pruned 1e-8) | 0.0007 | 14,645,577.3 | 0.0683 | 0.1914 |
| Adaptive Quadtree (3mm tier) | 0.0042 | 2,400,441.7 | 0.4166 | 0.2403 |
| Adaptive Quadtree (1mm tier) | 0.0045 | 2,199,010.4 | 0.4547 | 0.8741 |
| Fourier MLP (4x4, 2.7km) | 0.0324 | 308,562.9 | 3.2408 | 101.1036 |
| Thin Plate Spline (11km) | 0.3879 | 25,781.6 | 38.7874 | 1.2652 |
| Thin Plate Spline (4x4, 2.7km) | 0.2166 | 46,163.8 | 21.6620 | 0.0232 |
| PyTorch ResNet + Cyclic CellPos (4x4) | 0.0448 | 223,396.3 | 4.4764 | 126.0570 |
| PyTorch ResNet + Cyclic CellPos (Adaptive Hybrid) | 0.0436 | 229,215.3 | 4.3627 | 340.2506 |
| Bilinear + Neural Feature Grid (8x8) | 0.0038 | 2,613,286.0 | 0.3827 | 294.5727 |

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
| Thin Plate Spline (11km) | 8,477 | 123,489,234 | 151,553 | 0 | 117.9211 |
| Thin Plate Spline (4x4, 2.7km) | 8,477 | 123,489,234 | 3,032,295 | 0 | 120.6684 |
| PyTorch ResNet + Cyclic CellPos (4x4) | 24,277 | 4,280,113,426 | 2,321,218 | 0 | 4084.0711 |
| PyTorch ResNet + Cyclic CellPos (Adaptive Hybrid) | 24,277 | 4,280,113,426 | 2,321,218 | 0 | 4084.0711 |
| Bilinear + Neural Feature Grid (8x8) | 4,956 | 4,280,113,426 | 29,127 | 663,648 | 4082.4997 |
