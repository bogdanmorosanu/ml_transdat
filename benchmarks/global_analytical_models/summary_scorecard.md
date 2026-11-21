# Global Analytical Models Benchmark Summary

Precision evaluated on canonical corridor benchmark (`ro_corridor_snw`, 36,435 points). Models in this suite evaluate a **single closed-form global equation** without spatial lookups (0 MB grid lookup memory). All errors are reported in **millimeters (mm)**.

## 1. Comprehensive Precision Scorecard (mm)

| Model | Mean 2D Error (mm) | Std Dev 2D (mm) | RMSE 2D Error (mm) | Min 2D Error (mm) | P50 2D Error (mm) | P90 2D Error (mm) | P95 2D Error (mm) | P99 2D Error (mm) | Max 2D Error (mm) | Signed Bias dN (mm) | Signed Bias dE (mm) | Pass < 1 mm (%) | Pass < 1 cm (%) | Pass < 5 cm (%) | Pass < 10 cm (%) | Pass < 1 m (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Helmert 2D Conformal (4P) | 304.57 | 161.48 | 344.73 | 0.95 | 288.39 | 485.53 | 611.85 | 850.00 | 1060.01 | -41.82 | 1.31 | 0.00 | 0.12 | 2.91 | 8.39 | 99.81 |
| Linear Affine Plane (Deg 1, 6T) | 319.51 | 175.81 | 364.69 | 7.81 | 276.05 | 534.87 | 668.20 | 895.25 | 1101.77 | -12.03 | 11.21 | 0.00 | 0.01 | 0.88 | 5.66 | 99.63 |
| Quadratic Bowl (Deg 2, 12T) | 229.43 | 137.12 | 267.28 | 3.41 | 202.48 | 409.94 | 473.26 | 638.99 | 819.03 | 42.80 | 18.08 | 0.00 | 0.08 | 4.52 | 18.02 | 100.00 |
| Cubic Surface (Deg 3, 20T) | 216.39 | 123.81 | 249.31 | 0.90 | 197.74 | 389.42 | 425.72 | 545.67 | 718.08 | 31.07 | 17.93 | 0.00 | 0.17 | 4.94 | 20.83 | 100.00 |
| Quintic Surface (Deg 5, 42T) | 200.75 | 99.75 | 224.17 | 1.08 | 194.28 | 335.03 | 382.50 | 493.52 | 589.21 | 64.40 | 13.85 | 0.00 | 0.15 | 4.39 | 16.38 | 100.00 |
| Multi-Harmonic (Deg 8, 90T) | 148.68 | 81.55 | 169.58 | 0.50 | 134.20 | 261.63 | 304.38 | 402.76 | 454.43 | 1.39 | 5.42 | 0.01 | 0.31 | 7.87 | 30.76 | 100.00 |
| High-Order Limit (Deg 12, 182T) | 141.35 | 79.55 | 162.20 | 0.80 | 128.97 | 238.78 | 294.84 | 413.41 | 490.72 | -16.87 | -4.28 | 0.00 | 0.45 | 8.47 | 33.91 | 100.00 |
| Sparse Lasso (Deg 5, 24T) | 205.67 | 116.32 | 236.28 | 1.72 | 187.23 | 357.28 | 409.36 | 548.97 | 729.11 | 45.43 | 15.45 | 0.00 | 0.38 | 6.16 | 19.20 | 100.00 |

## 2. Performance Speed & Throughput Across Named Workloads

### Workload: `corridor_36k` (`ro_corridor_snw`, 36,435 pts)

| Model | Evaluation Time (s) | Throughput (pts/s) | Latency (µs/pt) | Fit Time (s) |
| :--- | :---: | :---: | :---: | :---: |
| Helmert 2D Conformal (4P) | 0.0005 | 72,623,081.5 | 0.0138 | 0.0314 |
| Linear Affine Plane (Deg 1, 6T) | 0.0012 | 30,853,586.3 | 0.0324 | 0.0021 |
| Quadratic Bowl (Deg 2, 12T) | 0.0023 | 16,096,041.7 | 0.0621 | 0.0014 |
| Cubic Surface (Deg 3, 20T) | 0.0033 | 10,968,120.7 | 0.0912 | 0.0016 |
| Quintic Surface (Deg 5, 42T) | 0.0068 | 5,332,367.4 | 0.1875 | 0.0021 |
| Multi-Harmonic (Deg 8, 90T) | 0.0157 | 2,327,758.0 | 0.4296 | 0.0031 |
| High-Order Limit (Deg 12, 182T) | 0.0246 | 1,479,894.4 | 0.6757 | 0.0046 |
| Sparse Lasso (Deg 5, 24T) | 0.0064 | 5,677,709.9 | 0.1761 | 0.0045 |

### Workload: `uniform_4x4` (`ro_grid_4x4`, 36,435 pts)

| Model | Evaluation Time (s) | Throughput (pts/s) | Latency (µs/pt) | Fit Time (s) |
| :--- | :---: | :---: | :---: | :---: |
| Helmert 2D Conformal (4P) | 0.0003 | 116,666,667.1 | 0.0086 | 0.0314 |
| Linear Affine Plane (Deg 1, 6T) | 0.0012 | 31,323,074.3 | 0.0319 | 0.0021 |
| Quadratic Bowl (Deg 2, 12T) | 0.0018 | 20,424,351.1 | 0.0490 | 0.0014 |
| Cubic Surface (Deg 3, 20T) | 0.0036 | 10,006,316.6 | 0.0999 | 0.0016 |
| Quintic Surface (Deg 5, 42T) | 0.0069 | 5,274,854.1 | 0.1896 | 0.0021 |
| Multi-Harmonic (Deg 8, 90T) | 0.0128 | 2,850,091.5 | 0.3509 | 0.0031 |
| High-Order Limit (Deg 12, 182T) | 0.0304 | 1,197,428.7 | 0.8351 | 0.0046 |
| Sparse Lasso (Deg 5, 24T) | 0.0072 | 5,037,677.2 | 0.1985 | 0.0045 |

### Workload: `single_cell_10k` (`single_cell_10k`, 10,000 pts)

| Model | Evaluation Time (s) | Throughput (pts/s) | Latency (µs/pt) | Fit Time (s) |
| :--- | :---: | :---: | :---: | :---: |
| Helmert 2D Conformal (4P) | 0.0001 | 90,661,831.8 | 0.0110 | 0.0314 |
| Linear Affine Plane (Deg 1, 6T) | 0.0005 | 18,556,318.4 | 0.0539 | 0.0021 |
| Quadratic Bowl (Deg 2, 12T) | 0.0008 | 12,626,262.6 | 0.0792 | 0.0014 |
| Cubic Surface (Deg 3, 20T) | 0.0014 | 6,897,978.9 | 0.1450 | 0.0016 |
| Quintic Surface (Deg 5, 42T) | 0.0027 | 3,671,880.7 | 0.2723 | 0.0021 |
| Multi-Harmonic (Deg 8, 90T) | 0.0045 | 2,217,245.7 | 0.4510 | 0.0031 |
| High-Order Limit (Deg 12, 182T) | 0.0073 | 1,369,656.6 | 0.7301 | 0.0046 |
| Sparse Lasso (Deg 5, 24T) | 0.0025 | 3,963,221.3 | 0.2523 | 0.0045 |

## 3. Storage & Memory Breakdown (4 Components)

Components:
- **Code Size**: Model implementation file on disk.
- **Library Dependencies**: Installed package footprint of external runtime dependencies.
- **Model Instance / Weights**: In-memory parameters, split trees, or index metadata.
- **Grid Lookup Data**: Retained control nodes or indexed cell tables (0 MB for all analytical models).

| Model | Code Size (bytes) | Dependencies Size (bytes) | Model Weights/Instance Size (bytes) | Grid Lookup Size (bytes) | Total Footprint (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Helmert 2D Conformal (4P) | 8,997 | 32,399,353 | 272 | 0 | 30.9073 |
| Linear Affine Plane (Deg 1, 6T) | 7,871 | 73,428,937 | 972 | 0 | 70.0357 |
| Quadratic Bowl (Deg 2, 12T) | 7,871 | 73,428,937 | 1,038 | 0 | 70.0358 |
| Cubic Surface (Deg 3, 20T) | 7,871 | 73,428,937 | 1,130 | 0 | 70.0359 |
| Quintic Surface (Deg 5, 42T) | 7,871 | 73,428,937 | 1,395 | 0 | 70.0361 |
| Multi-Harmonic (Deg 8, 90T) | 7,871 | 73,428,937 | 1,983 | 0 | 70.0367 |
| High-Order Limit (Deg 12, 182T) | 7,871 | 73,428,937 | 3,143 | 0 | 70.0378 |
| Sparse Lasso (Deg 5, 24T) | 7,871 | 73,428,937 | 1,519 | 0 | 70.0362 |
