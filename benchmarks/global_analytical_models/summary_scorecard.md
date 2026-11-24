# Global Analytical Models Benchmark Summary

Precision evaluated on `ro_corridor_snw` (36,435 points). Models in this suite evaluate a **single closed-form global equation** without spatial lookups (0 MB grid lookup memory). All error metrics are reported in **millimeters (mm)**.

## 1. Comprehensive Precision Scorecard (mm)

| Model | Mean 2D Error (mm) | Std Dev 2D (mm) | RMSE 2D Error (mm) | Centered RMSE 2D (mm) | IQR 2D Error (mm) | Min 2D Error (mm) | P50 2D Error (mm) | P90 2D Error (mm) | P95 2D Error (mm) | P99 2D Error (mm) | Max 2D Error (mm) | Signed Bias dN (mm) | Signed Bias dE (mm) | Roughness Grad RMS (mm/km) | Roughness Grad P95 (mm/km) | Local Jitter Ratio | Learnability Score (mm) | Pass < 1 mm (%) | Pass < 1 cm (%) | Pass < 5 cm (%) | Pass < 10 cm (%) | Pass < 1 m (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Helmert 2D Conformal (4P) | 304.57 | 161.48 | 344.73 | 342.18 | 184.60 | 0.95 | 288.39 | 485.53 | 611.85 | 850.00 | 1060.01 | -41.82 | 1.31 | 13.55 | 23.16 | 0.01 | 304.64 | 0.00 | 0.12 | 2.91 | 8.39 | 99.81 |
| Linear Affine Plane (Deg 1, 6T) | 319.51 | 175.81 | 364.69 | 364.32 | 217.98 | 7.81 | 276.05 | 534.87 | 668.20 | 895.25 | 1101.77 | -12.03 | 11.21 | 13.56 | 23.25 | 0.01 | 326.99 | 0.00 | 0.01 | 0.88 | 5.66 | 99.63 |
| Quadratic Bowl (Deg 2, 12T) | 229.43 | 137.12 | 267.28 | 263.21 | 198.90 | 3.41 | 202.48 | 409.94 | 473.26 | 638.99 | 819.03 | 42.80 | 18.08 | 13.32 | 22.89 | 0.02 | 237.25 | 0.00 | 0.08 | 4.52 | 18.02 | 100.00 |
| Cubic Surface (Deg 3, 20T) | 216.39 | 123.81 | 249.31 | 246.71 | 198.12 | 0.90 | 197.74 | 389.42 | 425.72 | 545.67 | 718.08 | 31.07 | 17.93 | 13.25 | 22.77 | 0.02 | 219.43 | 0.00 | 0.17 | 4.94 | 20.83 | 100.00 |
| Quintic Surface (Deg 5, 42T) | 200.75 | 99.75 | 224.17 | 214.27 | 127.84 | 1.08 | 194.28 | 335.03 | 382.50 | 493.52 | 589.21 | 64.40 | 13.85 | 13.28 | 22.80 | 0.02 | 194.59 | 0.00 | 0.15 | 4.39 | 16.38 | 100.00 |
| Multi-Harmonic (Deg 8, 90T) | 148.68 | 81.55 | 169.58 | 169.49 | 103.79 | 0.50 | 134.20 | 261.63 | 304.38 | 402.76 | 454.43 | 1.39 | 5.42 | 13.00 | 22.56 | 0.03 | 156.34 | 0.01 | 0.31 | 7.87 | 30.76 | 100.00 |
| High-Order Limit (Deg 12, 182T) | 141.35 | 79.55 | 162.20 | 161.26 | 97.04 | 0.80 | 128.97 | 238.78 | 294.84 | 413.41 | 490.72 | -16.87 | -4.28 | 13.01 | 22.64 | 0.03 | 150.33 | 0.00 | 0.45 | 8.47 | 33.91 | 100.00 |
| Sparse Lasso (Deg 5, 24T) | 205.67 | 116.32 | 236.28 | 231.36 | 165.49 | 1.72 | 187.23 | 357.28 | 409.36 | 548.97 | 729.11 | 45.43 | 15.45 | 13.22 | 22.70 | 0.02 | 208.46 | 0.00 | 0.38 | 6.16 | 19.20 | 100.00 |

## 2. Performance Speed & Throughput Across Named Workloads

### Workload: `corridor_36k` (`ro_corridor_snw`, 36,435 pts)

| Model | Evaluation Time (s) | Throughput (pts/s) | Latency (µs/pt) | Fit Time (s) |
| :--- | :---: | :---: | :---: | :---: |
| Helmert 2D Conformal (4P) | 0.0005 | 67,062,395.9 | 0.0149 | 0.0532 |
| Linear Affine Plane (Deg 1, 6T) | 0.0173 | 2,109,288.2 | 0.4741 | 0.0440 |
| Quadratic Bowl (Deg 2, 12T) | 0.0027 | 13,669,105.2 | 0.0732 | 0.0017 |
| Cubic Surface (Deg 3, 20T) | 0.0047 | 7,805,771.6 | 0.1281 | 0.0015 |
| Quintic Surface (Deg 5, 42T) | 0.0088 | 4,135,265.8 | 0.2418 | 0.0034 |
| Multi-Harmonic (Deg 8, 90T) | 0.0199 | 1,831,558.0 | 0.5460 | 0.0034 |
| High-Order Limit (Deg 12, 182T) | 0.0367 | 993,924.9 | 1.0061 | 0.0042 |
| Sparse Lasso (Deg 5, 24T) | 0.0085 | 4,273,699.8 | 0.2340 | 0.0066 |

### Workload: `uniform_4x4` (`ro_grid_4x4`, 36,435 pts)

| Model | Evaluation Time (s) | Throughput (pts/s) | Latency (µs/pt) | Fit Time (s) |
| :--- | :---: | :---: | :---: | :---: |
| Helmert 2D Conformal (4P) | 0.0004 | 99,115,885.5 | 0.0101 | 0.0532 |
| Linear Affine Plane (Deg 1, 6T) | 0.0014 | 26,238,657.5 | 0.0381 | 0.0440 |
| Quadratic Bowl (Deg 2, 12T) | 0.0031 | 11,676,761.8 | 0.0856 | 0.0017 |
| Cubic Surface (Deg 3, 20T) | 0.0047 | 7,825,890.8 | 0.1278 | 0.0015 |
| Quintic Surface (Deg 5, 42T) | 0.0094 | 3,863,896.7 | 0.2588 | 0.0034 |
| Multi-Harmonic (Deg 8, 90T) | 0.0183 | 1,988,886.0 | 0.5028 | 0.0034 |
| High-Order Limit (Deg 12, 182T) | 0.0490 | 743,869.0 | 1.3443 | 0.0042 |
| Sparse Lasso (Deg 5, 24T) | 0.0102 | 3,581,222.5 | 0.2792 | 0.0066 |

### Workload: `single_cell_10k` (`single_cell_10k`, 10,000 pts)

| Model | Evaluation Time (s) | Throughput (pts/s) | Latency (µs/pt) | Fit Time (s) |
| :--- | :---: | :---: | :---: | :---: |
| Helmert 2D Conformal (4P) | 0.0001 | 74,962,520.3 | 0.0133 | 0.0532 |
| Linear Affine Plane (Deg 1, 6T) | 0.0011 | 9,041,591.5 | 0.1106 | 0.0440 |
| Quadratic Bowl (Deg 2, 12T) | 0.0016 | 6,306,760.9 | 0.1586 | 0.0017 |
| Cubic Surface (Deg 3, 20T) | 0.0014 | 7,264,274.3 | 0.1377 | 0.0015 |
| Quintic Surface (Deg 5, 42T) | 0.0027 | 3,660,188.1 | 0.2732 | 0.0034 |
| Multi-Harmonic (Deg 8, 90T) | 0.0053 | 1,869,368.5 | 0.5349 | 0.0034 |
| High-Order Limit (Deg 12, 182T) | 0.0116 | 864,872.3 | 1.1562 | 0.0042 |
| Sparse Lasso (Deg 5, 24T) | 0.0033 | 3,015,499.7 | 0.3316 | 0.0066 |

## 3. Storage & Memory Breakdown (4 Components)

Components:
- **Code Size**: Model implementation file on disk.
- **Library Dependencies**: Installed package footprint of external runtime dependencies.
- **Model Instance / Weights**: In-memory parameters, split trees, or index metadata.
- **Grid Lookup Data**: Retained control nodes or indexed cell tables.

| Model | Code Size (bytes) | Dependencies Size (bytes) | Model Weights/Instance Size (bytes) | Grid Lookup Size (bytes) | Total Footprint (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Helmert 2D Conformal (4P) | 9,787 | 32,399,353 | 272 | 0 | 30.9080 |
| Linear Affine Plane (Deg 1, 6T) | 8,069 | 73,428,937 | 1,001 | 0 | 70.0359 |
| Quadratic Bowl (Deg 2, 12T) | 8,069 | 73,428,937 | 1,067 | 0 | 70.0360 |
| Cubic Surface (Deg 3, 20T) | 8,069 | 73,428,937 | 1,159 | 0 | 70.0361 |
| Quintic Surface (Deg 5, 42T) | 8,069 | 73,428,937 | 1,424 | 0 | 70.0363 |
| Multi-Harmonic (Deg 8, 90T) | 8,069 | 73,428,937 | 2,012 | 0 | 70.0369 |
| High-Order Limit (Deg 12, 182T) | 8,069 | 73,428,937 | 3,172 | 0 | 70.0380 |
| Sparse Lasso (Deg 5, 24T) | 8,069 | 73,428,937 | 1,556 | 0 | 70.0365 |
