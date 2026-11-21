# Global Analytical Models Benchmark Summary

Precision evaluated on `ro_corridor_snw` (36,435 points). Models in this suite evaluate a **single closed-form global equation** without spatial lookups (0 MB grid lookup memory). All error metrics are reported in **millimeters (mm)**.

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
| Helmert 2D Conformal (4P) | 0.0003 | 116,071,997.4 | 0.0086 | 0.0361 |
| Linear Affine Plane (Deg 1, 6T) | 0.0012 | 29,852,519.5 | 0.0335 | 0.0020 |
| Quadratic Bowl (Deg 2, 12T) | 0.0030 | 12,117,533.6 | 0.0825 | 0.0013 |
| Cubic Surface (Deg 3, 20T) | 0.0045 | 8,108,559.2 | 0.1233 | 0.0013 |
| Quintic Surface (Deg 5, 42T) | 0.0068 | 5,390,510.6 | 0.1855 | 0.0028 |
| Multi-Harmonic (Deg 8, 90T) | 0.0170 | 2,143,727.1 | 0.4665 | 0.0034 |
| High-Order Limit (Deg 12, 182T) | 0.0339 | 1,074,069.1 | 0.9310 | 0.0049 |
| Sparse Lasso (Deg 5, 24T) | 0.0081 | 4,520,583.6 | 0.2212 | 0.0053 |

### Workload: `uniform_4x4` (`ro_grid_4x4`, 36,435 pts)

| Model | Evaluation Time (s) | Throughput (pts/s) | Latency (µs/pt) | Fit Time (s) |
| :--- | :---: | :---: | :---: | :---: |
| Helmert 2D Conformal (4P) | 0.0004 | 86,605,656.8 | 0.0115 | 0.0361 |
| Linear Affine Plane (Deg 1, 6T) | 0.0016 | 23,423,336.6 | 0.0427 | 0.0020 |
| Quadratic Bowl (Deg 2, 12T) | 0.0027 | 13,603,778.5 | 0.0735 | 0.0013 |
| Cubic Surface (Deg 3, 20T) | 0.0037 | 9,852,356.6 | 0.1015 | 0.0013 |
| Quintic Surface (Deg 5, 42T) | 0.0063 | 5,742,497.8 | 0.1741 | 0.0028 |
| Multi-Harmonic (Deg 8, 90T) | 0.0154 | 2,359,703.4 | 0.4238 | 0.0034 |
| High-Order Limit (Deg 12, 182T) | 0.0254 | 1,436,263.9 | 0.6963 | 0.0049 |
| Sparse Lasso (Deg 5, 24T) | 0.0068 | 5,338,930.9 | 0.1873 | 0.0053 |

### Workload: `single_cell_10k` (`single_cell_10k`, 10,000 pts)

| Model | Evaluation Time (s) | Throughput (pts/s) | Latency (µs/pt) | Fit Time (s) |
| :--- | :---: | :---: | :---: | :---: |
| Helmert 2D Conformal (4P) | 0.0001 | 79,617,834.7 | 0.0126 | 0.0361 |
| Linear Affine Plane (Deg 1, 6T) | 0.0006 | 18,027,762.8 | 0.0555 | 0.0020 |
| Quadratic Bowl (Deg 2, 12T) | 0.0009 | 10,847,163.5 | 0.0922 | 0.0013 |
| Cubic Surface (Deg 3, 20T) | 0.0010 | 9,872,642.9 | 0.1013 | 0.0013 |
| Quintic Surface (Deg 5, 42T) | 0.0020 | 4,920,049.2 | 0.2032 | 0.0028 |
| Multi-Harmonic (Deg 8, 90T) | 0.0036 | 2,751,940.1 | 0.3634 | 0.0034 |
| High-Order Limit (Deg 12, 182T) | 0.0079 | 1,273,139.3 | 0.7855 | 0.0049 |
| Sparse Lasso (Deg 5, 24T) | 0.0030 | 3,379,634.3 | 0.2959 | 0.0053 |

## 3. Storage & Memory Breakdown (4 Components)

Components:
- **Code Size**: Model implementation file on disk.
- **Library Dependencies**: Installed package footprint of external runtime dependencies.
- **Model Instance / Weights**: In-memory parameters, split trees, or index metadata.
- **Grid Lookup Data**: Retained control nodes or indexed cell tables.

| Model | Code Size (bytes) | Dependencies Size (bytes) | Model Weights/Instance Size (bytes) | Grid Lookup Size (bytes) | Total Footprint (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Helmert 2D Conformal (4P) | 8,997 | 32,399,353 | 272 | 0 | 30.9073 |
| Linear Affine Plane (Deg 1, 6T) | 8,069 | 73,428,937 | 972 | 0 | 70.0359 |
| Quadratic Bowl (Deg 2, 12T) | 8,069 | 73,428,937 | 1,038 | 0 | 70.0360 |
| Cubic Surface (Deg 3, 20T) | 8,069 | 73,428,937 | 1,130 | 0 | 70.0361 |
| Quintic Surface (Deg 5, 42T) | 8,069 | 73,428,937 | 1,395 | 0 | 70.0363 |
| Multi-Harmonic (Deg 8, 90T) | 8,069 | 73,428,937 | 1,983 | 0 | 70.0369 |
| High-Order Limit (Deg 12, 182T) | 8,069 | 73,428,937 | 3,143 | 0 | 70.0380 |
| Sparse Lasso (Deg 5, 24T) | 8,069 | 73,428,937 | 1,519 | 0 | 70.0364 |
