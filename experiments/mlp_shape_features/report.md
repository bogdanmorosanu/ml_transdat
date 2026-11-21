# Experiment Report: Fourier MLP with Cell Shape Features (`mlp_shape_features`)

## 1. Objective & Hypothesis
This unit tests whether adding the **per-axis cell shape categories** (`cat_dn`, `cat_de` from `CellShapeFeatureExtractor` in `src/geometry/shape_classifier.py`) improves the champion Fourier MLP spatial regressor (`mlp_fourier_4x4` from `mlp_spatial_regressor`).

- **Hypothesis 1 (Input Feature)**: One-hot encoding `cat_dn` and `cat_de` (6 binary features) and concatenating them after the Fourier-encoded `(N, E)` coordinates will help the MLP distinguish Linear, Dome/Bowl, and S-Shape interior regimes.
- **Hypothesis 2 (Sample Weighting)**: Because S-Shape cells have nearly double the baseline RMSE of Dome/Bowl cells (`39.4 mm` vs. `21.1 mm` on `ro_test_adaptive_3mm_mid`), upweighting S-Shape training points (`2x`, `3x`) will focus network capacity on high-curvature regions without introducing cell-boundary discontinuities.

---

## 2. Experimental Setup & Methodology
- **Baseline**: `mlp_fourier_4x4` (`hidden_layer_sizes=(256, 128)`, `use_fourier_features=True`, `fourier_scale=3.0`, `num_fourier_frequencies=64`, `solver="adam"`, `batch_size=128`, `learning_rate_init=0.001`, `tol=1e-7`, `n_iter_no_change=25`, `max_iter=300`, `alpha=1e-6`), trained on `RO_GRID_4X4` (36,435 points, ~2.7 km).
- **Input-Feature Sweep**: Same architecture with `use_shape_features=True` at `shape_tol_mm` $\in \{3.0, 5.0, 10.0\}$.
- **Sample-Weighting Sweep**: Same architecture with `use_shape_features=False` and `s_shape_weight` $\in \{2, 3\}$ on 5 mm S-Shape cells (`cat_2d == 3`).
- **Evaluation**: `RO_TEST_UNIFORM_4X4` (36,752 points nationwide) and `RO_TEST_ADAPTIVE_3MM_MID` (176,350 points in high-curvature zones).

---

## 3. Quantitative Results

### 3.1 Nationwide Uniform (`ro_test_uniform_4x4`, 36,752 pts)

| Model | Mean 2D (mm) | RMSE 2D (mm) | P95 (mm) | Max 2D (mm) | < 1 cm (%) | Fit Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `mlp_fourier_4x4` (Baseline) | **10.3** | 13.0 | 23.1 | 374.7 | **57.9%** | 105.1 s |
| `mlp_fourier_shape_3mm_4x4` | 11.5 | 14.0 | 25.6 | 189.3 | 51.2% | 83.7 s |
| `mlp_fourier_shape_4x4` (5 mm) | 11.5 | 14.4 | 25.3 | 288.5 | 50.4% | 49.2 s |
| `mlp_fourier_shape_10mm_4x4` | 10.9 | 13.1 | 23.7 | **174.7** | 53.3% | 78.9 s |
| `mlp_fourier_sweight2_4x4` (2x) | **10.3** | 12.7 | 22.5 | 179.1 | 57.5% | 80.0 s |
| `mlp_fourier_sweight3_4x4` (3x) | 10.5 | **12.5** | **21.8** | 221.3 | 54.0% | 122.3 s |

### 3.2 Adaptive 3 mm Stress Test (`ro_test_adaptive_3mm_mid`, 176,350 pts)

| Model | Mean 2D (mm) | RMSE 2D (mm) | P95 (mm) | Max 2D (mm) | < 1 cm (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `mlp_fourier_4x4` (Baseline) | **18.4** | **32.5** | **54.3** | 514.6 | **40.9%** |
| `mlp_fourier_shape_3mm_4x4` | 20.2 | 34.1 | 57.4 | 501.9 | 33.9% |
| `mlp_fourier_shape_4x4` (5 mm) | 21.9 | 40.8 | 63.4 | 615.5 | 33.8% |
| `mlp_fourier_shape_10mm_4x4` | 19.8 | 36.1 | 59.4 | 646.1 | 38.2% |
| `mlp_fourier_sweight2_4x4` (2x) | 19.3 | 33.7 | 57.7 | 486.6 | 40.3% |
| `mlp_fourier_sweight3_4x4` (3x) | 19.1 | 34.1 | 58.2 | **485.8** | 39.1% |

### 3.3 Baseline Error Breakdown by 5 mm Cell Shape (`ro_test_adaptive_3mm_mid`)
- **Dome / Bowl** ($n = 79,242$): Mean = `14.9 mm`, RMSE = `21.1 mm`, P95 = `41.9 mm`, Max = `170.6 mm`, `<1 cm` = `47.2%`
- **S-Shape** ($n = 97,108$): Mean = `21.3 mm`, RMSE = `39.4 mm`, P95 = `62.6 mm`, Max = `514.6 mm`, `<1 cm` = `35.8%`

---

## 4. Visual Diagnostics
- [`plots/cell_transect_66km_mlp_shape.png`](plots/cell_transect_66km_mlp_shape.png): 66 km transect comparing the baseline Fourier MLP against the 3 mm, 5 mm, and 10 mm shape-feature variants.

---

## 5. Takeaways & Verdict
- **Verdict**: `Dead End` (for discrete cell-shape input features and simple S-Shape upweighting on `sklearn` `MLPRegressor`).
- **Key Findings**:
  1. Adding discrete per-axis cell shape categories (`cat_dn`, `cat_de`) as one-hot inputs consistently degrades RMSE and `< 1 cm` pass rate across `3 mm`, `5 mm`, and `10 mm` tolerances, because piecewise-constant cell labels introduce artificial discontinuities at $11\,\text{km}$ cell boundaries and trigger early stopping prematurely.
  2. Upweighting S-Shape cells (`2x`, `3x`) trims worst-case `Max` outliers (`374.7 mm` $\to$ `179.1 mm` on uniform) and slightly improves uniform RMSE (`13.0 mm` $\to$ `12.5 mm`), but does not improve `ro_test_adaptive_3mm_mid` RMSE (`32.5 mm` $\to$ `33.7–34.1 mm`).
  3. Future neural approaches should use **continuous local residual decomposition** (`shift - base_shift` from `(u, v)`) or **multi-scale Fourier features** in PyTorch rather than discrete cell-level labels.
