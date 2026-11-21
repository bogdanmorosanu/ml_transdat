# Experiment Report: Grid Cell Shape Classification & Feature Engineering (`cell_planar_features`)

## 1. Executive Summary & Problem Context
All prior transformation models in `ml_transdat` (Bivariate Polynomials, Decision Trees, Fourier MLPs, Thin Plate Splines) were trained strictly on raw 2D metric coordinates $(N, E)$. This forced every model to simultaneously learn two conflicting scales:
1. The macroscopic continental distortion ($800\,\text{km}$ span, $2.76\,\text{m}$ peak shift).
2. The microscopic $11\,\text{km}$ cell-interior bicubic patches and boundary kinks.

This study introduces a **Local Geometric Patch Decomposition**:
- Each cell is normalized to zero-offset by subtracting its base corner shift $(z_{00}^N, z_{00}^E)$.
- Query points are represented in local fractional coordinates $(u, v) \in [0, 1]^2$.
- Every cell is classified into a strict **3-Class Curvature Taxonomy** based on the polynomial degree required to model its interior surface:
  - **Category 1 (Linear)**: 1st-degree flat ramp.
  - **Category 2 (Dome / Bowl)**: 2nd-degree parabolic surface (monotonic curvature, no inflection).
  - **Category 3 (S-Shape)**: 3rd-degree cubic surface (inflection S-waves and high-order residuals).

---

## 2. Methodology & Mathematical Formulation

### 2.1 Interior Sampling & Order Evaluation
Inside every valid land cell, a regular $5 \times 5 = 25$ point interior grid is sampled:
$$u \in [0.1, 0.9], \quad v \in [0.1, 0.9]$$
Points are placed strictly in the interior to test genuine surface curvature away from the cell boundary.

At each point, ground-truth shifts $(\Delta N_{\text{true}}, \Delta E_{\text{true}})$ are queried from the official RomGEO bicubic engine (`pytransdat`) and compared against three successive polynomial orders:

1. **Order 1 (Bilinear Flat Ramp)**:
   $$P_1(u, v) = (1-u)(1-v)z_{00} + u(1-v)z_{01} + (1-u)v z_{10} + uv z_{11}$$
2. **Order 2 (Parabolic Surface - 6 Terms)**:
   $$P_2(u, v) = a_0 + a_1 u + a_2 v + a_3 u^2 + a_4 v^2 + a_5 uv$$
3. **Order 3 (Cubic Inflection Surface - 10 Terms)**:
   $$P_3(u, v) = P_2(u, v) + a_6 u^3 + a_7 v^3 + a_8 u^2 v + a_9 u v^2$$

### 2.2 Decision Metric: Maximum Absolute 2D Error ($L_\infty$ Norm)
$$\text{MaxError}_{\text{2D}} = \max_{i=1\dots 25} \sqrt{\left(\Delta N_{\text{true}, i} - P(u_i, v_i)_N\right)^2 + \left(\Delta E_{\text{true}, i} - P(u_i, v_i)_E\right)^2}$$

Worst-case peak deviation is chosen over RMSE because geodetic standards enforce strict spatial bounding tolerances.

---

## 3. Empirical Results Across Romania ($N = 2,297$ Interior Cells)

### 3.1 Tolerance Sweep Breakdown

| Tolerance ($\epsilon$) | Component | 1: Linear | 2: Dome / Bowl | 3: S-Shape | Strategic Interpretation |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **$2.0\,\text{mm}$** | $\Delta N$ | $0.0\%$ | $11.1\%$ | $88.9\%$ | Sub-millimeter tolerances require 3rd-order cubic terms nationwide. |
| | $\Delta E$ | $0.0\%$ | $16.8\%$ | $83.2\%$ | |
| | **2D** | **$0.0\%$** | **$2.0\%$** | **$98.0\%$** | |
| **$3.0\,\text{mm}$** | $\Delta N$ | $0.0\%$ | $32.1\%$ | $67.9\%$ | Official ANCPI corridor standard; Domes expand to one-third of territory. |
| *(ANCPI)* | $\Delta E$ | $0.1\%$ | $42.1\%$ | $57.8\%$ | |
| | **2D** | **$0.0\%$** | **$12.9\%$** | **$87.1\%$** | |
| **$5.0\,\text{mm}$** | $\Delta N$ | $0.7\%$ | $73.3\%$ | $26.0\%$ | **Primary Benchmark**: Romania cleanly splits into |
| *(Primary)* | $\Delta E$ | $2.5\%$ | $76.1\%$ | $21.4\%$ | **$\mathbf{53.2\%}$ Parabolic Domes vs. $\mathbf{46.8\%}$ Cubic S-Waves**. |
| | **2D** | **$0.0\%$** | **$53.2\%$** | **$46.8\%$** | High-order residuals completely vanish ($< 0.5\%$). |
| **$10.0\,\text{mm}$** | $\Delta N$ | $14.7\%$ | $83.2\%$ | $2.1\%$ | At $1\,\text{cm}$ precision, **$\mathbf{92.5\%}$ of Romania** |
| *(1 cm)* | $\Delta E$ | $22.1\%$ | $77.0\%$ | $1.0\%$ | **is modeled by a simple 2nd-degree parabola**. |
| | **2D** | **$2.7\%$** | **$92.5\%$** | **$4.8\%$** | |
| **$15.0\,\text{mm}$** | $\Delta N$ | $38.5\%$ | $61.1\%$ | $0.4\%$ | Linear ramps expand to $\approx 40\%$ on individual axes. |
| *(1.5 cm)* | $\Delta E$ | $45.2\%$ | $54.7\%$ | $0.1\%$ | |
| | **2D** | **$15.4\%$** | **$83.8\%$** | **$0.7\%$** | |

### 3.2 Axis Concordance ($\Delta N$ vs. $\Delta E$)
- **Agreement**: In **$68.7\%$** of cells at $5\,\text{mm}$ (and **$69.6\%$** at $10\,\text{mm}$), both coordinate axes share the exact same shape category.
- **Cross-Distribution at $5\,\text{mm}$**:
  - Both axes are **Dome/Bowl**: **$1,362\text{ cells}$ ($59.3\%$)**
  - Both axes are **S-Shape**: **$217\text{ cells}$ ($9.4\%$)**
  - One axis is **Dome** while the other is **S-Shape**: **$645\text{ cells}$ ($28.1\%$)**
  - Linear on one axis and S-Shape on the other: **$10\text{ cells}$ ($< 0.5\%$)**

---

## 4. Persistent Reusable Service (`CellShapeFeatureExtractor`)

To allow downstream ML models and evaluation pipelines to query these features dynamically, the functionality is externalized in [`src/geometry/shape_classifier.py`](../../src/geometry/shape_classifier.py).

### Python Usage Example:
```python
from src.geometry.shape_classifier import CellShapeFeatureExtractor, ShapeCategory

# 1. Initialize extractor (loads cached Parquet table in O(1) memory lookup)
extractor = CellShapeFeatureExtractor(default_tol_mm=5.0)

# 2. Single-point query
pt = extractor.lookup_point(northing=335000.0, easting=585000.0)
print(f"Fractional coords: u={pt['u']:.3f}, v={pt['v']:.3f}")
print(f"Shape Category N: {ShapeCategory(pt['cat_dn']).label}")
print(f"Shape Category E: {ShapeCategory(pt['cat_de']).label}")
print(f"Base Corner Shifts: dN={pt['base_dn']:.4f}m, dE={pt['base_de']:.4f}m")

# 3. High-speed vectorized batch query (for ML training arrays)
batch_features = extractor.extract_features(northing_arr, easting_arr, tol_mm=5.0)
# Returns dict of arrays: 'u', 'v', 'cat_dn', 'cat_de', 'cat_2d', 'base_dn', 'base_de', 'is_valid'
```

---

## 5. Generated Artifacts
- **Shared Feature Datasets (`data/features/`)**:
  - [`../../data/features/cell_planes.parquet`](../../data/features/cell_planes.parquet): Planar slope, aspect, and bilinear twist metrics across 2,556 valid cells.
  - [`../../data/features/cell_shape_metrics.parquet`](../../data/features/cell_shape_metrics.parquet): Continuous maximum errors for Degrees 1–3 and 3-class curvature labels across tolerances ($2, 3, 5, 10, 15\,\text{mm}$).
- **Local Unit Outputs**:
  - [`results/cell_planes.csv`](results/cell_planes.csv) and [`results/cell_shape_metrics.csv`](results/cell_shape_metrics.csv)
- **National Spatial Maps**:
  - [`plots/national_cell_slopes_map.png`](plots/national_cell_slopes_map.png), [`plots/national_cell_twist_map.png`](plots/national_cell_twist_map.png), [`plots/national_cell_aspect_distribution.png`](plots/national_cell_aspect_distribution.png)
  - [`plots/national_shape_map_5mm.png`](plots/national_shape_map_5mm.png): 3-panel map showing $\Delta N$, $\Delta E$, and 2D categories at $5\,\text{mm}$.
  - [`plots/national_shape_map_10mm.png`](plots/national_shape_map_10mm.png): 3-panel map at $10\,\text{mm}$ ($1\,\text{cm}$).
- **Unit Tests**:
  - [`tests/test_shape_classifier.py`](../../tests/test_shape_classifier.py): 4 unit tests covering synthetic shape recovery and vectorized spatial lookups (100% passing).
