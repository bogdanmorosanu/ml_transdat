# Experiment: Grid Cell Curvature & Shape Classification (`cell_planar_features`)

## 1. Executive Summary & Objective
This investigation establishes a functional geometric classification of Romania's official $11 \times 11\,\text{km}$ distortion grid (`rom_grid3d_25.09.spg`).

Rather than classifying cells by spatial orientation (slope, aspect) or 4-corner boundary metrics (corner twist), we classify each cell by its **internal polynomial order complexity**:
1. **Category 1 (Linear)**: 1st-degree / Bilinear corner ramp ($\text{MaxError} \le \text{tol}$).
2. **Category 2 (Dome / Bowl)**: 2nd-degree Parabolic surface (monotonic curvature, no inflection).
3. **Category 3 (S-Shape)**: 3rd-degree Cubic surface (inflection S-waves).
4. **Category 4 (High-Order)**: Residual high-order bicubic anomaly ($\text{MaxError} > \text{tol}$).

We evaluate 25 interior test points ($5 \times 5$ grid) inside every valid cell against the official RomGEO bicubic engine across geodetic tolerances ($2\,\text{mm}, 3\,\text{mm}, 5\,\text{mm}, 10\,\text{mm}, 15\,\text{mm}$).

---

## 2. Full-Grid Distribution Across Tolerances ($N = 2,297$ Interior Cells)

| Tolerance | Axis | 1: Linear | 2: Dome / Bowl | 3: S-Shape | 4: High-Order Anomaly | Key Takeaway |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **$2.0\,\text{mm}$** | $\Delta N$ | $0.0\%$ | $11.1\%$ | $81.8\%$ | $7.1\%$ | High-order appears only under strict sub-millimeter tolerances |
| | $\Delta E$ | $0.0\%$ | $16.8\%$ | $78.3\%$ | $5.0\%$ | |
| | **2D** | **$0.0\%$** | **$2.0\%$** | **$83.8\%$** | **$14.2\%$** | |
| **$3.0\,\text{mm}$** | $\Delta N$ | $0.0\%$ | $32.1\%$ | $65.7\%$ | $2.2\%$ | High-order drops to $< 3\%$ across Romania |
| | $\Delta E$ | $0.1\%$ | $42.1\%$ | $56.9\%$ | $0.9\%$ | |
| | **2D** | **$0.0\%$** | **$12.9\%$** | **$83.2\%$** | **$3.9\%$** | |
| **$5.0\,\text{mm}$** | $\Delta N$ | $0.7\%$ | $73.3\%$ | $25.6\%$ | $0.4\%$ | **High-order anomaly vanishes ($< 0.5\%$)** |
| *(Benchmark)* | $\Delta E$ | $2.5\%$ | $76.1\%$ | $21.3\%$ | $0.1\%$ | **Clean 53% Dome / 46% S-Shape split** |
| | **2D** | **$0.0\%$** | **$53.2\%$** | **$46.4\%$** | **$0.5\%$** | |
| **$10.0\,\text{mm}$** | $\Delta N$ | $14.7\%$ | $83.2\%$ | $2.0\%$ | $0.1\%$ | Within $1\,\text{cm}$, **over $92\%$ of Romania** |
| *(1 cm)* | $\Delta E$ | $22.1\%$ | $77.0\%$ | $1.0\%$ | $0.0\%$ | **is modeled by a simple 2nd-degree parabola** |
| | **2D** | **$2.7\%$** | **$92.5\%$** | **$4.7\%$** | **$0.2\%$** | |
| **$15.0\,\text{mm}$** | $\Delta N$ | $38.5\%$ | $61.1\%$ | $0.3\%$ | $0.1\%$ | Linear expands to $\approx 40\%$ on individual axes |
| *(1.5 cm)* | $\Delta E$ | $45.2\%$ | $54.7\%$ | $0.1\%$ | $0.0\%$ | |
| | **2D** | **$15.4\%$** | **$83.8\%$** | **$0.7\%$** | **$0.1\%$** | |

---

## 3. Axis Concordance Analysis ($\Delta N$ vs. $\Delta E$)

How often does the mathematical shape behavior on the Northing axis match the Easting axis?

### At $5.0\,\text{mm}$ Tolerance:
- **Exact Shape Agreement**: **$68.5\%$** of cells share the identical category on both axes.
- **Cross-Tabulation Matrix**:
  - Both axes are **Dome/Bowl**: **1,362 cells** ($59.3\%$)
  - Both axes are **S-Shape**: **210 cells** ($9.1\%$)
  - One axis is **Dome** and the other is **S-Shape**: **642 cells** ($28.0\%$)
  - Linear on one axis and S-shape on the other: **10 cells** ($< 0.5\%$)

### At $10.0\,\text{mm}$ Tolerance:
- **Exact Shape Agreement**: **$69.6\%$** of cells share the identical category.
- **Cross-Tabulation Matrix**:
  - Both axes are **Dome/Bowl**: **1,495 cells** ($65.1\%$)
  - Both axes are **Linear**: **98 cells** ($4.3\%$)
  - One axis is **Linear** while the other is **Dome**: **640 cells** ($27.9\%$)

---

## 4. Generated Artifacts & Maps
- **Datasets**:
  - `results/cell_shape_metrics.parquet`: Contains continuous maximum errors for Degrees 1, 2, and 3 for $\Delta N, \Delta E$, and 2D, plus classified categories for all tolerances ($2, 3, 5, 10, 15\,\text{mm}$).
  - `results/cell_shape_metrics.csv`
- **National Classification Maps**:
  - `plots/national_shape_map_5mm.png`: National 3-panel map showing $\Delta N$, $\Delta E$, and Combined 2D categories at $5\,\text{mm}$ tolerance.
  - `plots/national_shape_map_10mm.png`: National 3-panel map at $10\,\text{mm}$ ($1\,\text{cm}$) tolerance.
