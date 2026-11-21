# Grid Cell Curvature & Shape Classification (`cell_planar_features`)

- **Status**: `Validated`
- **Spawned From**: [`rbf_surface_regressor`](rbf_surface_regressor.md), [`mlp_spatial_regressor`](mlp_spatial_regressor.md)
- **Linked Sources**: [`experiments/cell_planar_features/`](../../experiments/cell_planar_features/report.md), [`src/geometry/`](../../src/geometry/shape_classifier.py), [task](../../.agents/tasks/cell-planar-features.md)

---

## Current Verdict
Establishes that cell planar slope magnitude and compass aspect do not govern coordinate transformation complexity—mathematical difficulty is strictly determined by **internal polynomial order curvature**. 

Confirms that under a **$5\,\text{mm}$ geodetic tolerance**, Romania's $11 \times 11\,\text{km}$ official distortion grid (`rom_grid3d_25.09.spg`) cleanly partitions into **$53.2\%$ 2nd-degree Parabolic Domes/Bowls** and **$46.8\%$ 3rd-degree Cubic S-Shapes**, with higher-order anomalies vanishing ($< 0.5\%$). At **$10\,\text{mm}$ ($1\,\text{cm}$)**, **$92.5\%$ of the entire country is captured by a simple 2nd-degree parabola**. 

Delivers [`CellShapeFeatureExtractor`](../../src/geometry/shape_classifier.py), providing high-speed $O(1)$ spatial feature extraction for arbitrary $(N, E)$ query points.

---

## Premise & Rationale
All prior continuous ML models in `ml_transdat` (Fourier MLPs, Bivariate Polynomials, Thin Plate Splines) were trained on raw 2D metric coordinates $(N, E)$. Feeding raw coordinates forced models to simultaneously resolve the $800\,\text{km}$ macroscopic continental trend and the microscopic $11\,\text{km}$ cell-interior bicubic patches.

To overcome this scale conflict, we investigated:
1. Re-parameterizing each cell into local fractional coordinates $(u, v) \in [0, 1]^2$.
2. Zero-offsetting base shifts by subtracting corner values $(z_{00}^N, z_{00}^E)$.
3. Progressive polynomial order testing (Degrees 1, 2, 3) on interior test points to establish a rigorous 3-class shape taxonomy.

---

## Key Strategic Findings

1. **Category 4 Dropped (Artifact of Overly Strict Tolerances)**:
   At $2\,\text{mm}$ (strict), $14.2\%$ of cells required high-order bicubic terms. Relaxing tolerance to the operational benchmark of $5\,\text{mm}$ caused this anomaly to vanish ($< 0.5\%$, only 9 to 11 cells nationwide on the Danube Delta border). Clamping all residuals into Category 3 establishes a permanent, robust **3-class taxonomy**:
   - `1: Linear` (Degree 1)
   - `2: Dome/Bowl` (Degree 2)
   - `3: S-Shape` (Degree 3)

2. **Full-Grid Multi-Tolerance Distribution ($N = 2,297$ Interior Cells)**:
   - **At $5.0\,\text{mm}$**: $53.2\%$ Dome/Bowl vs. $46.8\%$ S-Shape (clean, balanced functional split).
   - **At $10.0\,\text{mm}$ ($1\,\text{cm}$)**: $92.5\%$ Dome/Bowl, $4.8\%$ S-Shape, $2.7\%$ Linear.
   - **At $15.0\,\text{mm}$ ($1.5\,\text{cm}$)**: Linear expands to $15.4\%$ (2D) and $> 38\%$ on individual axes.

3. **High Axis Concordance ($\Delta N$ vs. $\Delta E$)**:
   In **$68.7\%$** of cells at $5\,\text{mm}$ (and **$69.6\%$** at $10\,\text{mm}$), both Northing and Easting coordinate shifts share the exact same shape category. In the remaining $\approx 31\%$, one axis is merely one degree simpler than the other (e.g. Dome + S-Shape). Extreme mismatches (Linear + S-Shape) account for $< 0.5\%$ nationwide.

4. **Production Reusable Service (`CellShapeFeatureExtractor`)**:
   Externalized in `src/geometry/shape_classifier.py` and backed by precomputed Parquet matrices, enabling downstream ML models to query local $(u, v)$, base offsets, and discrete shape categories for any $(N, E)$ coordinates in microseconds.
