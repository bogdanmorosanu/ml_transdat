# Research Unit 01: SPG 2D Distortion Grid Analysis

## Executive Summary
This unit analyzes the official Romanian 2D planimetric distortion field embedded in `rom_grid3d_25.09.spg` (Centrul Național de Cartografie). By evaluating `Grid2D.interp` across the 36,435 points of the canonical `ro_grid_3x3` benchmark, this unit profiles the spatial magnitude, component distributions, and directional vector orientations of the distortion that any candidate machine learning model must replicate.

---

## Key Statistical Findings

| Parameter | Minimum | Mean | Median (p50) | p95 | p99 | Maximum |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Northing Shift ($dN$)** | $-2.7364\,\text{m}$ | $-0.0043\,\text{m}$ | $-0.0125\,\text{m}$ | $+0.4812\,\text{m}$ | $+0.9234\,\text{m}$ | $+1.2293\,\text{m}$ |
| **Easting Shift ($dE$)** | $-0.9421\,\text{m}$ | $+0.0120\,\text{m}$ | $+0.0051\,\text{m}$ | $+0.5218\,\text{m}$ | $+0.8415\,\text{m}$ | $+1.1808\,\text{m}$ |
| **Combined Magnitude** | $+0.0017\,\text{m}$ | $+0.3716\,\text{m}$ | $+0.2642\,\text{m}$ | $+1.1420\,\text{m}$ | $+1.6841\,\text{m}$ | $+2.7570\,\text{m}$ |

---

## Visual Gallery

### 1. Vector Field Quiver (Direction & Magnitude)
Sampled every second grid node (22 km spacing) across the country. The arrows illustrate strong coherent rotational and divergent flow fields, with maximum displacement concentrated in western Romania (Banat/Crișana) pointing south-westward.

![Vector Quiver](plots/spg_shift_quiver.png)

---

### 2. Planimetric Distortion Shift Components ($dN$ & $dE$)
Diverging heatmaps centered at 0.0 m show spatial clustering of positive (red) and negative (blue) corrections:

| Northing Shift ($dN$) | Easting Shift ($dE$) |
|:---:|:---:|
| ![Shift dN](plots/spg_shift_dn.png) | ![Shift dE](plots/spg_shift_de.png) |

---

### 3. Total Shift Magnitude
Combined magnitude $\sqrt{dN^2 + dE^2}$ showing the peak distortion zones:

![Shift Magnitude](plots/spg_shift_mag.png)

---

### 4. Component Distributions & Residual Geometry

| Distribution (Hist & ECDF) | Signed Scatter ($dE$ vs $dN$) |
|:---:|:---:|
| ![Hist ECDF](plots/spg_grid_shifts_2d_bicubic_hist_ecdf.png) | ![Signed Scatter](plots/spg_grid_shifts_2d_bicubic_signed_scatter.png) |

---

## Scientific Takeaways for Modeling
1. **Locality is Essential**: Over 50% of Romania has distortion $< 0.26\,\text{m}$, but border regions reach up to $2.76\,\text{m}$. Global linear approximations will leave multi-meter residual errors.
2. **Vector Coherence**: The smooth directional continuity evident in `spg_shift_quiver.png` confirms that coordinate-conditioned spatial features (e.g. spatial embeddings or RBF networks) have strong regional signals to exploit.
