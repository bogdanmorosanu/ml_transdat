# Experiment Report: `submillimeter_exploration_v2`

- **Status**: `Validated` (Screening Complete in `905.1s`)
- **Evaluation Domain**: `box10_interior` (`10×10` interior SPG cells, `row_start=19, col_start=28`)
- **Test Benchmarks**: `RO_TEST_UNIFORM_16X16` (`25,600` pts) and `RO_TEST_ADAPTIVE_1MM_MID`

---

## 1. Objective & Hypothesis

This experiment executes the second-wave, exhaustive pre-finalization model screening defined in [`docs/exploration/submillimeter_v2_model_landscape.md`](../../docs/exploration/submillimeter_v2_model_landscape.md).

**Core Hypothesis**: Evaluating all **12 previously untested model families**—spanning modern zero-lookup Implicit Neural Representations (`KAN`, `WIRE`, `FINER`, `MFN`, `MoE`/`FBPINN`), structured neural-geometric hybrids (`NeuralBicubic`, `3-level FeatureGrid` on border-safe Stage-1 bases, `K-Planes`), and zero-PyTorch regressors (`RFF-ELM`, `LS-SVM`, `XGBoost` on residuals, `LinearLeafTree`, `LocalKriging`, `PU-RBF`, `LOESS deg=3`, `2D Wavelets`)—under an apples-to-apples interior benchmark will reveal:
1. Whether a border-safe Stage-1 base + 3-level `NeuralFeatureGrid` eliminates Stage-1 RBF border spikes while sharpening precision below `0.2 mm`.
2. Whether any modern zero-lookup (`0 MB`) INR breaks the `1.0 mm` `Adapt 1mm` RMSE barrier (`A2_Champion_ResNet_32x32` = `1.200 mm`).
3. Which **zero-PyTorch** families approach `~1–2 mm` accuracy and where their speed/memory Pareto trade-offs sit relative to `Adaptive Quadtree (1mm)` (`74.9 MB`).

---

## 2. Experimental Setup & Methodology

- **Domain (`box10_interior`)**: `10×10` interior SPG cells (`row_start=19, col_start=28`). Because Romania's national border (`*_BORDER`) sits $\ge 3$ grid nodes inside the `.spg` outer edge, screening focuses strictly on interior accuracy, `11 km` cell-wall kink resilience (`a1_wall_rmse_mm` within `5%` of cell borders), inference latency (`µs/pt`), and projected Full-Romania footprint (`est_full_ro_mb`, scaling subgrid-bound lookup/tree sizes by `23×`).
- **Stage-1 Bases (Fitted on Full `RO_GRID_4X4`)**: `lpoly_deg2_k16_tric` (default for residual learners), `cubic_k25_s0` (Anchor `A1`), `CellBilinearShiftModel`, and `ParameterizedBicubicGridShiftModel("catmull_rom")`.
- **Stage-2 Training**: `RO_GRID_32X32` (`103,041` pts) for neural/tree/ELM models; `8x8` (`6,561` pts) for `LocalKriging` / `PU-RBF`; `16x16` (`25,600` pts) for `LOESS deg=3` / `2D Haar Wavelet`.

---

## 3. Quantitative Results

### Consolidated Screening Scorecard (Sorted by `Adapt 1mm` RMSE)

| model | track | family | zero_pytorch | u16_mean_mm | u16_rmse_mm | u16_lt_1mm_pct | a1_mean_mm | a1_rmse_mm | a1_p95_mm | a1_max_mm | a1_lt_1mm_pct | a1_wall_rmse_mm | fit_time_s | latency_us_pt | throughput_kpts_s | weights_kb | lookup_kb | est_full_ro_mb |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| V2_6b_catmull_featgrid_3lvl | V2-6 | Micro-Grid Hybrid | No | 0.1110 | 0.1310 | 100.0000 | 0.1300 | 0.1540 | 0.2830 | 0.8520 | 100.0000 | 0.1600 | 43.3100 | 0.6650 | 1504.8000 | 85.4000 | 2760.4000 | 2.7800 |
| V2_6a_bilinear_featgrid_3lvl | V2-6 | Micro-Grid Hybrid | No | 0.1650 | 0.1950 | 100.0000 | 0.2130 | 0.2510 | 0.4700 | 1.0720 | 99.9800 | 0.2680 | 43.1800 | 0.3670 | 2725.2000 | 85.3000 | 1897.0000 | 1.9400 |
| A1_Champion_FeatGrid_32x32 | Anchor | Micro-Grid Hybrid | No | 0.3270 | 0.3930 | 98.6300 | 0.4130 | 0.4920 | 0.9360 | 2.1190 | 96.2500 | 0.5270 | 43.5300 | 37.3500 | 26.8000 | 3044.8000 | 588.5000 | 3.5500 |
| A3_Quadtree_1mm | Anchor | Geometric Lookup | Yes | 0.5110 | 0.6190 | 93.4800 | 0.4440 | 0.5550 | 1.0280 | 3.8350 | 94.1800 | 0.5180 | 0.0000 | 1.4730 | 678.9000 | 9.6000 | 76,688.8 | 74.9100 |
| V2_2a_lpoly_wire | V2-2 | Modern INR (Wavelet) | No | 0.4650 | 0.5820 | 92.8400 | 0.7520 | 0.9450 | 1.8980 | 5.2920 | 74.6900 | 1.0660 | 107.6900 | 9.0500 | 110.5000 | 2703.6000 | 0.0000 | 2.6500 |
| V2_2b_lpoly_finer | V2-2 | Modern INR (Periodic) | No | 0.5790 | 0.6950 | 88.2200 | 0.8390 | 1.0230 | 1.9720 | 5.7530 | 69.9700 | 1.1380 | 44.7700 | 8.8490 | 113.0000 | 2670.8000 | 0.0000 | 2.6100 |
| V2_1a_lpoly_rbf_kan | V2-1 | Modern INR (KAN) | No | 0.4590 | 0.5870 | 92.4400 | 0.8380 | 1.0580 | 2.1070 | 6.2480 | 68.5100 | 1.2020 | 68.7900 | 13.7770 | 72.6000 | 3012.0000 | 0.0000 | 2.9500 |
| V2_4b_lpoly_fbpinn | V2-4 | Modern INR (FBPINN) | No | 0.4890 | 0.6600 | 89.1500 | 0.8350 | 1.1060 | 2.2400 | 8.3360 | 70.2100 | 1.2400 | 45.6300 | 13.2010 | 75.8000 | 2954.2000 | 0.0000 | 2.8900 |
| V2_4a_lpoly_spatial_moe | V2-4 | Modern INR (MoE) | No | 0.5020 | 0.6730 | 88.5500 | 0.8490 | 1.1160 | 2.2390 | 8.4770 | 69.0400 | 1.2520 | 46.8100 | 12.8460 | 77.8000 | 2826.8000 | 0.0000 | 2.7700 |
| A2_Champion_ResNet_32x32 | Anchor | Zero-Lookup ResNet | No | 0.5720 | 0.7400 | 85.8500 | 0.9280 | 1.2000 | 2.3800 | 8.4710 | 65.0400 | 1.3380 | 65.4800 | 12.7270 | 78.6000 | 2945.1000 | 0.0000 | 2.8800 |
| V2_7_lpoly_kplanes | V2-7 | Structured Hybrid | No | 0.8220 | 1.0150 | 71.2500 | 1.1200 | 1.3910 | 2.7370 | 10.0530 | 54.8200 | 1.4940 | 34.3100 | 9.0770 | 110.2000 | 2490.6000 | 16.0000 | 2.4500 |
| V2_1b_lpoly_fourier_kan | V2-1 | Modern INR (KAN) | No | 1.0680 | 1.3030 | 56.3700 | 1.2580 | 1.5070 | 2.8650 | 10.8660 | 45.5300 | 1.5390 | 77.7700 | 12.5310 | 79.8000 | 3252.1000 | 0.0000 | 3.1800 |
| V2_10a_lpoly_xgboost | V2-10 | Zero-PT Tree Ensemble | Yes | 0.9040 | 1.2420 | 68.8600 | 1.3500 | 1.7710 | 3.6140 | 10.9310 | 48.7600 | 1.9140 | 3.3800 | 10.8630 | 92.1000 | 8190.4000 | 0.0000 | 183.9700 |
| V2_5a_neural_bicubic_grid | V2-5 | Structured Hybrid | No | 0.7270 | 1.0730 | 77.2000 | 1.3920 | 2.0430 | 4.1000 | 22.1530 | 51.8400 | 2.3150 | 27.7600 | 0.3570 | 2798.1000 | 1.9000 | 521.1000 | 0.5200 |
| V2_11_local_kriging_8x8 | V2-11 | Zero-PT Geostatistics | Yes | 1.1030 | 1.5480 | 60.9300 | 1.6920 | 2.1990 | 4.6170 | 11.8530 | 37.3600 | 2.3980 | 0.0600 | 47.2310 | 21.2000 | 2894.5000 | 0.0000 | 65.0200 |
| V2_12a_pu_rbf_8x8 | V2-12 | Zero-PT Patch-RBF | Yes | 1.1620 | 1.5930 | 57.9300 | 1.7260 | 2.2310 | 4.6580 | 11.9400 | 36.2800 | 2.4280 | 0.1000 | 11.5400 | 86.7000 | 3223.8000 | 0.0000 | 72.4100 |
| V2_10b_lpoly_linear_leaf_tree | V2-10 | Zero-PT Model Tree | Yes | 1.5190 | 2.2020 | 49.6600 | 2.1430 | 2.8180 | 5.8200 | 21.5790 | 30.8400 | 2.8750 | 1.1100 | 9.0100 | 111.0000 | 2855.5000 | 0.0000 | 64.1400 |
| V2_12b_loess_deg3_16x16 | V2-12 | Zero-PT Local Poly | Yes | 0.2680 | 0.6260 | 91.9800 | 2.0380 | 2.9420 | 6.3020 | 18.0050 | 43.5700 | 3.4360 | 0.0100 | 18.2900 | 54.7000 | 1688.7000 | 0.0000 | 37.9300 |
| V2_3a_lpoly_gabor_mfn | V2-3 | Modern INR (MFN) | No | 2.5460 | 3.6010 | 34.3000 | 2.5990 | 3.5600 | 7.3140 | 23.2870 | 29.0900 | 3.0270 | 102.7100 | 8.4030 | 119.0000 | 2727.9000 | 0.0000 | 2.6700 |
| V2_8_lpoly_rff_elm | V2-8 | Zero-PT Closed-Form | Yes | 2.1900 | 2.6640 | 21.7000 | 3.4400 | 4.0740 | 7.5760 | 18.8790 | 9.3700 | 4.4410 | 11.4400 | 58.8600 | 17.0000 | 2512.1000 | 0.0000 | 2.4600 |
| V2_5b_neural_bicubic_mlp | V2-5 | Structured Hybrid | No | 2.3880 | 2.8980 | 18.4700 | 3.4530 | 4.1360 | 7.8170 | 20.0560 | 9.9600 | 4.4430 | 54.4400 | 12.0600 | 82.9000 | 2803.9000 | 0.0000 | 2.7400 |
| V2_9_lpoly_lssvm_nystroem | V2-9 | Zero-PT Kernel ML | Yes | 3.3950 | 4.7450 | 31.2500 | 3.2870 | 4.5440 | 9.6100 | 25.4830 | 26.5000 | 3.7310 | 8.5200 | 56.9700 | 17.6000 | 20,241.8 | 0.0000 | 19.7700 |
| A0_Stage1_lpoly_only | Anchor | Stage-1 Classical | Yes | 3.4320 | 4.8090 | 31.2300 | 3.3020 | 4.5830 | 9.6290 | 26.9940 | 27.0000 | 3.7510 | 0.0000 | 10.2720 | 97.4000 | 2462.8000 | 0.0000 | 2.4100 |
| V2_3b_lpoly_fourier_mfn | V2-3 | Modern INR (MFN) | No | 4.5220 | 5.3600 | 5.2900 | 4.4100 | 5.2140 | 9.7130 | 23.5320 | 5.3600 | 4.9760 | 42.5800 | 8.2540 | 121.2000 | 2695.5000 | 0.0000 | 2.6400 |
| V2_12c_wavelet_dwt_16x16 | V2-12 | Zero-PT Signal Proc | Yes | 4.5860 | 7.2370 | 27.3100 | 7.0050 | 11.4480 | 25.4660 | 101.9610 | 16.9300 | 9.5590 | 0.0100 | 0.2600 | 3840.2000 | 406.1000 | 450.0000 | 10.5100 |

---

## 4. Visual Diagnostics
- ![Accuracy vs. Inference Latency Pareto](plots/v2_accuracy_vs_latency_pareto.png)
- ![Accuracy vs. Projected Full-Romania Memory Pareto](plots/v2_accuracy_vs_memory_pareto.png)
- ![Family Accuracy & Wall-Kink Stress Comparison](plots/v2_family_comparison_bar.png)

---

## 5. Takeaways & Per-Track Verdicts

1. **Micro-Grid Hybrid Breakthrough (`V2-6` — `Promote to Full-Romania`)**:
   - `V2_6b_catmull_featgrid_3lvl` (`0.111 mm` `U16` mean, `0.154 mm` `A1` RMSE, **`100.00% < 1 mm`**, `0.852 mm` max, `0.665 µs/pt`, `2.78 MB`) and `V2_6a_bilinear_featgrid_3lvl` (`0.165 mm` `U16` mean, `0.251 mm` `A1` RMSE, `99.98% < 1 mm`, `0.367 µs/pt`, `1.94 MB`) beat `A1_Champion_FeatGrid_32x32` by **2×–3× in accuracy** and **50×–100× in inference speed**, while eliminating Stage-1 RBF border spikes.
2. **Zero-Lookup (`0 MB`) Neural INRs (`V2-2`, `V2-1`, `V2-4` — `Promote WIRE / RBF-KAN / FBPINN`)**:
   - Complex Gabor `WIRE` (`V2_2a_lpoly_wire`) broke `1.0 mm` `A1` RMSE (`0.945 mm`, `74.69% < 1 mm`, `0.465 mm` `U16` mean), outperforming `A2_Champion_ResNet_32x32` (`1.200 mm`). `FINER` (`1.023 mm`), `RBF-KAN` (`1.058 mm`, `0.459 mm` `U16` mean), `FBPINN` (`1.106 mm`), and `SpatialMoE` (`1.116 mm`) all beat the `ResNet` anchor cleanly.
   - `MFN` (`GaborNet`/`FourierNet`, `3.56–5.21 mm`) is a confirmed `Dead End`.
3. **Structured Hybrids (`V2-5`, `V2-7` — `Viable Niche` / `Dead End`)**:
   - `V2_5a_neural_bicubic_grid` is ultra-compact (`0.52 MB`) and ultra-fast (`0.357 µs/pt`), hitting `0.727 mm` `U16` mean (`2.043 mm` `A1` RMSE), but `V2_6a/b` are ~10× more accurate for ~1.4–2.2 MB more. `V2_5b_neural_bicubic_mlp` (`4.136 mm`) is a `Dead End`. `V2_7_lpoly_kplanes` (`1.391 mm` `A1` RMSE) sits slightly behind `ResNet`.
4. **Zero-PyTorch Models (`V2-8`..`V2-12`)**:
   - `Adaptive Quadtree (1mm)` (`0.555 mm` `A1` RMSE, `1.47 µs/pt`, `74.9 MB`) remains the overall Zero-PyTorch accuracy champion.
   - `LocalPoly deg=3` (`V2_12b`, `0.268 mm` `U16` mean / `2.942 mm` `A1` RMSE), `XGBoost` on residuals (`V2_10a`, `1.771 mm` `A1` RMSE, `10.86 µs/pt`), `LocalKriging` (`2.199 mm`), `PU-RBF` (`2.231 mm`), and `LinearLeafTree` (`2.818 mm`) are viable `1–3 mm` Zero-PyTorch candidates if memory/compression is tuned; `RFF-ELM` (`4.07 mm`), `LS-SVM` (`4.54 mm`), and `2D Haar Wavelets` (`11.45 mm`) are confirmed `Dead End`s.

---

## 6. Next-Step Full-Romania Promotion Candidates
*(Evaluated in Section 7 below; ready for dedicated finalization/fine-tuning):*
1. **Micro-Grid Tier**: `V2_6a_bilinear_featgrid_3lvl` & `V2_6b_catmull_featgrid_3lvl`
2. **Zero-Lookup Neural Tier (`0 MB`)**: `V2_4b_lpoly_fbpinn` (`FBPINN` scaled cleanly nationwide, overtaking `WIRE` and `RBF-KAN`)
3. **Zero-PyTorch Tier**: `LocalPoly deg=3 (16x16)` (`38.69 MB`) & `XGBoost` (`59.40 MB`) vs. `Adaptive Quadtree (1mm)` (`74.91 MB`)

---

## 7. Full-Romania Validation Check (`--full-ro`, `*_BORDER` Evaluation)

Trained on Full Romania `RO_GRID_32X32` (`2.35M` pts; `RO_GRID_16X16` for `LocalPoly deg=3`) and evaluated on `RO_TEST_UNIFORM_16X16_BORDER` and `RO_TEST_ADAPTIVE_1MM_MID_BORDER` (`2048.9s` total):

| model | track | family | zero_pytorch | u16_mean_mm | u16_rmse_mm | u16_lt_1mm_pct | a1_mean_mm | a1_rmse_mm | a1_p95_mm | a1_max_mm | a1_lt_1mm_pct | a1_wall_rmse_mm | fit_time_s | latency_us_pt | throughput_kpts_s | weights_kb | lookup_kb | est_full_ro_mb |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| V2_6a_bilinear_featgrid_3lvl_FullRO | V2-6 | Micro-Grid Hybrid | No | 0.0680 | 0.0810 | 100.0000 | 0.0820 | 0.0980 | 0.1860 | 0.7890 | 100.0000 | 0.1040 | 107.0500 | 0.5880 | 1699.9000 | 85.3000 | 1897.0000 | 1.9400 |
| V2_6b_catmull_featgrid_3lvl_FullRO | V2-6 | Micro-Grid Hybrid | No | 0.0680 | 0.0820 | 100.0000 | 0.0840 | 0.1030 | 0.1960 | 1.1060 | 100.0000 | 0.1110 | 93.1600 | 0.7830 | 1277.3000 | 85.4000 | 2760.4000 | 2.7800 |
| A3_Quadtree_1mm | Anchor | Geometric Lookup | Yes | 0.4950 | 0.5960 | 94.0800 | 0.4800 | 0.5990 | 1.0830 | 9.0970 | 92.3600 | 0.5910 | 0.0000 | 1.3060 | 766.0000 | 9.6000 | 76,688.8 | 74.9100 |
| V2_4b_lpoly_fbpinn_FullRO | V2-4 | Modern INR (FBPINN) | No | 0.4230 | 0.5590 | 92.9800 | 0.6630 | 0.8930 | 1.8130 | 10.9170 | 80.1200 | 1.0120 | 1088.2500 | 18.8660 | 53.0000 | 5379.5000 | 0.0000 | 5.2600 |
| V2_12b_loess_deg3_16x16_FullRO | V2-12 | Zero-PT Local Poly | Yes | 0.2450 | 0.5460 | 91.8900 | 0.9510 | 1.4020 | 3.0350 | 11.0770 | 63.3000 | 1.6720 | 0.1700 | 20.4730 | 48.8000 | 39,617.0 | 0.0000 | 38.6900 |
| V2_10a_lpoly_xgboost_FullRO | V2-10 | Zero-PT Tree Ensemble | Yes | 0.8160 | 1.1000 | 71.1800 | 1.1370 | 1.5590 | 2.9980 | 77.7060 | 56.7100 | 1.6680 | 187.8100 | 14.9030 | 67.1000 | 60,822.3 | 0.0000 | 59.4000 |
| V2_1a_lpoly_rbf_kan_FullRO | V2-1 | Modern INR (KAN) | No | 2.1870 | 2.3940 | 8.4400 | 2.3280 | 2.6180 | 4.5180 | 13.5030 | 10.8700 | 2.6580 | 292.7500 | 12.7410 | 78.5000 | 4270.3000 | 0.0000 | 4.1800 |
| V2_2a_lpoly_wire_FullRO | V2-2 | Modern INR (Wavelet) | No | 2.1810 | 3.1740 | 41.7000 | 2.2870 | 3.2600 | 6.9920 | 27.1820 | 37.3600 | 2.7030 | 189.2900 | 11.6610 | 85.8000 | 3366.6000 | 0.0000 | 3.2900 |

### Nationwide Key Findings
- **Sub-100-Micrometer Micro-Grid Champion**: `V2_6a_bilinear_featgrid_3lvl_FullRO` reaches **`0.068 mm` `U16` mean / `0.098 mm` `A1` RMSE (`100.00% < 1 mm`, `max = 0.789 mm`, `0.588 µs/pt`, `1.94 MB`)** across all of Romania inside the border!
- **Zero-Lookup (`0 MB`) Scaling Surprise**: `WIRE` (`3.260 mm`) and `RBF-KAN` (`2.618 mm`) hit a nationwide capacity wall when scaling from 100 to 2,297 cells, whereas **`FBPINN` (`8×10` overlapping subdomain MLPs)** scaled cleanly to **`0.423 mm` `U16` mean / `0.893 mm` `A1` RMSE (`80.12% < 1 mm`, `5.26 MB`)**.
- **Zero-PyTorch Nationwide**: `LocalPoly deg=3 (16x16)` reaches `0.245 mm` `U16` mean / `1.402 mm` `A1` RMSE at `38.69 MB` (`0.17s` fit), and `XGBoost` reaches `0.816 mm` `U16` mean / `1.559 mm` `A1` RMSE at `59.40 MB`—both trailing `Adaptive Quadtree (1mm)` (`0.599 mm` `A1` RMSE, `74.91 MB`, `1.306 µs/pt`).

---

## 8. Takeaways & Proposed Next Experiments

### Key Scientific Takeaways
1. **Border-Safe Stage-1 + 3-Level Feature Grid Solves Both Accuracy and Speed**: Replacing `cubic_k25_s0` with `CellBilinear` or `Catmull-Rom` and adding a 3rd (`~2.75 km`) grid level (`V2_6a` / `V2_6b`) achieves **`0.068 mm` (`68 µm`) `U16` mean and `0.098 mm` `A1` RMSE (`100.00% < 1 mm`, `max = 0.789 mm`)** across all of Romania inside the border at **`0.588 µs/pt` (~1.7M pts/s)** and **`1.94 MB`** total footprint.
2. **Domain Decomposition (`FBPINN`) Is the Key to Zero-Lookup (`0 MB`) Nationwide Scaling**: While `WIRE` and `RBF-KAN` beat `FBPINN` on `box10_interior`, they hit a capacity ceiling at Full-Romania scale (`3.26 mm` and `2.62 mm`). Partitioning Romania into `8×10` overlapping subdomain MLPs (`FBPINN`) scaled cleanly to **`0.423 mm` `U16` mean / `0.893 mm` `A1` RMSE (`80.12% < 1 mm`, `5.26 MB`)**.
3. **Zero-PyTorch Deployment Frontier**: `Adaptive Quadtree (1mm)` (`0.599 mm` `A1` RMSE, `1.31 µs/pt`, `74.91 MB`) remains the strongest non-neural baseline; `LocalPoly deg=3` (`38.69 MB`, `0.245 mm` `U16` mean) and `XGBoost` (`59.40 MB`, `1.559 mm` `A1` RMSE) show that non-neural models can trade accuracy for smaller lookups or fast CPU/tree deployment.

### Proposed Next Experiments (Ordered by Priority)

1. **Priority 1 — `bilinear_featgrid_3lvl_finalization` (Champion Micro-Grid Refinement & Bidirectional Validation)**
   - **Target Model**: `V2_6a_bilinear_featgrid_3lvl` (`1.94 MB`, `0.068 mm` `U16` mean, `0.098 mm` `A1` RMSE) and `V2_6b_catmull_featgrid_3lvl` (`2.78 MB`).
   - **Scope**: Fine-tune grid resolutions, `feature_dim`, `hidden_dim`, and training density (`16x16` vs. `32x32` vs. adaptive-enriched), benchmark full **bidirectional `ETRS89 <-> Stereo70`** round-trip precision via `ResidualCoordinateTransformer`, and promote the winner to `benchmarks/grid_shift_models/` as the overall project champion.
2. **Priority 2 — `fbpinn_zero_lookup_refinement` (Zero-Lookup `0 MB` Neural Champion Optimization)**
   - **Target Model**: `V2_4b_lpoly_fbpinn` (`0 MB` lookup, `5.26 MB` weights, `0.423 mm` `U16` mean / `0.893 mm` `A1` RMSE).
   - **Scope**: Test denser subdomain grids (`10×12` or `12×16`), adaptive curvature-weighted training (`RO_TRAIN_ADAPTIVE_HYBRID`), and faster inference batching/compilation to push `Adapt 1mm < 1 mm` pass rate from `80.1%` toward `> 95%` with zero spatial lookup tables.
3. **Priority 3 — `zero_pytorch_compression_and_speed` (Non-Neural / Pure-NumPy Deployment Study)**
   - **Target Models**: `LocalPolynomialSurfaceRegressor(degree=3)` (`38.69 MB`), `XGBoost` / `LinearLeafTree` (pruned/compressed), vs. `Adaptive Quadtree (1mm)` (`74.91 MB`) and a pure-NumPy exported `V2_6a` inference runtime (eliminating PyTorch runtime dependency at inference!).
   - **Scope**: Compare pure-NumPy inference export of our 3-level FeatureGrid MLP (which only needs 2 tiny matrix multiplies + `SiLU` in NumPy!) against pruned `XGBoost` / `LinearLeafTree` and `Quadtree` to deliver a sub-millimeter, zero-PyTorch production runtime.
