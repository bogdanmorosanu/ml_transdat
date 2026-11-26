| model | ro_test_uniform_16x16_mean_mm | ro_test_uniform_16x16_rmse_mm | delta_ro_test_uniform_16x16_rmse_mm | ro_test_uniform_16x16_lt_1mm_pct | ro_test_adaptive_1mm_mid_mean_mm | ro_test_adaptive_1mm_mid_rmse_mm | delta_ro_test_adaptive_1mm_mid_rmse_mm | ro_test_adaptive_1mm_mid_lt_1mm_pct | total_size_mb |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Stage1_lpoly_deg2_k16_tric_only | 2.2900 | 5.1700 | 0.0000 | 41.9800 | 5.9300 | 23.4700 | 0.0000 | 29.6600 | 2.4050 |
| TorchMLP_32x32_only | 1.2200 | 2.8300 | -2.3400 | 50.8900 | 3.6800 | 14.2800 | -9.1900 | 23.0400 | 0.4710 |
| Hybrid_lpoly_deg2_k16_tric_plus_resnet_32x32 | 0.5700 | 2.6200 | -2.5500 | 89.7100 | 2.1300 | 13.8200 | -9.6600 | 62.1300 | 2.8760 |
| Hybrid_lpoly_deg2_k16_tric_plus_resnet_32x32_h256 | 0.3300 | 2.8200 | -2.3500 | 97.9600 | 1.7600 | 15.1500 | -8.3200 | 75.8600 | 4.0820 |
| Hybrid_lpoly_deg2_k16_tric_plus_resnet_32x32_h512 | 0.3300 | 2.9100 | -2.2600 | 98.2000 | 1.7600 | 15.2500 | -8.2300 | 77.1800 | 8.7430 |
| Hybrid_lpoly_deg2_k16_tric_plus_resnet_32x32_h256_ms_rff | 0.3100 | 2.9700 | -2.2000 | 98.3900 | 1.6600 | 15.3200 | -8.1500 | 79.3100 | 4.0820 |