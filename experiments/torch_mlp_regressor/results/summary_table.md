| Model | Dataset | Direction | Count | Mean 2D (mm) | RMSE 2D (mm) | P95 (mm) | P99 (mm) | Max 2D (mm) | <1cm (%) | <5cm (%) | Time (s) | Size (MB) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| torch_res_relu_4x4 | ro_test_uniform_4x4 | grid_shift_2d | 36,752 | 4.8894 | 7.1864 | 11.1010 | 17.8943 | 310.7251 | 92.9555 | 99.8884 | 0.2303 | 4083.0084 |
| torch_res_relu_4x4 | ro_test_adaptive_3mm_mid | grid_shift_2d | 176,350 | 15.7836 | 42.8256 | 54.9210 | 166.3974 | 804.9297 | 67.9200 | 94.2659 | 0.4513 | 4083.0084 |
| torch_res_relu_cellpos_4x4 | ro_test_uniform_4x4 | grid_shift_2d | 36,752 | 4.2309 | 6.7993 | 9.6557 | 16.4044 | 294.5994 | 95.5540 | 99.8640 | 0.1245 | 4083.0123 |
| torch_res_relu_cellpos_4x4 | ro_test_adaptive_3mm_mid | grid_shift_2d | 176,350 | 15.5528 | 44.1843 | 59.3884 | 160.6359 | 838.0128 | 71.3303 | 93.5775 | 0.5538 | 4083.0123 |
| torch_res_relu_cellpos_hermite_slopes_4x4 | ro_test_uniform_4x4 | grid_shift_2d | 36,752 | 2.9592 | 6.5271 | 6.3365 | 11.5585 | 411.7822 | 98.6341 | 99.8721 | 0.1897 | 4083.0445 |
| torch_res_relu_cellpos_hermite_slopes_4x4 | ro_test_adaptive_3mm_mid | grid_shift_2d | 176,350 | 14.5196 | 49.4778 | 57.5121 | 196.3364 | 939.4180 | 76.8188 | 94.1537 | 1.0502 | 4083.0445 |