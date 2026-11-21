| Model | Dataset | Direction | Count | Mean 2D (mm) | RMSE 2D (mm) | P95 (mm) | P99 (mm) | Max 2D (mm) | <1cm (%) | <5cm (%) | Time (s) | Size (MB) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| mlp_fourier_4x4 | ro_test_uniform_4x4 | grid_shift_2d | 36,752 | 10.3404 | 13.0488 | 23.1000 | 34.4889 | 374.6679 | 57.8989 | 99.7796 | 0.1293 | 72.0880 |
| mlp_fourier_4x4 | ro_test_adaptive_3mm_mid | grid_shift_2d | 176,350 | 18.4233 | 32.4599 | 54.2919 | 115.6803 | 514.6135 | 40.9033 | 94.2393 | 0.6166 | 72.0880 |
| mlp_fourier_center_dist_4x4 | ro_test_uniform_4x4 | grid_shift_2d | 36,752 | 10.3773 | 13.6861 | 22.0594 | 34.5079 | 320.7397 | 57.6948 | 99.5592 | 0.1640 | 73.1594 |
| mlp_fourier_center_dist_4x4 | ro_test_adaptive_3mm_mid | grid_shift_2d | 176,350 | 21.1426 | 41.9698 | 61.4334 | 204.0392 | 584.2942 | 38.9305 | 92.9668 | 0.8669 | 73.1594 |
| mlp_fourier_center_dist_slopes_4x4 | ro_test_uniform_4x4 | grid_shift_2d | 36,752 | 12.4376 | 15.5440 | 28.8966 | 44.8079 | 214.0009 | 47.6518 | 99.3715 | 0.2106 | 73.4065 |
| mlp_fourier_center_dist_slopes_4x4 | ro_test_adaptive_3mm_mid | grid_shift_2d | 176,350 | 25.7002 | 47.3126 | 67.1106 | 178.0042 | 755.2548 | 27.9966 | 90.1934 | 1.0238 | 73.4065 |
| mlp_fourier_center_dist_all_4x4 | ro_test_uniform_4x4 | grid_shift_2d | 36,752 | 10.9137 | 14.4823 | 26.3237 | 43.0369 | 259.5170 | 57.5370 | 99.3987 | 0.1951 | 73.4706 |
| mlp_fourier_center_dist_all_4x4 | ro_test_adaptive_3mm_mid | grid_shift_2d | 176,350 | 24.4183 | 47.0387 | 67.8348 | 197.6465 | 835.6768 | 33.6671 | 90.8812 | 0.9683 | 73.4706 |