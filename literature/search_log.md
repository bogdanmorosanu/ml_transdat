# Literature Search Log & Knowledge Gaps

This log records all web search queries executed during the initial literature survey, the target domains, and explicit **negative results** (topics searched where no relevant published ML coordinate transformation work was found).

---

## 1. Executed Search Queries

| Axis | Query | Domain Filter | Outcome |
| :--- | :--- | :--- | :--- |
| **A. Direct Target** | `"Stereo 70" OR "Stereo70" OR "TransDatRO" "machine learning" OR "neural network"` | All | **Not Found** (No ML/NN papers for Stereo 70 or TransDatRO; only adjacent GIS/orthophoto CNNs) |
| **B. Stereographic + ML** | `"stereographic projection" "neural network" OR "machine learning" coordinate transformation geodesy` | All | **Partial** (General conformal/stereographic surrogate & spatial encoding concepts; no dedicated national stereographic ML benchmark) |
| **C. Shift Grids + ML** | `"NTv2" OR "NADCON" OR "distortion grid" "neural network" OR "machine learning" datum transformation` | All | **Found (Conceptual/Academic)** (Neural surrogates for continuous distortion grids & crustal motion; no production NTv2/NADCON replacement) |
| **C. Shift Grids + ML** | `"grid shift" OR "shift grid" "machine learning" OR "neural network" geodetic coordinate transformation` | All | **Found** (BPANN, RBFNN, ANFIS for learning regional distortion grids & local non-linearities) |
| **D. Datum + ML** | `"coordinate transformation" "neural network" geodesy datum` | `mdpi.com` | **Found** (BPANN, RBFNN, GRNN, K-fold CV) |
| **D. Datum + ML** | `"coordinate transformation" "neural network" datum geodesy` | `researchgate.net` | **Found** (Regional case studies: Ghana, Egypt, Turkey, Iraq; MLP/RBFNN) |
| **D. Datum + ML** | `"coordinate transformation" "neural network" geodesy datum` | `sciencedirect.com` | **Found** (RBNC hybrid residual modeling in *Geodesy and Geodynamics*) |
| **D. Datum + ML** | `"coordinate transformation" "neural network" OR "machine learning" geodesy datum` | `springer.com` | **Found** (MLP, BPANN, RBFNN, ELM, TLS-RBFNN) |
| **D. Datum + ML** | `"coordinate transformation" "neural network" OR "machine learning" geodetic OR geodesy` | `arxiv.org` | **Found** (RBFNN, *GeodesyNets*, *GTS Forecaster*) |
| **E. Specialized ML** | `"Residual-based Neural Network Correction" geodesy OR datum` | All | **Found** (RBNC hybrid architecture: geometric baseline + NN residual) |
| **E. Specialized ML** | `"Extreme Learning Machine" "coordinate transformation" datum geodesy` | All | **Found** (ELM vs BPNN/RBFNN for 2D/3D datum conversions) |
| **E. Specialized ML** | `"Support Vector" OR "Gaussian Process" "coordinate transformation" datum geodesy` | All | **Found** (LS-SVM & GP/Kriging for residual prediction & uncertainty quantification) |
| **E. Specialized ML** | `"physics-informed neural network" "conformal mapping" OR "map projection" OR "coordinate transformation"` | All | **Found (Adjacent)** (CoCo-PINNs for PDEs/conformal domains; not applied to geodetic datum shifts) |
| **E. Specialized ML** | `"Radial Basis Function" "coordinate transformation" datum WGS84 OR ETRS89` | All | **Found** (RBFNN & Thin Plate Splines for WGS84/ETRS89 to local datums) |
| **F. Tree-Based ML** | `"XGBoost" OR "Random Forest" OR "Gradient Boosting" "coordinate transformation" geodesy datum` | All | **Found** (Per-axis XGBoost/RF for 3D datum conversions; high $R^2$ but poor extrapolation & bulky models) |
| **F. Tree-Based ML** | `"decision tree" OR "random forest" OR "XGBoost" "datum transformation" OR "distortion grid" geodesy` | All | **Found** (XGBoost/RF as residual correctors; requires spatial block CV) |
| **F. Tree-Based ML** | `"XGBoost" OR "Random Forest" "geoid" OR "grid shift" interpolation geodesy` | All | **Found** (XGBoost & Random Forest Spatial Interpolation - RFSI for geoid & grid-shift surfaces) |
| **G. Grid Interpolation Standards** | `"NTv2" interpolation "bilinear" "bicubic" Geodetic Survey of Canada` | All | **Found** (NTv2 mandates bilinear C0; bicubic rejected to prevent oscillation/overshoot) |
| **G. Grid Interpolation Standards** | `"NADCON" OR "NADCON5" "interpolation" "biquadratic" "bicubic" "bilinear" NGS` | All | **Found** (NADCON 1-4 bilinear; NADCON5 biquadratic 9-node; bicubic rejected for RAM/stability) |
| **G. Grid Interpolation Standards** | `"OSTN15" OR "OSTN02" "interpolation" "bilinear" "bicubic" "tps" "grid" Ordnance Survey` | All | **Found** (Ordnance Survey uses dense 1 km grid with bilinear interpolation) |
| **G. Grid Interpolation Standards** | `"PROJ" "gridshift" "bilinear" "bicubic" "interpolation" PROJ.4` | All | **Found** (PROJ defaults to bilinear/biquadratic; restricts unconstrained bicubic) |
| **G. Grid Interpolation Standards** | `"bcucof" "Numerical Recipes" bicubic interpolation derivatives` | All | **Found** (Classic 16-coeff patch formula; assembly from local finite differences) |

---

## 2. Searched & Not Found (Knowledge Gaps)

1. **ETRS89 ↔ Stereo 70 (Romania) via Machine Learning**:
   - No published papers, preprints, or technical reports were found applying ML/ANNs to `Stereo 70`, `TransDatRO`, or Romania's `.spg` distortion grid.
2. **Production Replacement of NTv2 / NADCON Binary Grids**:
   - While academic papers propose ML surrogates for regional distortion fields, no published national geodetic agency standard or production benchmark replacing `.gsb` (NTv2) or NADCON grids with neural networks was found.
3. **Physics-Informed Neural Networks (PINNs) for Geodetic Datum Transformations**:
   - PINNs with conformal mapping (`CoCo-PINNs`) exist for solving PDEs on complex 2D geometries, but no published work was found applying PINNs to geodetic map projections or conformal datum transformations.
