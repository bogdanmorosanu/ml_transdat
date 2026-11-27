"""Unified model interfaces and spatial transformation implementations."""

from src.models.base import (
    BaseCoordinateTransformer,
    BaseGridShiftModel,
    BaseSpatialModel,
    MemoryBreakdown,
    ResidualCoordinateTransformer,
)
from src.models.bspline_surface import BSplineSurfaceRegressor
from src.models.composite_shift import (
    CellBilinearShiftModel,
    CompositeResidualShiftModel,
)
from src.models.decision_tree_regressor import DecisionTreeSpatialRegressor
from src.models.geostat_and_wavelet import (
    LocalKrigingShiftModel,
    PartitionOfUnityRBFShiftModel,
    WaveletGridShiftModel,
)
from src.models.grid_bicubic_variants import (
    BicubicScheme,
    ParameterizedBicubicGridShiftModel,
)
from src.models.helmert2d import (
    FittedHelmert2D,
    Helmert2DConformalShiftModel,
    Helmert2DParams,
    TwoStepHelmert2DTransformer,
    fit_helmert2d,
    load_embedded_spg_helmert_params,
)
from src.models.hist_gbr import HistGBRGridShiftModel
from src.models.knn_regressor import KNNSpatialRegressor
from src.models.local_poly_surface import LocalPolynomialSurfaceRegressor
from src.models.mlp import MLPGridShiftModel
from src.models.neural_feature_grid import NeuralFeatureGridShiftModel
from src.models.neural_inrs import (
    KANGridShiftModel,
    MFNGridShiftModel,
    SpatialMoEGridShiftModel,
    WaveletINRGridShiftModel,
)
from src.models.non_neural_ml import (
    LinearLeafTreeShiftModel,
    LSSVMGridShiftModel,
    RFFELMGridShiftModel,
    XGBoostGridShiftModel,
)
from src.models.polynomial_surface import PolynomialSurfaceRegressor
from src.models.pyproj_transformer import PyprojEpsg3844Transformer
from src.models.quadtree_regressor import AdaptiveQuadtreeRegressor
from src.models.rbf_surface import RBFSurfaceRegressor
from src.models.structured_neural_grids import (
    KPlanesGridShiftModel,
    NeuralBicubicHeadShiftModel,
)
from src.models.torch_mlp import TorchMLPGridShiftModel, TorchMLPNet

__all__ = [
    "AdaptiveQuadtreeRegressor",
    "BSplineSurfaceRegressor",
    "BaseCoordinateTransformer",
    "BaseGridShiftModel",
    "BaseSpatialModel",
    "BicubicScheme",
    "CellBilinearShiftModel",
    "CompositeResidualShiftModel",
    "DecisionTreeSpatialRegressor",
    "FittedHelmert2D",
    "Helmert2DConformalShiftModel",
    "Helmert2DParams",
    "HistGBRGridShiftModel",
    "KANGridShiftModel",
    "KNNSpatialRegressor",
    "KPlanesGridShiftModel",
    "LSSVMGridShiftModel",
    "LinearLeafTreeShiftModel",
    "LocalKrigingShiftModel",
    "LocalPolynomialSurfaceRegressor",
    "MFNGridShiftModel",
    "MLPGridShiftModel",
    "MemoryBreakdown",
    "NeuralBicubicHeadShiftModel",
    "NeuralFeatureGridShiftModel",
    "ParameterizedBicubicGridShiftModel",
    "PartitionOfUnityRBFShiftModel",
    "PolynomialSurfaceRegressor",
    "PyprojEpsg3844Transformer",
    "RBFSurfaceRegressor",
    "RFFELMGridShiftModel",
    "ResidualCoordinateTransformer",
    "SpatialMoEGridShiftModel",
    "TorchMLPGridShiftModel",
    "TorchMLPNet",
    "TwoStepHelmert2DTransformer",
    "WaveletGridShiftModel",
    "WaveletINRGridShiftModel",
    "XGBoostGridShiftModel",
    "fit_helmert2d",
    "load_embedded_spg_helmert_params",
]
