"""Geometric and Planar Spatial Analysis Tools for Coordinate Grids.

Provides modular, reusable classes and functions to fit, analyze, and evaluate
local 3D planes, inclinations, slopes, and bilinear surfaces on coordinate grid cells.
"""

from src.geometry.cell_uv import (
    compute_spg_cell_uv,
    encode_cyclic_cell_uv,
    get_spg_grid_bounds,
)
from src.geometry.plane import (
    CellPlane,
    CellPlane2D,
    extract_grid_cell_planes,
    fit_cell_plane,
)
from src.geometry.shape_classifier import (
    CellShapeFeatureExtractor,
    CellShapeMetrics,
    ShapeCategory,
    classify_cells_dataset,
    compute_shape_polynomial_basis,
    evaluate_cell_polynomial_errors,
)

__all__ = [
    "CellPlane",
    "CellPlane2D",
    "CellShapeFeatureExtractor",
    "CellShapeMetrics",
    "ShapeCategory",
    "classify_cells_dataset",
    "compute_shape_polynomial_basis",
    "compute_spg_cell_uv",
    "encode_cyclic_cell_uv",
    "evaluate_cell_polynomial_errors",
    "extract_grid_cell_planes",
    "fit_cell_plane",
    "get_spg_grid_bounds",
]
