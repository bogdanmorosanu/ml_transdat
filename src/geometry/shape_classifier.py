"""Cell Shape Classification and Polynomial Order Modeling.

Classifies grid cells into geometric curvature archetypes:
1. LINEAR (Degree 1 / Bilinear corner ramp)
2. DOME_BOWL (Degree 2 / Parabolic monotonic curvature)
3. S_SHAPE (Degree 3 / Cubic inflection wave and non-linear residual)

Evaluates maximum error against geodetic tolerances on interior test points
and provides high-speed O(1) spatial feature extraction for arbitrary (N, E) coordinates.
"""

import os
from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pytransdat.exceptions import PyTransDatError


class ShapeCategory(IntEnum):
    """Geometric shape category of a grid cell (3-class taxonomy)."""

    LINEAR = 1
    DOME_BOWL = 2
    S_SHAPE = 3

    @property
    def label(self) -> str:
        """Human-readable label for the shape category."""
        labels = {
            ShapeCategory.LINEAR: "Linear",
            ShapeCategory.DOME_BOWL: "Dome/Bowl",
            ShapeCategory.S_SHAPE: "S-Shape",
        }
        return labels[self]


@dataclass(frozen=True)
class CellShapeMetrics:
    """Stores polynomial order approximation errors for a single cell."""

    row: int
    col: int
    center_n: float
    center_e: float
    min_n: float
    max_n: float
    min_e: float
    max_e: float

    # Base corner offsets (meters)
    base_dn: float
    base_de: float

    # Northing errors (mm) across interior test points
    err_n_deg1: float
    err_n_deg2: float
    err_n_deg3: float

    # Easting errors (mm) across interior test points
    err_e_deg1: float
    err_e_deg2: float
    err_e_deg3: float

    # Combined 2D Euclidean distance errors (mm)
    err_2d_deg1: float
    err_2d_deg2: float
    err_2d_deg3: float

    def get_category_dn(self, tol_mm: float) -> ShapeCategory:
        """Determines Northing shift shape category at the given tolerance.

        Clamps any residual higher-order errors into S_SHAPE.
        """
        if self.err_n_deg1 <= tol_mm:
            return ShapeCategory.LINEAR
        if self.err_n_deg2 <= tol_mm:
            return ShapeCategory.DOME_BOWL
        return ShapeCategory.S_SHAPE

    def get_category_de(self, tol_mm: float) -> ShapeCategory:
        """Determines Easting shift shape category at the given tolerance.

        Clamps any residual higher-order errors into S_SHAPE.
        """
        if self.err_e_deg1 <= tol_mm:
            return ShapeCategory.LINEAR
        if self.err_e_deg2 <= tol_mm:
            return ShapeCategory.DOME_BOWL
        return ShapeCategory.S_SHAPE

    def get_category_2d(self, tol_mm: float) -> ShapeCategory:
        """Determines combined 2D shape category at the given tolerance.

        Clamps any residual higher-order errors into S_SHAPE.
        """
        if self.err_2d_deg1 <= tol_mm:
            return ShapeCategory.LINEAR
        if self.err_2d_deg2 <= tol_mm:
            return ShapeCategory.DOME_BOWL
        return ShapeCategory.S_SHAPE

    def to_dict(self, default_tol_mm: float = 5.0) -> dict[str, Any]:
        """Serializes metrics and categories at a default tolerance to a dictionary."""
        return {
            "row": self.row,
            "col": self.col,
            "center_n": self.center_n,
            "center_e": self.center_e,
            "min_n": self.min_n,
            "max_n": self.max_n,
            "min_e": self.min_e,
            "max_e": self.max_e,
            "base_dn": self.base_dn,
            "base_de": self.base_de,
            # Continuous error metrics in mm
            "err_n_deg1_mm": self.err_n_deg1,
            "err_n_deg2_mm": self.err_n_deg2,
            "err_n_deg3_mm": self.err_n_deg3,
            "err_e_deg1_mm": self.err_e_deg1,
            "err_e_deg2_mm": self.err_e_deg2,
            "err_e_deg3_mm": self.err_e_deg3,
            "err_2d_deg1_mm": self.err_2d_deg1,
            "err_2d_deg2_mm": self.err_2d_deg2,
            "err_2d_deg3_mm": self.err_2d_deg3,
            # Classified categories at default tolerance (3-class taxonomy)
            "cat_dn": int(self.get_category_dn(default_tol_mm)),
            "cat_de": int(self.get_category_de(default_tol_mm)),
            "cat_2d": int(self.get_category_2d(default_tol_mm)),
            "cat_dn_label": self.get_category_dn(default_tol_mm).label,
            "cat_de_label": self.get_category_de(default_tol_mm).label,
            "cat_2d_label": self.get_category_2d(default_tol_mm).label,
        }


def compute_shape_polynomial_basis(
    u: np.ndarray,
    v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Precomputes degree-2 and degree-3 polynomial basis matrices for coordinates (u, v)."""
    ones = np.ones_like(u)
    x_deg2 = np.column_stack([ones, u, v, u**2, v**2, u * v])
    x_deg3 = np.column_stack(
        [
            x_deg2,
            u**3,
            v**3,
            (u**2) * v,
            u * (v**2),
        ]
    )
    return x_deg2, x_deg3


def evaluate_cell_polynomial_errors(
    cell: Any,
    dn_true: np.ndarray,
    de_true: np.ndarray,
    u_pts: np.ndarray,
    v_pts: np.ndarray,
    x_deg2: np.ndarray,
    x_deg3: np.ndarray,
) -> CellShapeMetrics:
    """Computes max approximation errors for Degree 1, Degree 2, and Degree 3."""
    # 1. Degree 1: Bilinear corner model
    dn_deg1 = np.array(
        [cell.plane_dn.eval_bilinear(u, v) for u, v in zip(u_pts, v_pts, strict=True)]
    )
    de_deg1 = np.array(
        [cell.plane_de.eval_bilinear(u, v) for u, v in zip(u_pts, v_pts, strict=True)]
    )

    diff_n_deg1 = dn_true - dn_deg1
    diff_e_deg1 = de_true - de_deg1
    err_n_deg1 = float(np.max(np.abs(diff_n_deg1)) * 1000.0)
    err_e_deg1 = float(np.max(np.abs(diff_e_deg1)) * 1000.0)
    err_2d_deg1 = float(np.max(np.hypot(diff_n_deg1, diff_e_deg1)) * 1000.0)

    # 2. Degree 2: Parabolic surface fit
    coeffs_n2, _, _, _ = np.linalg.lstsq(x_deg2, dn_true, rcond=None)
    coeffs_e2, _, _, _ = np.linalg.lstsq(x_deg2, de_true, rcond=None)
    pred_n_deg2 = x_deg2 @ coeffs_n2
    pred_e_deg2 = x_deg2 @ coeffs_e2

    diff_n_deg2 = dn_true - pred_n_deg2
    diff_e_deg2 = de_true - pred_e_deg2
    err_n_deg2 = float(np.max(np.abs(diff_n_deg2)) * 1000.0)
    err_e_deg2 = float(np.max(np.abs(diff_e_deg2)) * 1000.0)
    err_2d_deg2 = float(np.max(np.hypot(diff_n_deg2, diff_e_deg2)) * 1000.0)

    # 3. Degree 3: Cubic inflection fit
    coeffs_n3, _, _, _ = np.linalg.lstsq(x_deg3, dn_true, rcond=None)
    coeffs_e3, _, _, _ = np.linalg.lstsq(x_deg3, de_true, rcond=None)
    pred_n_deg3 = x_deg3 @ coeffs_n3
    pred_e_deg3 = x_deg3 @ coeffs_e3

    diff_n_deg3 = dn_true - pred_n_deg3
    diff_e_deg3 = de_true - pred_e_deg3
    err_n_deg3 = float(np.max(np.abs(diff_n_deg3)) * 1000.0)
    err_e_deg3 = float(np.max(np.abs(diff_e_deg3)) * 1000.0)
    err_2d_deg3 = float(np.max(np.hypot(diff_n_deg3, diff_e_deg3)) * 1000.0)

    return CellShapeMetrics(
        row=cell.row,
        col=cell.col,
        center_n=cell.center_n,
        center_e=cell.center_e,
        min_n=cell.min_n,
        max_n=cell.max_n,
        min_e=cell.min_e,
        max_e=cell.max_e,
        base_dn=cell.plane_dn.z00,
        base_de=cell.plane_de.z00,
        err_n_deg1=err_n_deg1,
        err_n_deg2=err_n_deg2,
        err_n_deg3=err_n_deg3,
        err_e_deg1=err_e_deg1,
        err_e_deg2=err_e_deg2,
        err_e_deg3=err_e_deg3,
        err_2d_deg1=err_2d_deg1,
        err_2d_deg2=err_2d_deg2,
        err_2d_deg3=err_2d_deg3,
    )


def classify_cells_dataset(
    cells: Sequence[Any],
    grid_interp_fn: Any,
    grid_res: int = 5,
) -> list[CellShapeMetrics]:
    """Iterates through all cells and evaluates polynomial order metrics on interior points."""
    u_vals = np.linspace(0.1, 0.9, grid_res)
    v_vals = np.linspace(0.1, 0.9, grid_res)
    u_grid, v_grid = np.meshgrid(u_vals, v_vals)
    u_pts = u_grid.flatten()
    v_pts = v_grid.flatten()

    x_deg2, x_deg3 = compute_shape_polynomial_basis(u_pts, v_pts)
    results: list[CellShapeMetrics] = []

    for cell in cells:
        pts_n = list(cell.min_n + v_pts * cell.step_n)
        pts_e = list(cell.min_e + u_pts * cell.step_e)

        try:
            corrs = grid_interp_fn(pts_n, pts_e)
        except (PyTransDatError, ValueError, RuntimeError):
            continue

        dn_true = np.array(corrs[0], dtype=np.float64)
        de_true = np.array(corrs[1], dtype=np.float64)

        metrics = evaluate_cell_polynomial_errors(
            cell=cell,
            dn_true=dn_true,
            de_true=de_true,
            u_pts=u_pts,
            v_pts=v_pts,
            x_deg2=x_deg2,
            x_deg3=x_deg3,
        )
        results.append(metrics)

    return results


class CellShapeFeatureExtractor:
    """Production-ready O(1) spatial feature extractor for arbitrary (N, E) coordinates.

    Maps any query point (Northing, Easting) directly to its bounding cell and returns:
    1. Local fractional coordinates: u, v in [0, 1]
    2. Discrete shape category features: cat_dn, cat_de, cat_2d (1=Linear, 2=Dome, 3=S-Shape)
    3. Cell base shift offsets: base_dn, base_de
    4. Cell row, col indices

    Can be loaded directly from a saved Parquet table or initialized from metrics list.
    """

    def __init__(
        self,
        metrics_df_or_path: pd.DataFrame | str | Path | None = None,
        default_tol_mm: float = 5.0,
    ) -> None:
        """Initializes the feature extractor and builds 2D matrix lookups for O(1) retrieval."""
        self.default_tol_mm: float = default_tol_mm

        if metrics_df_or_path is None:
            # Auto-resolve canonical parquet path in experiments/
            base_dir = Path(__file__).resolve().parents[2]
            metrics_df_or_path = (
                base_dir
                / "experiments"
                / "cell_planar_features"
                / "results"
                / "cell_shape_metrics.parquet"
            )

        if isinstance(metrics_df_or_path, (str, Path)):
            if not os.path.exists(metrics_df_or_path):
                raise FileNotFoundError(
                    f"Shape metrics dataset not found at: {metrics_df_or_path}"
                )
            self.df: pd.DataFrame = pd.read_parquet(metrics_df_or_path)
        else:
            self.df = metrics_df_or_path.copy()

        # Grid matrix extents
        self.step_n: float = 11000.0
        self.step_e: float = 11000.0
        self.min_n: float = 213634.564
        self.min_e: float = 109783.040
        self.nrows: int = 53
        self.ncols: int = 72

        # Build dense lookup matrices for O(1) indexing: (nrows, ncols)
        # NaN sentinel for unmapped / offshore cells
        self._err_n1 = np.full((self.nrows, self.ncols), np.nan, dtype=np.float32)
        self._err_n2 = np.full((self.nrows, self.ncols), np.nan, dtype=np.float32)
        self._err_e1 = np.full((self.nrows, self.ncols), np.nan, dtype=np.float32)
        self._err_e2 = np.full((self.nrows, self.ncols), np.nan, dtype=np.float32)
        self._err_2d1 = np.full((self.nrows, self.ncols), np.nan, dtype=np.float32)
        self._err_2d2 = np.full((self.nrows, self.ncols), np.nan, dtype=np.float32)
        self._base_dn = np.full((self.nrows, self.ncols), np.nan, dtype=np.float32)
        self._base_de = np.full((self.nrows, self.ncols), np.nan, dtype=np.float32)
        self._valid_mask = np.zeros((self.nrows, self.ncols), dtype=bool)

        for _, row in self.df.iterrows():
            r = int(row["row"])
            c = int(row["col"])
            self._err_n1[r, c] = row["err_n_deg1_mm"]
            self._err_n2[r, c] = row["err_n_deg2_mm"]
            self._err_e1[r, c] = row["err_e_deg1_mm"]
            self._err_e2[r, c] = row["err_e_deg2_mm"]
            self._err_2d1[r, c] = row["err_2d_deg1_mm"]
            self._err_2d2[r, c] = row["err_2d_deg2_mm"]
            self._base_dn[r, c] = row.get("base_dn", 0.0)
            self._base_de[r, c] = row.get("base_de", 0.0)
            self._valid_mask[r, c] = True

    def _coords_to_grid_indices(
        self,
        n: np.ndarray,
        e: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Converts metric coordinates (N, E) to grid row, col and fractional (u, v)."""
        r_float = (n - self.min_n) / self.step_n
        c_float = (e - self.min_e) / self.step_e

        r_idx = np.floor(r_float).astype(np.int64)
        c_idx = np.floor(c_float).astype(np.int64)

        u = c_float - c_idx
        v = r_float - r_idx

        return r_idx, c_idx, u, v

    def extract_features(
        self,
        northing: Sequence[float] | np.ndarray,
        easting: Sequence[float] | np.ndarray,
        tol_mm: float | None = None,
    ) -> dict[str, np.ndarray]:
        """Extracts complete local planar and shape category features for arbitrary points.

        :param northing: Array of Northing coordinates in Stereo 70 meters
        :param easting: Array of Easting coordinates in Stereo 70 meters
        :param tol_mm: Geodetic tolerance for shape classification (default: 5.0 mm)
        :return: Dictionary of NumPy arrays:
            - 'u': Fractional Easting coordinate in [0, 1]
            - 'v': Fractional Northing coordinate in [0, 1]
            - 'cat_dn': Integer category for Northing (1=Linear, 2=Dome, 3=S-Shape)
            - 'cat_de': Integer category for Easting (1=Linear, 2=Dome, 3=S-Shape)
            - 'cat_2d': Combined 2D category
            - 'base_dn': Corner base shift for Northing (m)
            - 'base_de': Corner base shift for Easting (m)
            - 'is_valid': Boolean mask indicating if coordinate falls in a valid cell
        """
        tol = self.default_tol_mm if tol_mm is None else tol_mm

        n_arr = np.asarray(northing, dtype=np.float64)
        e_arr = np.asarray(easting, dtype=np.float64)

        r_idx, c_idx, u, v = self._coords_to_grid_indices(n_arr, e_arr)

        # Boundary bounds check
        in_bounds = (
            (r_idx >= 0)
            & (r_idx < self.nrows - 1)
            & (c_idx >= 0)
            & (c_idx < self.ncols - 1)
        )
        r_clamped = np.clip(r_idx, 0, self.nrows - 1)
        c_clamped = np.clip(c_idx, 0, self.ncols - 1)

        is_valid = in_bounds & self._valid_mask[r_clamped, c_clamped]

        # Retrieve polynomial error matrices
        n1 = self._err_n1[r_clamped, c_clamped]
        n2 = self._err_n2[r_clamped, c_clamped]
        e1 = self._err_e1[r_clamped, c_clamped]
        e2 = self._err_e2[r_clamped, c_clamped]
        t1 = self._err_2d1[r_clamped, c_clamped]
        t2 = self._err_2d2[r_clamped, c_clamped]

        base_dn = self._base_dn[r_clamped, c_clamped]
        base_de = self._base_de[r_clamped, c_clamped]

        # Classify categories: 1=Linear, 2=Dome, 3=S-Shape
        cat_dn = np.where(
            n1 <= tol,
            int(ShapeCategory.LINEAR),
            np.where(
                n2 <= tol, int(ShapeCategory.DOME_BOWL), int(ShapeCategory.S_SHAPE)
            ),
        )
        cat_de = np.where(
            e1 <= tol,
            int(ShapeCategory.LINEAR),
            np.where(
                e2 <= tol, int(ShapeCategory.DOME_BOWL), int(ShapeCategory.S_SHAPE)
            ),
        )
        cat_2d = np.where(
            t1 <= tol,
            int(ShapeCategory.LINEAR),
            np.where(
                t2 <= tol, int(ShapeCategory.DOME_BOWL), int(ShapeCategory.S_SHAPE)
            ),
        )

        # Mask invalid points with 0
        cat_dn = np.where(is_valid, cat_dn, 0)
        cat_de = np.where(is_valid, cat_de, 0)
        cat_2d = np.where(is_valid, cat_2d, 0)

        return {
            "u": u,
            "v": v,
            "cat_dn": cat_dn,
            "cat_de": cat_de,
            "cat_2d": cat_2d,
            "base_dn": base_dn,
            "base_de": base_de,
            "is_valid": is_valid,
            "row": r_idx,
            "col": c_idx,
        }

    def lookup_point(
        self,
        northing: float,
        easting: float,
        tol_mm: float | None = None,
    ) -> dict[str, Any]:
        """Convenience single-point lookup returning a typed dictionary."""
        feats = self.extract_features([northing], [easting], tol_mm=tol_mm)
        return {k: v[0] if isinstance(v, np.ndarray) else v for k, v in feats.items()}
