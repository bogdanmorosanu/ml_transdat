"""2D Tensor-Product Smooth B-Spline Surface Regressor for spatial distortion shifts."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.interpolate import SmoothBivariateSpline

from src.models.base import BaseGridShiftModel


class BSplineSurfaceRegressor(BaseGridShiftModel):
    """Smooth tensor-product B-spline surface regressor for 2D coordinate shifts.

    Fits independent bivariate smoothing splines for Northing and Easting shifts
    using :class:`scipy.interpolate.SmoothBivariateSpline` on normalized ``[-1, 1]``
    coordinates.

    :param degree: Spline polynomial degree along both axes (``1``..``5``, typically ``3`` or ``5``).
    :type degree: int
    :param smoothing: Positive FITPACK smoothing factor ``s`` (smaller = tighter fit).
    :type smoothing: float
    :param normalize_coords: Whether to normalize coordinates to centered ``[-1, 1]``.
    :type normalize_coords: bool
    """

    def __init__(
        self,
        degree: int = 3,
        smoothing: float = 1e-2,
        normalize_coords: bool = True,
    ) -> None:
        super().__init__()
        if not 1 <= degree <= 5:
            raise ValueError(f"degree must be in [1, 5], got {degree}")
        if smoothing < 0.0:
            raise ValueError(f"smoothing must be >= 0.0, got {smoothing}")

        self.degree = degree
        self.smoothing = smoothing
        self.normalize_coords = normalize_coords

        self._spline_n: SmoothBivariateSpline | None = None
        self._spline_e: SmoothBivariateSpline | None = None
        self._center: np.ndarray | None = None
        self._scale: float = 1.0

    def _fit_impl(self, coords: np.ndarray, shifts: np.ndarray) -> None:
        if self.normalize_coords:
            self._center = np.mean(coords, axis=0)
            ptp_n = float(np.ptp(coords[:, 0]))
            ptp_e = float(np.ptp(coords[:, 1]))
            self._scale = 0.5 * max(ptp_n, ptp_e)
            if self._scale <= 0.0:
                self._scale = 1.0
            norm_coords = (coords - self._center) / self._scale
        else:
            self._center = np.zeros(2, dtype=np.float64)
            self._scale = 1.0
            norm_coords = coords

        x = np.asarray(norm_coords[:, 0], dtype=np.float64)
        y = np.asarray(norm_coords[:, 1], dtype=np.float64)
        zn = np.asarray(shifts[:, 0], dtype=np.float64)
        ze = np.asarray(shifts[:, 1], dtype=np.float64)

        # FITPACK requires s > 0 for large scattered point clouds to avoid knot exhaustion
        eff_s = max(float(self.smoothing), 1e-5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._spline_n = SmoothBivariateSpline(
                x, y, zn, kx=self.degree, ky=self.degree, s=eff_s
            )
            self._spline_e = SmoothBivariateSpline(
                x, y, ze, kx=self.degree, ky=self.degree, s=eff_s
            )

    def _predict_impl(self, coords: np.ndarray) -> np.ndarray:
        if self._spline_n is None or self._spline_e is None:
            raise RuntimeError("Model is not fitted. Call fit() before predict().")

        if self.normalize_coords and self._center is not None:
            norm_coords = (coords - self._center) / self._scale
        else:
            norm_coords = coords

        x = np.asarray(norm_coords[:, 0], dtype=np.float64)
        y = np.asarray(norm_coords[:, 1], dtype=np.float64)
        pred_n = self._spline_n.ev(x, y)
        pred_e = self._spline_e.ev(x, y)
        return np.column_stack([pred_n, pred_e]).astype(np.float64)

    def _runtime_dependencies(self) -> tuple[str, ...]:
        return ("scipy", "numpy")
