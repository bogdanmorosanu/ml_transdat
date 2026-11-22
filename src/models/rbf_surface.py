"""2D Radial Basis Function (RBF) & Thin Plate Spline (TPS) Regressor for spatial distortion shifts.

Fits continuous localized surface deformation fields (Stereo70 Northing, Easting -> shift_dn_m, shift_de_m)
using Thin Plate Splines, Multiquadrics, or Gaussian kernels with optional low-degree polynomial trends
and nugget smoothing.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree

from src.models.base import BaseGridShiftModel


class RBFSurfaceRegressor(BaseGridShiftModel):
    """Radial Basis Function and Thin Plate Spline regressor for 2D spatial coordinate shifts.

    Models coordinate shift residuals as an elastic continuous sheet with localized bending:
    .. math::
        s(\\mathbf{x}) = p(\\mathbf{x}) + \\sum_{j=1}^M w_j \\phi(\\|\\mathbf{x} - \\mathbf{c}_j\\|)

    where :math:`p(\\mathbf{x})` is a polynomial trend of degree :math:`D` (e.g., affine, quadratic, cubic),
    and :math:`\\phi(r)` is a radial basis kernel centered at anchor points :math:`\\mathbf{c}_j`.

    Supports exact interpolation (:math:`\\lambda = 0`) or regularized smoothing (:math:`\\lambda > 0`),
    and local neighborhood acceleration via k-d tree.
    """

    SUPPORTED_KERNELS = (
        "thin_plate_spline",
        "multiquadric",
        "inverse_multiquadric",
        "gaussian",
        "linear",
        "cubic",
        "quintic",
        "matern32",
        "matern52",
    )

    def __init__(
        self,
        kernel: str = "thin_plate_spline",
        degree: int = 1,
        smoothing: float = 0.0,
        epsilon: float | None = None,
        neighbors: int | None = None,
        normalize_coords: bool = True,
    ) -> None:
        """Initialize the 2D RBF Surface Regressor.

        :param kernel: Radial basis function kernel type. Options include:
            ``'thin_plate_spline'`` (scale-free bending energy minimizer),
            ``'multiquadric'``, ``'inverse_multiquadric'``, ``'gaussian'``, etc.
        :type kernel: str
        :param degree: Degree of added polynomial trend.
            ``-1``: No polynomial trend.
            ``0``: Constant offset.
            ``1``: Affine plane (default for TPS null space).
            ``2``: Quadratic bowl / saddle.
            ``3``: Cubic inflection surface.
        :type degree: int
        :param smoothing: Nugget smoothing parameter :math:`\\lambda \\ge 0.0`.
            ``0.0`` indicates exact interpolation at anchor nodes.
        :type smoothing: float
        :param epsilon: Shape parameter for kernels requiring scaling (multiquadric, gaussian).
            When coordinates are normalized to :math:`[-1, 1]`, epsilon scales normalized distance.
        :type epsilon: float | None
        :param neighbors: Optional number of nearest neighbors for localized k-d tree evaluation.
            If None (default), all anchor centers are used globally.
        :type neighbors: int | None
        :param normalize_coords: Whether to normalize input coordinates to centered :math:`[-1, 1]`.
        :type normalize_coords: bool
        """
        super().__init__()
        kernel_norm = kernel.lower()
        if kernel_norm not in self.SUPPORTED_KERNELS:
            raise ValueError(
                f"Unsupported kernel '{kernel}'. Supported: {self.SUPPORTED_KERNELS}"
            )
        if smoothing < 0.0:
            raise ValueError(f"smoothing must be non-negative, got {smoothing}")
        if degree < -1:
            raise ValueError(f"degree must be >= -1, got {degree}")
        if neighbors is not None and neighbors <= 0:
            raise ValueError(f"neighbors must be a positive integer, got {neighbors}")
        if epsilon is not None and epsilon <= 0.0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        self.kernel = kernel_norm
        self.degree = degree
        self.smoothing = smoothing
        self.epsilon = epsilon
        self.neighbors = neighbors
        self.normalize_coords = normalize_coords

        self._rbf: RBFInterpolator | None = None
        self._kdtree: cKDTree | None = None
        self._norm_coords: np.ndarray | None = None
        self._shifts: np.ndarray | None = None
        self._center: np.ndarray | None = None
        self._scale: float = 1.0
        self._n_anchors: int = 0

    @property
    def n_anchors(self) -> int:
        """Return number of anchor control centers used during fitting."""
        return self._n_anchors

    def _matern_kernel(self, r: np.ndarray) -> np.ndarray:
        """Evaluate Matérn 3/2 or 5/2 radial kernel at normalized distances ``r``."""
        eps = self.epsilon if self.epsilon is not None else 10.0
        if self.kernel == "matern32":
            s = np.sqrt(3.0) * eps * r
            return (1.0 + s) * np.exp(-s)
        s = np.sqrt(5.0) * eps * r
        return (1.0 + s + (s * s) / 3.0) * np.exp(-s)

    def _fit_impl(self, coords: np.ndarray, shifts: np.ndarray) -> None:
        """Fit radial basis function interpolation / smoothing system."""
        self._n_anchors = coords.shape[0]

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

        if self.kernel in ("matern32", "matern52"):
            self._kdtree = cKDTree(norm_coords)
            self._norm_coords = np.asarray(norm_coords, dtype=np.float64)
            self._shifts = np.asarray(shifts, dtype=np.float64)
            self._rbf = None
            return

        # Instantiate Scipy RBFInterpolator
        self._rbf = RBFInterpolator(
            y=norm_coords,
            d=shifts,
            neighbors=self.neighbors,
            smoothing=self.smoothing,
            kernel=self.kernel,
            epsilon=self.epsilon,
            degree=self.degree,
        )

    def _predict_impl(self, coords: np.ndarray) -> np.ndarray:
        """Evaluate continuous RBF surface at target coordinates."""
        if self.normalize_coords and self._center is not None:
            norm_coords = (coords - self._center) / self._scale
        else:
            norm_coords = coords

        if self.kernel in ("matern32", "matern52"):
            if self._kdtree is None or self._norm_coords is None or self._shifts is None:
                raise RuntimeError("Model is not fitted. Call fit() before predict().")
            k = min(self.neighbors or 60, self._n_anchors)
            n_q = len(norm_coords)
            preds = np.empty((n_q, 2), dtype=np.float64)
            nugget = max(self.smoothing, 1e-10)
            eye_k = np.eye(k, dtype=np.float64) * nugget
            chunk = 2048
            for start in range(0, n_q, chunk):
                end = min(start + chunk, n_q)
                q_batch = norm_coords[start:end]
                dists, idxs = self._kdtree.query(q_batch, k=k)
                if k == 1:
                    dists = dists[:, None]
                    idxs = idxs[:, None]
                pts_loc = self._norm_coords[idxs]  # (B, k, 2)
                y_loc = self._shifts[idxs]  # (B, k, 2)
                diff = pts_loc[:, :, None, :] - pts_loc[:, None, :, :]
                r_mat = np.sqrt(np.sum(diff * diff, axis=-1))  # (B, k, k)
                k_mat = self._matern_kernel(r_mat) + eye_k[None, :, :]
                k_vec = self._matern_kernel(dists)  # (B, k)
                weights = np.linalg.solve(k_mat, y_loc)  # (B, k, 2)
                preds[start:end] = np.einsum("bk,bkd->bd", k_vec, weights)
            return preds

        if self._rbf is None:
            raise RuntimeError("Model is not fitted. Call fit() before predict().")

        preds = self._rbf(norm_coords)
        return np.asarray(preds, dtype=np.float64)

    def _runtime_dependencies(self) -> tuple[str, ...]:
        """Return external runtime dependencies."""
        return ("scipy", "numpy")
