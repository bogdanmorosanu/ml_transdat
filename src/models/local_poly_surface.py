"""2D Weighted Local Polynomial (LOESS / Moving Least Squares) Surface Regressor."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from src.models.base import BaseGridShiftModel


class LocalPolynomialSurfaceRegressor(BaseGridShiftModel):
    """2D weighted local polynomial (LOESS / moving least squares) surface regressor.

    At each query point, fits a distance-weighted bivariate polynomial of degree
    ``1``, ``2``, or ``3`` over the ``neighbors`` nearest training anchors and
    evaluates the local surface at the query origin.

    :param degree: Local polynomial degree (``1``, ``2``, or ``3``).
    :type degree: int
    :param neighbors: Number of nearest 2D neighbors per local fit.
    :type neighbors: int
    :param weight: Distance weighting kernel (``'tricube'`` or ``'gaussian'``).
    :type weight: str
    :param ridge: Small diagonal Tikhonov stabilizer for numerical robustness.
    :type ridge: float
    """

    SUPPORTED_WEIGHTS = ("tricube", "gaussian")

    def __init__(
        self,
        degree: int = 2,
        neighbors: int = 25,
        weight: str = "tricube",
        ridge: float = 1e-8,
    ) -> None:
        super().__init__()
        if degree not in (1, 2, 3):
            raise ValueError(f"degree must be 1, 2, or 3, got {degree}")
        min_k = {1: 3, 2: 6, 3: 10}[degree]
        if neighbors < min_k:
            raise ValueError(
                f"neighbors ({neighbors}) must be >= {min_k} for degree={degree}"
            )
        w_norm = weight.lower()
        if w_norm not in self.SUPPORTED_WEIGHTS:
            raise ValueError(
                f"Unsupported weight '{weight}'. Supported: {self.SUPPORTED_WEIGHTS}"
            )
        if ridge < 0.0:
            raise ValueError(f"ridge must be >= 0.0, got {ridge}")

        self.degree = degree
        self.neighbors = neighbors
        self.weight = w_norm
        self.ridge = ridge

        self._kdtree: cKDTree | None = None
        self._coords: np.ndarray | None = None
        self._shifts: np.ndarray | None = None

    def _fit_impl(self, coords: np.ndarray, shifts: np.ndarray) -> None:
        self._coords = np.asarray(coords, dtype=np.float64)
        self._shifts = np.asarray(shifts, dtype=np.float64)
        self._kdtree = cKDTree(self._coords)

    def _design_matrix(self, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
        """Build local 2D polynomial Vandermonde tensor of shape ``(B, k, P)``."""
        ones = np.ones_like(dx)
        cols = [ones, dx, dy]
        if self.degree >= 2:
            cols.extend([dx * dx, dx * dy, dy * dy])
        if self.degree >= 3:
            cols.extend([dx * dx * dx, dx * dx * dy, dx * dy * dy, dy * dy * dy])
        return np.stack(cols, axis=-1)

    def _weights(self, dists: np.ndarray) -> np.ndarray:
        """Compute local distance weights of shape ``(B, k)``."""
        h = np.max(dists, axis=1, keepdims=True) + 1e-12
        u = np.clip(dists / h, 0.0, 1.0)
        if self.weight == "tricube":
            w = (1.0 - u**3) ** 3
            return np.maximum(w, 1e-6)
        # Gaussian kernel with bandwidth = 0.5 * neighborhood radius
        return np.exp(-0.5 * (u / 0.5) ** 2)

    def _predict_impl(self, coords: np.ndarray) -> np.ndarray:
        if self._kdtree is None or self._coords is None or self._shifts is None:
            raise RuntimeError("Model is not fitted. Call fit() before predict().")

        q_all = np.asarray(coords, dtype=np.float64)
        n_q = len(q_all)
        k = min(self.neighbors, len(self._coords))
        n_terms = {1: 3, 2: 6, 3: 10}[self.degree]
        eye_p = np.eye(n_terms, dtype=np.float64) * max(self.ridge, 1e-12)

        preds = np.empty((n_q, 2), dtype=np.float64)
        chunk = 4096
        for start in range(0, n_q, chunk):
            end = min(start + chunk, n_q)
            q = q_all[start:end]  # (B, 2)
            dists, idxs = self._kdtree.query(q, k=k)
            pts_loc = self._coords[idxs]  # (B, k, 2)
            y_loc = self._shifts[idxs]  # (B, k, 2)

            h = np.max(dists, axis=1, keepdims=True) + 1e-12  # (B, 1)
            dx = (pts_loc[:, :, 0] - q[:, 0:1]) / h
            dy = (pts_loc[:, :, 1] - q[:, 1:2]) / h

            a_mat = self._design_matrix(dx, dy)  # (B, k, P)
            w = self._weights(dists)  # (B, k)
            aw = a_mat * w[:, :, None]  # (B, k, P)

            ata = np.einsum("bkp,bkq->bpq", aw, a_mat) + eye_p[None, :, :]
            aty = np.einsum("bkp,bkd->bpd", aw, y_loc)
            coeffs = np.linalg.solve(ata, aty)  # (B, P, 2)
            # Local polynomial evaluated at (dx=0, dy=0) is simply the intercept term
            preds[start:end] = coeffs[:, 0, :]

        return preds

    def _runtime_dependencies(self) -> tuple[str, ...]:
        return ("scipy", "numpy")
