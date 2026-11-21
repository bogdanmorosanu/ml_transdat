"""2D Polynomial Surface Regressor for spatial coordinate distortion shifts.

Fits multi-degree bivariate polynomial surfaces (Stereo70 Northing, Easting -> shift_dn_m, shift_de_m)
using normalized coordinates and Ridge least squares.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures

from src.models.base import BaseGridShiftModel


class PolynomialSurfaceRegressor(BaseGridShiftModel):
    """Bivariate polynomial surface regressor for 2D spatial coordinate shifts.

    Fits smooth polynomial surfaces of degree :math:`D`:
    .. math::
        \\Delta N(u, v) = \\sum_{p+q \\le D} c^{(n)}_{pq} u^p v^q, \\quad
        \\Delta E(u, v) = \\sum_{p+q \\le D} c^{(e)}_{pq} u^p v^q

    where :math:`(u, v) \\in [-1, 1]` are aspect-ratio preserved centered coordinates.
    Supports L2 Ridge regularization and L1 Lasso sparse feature selection.
    """

    def __init__(
        self,
        degree: int = 3,
        alpha: float = 1e-8,
        regressor: str = "ridge",
        normalize_coords: bool = True,
        solver: str = "auto",
        max_iter: int = 10000,
    ) -> None:
        """Initialize the 2D Polynomial Surface Regressor.

        :param degree: Polynomial degree :math:`D \\ge 1`.
        :type degree: int
        :param alpha: Regularization strength (L2 penalty for Ridge, L1 for Lasso).
        :type alpha: float
        :param regressor: Linear model type, either ``'ridge'`` (default) or ``'lasso'``.
        :type regressor: str
        :param normalize_coords: Whether to normalize coordinates to centered :math:`[-1, 1]`.
        :type normalize_coords: bool
        :param solver: Solver for Ridge regression (e.g., 'auto', 'svd', 'cholesky').
        :type solver: str
        :param max_iter: Maximum iterations for iterative solvers (used by Lasso).
        :type max_iter: int
        """
        super().__init__()
        if degree < 1:
            raise ValueError(f"degree must be at least 1, got {degree}")
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")
        if regressor.lower() not in ("ridge", "lasso"):
            raise ValueError(f"regressor must be 'ridge' or 'lasso', got {regressor}")

        self.degree = degree
        self.alpha = alpha
        self.regressor = regressor.lower()
        self.normalize_coords = normalize_coords
        self.solver = solver
        self.max_iter = max_iter

        self._poly: PolynomialFeatures | None = None
        self._model: Ridge | Lasso | None = None
        self._center: np.ndarray | None = None
        self._scale: float = 1.0
        self._feature_names: list[str] = []

    def _fit_impl(self, coords: np.ndarray, shifts: np.ndarray) -> None:
        if self.normalize_coords:
            self._center = np.mean(coords, axis=0)
            ptp_n = float(np.ptp(coords[:, 0]))
            ptp_e = float(np.ptp(coords[:, 1]))
            self._scale = 0.5 * max(ptp_n, ptp_e)
            if self._scale <= 0:
                self._scale = 1.0
            norm_coords = (coords - self._center) / self._scale
        else:
            self._center = np.zeros(2, dtype=np.float64)
            self._scale = 1.0
            norm_coords = coords

        self._poly = PolynomialFeatures(degree=self.degree, include_bias=True)
        feat_matrix = self._poly.fit_transform(norm_coords)

        if self.regressor == "lasso":
            self._model = Lasso(
                alpha=self.alpha,
                fit_intercept=False,
                max_iter=self.max_iter,
            )
        else:
            self._model = Ridge(
                alpha=self.alpha,
                fit_intercept=False,
                solver=self.solver,
            )
        self._model.fit(feat_matrix, shifts)

        # Store readable feature names (u, v)
        self._feature_names = self._poly.get_feature_names_out(["u", "v"]).tolist()

    def _predict_impl(self, coords: np.ndarray) -> np.ndarray:
        if self._model is None or self._poly is None:
            raise RuntimeError("Model is not fitted. Call fit() before predict().")

        if self.normalize_coords and self._center is not None:
            norm_coords = (coords - self._center) / self._scale
        else:
            norm_coords = coords

        feat_matrix = self._poly.transform(norm_coords)
        preds = self._model.predict(feat_matrix)
        return np.asarray(preds, dtype=np.float64)

    def _runtime_dependencies(self) -> tuple[str, ...]:
        return ("scikit-learn", "numpy")

    @property
    def n_terms(self) -> int:
        """Return total number of polynomial monomial terms per coordinate axis."""
        return (self.degree + 1) * (self.degree + 2) // 2

    @property
    def n_nonzero_terms(self) -> tuple[int, int]:
        """Return count of non-zero terms (dN, dE) with absolute coefficient > 1e-6."""
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Model is not fitted.")
        nz_n = int(np.sum(np.abs(self._model.coef_[0]) > 1e-6))
        nz_e = int(np.sum(np.abs(self._model.coef_[1]) > 1e-6))
        return nz_n, nz_e

    @property
    def feature_names(self) -> list[str]:
        """Return list of monomial names :math:`[1, u, v, u^2, u v, v^2, \\dots]`."""
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted.")
        return list(self._feature_names)

    @property
    def coef_(self) -> np.ndarray:
        """Return fitted coefficient matrix of shape ``(2, n_terms)``."""
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Model is not fitted.")
        return np.asarray(self._model.coef_, dtype=np.float64)

    def get_formula(self, axis: str = "n", max_terms: int = 8, threshold: float = 1e-5) -> str:
        """Return formatted mathematical string of the fitted polynomial surface.

        :param axis: Coordinate axis, either 'n' (Northing shift) or 'e' (Easting shift).
        :type axis: str
        :param max_terms: Maximum number of dominant terms to format in the string.
        :type max_terms: int
        :param threshold: Minimum absolute coefficient magnitude to include a term.
        :type threshold: float
        :return: Formatted formula representation.
        :rtype: str
        """
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Model is not fitted.")
        idx = 0 if axis.lower().startswith("n") else 1
        axis_name = "\\Delta N" if idx == 0 else "\\Delta E"
        coefs = self._model.coef_[idx]

        # Filter and sort terms by absolute magnitude
        active_indices = [i for i in range(len(coefs)) if abs(coefs[i]) > threshold]
        ranked_indices = sorted(active_indices, key=lambda i: abs(coefs[i]), reverse=True)

        terms: list[str] = []
        for i in ranked_indices[:max_terms]:
            val = coefs[i]
            fname = self._feature_names[i]
            terms.append(f"{val:+.5f}*{fname}")

        formula = " ".join(terms) if terms else "0.0"
        if len(ranked_indices) > max_terms:
            formula += f" + ... ({len(ranked_indices) - max_terms} more terms)"
        return f"{axis_name}(u, v) = {formula}"

    def get_params(self) -> dict[str, Any]:
        """Return hyperparameter and diagnostic dictionary."""
        nz_n, nz_e = self.n_nonzero_terms if self._is_fitted else (0, 0)
        return {
            "degree": self.degree,
            "alpha": self.alpha,
            "regressor": self.regressor,
            "normalize_coords": self.normalize_coords,
            "solver": self.solver,
            "n_terms": self.n_terms,
            "n_nonzero_terms": (nz_n, nz_e),
            "is_fitted": self._is_fitted,
        }
