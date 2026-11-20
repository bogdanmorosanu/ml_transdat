"""K-Nearest Neighbors (KNN) spatial regression for 2D geodetic distortion grid shifts."""

from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.neighbors import KNeighborsRegressor

from src.models.base import BaseGridShiftModel


class KNNSpatialRegressor(BaseGridShiftModel):
    """K-Nearest Neighbors regressor for 2D planar coordinate distortion shifts."""

    def __init__(
        self,
        n_neighbors: int = 4,
        weights: Literal["uniform", "distance"] = "distance",
        algorithm: str = "kd_tree",
        leaf_size: int = 30,
    ) -> None:
        super().__init__()
        self.n_neighbors = int(n_neighbors)
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size

        self._model = KNeighborsRegressor(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            metric="euclidean",
        )

    def _fit_impl(self, coords: np.ndarray, shifts: np.ndarray) -> None:
        if coords.shape[0] < self.n_neighbors:
            raise ValueError(
                f"Sample count {coords.shape[0]} is less than n_neighbors {self.n_neighbors}"
            )
        self._model.fit(coords, shifts)

    def _predict_impl(self, coords: np.ndarray) -> np.ndarray:
        return np.asarray(self._model.predict(coords), dtype=np.float64)

    def kneighbors(
        self,
        coords: np.ndarray,
        n_neighbors: int | None = None,
        return_distance: bool = True,
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        """Find the K-neighbors of query coordinates.

        :param coords: Query coordinates ``[st70_n, st70_e]`` in meters, shape ``(M, 2)``.
        :type coords: np.ndarray
        :param n_neighbors: Optional number of neighbors (defaults to ``self.n_neighbors``).
        :type n_neighbors: int | None
        :param return_distance: Whether to return distances alongside neighbor indices.
        :type return_distance: bool
        :return: Distances and indices (or indices only if ``return_distance=False``).
        :rtype: tuple[np.ndarray, np.ndarray] | np.ndarray
        :raises RuntimeError: If called before fitting.
        """
        if not self._is_fitted:
            raise RuntimeError("Estimator must be fitted before calling kneighbors().")
        coords_arr = self._validate_coords(coords, "coords")
        return self._model.kneighbors(
            coords_arr,
            n_neighbors=n_neighbors,
            return_distance=return_distance,
        )

    def _runtime_dependencies(self) -> tuple[str, ...]:
        return ("scikit-learn",)

    def _grid_lookup_bytes(self) -> int:
        if self._is_fitted and hasattr(self._model, "_fit_X"):
            return int(
                self._model._fit_X.nbytes
                + getattr(self._model, "_y", np.empty(0)).nbytes
            )
        return 0
