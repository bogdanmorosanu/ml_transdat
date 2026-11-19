"""K-Nearest Neighbors (KNN) spatial regression for 2D geodetic distortion grid shifts."""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
from sklearn.neighbors import KNeighborsRegressor


class KNNSpatialRegressor:
    """K-Nearest Neighbors regressor for 2D planar coordinate distortion shifts.

    Maps planar coordinates :math:`(N, E) \\mapsto (\\delta N, \\delta E)`
    using Euclidean distance queries over known spatial anchor nodes.

    Supports both uniform neighborhood averaging and inverse distance weighting (IDW).
    """

    def __init__(
        self,
        n_neighbors: int = 4,
        weights: Literal["uniform", "distance"] = "distance",
        algorithm: str = "kd_tree",
        leaf_size: int = 30,
    ) -> None:
        """Initialize the spatial KNN regressor.

        :param n_neighbors: Number of nearest spatial neighbors ($K$).
        :param weights: Weighting scheme: ``'uniform'`` or ``'distance'`` (IDW).
        :param algorithm: Nearest-neighbor search algorithm (default ``'kd_tree'``).
        :param leaf_size: Leaf size passed to the KDTree.
        """
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
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        """Return True if the estimator has been fitted."""
        return self._is_fitted

    def fit(
        self,
        st70_n: Sequence[float] | np.ndarray,
        st70_e: Sequence[float] | np.ndarray,
        dn: Sequence[float] | np.ndarray,
        de: Sequence[float] | np.ndarray,
    ) -> KNNSpatialRegressor:
        """Fit the KNN regressor on known spatial anchor points.

        :param st70_n: Stereo 70 Northing coordinates (meters), shape ``(N,)``.
        :param st70_e: Stereo 70 Easting coordinates (meters), shape ``(N,)``.
        :param dn: Planimetric Northing shift :math:`\\delta N` (meters), shape ``(N,)``.
        :param de: Planimetric Easting shift :math:`\\delta E` (meters), shape ``(N,)``.
        :return: Fitted self instance.
        """
        coords = np.column_stack([
            np.asarray(st70_n, dtype=np.float64).ravel(),
            np.asarray(st70_e, dtype=np.float64).ravel(),
        ])
        targets = np.column_stack([
            np.asarray(dn, dtype=np.float64).ravel(),
            np.asarray(de, dtype=np.float64).ravel(),
        ])

        if coords.shape[0] != targets.shape[0]:
            raise ValueError(
                f"Mismatch in sample counts: coords={coords.shape[0]}, targets={targets.shape[0]}"
            )
        if coords.shape[0] < self.n_neighbors:
            raise ValueError(
                f"Sample count {coords.shape[0]} is less than n_neighbors {self.n_neighbors}"
            )

        self._model.fit(coords, targets)
        self._is_fitted = True
        return self

    def predict(
        self,
        st70_n: Sequence[float] | np.ndarray,
        st70_e: Sequence[float] | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict planimetric shifts :math:`(\\delta N, \\delta E)` at query coordinates.

        :param st70_n: Query Northing coordinates (meters), shape ``(M,)``.
        :param st70_e: Query Easting coordinates (meters), shape ``(M,)``.
        :return: Tuple of ``(pred_dn, pred_de)`` arrays of shape ``(M,)``.
        """
        if not self._is_fitted:
            raise RuntimeError("Estimator must be fitted before calling predict().")

        coords = np.column_stack([
            np.asarray(st70_n, dtype=np.float64).ravel(),
            np.asarray(st70_e, dtype=np.float64).ravel(),
        ])
        preds = self._model.predict(coords)
        pred_dn = preds[:, 0]
        pred_de = preds[:, 1]
        return pred_dn, pred_de

    def kneighbors(
        self,
        st70_n: Sequence[float] | np.ndarray,
        st70_e: Sequence[float] | np.ndarray,
        n_neighbors: int | None = None,
        return_distance: bool = True,
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        """Find the K-neighbors of a point.

        :param st70_n: Query Northing coordinates (meters).
        :param st70_e: Query Easting coordinates (meters).
        :param n_neighbors: Number of neighbors required (defaults to model's ``n_neighbors``).
        :param return_distance: Whether to return distances.
        :return: Distances and indices of the nearest neighbors.
        """
        if not self._is_fitted:
            raise RuntimeError("Estimator must be fitted before calling kneighbors().")

        coords = np.column_stack([
            np.asarray(st70_n, dtype=np.float64).ravel(),
            np.asarray(st70_e, dtype=np.float64).ravel(),
        ])
        return self._model.kneighbors(
            coords,
            n_neighbors=n_neighbors,
            return_distance=return_distance,
        )
