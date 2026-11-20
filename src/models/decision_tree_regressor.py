"""Decision Tree Spatial Regressor for 2D coordinate distortion shifts.

Wraps scikit-learn's DecisionTreeRegressor for multi-output regression
(Stereo70 Northing, Easting -> shift_dn_m, shift_de_m).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.tree import DecisionTreeRegressor

from src.models.base import BaseGridShiftModel


class DecisionTreeSpatialRegressor(BaseGridShiftModel):
    """Multi-output Decision Tree regressor for 2D spatial distortion shifts.

    Fits orthogonal axis-aligned partition trees on known coordinates ``(N, E)``
    to predict coordinate distortion shifts ``(dN, dE)``.
    """

    def __init__(
        self,
        criterion: str = "squared_error",
        max_depth: int | None = None,
        min_samples_split: float = 2,
        min_samples_leaf: float = 1,
        min_impurity_decrease: float = 0.0,
        random_state: int = 42,
    ) -> None:
        """Initialize the spatial Decision Tree regressor.

        :param criterion: Impurity criterion (e.g., ``'squared_error'``, ``'absolute_error'``).
        :type criterion: str
        :param max_depth: Maximum tree depth (``None`` expands until pure).
        :type max_depth: int | None
        :param min_samples_split: Minimum samples required to split an internal node.
        :type min_samples_split: int | float
        :param min_samples_leaf: Minimum samples required at a leaf node.
        :type min_samples_leaf: int | float
        :param min_impurity_decrease: Minimum impurity decrease required for a split.
        :type min_impurity_decrease: float
        :param random_state: Seed for reproducible split tie-breaking.
        :type random_state: int
        """
        super().__init__()
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state

        self.model: DecisionTreeRegressor | None = None

    def _fit_impl(self, coords: np.ndarray, shifts: np.ndarray) -> None:
        self.model = DecisionTreeRegressor(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
            random_state=self.random_state,
        )
        self.model.fit(coords, shifts)

    def _predict_impl(self, coords: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted. Call fit() before predict().")
        return np.asarray(self.model.predict(coords), dtype=np.float64)

    def _runtime_dependencies(self) -> tuple[str, ...]:
        return ("scikit-learn",)

    @property
    def n_leaves(self) -> int:
        """Return the number of leaf nodes in the tree."""
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Model is not fitted.")
        return int(self.model.get_n_leaves())

    @property
    def depth(self) -> int:
        """Return the maximum depth of the tree."""
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Model is not fitted.")
        return int(self.model.get_depth())

    def get_params(self) -> dict[str, Any]:
        """Return dictionary of hyperparameters and tree diagnostics."""
        params: dict[str, Any] = {
            "criterion": self.criterion,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "min_impurity_decrease": self.min_impurity_decrease,
            "random_state": self.random_state,
            "is_fitted": self._is_fitted,
        }
        if self._is_fitted and self.model is not None:
            params["actual_depth"] = self.depth
            params["actual_n_leaves"] = self.n_leaves
        return params
