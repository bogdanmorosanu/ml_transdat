"""Decision Tree Spatial Regressor for 2D coordinate distortion shifts.

Wraps scikit-learn's DecisionTreeRegressor for multi-output regression
(Stereo70 Northing, Easting -> shift_dn_m, shift_de_m).
"""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from sklearn.tree import DecisionTreeRegressor


class DecisionTreeSpatialRegressor:
    """Multi-output Decision Tree regressor for 2D spatial distortion shifts.

    Fits orthogonal axis-aligned partition trees on known coordinates (N, E)
    to predict coordinate distortion shifts (dN, dE).

    Attributes:
        criterion: Impurity criterion ('squared_error', 'absolute_error', etc.).
        max_depth: Maximum tree depth. None allows leaves to expand until pure.
        min_samples_split: Minimum samples required to split an internal node.
        min_samples_leaf: Minimum samples required at a leaf node.
        min_impurity_decrease: Minimum impurity decrease required for a split.
        random_state: Seed for reproducible tie-breaking.
        model: Underlying fitted sklearn DecisionTreeRegressor instance.
    """

    def __init__(
        self,
        criterion: str = "squared_error",
        max_depth: Optional[int] = None,
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: Union[int, float] = 1,
        min_impurity_decrease: float = 0.0,
        random_state: int = 42,
    ) -> None:
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state

        self.model: Optional[DecisionTreeRegressor] = None
        self._is_fitted: bool = False

    def fit(self, coords: np.ndarray, shifts: np.ndarray) -> "DecisionTreeSpatialRegressor":
        """Fit decision tree on reference coordinate anchors.

        Args:
            coords: Array of shape (N, 2) containing [northing, easting].
            shifts: Array of shape (N, 2) containing [shift_dn, shift_de].

        Returns:
            Self (fitted regressor).

        Raises:
            ValueError: If coordinate or shift array dimensions are invalid.
        """
        coords_arr = np.asarray(coords, dtype=np.float64)
        shifts_arr = np.asarray(shifts, dtype=np.float64)

        if coords_arr.ndim != 2 or coords_arr.shape[1] != 2:
            raise ValueError(f"coords must have shape (N, 2), got {coords_arr.shape}")
        if shifts_arr.ndim != 2 or shifts_arr.shape[1] != 2:
            raise ValueError(f"shifts must have shape (N, 2), got {shifts_arr.shape}")
        if coords_arr.shape[0] != shifts_arr.shape[0]:
            raise ValueError(
                f"Sample count mismatch: coords has {coords_arr.shape[0]}, "
                f"shifts has {shifts_arr.shape[0]}"
            )

        self.model = DecisionTreeRegressor(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
            random_state=self.random_state,
        )
        self.model.fit(coords_arr, shifts_arr)
        self._is_fitted = True
        return self

    def predict(self, coords: np.ndarray) -> np.ndarray:
        """Predict 2D shifts for query coordinates.

        Args:
            coords: Array of shape (M, 2) containing [northing, easting].

        Returns:
            Array of shape (M, 2) containing predicted [shift_dn, shift_de].

        Raises:
            RuntimeError: If model is called before fitting.
            ValueError: If query coords array dimensions are invalid.
        """
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Model is not fitted. Call fit() before predict().")

        coords_arr = np.asarray(coords, dtype=np.float64)
        if coords_arr.ndim != 2 or coords_arr.shape[1] != 2:
            raise ValueError(f"Query coords must have shape (M, 2), got {coords_arr.shape}")

        return self.model.predict(coords_arr)

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

    def get_params(self) -> Dict[str, Any]:
        """Return dictionary of hyperparameters and tree diagnostics."""
        params: Dict[str, Any] = {
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
