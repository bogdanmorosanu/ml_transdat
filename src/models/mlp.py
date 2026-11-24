"""Multilayer Perceptron (MLP) neural network regressor for 2D geodetic distortion grid shifts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPRegressor

from src.models._features import SpatialFeatureEncoder
from src.models.base import BaseGridShiftModel


class MLPGridShiftModel(BaseGridShiftModel):
    """Multilayer Perceptron neural network for 2D planar coordinate distortion shifts.

    Approximates the continuous mapping:
        (st70_n, st70_e) -> (shift_n, shift_e)

    Supports both raw coordinate inputs (normalized to [-1, 1]) and Random Fourier
    Feature (RFF) positional encodings to overcome spectral bias in spatial coordinate regression.
    """

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (128, 128),
        activation: str = "relu",
        solver: str = "lbfgs",
        alpha: float = 1e-4,
        max_iter: int = 500,
        random_state: int = 42,
        normalize_inputs: bool = True,
        use_fourier_features: bool = False,
        fourier_scale: float = 3.0,
        num_fourier_frequencies: int = 32,
        batch_size: int | str = "auto",
        learning_rate_init: float = 0.001,
        tol: float = 1e-4,
        n_iter_no_change: int = 10,
        use_shape_features: bool = False,
        shape_tol_mm: float = 5.0,
        s_shape_weight: int = 1,
        use_neighbor_features: bool = False,
        neighbor_feature_mode: str = "all",
        neighbor_features_path: str | Path = "data/features/cell_neighbor_features.parquet",
        scale_neighbor_features: bool = True,
        use_cell_center_distance: bool = False,
    ) -> None:
        """Initialize the MLP neural network spatial regressor."""
        super().__init__()
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.activation = activation
        self.solver = solver
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.random_state = int(random_state)
        self.normalize_inputs = bool(normalize_inputs)
        self.use_fourier_features = bool(use_fourier_features)
        self.fourier_scale = float(fourier_scale)
        self.num_fourier_frequencies = int(num_fourier_frequencies)
        self.batch_size = batch_size
        self.learning_rate_init = float(learning_rate_init)
        self.tol = float(tol)
        self.n_iter_no_change = int(n_iter_no_change)
        self.use_shape_features = bool(use_shape_features)
        self.shape_tol_mm = float(shape_tol_mm)
        self.s_shape_weight = int(s_shape_weight)

        if neighbor_feature_mode not in ("all", "slopes"):
            raise ValueError(
                f"Unsupported neighbor_feature_mode={neighbor_feature_mode!r}; "
                "expected 'all' or 'slopes'."
            )
        self.use_neighbor_features = bool(use_neighbor_features)
        self.neighbor_feature_mode = str(neighbor_feature_mode)
        self.neighbor_features_path = str(neighbor_features_path)
        self.scale_neighbor_features = bool(scale_neighbor_features)
        self.use_cell_center_distance = bool(use_cell_center_distance)

        self._feature_encoder = SpatialFeatureEncoder(
            normalize_inputs=self.normalize_inputs,
            use_fourier_features=self.use_fourier_features,
            fourier_scale=self.fourier_scale,
            num_fourier_frequencies=self.num_fourier_frequencies,
            random_state=self.random_state,
            use_shape_features=self.use_shape_features,
            shape_tol_mm=self.shape_tol_mm,
            use_neighbor_features=self.use_neighbor_features,
            neighbor_feature_mode=self.neighbor_feature_mode,
            neighbor_features_path=self.neighbor_features_path,
            scale_neighbor_features=self.scale_neighbor_features,
            use_cell_center_distance=self.use_cell_center_distance,
        )

        self._model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver=self.solver,
            alpha=self.alpha,
            max_iter=self.max_iter,
            random_state=self.random_state,
            batch_size=self.batch_size,
            learning_rate_init=self.learning_rate_init,
            tol=self.tol,
            n_iter_no_change=self.n_iter_no_change,
        )

    def _transform_features(self, coords: np.ndarray) -> np.ndarray:
        """Scale coordinates to [-1, 1] and apply configured spatial encodings."""
        return self._feature_encoder.transform(coords)

    def _fit_impl(self, coords: np.ndarray, shifts: np.ndarray) -> None:
        """Fit the MLP regressor on (coords, shifts)."""
        x_feats = self._feature_encoder.fit_transform(coords)
        fit_x, fit_y = x_feats, shifts
        if self.s_shape_weight > 1:
            feats = self._feature_encoder.extract_shape_categories(coords)
            s_mask = feats["cat_2d"] == 3
            if np.any(s_mask):
                extra_x = np.repeat(x_feats[s_mask], self.s_shape_weight - 1, axis=0)
                extra_y = np.repeat(shifts[s_mask], self.s_shape_weight - 1, axis=0)
                fit_x = np.concatenate([x_feats, extra_x], axis=0)
                fit_y = np.concatenate([shifts, extra_y], axis=0)
                rng = np.random.RandomState(self.random_state)
                perm = rng.permutation(len(fit_x))
                fit_x = fit_x[perm]
                fit_y = fit_y[perm]

        self._model.fit(fit_x, fit_y)

    def _predict_impl(self, coords: np.ndarray) -> np.ndarray:
        """Predict shift_n and shift_e for given query coordinates."""
        x_feats = self._feature_encoder.transform(coords)
        preds = self._model.predict(x_feats)
        return np.asarray(preds, dtype=np.float64)

    def _runtime_dependencies(self) -> tuple[str, ...]:
        return ("scikit-learn", "numpy")

    def _grid_lookup_bytes(self) -> int:
        """MLP is a parametric neural network; it has zero external lookup tables."""
        return 0

    @property
    def total_parameters(self) -> int:
        """Calculate total number of trainable weights and biases in the network."""
        if not self._is_fitted or not hasattr(self._model, "coefs_"):
            return 0
        w_params = sum(coef.size for coef in self._model.coefs_)
        b_params = sum(intercept.size for intercept in self._model.intercepts_)
        return w_params + b_params
