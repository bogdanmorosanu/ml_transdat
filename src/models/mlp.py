"""Multilayer Perceptron (MLP) neural network regressor for 2D geodetic distortion grid shifts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.neural_network import MLPRegressor

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
        """Initialize the MLP neural network spatial regressor.

        :param hidden_layer_sizes: Tuple representing neuron counts per hidden layer.
        :param activation: Activation function ('relu', 'tanh', 'identity', 'logistic').
        :param solver: Optimization solver ('lbfgs', 'adam', 'sgd').
        :param alpha: L2 regularization penalty parameter.
        :param max_iter: Maximum optimization iterations.
        :param random_state: Reproducibility seed.
        :param normalize_inputs: Whether to scale coordinates to [-1, 1].
        :param use_fourier_features: Whether to expand inputs using Random Fourier Features.
        :param fourier_scale: Standard deviation Gaussian scale for Fourier projection frequencies.
        :param num_fourier_frequencies: Number of projection dimensions (generates 2 * frequencies features).
        :param batch_size: Size of mini-batches for stochastic optimizers ('auto' or int).
        :param learning_rate_init: Initial learning rate for Adam or SGD.
        :param tol: Optimization convergence tolerance threshold.
        :param n_iter_no_change: Maximum number of epochs without improvement before early stopping.
        """
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
        self._shape_extractor: object | None = None

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
        self._neighbor_extractor: Any = None
        self._neighbor_col_indices: np.ndarray | None = None
        self._neighbor_mean: np.ndarray | None = None
        self._neighbor_std: np.ndarray | None = None

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
        self._min_coords: np.ndarray | None = None
        self._max_coords: np.ndarray | None = None
        self._fourier_basis: np.ndarray | None = None

    def _transform_features(self, coords: np.ndarray) -> np.ndarray:
        """Scale coordinates to [-1, 1] and optionally apply Fourier positional encoding."""
        if self.normalize_inputs:
            if self._min_coords is None or self._max_coords is None:
                raise RuntimeError(
                    "Model fitted state missing normalization parameters."
                )
            diff = self._max_coords - self._min_coords
            diff[diff == 0.0] = 1.0
            x_norm = 2.0 * (coords - self._min_coords) / diff - 1.0
        else:
            x_norm = coords

        if not self.use_fourier_features:
            x_spatial = x_norm
        else:
            if self._fourier_basis is None:
                raise RuntimeError("Fourier projection matrix B is uninitialized.")

            proj = 2.0 * np.pi * (x_norm @ self._fourier_basis)
            x_spatial = np.concatenate([x_norm, np.sin(proj), np.cos(proj)], axis=1)

        if self.use_shape_features:
            if self._shape_extractor is None:
                from src.geometry.shape_classifier import CellShapeFeatureExtractor

                self._shape_extractor = CellShapeFeatureExtractor(
                    default_tol_mm=self.shape_tol_mm
                )

            feats = self._shape_extractor.extract_features(  # type: ignore[union-attr]
                coords[:, 0], coords[:, 1], tol_mm=self.shape_tol_mm
            )
            cat_dn = feats["cat_dn"]
            cat_de = feats["cat_de"]
            shape_onehot = np.column_stack(
                [
                    (cat_dn == 1).astype(np.float64),
                    (cat_dn == 2).astype(np.float64),
                    (cat_dn == 3).astype(np.float64),
                    (cat_de == 1).astype(np.float64),
                    (cat_de == 2).astype(np.float64),
                    (cat_de == 3).astype(np.float64),
                ]
            )
            x_spatial = np.concatenate([x_spatial, shape_onehot], axis=1)

        if self.use_neighbor_features:
            neighbor_feats = self._extract_neighbor_features(coords)
            if self.scale_neighbor_features:
                if self._neighbor_mean is None or self._neighbor_std is None:
                    raise RuntimeError(
                        "Model fitted state missing neighbor feature scaling parameters."
                    )
                neighbor_feats = (
                    neighbor_feats - self._neighbor_mean
                ) / self._neighbor_std
            x_spatial = np.concatenate([x_spatial, neighbor_feats], axis=1)

        if self.use_cell_center_distance:
            self._ensure_neighbor_extractor()
            center_dists = self._neighbor_extractor.lookup_center_distances(
                coords[:, 0], coords[:, 1]
            )
            x_spatial = np.concatenate([x_spatial, center_dists], axis=1)

        return x_spatial

    def _ensure_neighbor_extractor(self) -> None:
        """Lazily initialize the CellNeighborFeatureExtractor and column indices."""
        if self._neighbor_extractor is None:
            from src.geometry.neighbor_features import (
                NEIGHBOR_FEATURE_COLS,
                CellNeighborFeatureExtractor,
            )

            self._neighbor_extractor = CellNeighborFeatureExtractor(
                self.neighbor_features_path
            )
            if self.neighbor_feature_mode == "slopes":
                self._neighbor_col_indices = np.array(
                    [i for i, c in enumerate(NEIGHBOR_FEATURE_COLS) if "slope" in c],
                    dtype=np.int64,
                )
            else:
                self._neighbor_col_indices = np.arange(
                    len(NEIGHBOR_FEATURE_COLS), dtype=np.int64
                )

    def _extract_neighbor_features(self, coords: np.ndarray) -> np.ndarray:
        """Extract selected 3x3 neighbor features (all 40 or 32 slopes) with NaN->0 fallback."""
        self._ensure_neighbor_extractor()
        feats = self._neighbor_extractor.lookup_points(coords[:, 0], coords[:, 1])
        feats = feats[:, self._neighbor_col_indices]
        return np.nan_to_num(feats, nan=0.0)

    def _fit_impl(self, coords: np.ndarray, shifts: np.ndarray) -> None:
        """Fit the MLP regressor on (coords, shifts)."""
        if self.normalize_inputs:
            self._min_coords = np.min(coords, axis=0)
            self._max_coords = np.max(coords, axis=0)

        if self.use_fourier_features:
            rng = np.random.RandomState(self.random_state)
            self._fourier_basis = (
                rng.randn(2, self.num_fourier_frequencies) * self.fourier_scale
            )

        if self.use_neighbor_features and self.scale_neighbor_features:
            raw_neighbor = self._extract_neighbor_features(coords)
            self._neighbor_mean = np.mean(raw_neighbor, axis=0)
            std = np.std(raw_neighbor, axis=0)
            std[std == 0.0] = 1.0
            self._neighbor_std = std

        x_feats = self._transform_features(coords)
        fit_x, fit_y = x_feats, shifts
        if self.s_shape_weight > 1:
            if self._shape_extractor is None:
                from src.geometry.shape_classifier import CellShapeFeatureExtractor

                self._shape_extractor = CellShapeFeatureExtractor(
                    default_tol_mm=self.shape_tol_mm
                )
            feats = self._shape_extractor.extract_features(  # type: ignore[union-attr]
                coords[:, 0], coords[:, 1], tol_mm=self.shape_tol_mm
            )
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
        x_feats = self._transform_features(coords)
        preds = self._model.predict(x_feats)
        return np.asarray(preds, dtype=np.float64)

    def _runtime_dependencies(self) -> tuple[str, ...]:
        return ("scikit-learn", "numpy")

    def _grid_lookup_bytes(self) -> int:
        """MLP is a parametric neural network; it has zero external lookup tables."""
        return 0

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_shape_extractor"] = None
        state["_neighbor_extractor"] = None
        return state

    @property
    def total_parameters(self) -> int:
        """Calculate total number of trainable weights and biases in the network."""
        if not self._is_fitted or not hasattr(self._model, "coefs_"):
            return 0
        w_params = sum(coef.size for coef in self._model.coefs_)
        b_params = sum(intercept.size for intercept in self._model.intercepts_)
        return w_params + b_params
