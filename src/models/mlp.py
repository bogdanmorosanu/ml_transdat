"""Multilayer Perceptron (MLP) neural network regressor for 2D geodetic distortion grid shifts."""

from __future__ import annotations

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
            return x_norm

        if self._fourier_basis is None:
            raise RuntimeError("Fourier projection matrix B is uninitialized.")

        proj = 2.0 * np.pi * (x_norm @ self._fourier_basis)
        return np.concatenate([x_norm, np.sin(proj), np.cos(proj)], axis=1)

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

        x_feats = self._transform_features(coords)
        self._model.fit(x_feats, shifts)

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

    @property
    def total_parameters(self) -> int:
        """Calculate total number of trainable weights and biases in the network."""
        if not self._is_fitted or not hasattr(self._model, "coefs_"):
            return 0
        w_params = sum(coef.size for coef in self._model.coefs_)
        b_params = sum(intercept.size for intercept in self._model.intercepts_)
        return w_params + b_params
