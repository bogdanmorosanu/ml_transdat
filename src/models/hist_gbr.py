"""Histogram-based Gradient Boosted Decision Tree regressor for 2D distortion shifts."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from pytransdat.spg_reader import SpgReader
from src.models.base import BaseGridShiftModel


class HistGBRGridShiftModel(BaseGridShiftModel):
    """Per-axis Histogram Gradient Boosted Trees regressor for 2D grid shifts."""

    def __init__(
        self,
        max_iter: int = 500,
        max_leaf_nodes: int = 63,
        learning_rate: float = 0.05,
        l2_regularization: float = 1e-4,
        use_cyclic_cell_pos: bool = True,
        random_state: int = 42,
        spg_path: str | None = None,
    ) -> None:
        super().__init__()
        self.max_iter = int(max_iter)
        self.max_leaf_nodes = int(max_leaf_nodes)
        self.learning_rate = float(learning_rate)
        self.l2_regularization = float(l2_regularization)
        self.use_cyclic_cell_pos = bool(use_cyclic_cell_pos)
        self.random_state = int(random_state)
        self.spg_path = spg_path

        self._min_n: float = 0.0
        self._min_e: float = 0.0
        self._step_n: float = 11000.0
        self._step_e: float = 11000.0
        self._model_dn: HistGradientBoostingRegressor | None = None
        self._model_de: HistGradientBoostingRegressor | None = None

    def _build_features(self, coords: np.ndarray) -> np.ndarray:
        if not self.use_cyclic_cell_pos:
            return coords
        n_arr = coords[:, 0]
        e_arr = coords[:, 1]
        r_idx = np.floor((n_arr - self._min_n) / self._step_n)
        c_idx = np.floor((e_arr - self._min_e) / self._step_e)
        u = (n_arr - (self._min_n + r_idx * self._step_n)) / self._step_n
        v = (e_arr - (self._min_e + c_idx * self._step_e)) / self._step_e
        return np.column_stack(
            [
                n_arr,
                e_arr,
                np.sin(2.0 * np.pi * u),
                np.cos(2.0 * np.pi * u),
                np.sin(2.0 * np.pi * v),
                np.cos(2.0 * np.pi * v),
            ]
        )

    def _fit_impl(self, coords: np.ndarray, shifts: np.ndarray) -> None:
        if self.use_cyclic_cell_pos:
            reader = SpgReader(self.spg_path) if self.spg_path is not None else SpgReader()
            min_e, _, min_n, _, step_e, step_n = reader.shift_bounds
            self._min_n = float(min_n)
            self._min_e = float(min_e)
            self._step_n = float(step_n)
            self._step_e = float(step_e)

        x_feats = self._build_features(coords)
        kwargs = {
            "max_iter": self.max_iter,
            "max_leaf_nodes": self.max_leaf_nodes,
            "learning_rate": self.learning_rate,
            "l2_regularization": self.l2_regularization,
            "random_state": self.random_state,
        }
        self._model_dn = HistGradientBoostingRegressor(**kwargs).fit(x_feats, shifts[:, 0])
        self._model_de = HistGradientBoostingRegressor(**kwargs).fit(x_feats, shifts[:, 1])

    def _predict_impl(self, coords: np.ndarray) -> np.ndarray:
        if self._model_dn is None or self._model_de is None:
            raise RuntimeError("Model is not fitted yet.")
        x_feats = self._build_features(coords)
        return np.column_stack(
            [self._model_dn.predict(x_feats), self._model_de.predict(x_feats)]
        ).astype(np.float64)

    def _runtime_dependencies(self) -> tuple[str, ...]:
        return ("scikit-learn", "numpy")
