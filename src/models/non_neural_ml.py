"""Zero-PyTorch Machine Learning shift regressors.

Implements:
- ``RFFELMGridShiftModel`` (V2-8): Extreme Learning Machine solved analytically via Ridge.
- ``LSSVMGridShiftModel`` (V2-9): Nystrom-approximated Least-Squares SVM / Kernel Ridge.
- ``XGBoostGridShiftModel`` (V2-10a): Per-axis XGBoost histogram trees on residual + (u, v).
- ``LinearLeafTreeShiftModel`` (V2-10b): Piecewise-linear/quadratic model tree.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.geometry.cell_uv import compute_spg_cell_uv, encode_cyclic_cell_uv
from src.models._features import SpatialFeatureEncoder
from src.models.base import BaseGridShiftModel


class RFFELMGridShiftModel(BaseGridShiftModel):
    """Extreme Learning Machine with multi-scale RFF solved via closed-form Ridge."""

    def __init__(
        self,
        n_features: int = 2048,
        fourier_scales: tuple[float, ...] = (1.0, 3.0, 10.0),
        cyclic_cell_harmonics: int = 2,
        alpha: float = 1e-4,
        random_state: int = 42,
    ) -> None:
        super().__init__()
        self.n_features = int(n_features)
        self.fourier_scales = tuple(float(s) for s in fourier_scales)
        self.cyclic_cell_harmonics = int(cyclic_cell_harmonics)
        self.alpha = float(alpha)
        self.random_state = int(random_state)

        self._encoder: SpatialFeatureEncoder | None = None
        self._weights: np.ndarray | None = None

    def _fit_impl(self, coords: np.ndarray, shifts: np.ndarray) -> None:
        self._encoder = SpatialFeatureEncoder(
            normalize_inputs=True,
            use_fourier_features=True,
            fourier_scales=self.fourier_scales,
            num_fourier_frequencies=self.n_features // 2,
            use_cyclic_cell_pos=self.cyclic_cell_harmonics > 0,
            cyclic_cell_harmonics=max(self.cyclic_cell_harmonics, 1),
            random_state=self.random_state,
        )
        h_mat = self._encoder.fit_transform(coords)
        h_aug = np.column_stack([np.ones((len(h_mat), 1), dtype=np.float64), h_mat])
        d_feat = h_aug.shape[1]
        gram = h_aug.T @ h_aug
        gram[np.diag_indices(d_feat)] += self.alpha
        rhs = h_aug.T @ np.asarray(shifts, dtype=np.float64)
        self._weights = solve(gram, rhs, assume_a="pos")

    def _predict_impl(self, coords: np.ndarray) -> np.ndarray:
        if self._encoder is None or self._weights is None:
            raise RuntimeError("Model is not fitted yet.")
        h_mat = self._encoder.transform(coords)
        h_aug = np.column_stack([np.ones((len(h_mat), 1), dtype=np.float64), h_mat])
        return h_aug @ self._weights

    def _runtime_dependencies(self) -> tuple[str, ...]:
        return ("numpy", "scipy")


class LSSVMGridShiftModel(BaseGridShiftModel):
    """Nystrom-approximated Least-Squares SVM (Kernel Ridge) shift regressor."""

    def __init__(
        self,
        n_components: int = 1500,
        gamma: float = 2.0,
        alpha: float = 1e-4,
        cyclic_cell_harmonics: int = 2,
        random_state: int = 42,
    ) -> None:
        super().__init__()
        self.n_components = int(n_components)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.cyclic_cell_harmonics = int(cyclic_cell_harmonics)
        self.random_state = int(random_state)

        self._encoder: SpatialFeatureEncoder | None = None
        self._nystroem: Nystroem | None = None
        self._ridge: Ridge | None = None

    def _fit_impl(self, coords: np.ndarray, shifts: np.ndarray) -> None:
        self._encoder = SpatialFeatureEncoder(
            normalize_inputs=True,
            use_fourier_features=False,
            use_cyclic_cell_pos=self.cyclic_cell_harmonics > 0,
            cyclic_cell_harmonics=max(self.cyclic_cell_harmonics, 1),
            random_state=self.random_state,
        )
        x_feat = self._encoder.fit_transform(coords)
        n_comp = min(self.n_components, len(x_feat))
        self._nystroem = Nystroem(
            kernel="rbf",
            gamma=self.gamma,
            n_components=n_comp,
            random_state=self.random_state,
        )
        z_feat = self._nystroem.fit_transform(x_feat)
        self._ridge = Ridge(alpha=self.alpha)
        self._ridge.fit(z_feat, np.asarray(shifts, dtype=np.float64))

    def _predict_impl(self, coords: np.ndarray) -> np.ndarray:
        if self._encoder is None or self._nystroem is None or self._ridge is None:
            raise RuntimeError("Model is not fitted yet.")
        x_feat = self._encoder.transform(coords)
        z_feat = self._nystroem.transform(x_feat)
        return np.asarray(self._ridge.predict(z_feat), dtype=np.float64)

    def _runtime_dependencies(self) -> tuple[str, ...]:
        return ("numpy", "scikit-learn")


class XGBoostGridShiftModel(BaseGridShiftModel):
    """Per-axis XGBoost histogram regressor with intra-cell ``(u, v)`` features."""

    def __init__(
        self,
        n_estimators: int = 600,
        max_depth: int = 8,
        learning_rate: float = 0.08,
        subsample: float = 0.8,
        colsample_bytree: float = 0.9,
        reg_lambda: float = 1.0,
        cyclic_cell_harmonics: int = 2,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        super().__init__()
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.learning_rate = float(learning_rate)
        self.subsample = float(subsample)
        self.colsample_bytree = float(colsample_bytree)
        self.reg_lambda = float(reg_lambda)
        self.cyclic_cell_harmonics = int(cyclic_cell_harmonics)
        self.random_state = int(random_state)
        self.n_jobs = int(n_jobs)

        self._min_coords: np.ndarray | None = None
        self._max_coords: np.ndarray | None = None
        self._model_dn: XGBRegressor | None = None
        self._model_de: XGBRegressor | None = None

    def _build_features(self, coords: np.ndarray) -> np.ndarray:
        assert self._min_coords is not None and self._max_coords is not None
        diff = np.where(self._max_coords > self._min_coords, self._max_coords - self._min_coords, 1.0)
        xy_norm = 2.0 * (coords - self._min_coords) / diff - 1.0
        u, v = compute_spg_cell_uv(coords)
        parts = [xy_norm, u[:, None], v[:, None]]
        if self.cyclic_cell_harmonics > 0:
            parts.append(
                encode_cyclic_cell_uv(
                    coords, harmonics=self.cyclic_cell_harmonics, include_cross_terms=False
                )
            )
        return np.hstack(parts).astype(np.float32)

    def _fit_impl(self, coords: np.ndarray, shifts: np.ndarray) -> None:
        self._min_coords = np.min(coords, axis=0)
        self._max_coords = np.max(coords, axis=0)
        feats = self._build_features(coords)
        y = np.asarray(shifts, dtype=np.float32)

        common_kw = dict(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            tree_method="hist",
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self._model_dn = XGBRegressor(**common_kw)
        self._model_de = XGBRegressor(**common_kw)
        self._model_dn.fit(feats, y[:, 0])
        self._model_de.fit(feats, y[:, 1])

    def _predict_impl(self, coords: np.ndarray) -> np.ndarray:
        if self._model_dn is None or self._model_de is None:
            raise RuntimeError("Model is not fitted yet.")
        feats = self._build_features(coords)
        return np.column_stack(
            [self._model_dn.predict(feats), self._model_de.predict(feats)]
        ).astype(np.float64)

    def _runtime_dependencies(self) -> tuple[str, ...]:
        return ("numpy", "xgboost")


class LinearLeafTreeShiftModel(BaseGridShiftModel):
    """Piecewise-linear or piecewise-quadratic Decision Tree regressor."""

    def __init__(
        self,
        max_depth: int = 12,
        min_samples_leaf: int = 25,
        leaf_degree: int = 2,
        ridge: float = 1e-6,
        random_state: int = 42,
    ) -> None:
        super().__init__()
        if leaf_degree not in (1, 2):
            raise ValueError(f"leaf_degree must be 1 or 2, got {leaf_degree}")
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.leaf_degree = int(leaf_degree)
        self.ridge = float(ridge)
        self.random_state = int(random_state)

        self._tree: DecisionTreeRegressor | None = None
        self._leaf_coeffs: np.ndarray | None = None
        self._leaf_centers: np.ndarray | None = None
        self._leaf_scales: np.ndarray | None = None

    def _leaf_design(self, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
        cols = [np.ones_like(dx), dx, dy]
        if self.leaf_degree == 2:
            cols.extend([dx * dx, dx * dy, dy * dy])
        return np.column_stack(cols)

    def _fit_impl(self, coords: np.ndarray, shifts: np.ndarray) -> None:
        x_in = np.asarray(coords, dtype=np.float64)
        y_in = np.asarray(shifts, dtype=np.float64)
        self._tree = DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )
        self._tree.fit(x_in, y_in)
        leaf_ids = self._tree.apply(x_in)
        n_nodes = self._tree.tree_.node_count
        n_terms = 3 if self.leaf_degree == 1 else 6

        coeffs = np.zeros((n_nodes, n_terms, 2), dtype=np.float64)
        centers = np.zeros((n_nodes, 2), dtype=np.float64)
        scales = np.ones((n_nodes, 2), dtype=np.float64)
        eye_p = np.eye(n_terms, dtype=np.float64) * self.ridge

        order = np.argsort(leaf_ids)
        sorted_ids = leaf_ids[order]
        uniq_ids, starts, counts = np.unique(sorted_ids, return_index=True, return_counts=True)
        for lid, start, cnt in zip(uniq_ids, starts, counts):
            idx = order[start : start + cnt]
            pts = x_in[idx]
            tgt = y_in[idx]
            c = np.mean(pts, axis=0)
            s = np.maximum(np.max(np.abs(pts - c), axis=0), 1.0)
            centers[lid] = c
            scales[lid] = s
            dx = (pts[:, 0] - c[0]) / s[0]
            dy = (pts[:, 1] - c[1]) / s[1]
            a_mat = self._leaf_design(dx, dy)
            ata = a_mat.T @ a_mat + eye_p
            aty = a_mat.T @ tgt
            coeffs[lid] = np.linalg.solve(ata, aty)

        self._leaf_coeffs = coeffs
        self._leaf_centers = centers
        self._leaf_scales = scales

    def _predict_impl(self, coords: np.ndarray) -> np.ndarray:
        if (
            self._tree is None
            or self._leaf_coeffs is None
            or self._leaf_centers is None
            or self._leaf_scales is None
        ):
            raise RuntimeError("Model is not fitted yet.")
        x_in = np.asarray(coords, dtype=np.float64)
        lids = self._tree.apply(x_in)
        c = self._leaf_centers[lids]
        s = self._leaf_scales[lids]
        dx = (x_in[:, 0] - c[:, 0]) / s[:, 0]
        dy = (x_in[:, 1] - c[:, 1]) / s[:, 1]
        a_mat = self._leaf_design(dx, dy)  # (M, P)
        w = self._leaf_coeffs[lids]  # (M, P, 2)
        return np.einsum("mp,mpd->md", a_mat, w)

    def _runtime_dependencies(self) -> tuple[str, ...]:
        return ("numpy", "scikit-learn")
