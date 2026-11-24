"""Shared spatial coordinate normalization, RFF, cell-phase, and neighbor feature encoder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.geometry.cell_uv import compute_spg_cell_uv, encode_cyclic_cell_uv


class SpatialFeatureEncoder:
    """Reusable spatial feature encoder for neural grid-shift regressors.

    Handles ``[-1, 1]`` coordinate scaling, single/multi-scale Random Fourier
    Features (RFF), 11 km cell shape one-hot features, ``3x3`` neighbor slopes/twists
    (with optional Hermite weighting or 6-term Hermite summation), cell-center
    distance, and smooth cyclic ``(u, v)`` cell-phase harmonics.
    """

    def __init__(
        self,
        *,
        normalize_inputs: bool = True,
        use_fourier_features: bool = False,
        fourier_scale: float = 3.0,
        fourier_scales: tuple[float, ...] | None = None,
        num_fourier_frequencies: int = 32,
        random_state: int = 42,
        use_shape_features: bool = False,
        shape_tol_mm: float = 5.0,
        use_neighbor_features: bool = False,
        neighbor_feature_mode: str = "all",
        neighbor_features_path: str | Path = "data/features/cell_neighbor_features.parquet",
        scale_neighbor_features: bool = True,
        use_hermite_weights: bool = False,
        use_cell_center_distance: bool = False,
        use_cyclic_cell_pos: bool = False,
        cyclic_cell_harmonics: int = 1,
    ) -> None:
        if neighbor_feature_mode not in ("all", "slopes", "hermite_sum"):
            raise ValueError(
                f"Unsupported neighbor_feature_mode={neighbor_feature_mode!r}; "
                "expected 'all', 'slopes', or 'hermite_sum'."
            )
        self.normalize_inputs = bool(normalize_inputs)
        self.use_fourier_features = bool(use_fourier_features)
        self.fourier_scale = float(fourier_scale)
        self.fourier_scales = (
            tuple(float(s) for s in fourier_scales) if fourier_scales is not None else None
        )
        self.num_fourier_frequencies = int(num_fourier_frequencies)
        self.random_state = int(random_state)
        self.use_shape_features = bool(use_shape_features)
        self.shape_tol_mm = float(shape_tol_mm)
        self.use_neighbor_features = bool(use_neighbor_features)
        self.neighbor_feature_mode = str(neighbor_feature_mode)
        self.neighbor_features_path = str(neighbor_features_path)
        self.scale_neighbor_features = bool(scale_neighbor_features)
        self.use_hermite_weights = bool(use_hermite_weights)
        self.use_cell_center_distance = bool(use_cell_center_distance)
        self.use_cyclic_cell_pos = bool(use_cyclic_cell_pos)
        self.cyclic_cell_harmonics = int(cyclic_cell_harmonics)

        self._min_coords: np.ndarray | None = None
        self._max_coords: np.ndarray | None = None
        self._fourier_basis: np.ndarray | None = None
        self._shape_extractor: Any = None
        self._neighbor_extractor: Any = None
        self._neighbor_col_indices: np.ndarray | None = None
        self._neighbor_mean: np.ndarray | None = None
        self._neighbor_std: np.ndarray | None = None

    def _ensure_shape_extractor(self) -> None:
        if self._shape_extractor is None:
            from src.geometry.shape_classifier import CellShapeFeatureExtractor

            self._shape_extractor = CellShapeFeatureExtractor(
                default_tol_mm=self.shape_tol_mm
            )

    def _ensure_neighbor_extractor(self) -> None:
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
        self._ensure_neighbor_extractor()
        feats = self._neighbor_extractor.lookup_points(coords[:, 0], coords[:, 1])
        feats = feats[:, self._neighbor_col_indices]
        return np.nan_to_num(feats, nan=0.0)

    def _compute_hermite_weights(self, coords: np.ndarray) -> np.ndarray:
        from src.geometry.neighbor_features import NEIGHBOR_FEATURE_COLS

        u, v = compute_spg_cell_uv(coords)
        h0_u = (1.0 + 2.0 * u) * (1.0 - u) ** 2
        h1_u = (u**2) * (3.0 - 2.0 * u)
        h10_u = u * (1.0 - u) ** 2
        h11_u = (u**2) * (u - 1.0)

        h0_v = (1.0 + 2.0 * v) * (1.0 - v) ** 2
        h1_v = (v**2) * (3.0 - 2.0 * v)
        h10_v = v * (1.0 - v) ** 2
        h11_v = (v**2) * (v - 1.0)

        corner_basis = {
            "bl": (h0_u, h10_u, h0_v, h10_v),
            "br": (h0_u, h10_u, h1_v, h11_v),
            "tl": (h1_u, h11_u, h0_v, h10_v),
            "tr": (h1_u, h11_u, h1_v, h11_v),
        }

        assert self._neighbor_col_indices is not None
        selected_cols = [NEIGHBOR_FEATURE_COLS[i] for i in self._neighbor_col_indices]
        weights = np.empty((len(coords), len(selected_cols)), dtype=np.float64)
        for j, col in enumerate(selected_cols):
            _, corner, feat = col.split("_", 2)
            hu, hdu, hv, hdv = corner_basis[corner]
            if feat in ("left_slope", "right_slope"):
                weights[:, j] = hu * hdv
            elif feat in ("bottom_slope", "top_slope"):
                weights[:, j] = hdu * hv
            else:
                weights[:, j] = hdu * hdv
        return weights

    def _compute_hermite_sum_features(
        self, coords: np.ndarray, scaled_all: np.ndarray
    ) -> np.ndarray:
        weights = self._compute_hermite_weights(coords)
        weighted = scaled_all * weights
        out = np.empty((len(coords), 6), dtype=np.float64)
        for ch_idx in range(2):
            base = ch_idx * 20
            block = weighted[:, base : base + 20].reshape(-1, 4, 5)
            dv_sum = np.sum(0.5 * (block[:, :, 0] + block[:, :, 1]), axis=1)
            du_sum = np.sum(0.5 * (block[:, :, 2] + block[:, :, 3]), axis=1)
            tw_sum = np.sum(block[:, :, 4], axis=1)
            out[:, ch_idx * 3 + 0] = du_sum
            out[:, ch_idx * 3 + 1] = dv_sum
            out[:, ch_idx * 3 + 2] = tw_sum
        return out

    def extract_shape_categories(self, coords: np.ndarray) -> dict[str, np.ndarray]:
        """Extract 11 km cell shape classifications for sample weighting or features."""
        self._ensure_shape_extractor()
        return self._shape_extractor.extract_features(
            coords[:, 0], coords[:, 1], tol_mm=self.shape_tol_mm
        )

    def fit(self, coords: np.ndarray) -> SpatialFeatureEncoder:
        """Fit coordinate normalization, Fourier basis, and neighbor scaling statistics."""
        if self.normalize_inputs:
            self._min_coords = np.min(coords, axis=0)
            self._max_coords = np.max(coords, axis=0)

        if self.use_fourier_features:
            rng = np.random.RandomState(self.random_state)
            if self.fourier_scales is not None and len(self.fourier_scales) > 0:
                n_bands = len(self.fourier_scales)
                base_n = self.num_fourier_frequencies // n_bands
                rem = self.num_fourier_frequencies % n_bands
                bands: list[np.ndarray] = []
                for idx, sc in enumerate(self.fourier_scales):
                    n_k = base_n + (1 if idx < rem else 0)
                    if n_k > 0:
                        bands.append(rng.randn(2, n_k) * sc)
                self._fourier_basis = np.hstack(bands)
            else:
                self._fourier_basis = (
                    rng.randn(2, self.num_fourier_frequencies) * self.fourier_scale
                )

        if self.use_neighbor_features and self.scale_neighbor_features:
            raw_neighbor = self._extract_neighbor_features(coords)
            self._neighbor_mean = np.mean(raw_neighbor, axis=0)
            std = np.std(raw_neighbor, axis=0)
            std[std == 0.0] = 1.0
            self._neighbor_std = std

        return self

    def transform(self, coords: np.ndarray) -> np.ndarray:
        """Transform ``(N, 2)`` Stereo 70 coordinates into encoded feature matrix."""
        if self.normalize_inputs:
            if self._min_coords is None or self._max_coords is None:
                raise RuntimeError("SpatialFeatureEncoder missing normalization parameters.")
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
            feats = self.extract_shape_categories(coords)
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
                    raise RuntimeError("Missing neighbor feature scaling parameters.")
                neighbor_feats = (neighbor_feats - self._neighbor_mean) / self._neighbor_std
            if self.neighbor_feature_mode == "hermite_sum":
                neighbor_feats = self._compute_hermite_sum_features(coords, neighbor_feats)
            elif self.use_hermite_weights:
                neighbor_feats = neighbor_feats * self._compute_hermite_weights(coords)
            x_spatial = np.concatenate([x_spatial, neighbor_feats], axis=1)

        if self.use_cell_center_distance:
            self._ensure_neighbor_extractor()
            center_dists = self._neighbor_extractor.lookup_center_distances(
                coords[:, 0], coords[:, 1]
            )
            x_spatial = np.concatenate([x_spatial, center_dists], axis=1)

        if self.use_cyclic_cell_pos:
            cyclic_pos = encode_cyclic_cell_uv(
                coords,
                harmonics=self.cyclic_cell_harmonics,
                include_cross_terms=self.cyclic_cell_harmonics > 1,
            )
            x_spatial = np.concatenate([x_spatial, cyclic_pos], axis=1)

        return x_spatial

    def fit_transform(self, coords: np.ndarray) -> np.ndarray:
        """Fit encoder state on ``coords`` and return transformed features."""
        return self.fit(coords).transform(coords)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_shape_extractor"] = None
        state["_neighbor_extractor"] = None
        return state
