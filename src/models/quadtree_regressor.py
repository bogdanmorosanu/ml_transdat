"""Adaptive Quadtree bilinear lookup regressor with depth-bucketed lattice search."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.models.base import BaseGridShiftModel

_KEY_SHIFT: int = 24
_MAX_INDEX: int = (1 << _KEY_SHIFT) - 1


@dataclass(frozen=True)
class _LatticeLevel:
    """Sorted regular lattice of quadtree leaf boxes at a single cell resolution."""

    origin_n: float
    origin_e: float
    step_n: float
    step_e: float
    keys: np.ndarray
    n_min: np.ndarray
    n_max: np.ndarray
    e_min: np.ndarray
    e_max: np.ndarray
    v00: np.ndarray
    v01: np.ndarray
    v10: np.ndarray
    v11: np.ndarray

    def nbytes(self) -> int:
        """Total bytes occupied by all arrays in this lattice level."""
        return int(
            self.keys.nbytes
            + self.n_min.nbytes
            + self.n_max.nbytes
            + self.e_min.nbytes
            + self.e_max.nbytes
            + self.v00.nbytes
            + self.v01.nbytes
            + self.v10.nbytes
            + self.v11.nbytes
        )


@dataclass(frozen=True)
class _FallbackBoxes:
    """Arbitrary non-lattice boxes queried sequentially only if a lattice misses."""

    n_min: np.ndarray
    n_max: np.ndarray
    e_min: np.ndarray
    e_max: np.ndarray
    v00: np.ndarray
    v01: np.ndarray
    v10: np.ndarray
    v11: np.ndarray

    def nbytes(self) -> int:
        """Total bytes occupied by fallback box arrays."""
        return int(
            self.n_min.nbytes
            + self.n_max.nbytes
            + self.e_min.nbytes
            + self.e_max.nbytes
            + self.v00.nbytes
            + self.v01.nbytes
            + self.v10.nbytes
            + self.v11.nbytes
        )


@dataclass(frozen=True)
class _QuadtreeComponentIndex:
    """Indexed lookup structure for a single shift component (``dN`` or ``dE``)."""

    levels: tuple[_LatticeLevel, ...]
    fallback: _FallbackBoxes | None

    def nbytes(self) -> int:
        """Total bytes occupied by lattice levels and optional fallback boxes."""
        total = sum(lvl.nbytes() for lvl in self.levels)
        if self.fallback is not None:
            total += self.fallback.nbytes()
        return total


class AdaptiveQuadtreeRegressor(BaseGridShiftModel):
    """2D piecewise bilinear interpolator over adaptive quadtree cells.

    Organizes quadtree leaves into sorted depth-bucketed lattice levels for
    vectorized :math:`O(D \\log N)` lookup via ``np.searchsorted``, falling back
    to exact closed-box search for any non-lattice boxes.
    """

    def __init__(
        self,
        dn_grid: pd.DataFrame | None = None,
        de_grid: pd.DataFrame | None = None,
    ) -> None:
        super().__init__()
        self._dn_index: _QuadtreeComponentIndex | None = None
        self._de_index: _QuadtreeComponentIndex | None = None
        if dn_grid is not None and de_grid is not None:
            self.load_grids(dn_grid, de_grid)

    @staticmethod
    def _build_component_index(grid: pd.DataFrame) -> _QuadtreeComponentIndex:
        required = ("n_min", "n_max", "e_min", "e_max", "v00", "v01", "v10", "v11")
        missing = [col for col in required if col not in grid.columns]
        if missing:
            raise ValueError(
                f"Adaptive grid DataFrame missing required columns: {missing}"
            )

        n_min = grid["n_min"].to_numpy(dtype=np.float64)
        n_max = grid["n_max"].to_numpy(dtype=np.float64)
        e_min = grid["e_min"].to_numpy(dtype=np.float64)
        e_max = grid["e_max"].to_numpy(dtype=np.float64)
        v00 = grid["v00"].to_numpy(dtype=np.float64)
        v01 = grid["v01"].to_numpy(dtype=np.float64)
        v10 = grid["v10"].to_numpy(dtype=np.float64)
        v11 = grid["v11"].to_numpy(dtype=np.float64)

        if len(n_min) == 0:
            return _QuadtreeComponentIndex(levels=(), fallback=None)

        dn = n_max - n_min
        de = e_max - e_min
        if not (np.all(dn > 0.0) and np.all(de > 0.0)):
            raise ValueError("All adaptive grid cells must have positive extents.")

        dims = np.column_stack([np.round(dn, 6), np.round(de, 6)])
        _, inverse = np.unique(dims, axis=0, return_inverse=True)

        levels: list[_LatticeLevel] = []
        fallback_masks: list[np.ndarray] = []

        for group_id in range(int(inverse.max()) + 1):
            mask = inverse == group_id
            step_n = float(np.mean(dn[mask]))
            step_e = float(np.mean(de[mask]))
            ln_min = n_min[mask]
            le_min = e_min[mask]
            origin_n = float(ln_min.min())
            origin_e = float(le_min.min())

            fx = (ln_min - origin_n) / step_n
            fy = (le_min - origin_e) / step_e
            rx = np.round(fx)
            ry = np.round(fy)

            aligned = bool(
                np.all(np.abs(fx - rx) < 1e-4)
                and np.all(np.abs(fy - ry) < 1e-4)
                and int(rx.max()) <= _MAX_INDEX
                and int(ry.max()) <= _MAX_INDEX
            )
            if not aligned:
                fallback_masks.append(mask)
                continue

            ix = rx.astype(np.int64)
            iy = ry.astype(np.int64)
            keys = (ix << _KEY_SHIFT) | iy
            order = np.argsort(keys, kind="mergesort")
            sorted_keys = keys[order]

            if len(sorted_keys) > 1 and not np.all(sorted_keys[1:] > sorted_keys[:-1]):
                fallback_masks.append(mask)
                continue

            levels.append(
                _LatticeLevel(
                    origin_n=origin_n,
                    origin_e=origin_e,
                    step_n=step_n,
                    step_e=step_e,
                    keys=sorted_keys,
                    n_min=ln_min[order],
                    n_max=n_max[mask][order],
                    e_min=le_min[order],
                    e_max=e_max[mask][order],
                    v00=v00[mask][order],
                    v01=v01[mask][order],
                    v10=v10[mask][order],
                    v11=v11[mask][order],
                )
            )

        # Search densest levels first to resolve most query points in fewest probes
        levels.sort(key=lambda lvl: len(lvl.keys), reverse=True)

        fallback: _FallbackBoxes | None = None
        if fallback_masks:
            fb_mask = np.logical_or.reduce(fallback_masks)
            fallback = _FallbackBoxes(
                n_min=n_min[fb_mask],
                n_max=n_max[fb_mask],
                e_min=e_min[fb_mask],
                e_max=e_max[fb_mask],
                v00=v00[fb_mask],
                v01=v01[fb_mask],
                v10=v10[fb_mask],
                v11=v11[fb_mask],
            )

        return _QuadtreeComponentIndex(levels=tuple(levels), fallback=fallback)

    def load_grids(
        self, dn_grid: pd.DataFrame, de_grid: pd.DataFrame
    ) -> AdaptiveQuadtreeRegressor:
        """Build vectorized spatial indices from ``dN`` and ``dE`` quadtree tables."""
        t0 = time.perf_counter()
        self._dn_index = self._build_component_index(dn_grid)
        self._de_index = self._build_component_index(de_grid)
        self._fit_time_sec = time.perf_counter() - t0
        self._is_fitted = True
        return self

    def _fit_impl(self, coords: np.ndarray, shifts: np.ndarray) -> None:
        """Quadtree cells are pre-partitioned via :meth:`load_grids`; not a point-cloud fit."""
        raise NotImplementedError(
            "AdaptiveQuadtreeRegressor uses pre-built quadtree cell tables via load_grids(dn_grid, de_grid)."
        )

    @staticmethod
    def _probe_lattice_level(
        lvl: _LatticeLevel,
        n_vals: np.ndarray,
        e_vals: np.ndarray,
        unresolved: np.ndarray,
        preds: np.ndarray,
        dn_off: int,
        de_off: int,
    ) -> None:
        u_idx = np.flatnonzero(unresolved)
        if len(u_idx) == 0:
            return

        qn = n_vals[u_idx]
        qe = e_vals[u_idx]
        ix = np.floor((qn - lvl.origin_n) / lvl.step_n).astype(np.int64) + dn_off
        iy = np.floor((qe - lvl.origin_e) / lvl.step_e).astype(np.int64) + de_off
        valid_key = (ix >= 0) & (iy >= 0) & (ix <= _MAX_INDEX) & (iy <= _MAX_INDEX)

        q_keys = np.where(valid_key, (ix << _KEY_SHIFT) | iy, 0)
        pos = np.searchsorted(lvl.keys, q_keys)
        pos_clipped = np.minimum(pos, len(lvl.keys) - 1)
        hit = valid_key & (lvl.keys[pos_clipped] == q_keys)
        if not hit.any():
            return

        h_idx = np.flatnonzero(hit)
        p = pos_clipped[h_idx]
        hn = qn[h_idx]
        he = qe[h_idx]
        in_box = (
            (lvl.n_min[p] <= hn)
            & (hn <= lvl.n_max[p])
            & (lvl.e_min[p] <= he)
            & (he <= lvl.e_max[p])
        )
        if not in_box.any():
            return

        bp = p[in_box]
        bu_idx = u_idx[h_idx[in_box]]
        u = (hn[in_box] - lvl.n_min[bp]) / (lvl.n_max[bp] - lvl.n_min[bp])
        v = (he[in_box] - lvl.e_min[bp]) / (lvl.e_max[bp] - lvl.e_min[bp])
        preds[bu_idx] = (
            (1.0 - u) * (1.0 - v) * lvl.v00[bp]
            + (1.0 - u) * v * lvl.v01[bp]
            + u * (1.0 - v) * lvl.v10[bp]
            + u * v * lvl.v11[bp]
        )
        unresolved[bu_idx] = False

    @classmethod
    def _query_index(
        cls,
        index: _QuadtreeComponentIndex,
        n_vals: np.ndarray,
        e_vals: np.ndarray,
    ) -> np.ndarray:
        preds = np.full(len(n_vals), np.nan, dtype=np.float64)
        unresolved = np.ones(len(n_vals), dtype=bool)

        # Pass 1: primary (0, 0) cell lookup across all levels
        for lvl in index.levels:
            if not unresolved.any():
                break
            cls._probe_lattice_level(lvl, n_vals, e_vals, unresolved, preds, 0, 0)

        # Pass 2: closed-boundary offsets for exact upper-edge / corner points
        if unresolved.any():
            for dn_off, de_off in ((-1, 0), (0, -1), (-1, -1)):
                for lvl in index.levels:
                    if not unresolved.any():
                        break
                    cls._probe_lattice_level(
                        lvl, n_vals, e_vals, unresolved, preds, dn_off, de_off
                    )

        # Pass 3: exact linear scan only for any non-lattice fallback cells
        if unresolved.any() and index.fallback is not None:
            fb = index.fallback
            for i in np.flatnonzero(unresolved):
                n = n_vals[i]
                e = e_vals[i]
                mask = (
                    (fb.n_min <= n)
                    & (n <= fb.n_max)
                    & (fb.e_min <= e)
                    & (e <= fb.e_max)
                )
                if not mask.any():
                    continue
                idx = int(np.argmax(mask))
                u = (n - fb.n_min[idx]) / (fb.n_max[idx] - fb.n_min[idx])
                v = (e - fb.e_min[idx]) / (fb.e_max[idx] - fb.e_min[idx])
                preds[i] = (
                    (1.0 - u) * (1.0 - v) * fb.v00[idx]
                    + (1.0 - u) * v * fb.v01[idx]
                    + u * (1.0 - v) * fb.v10[idx]
                    + u * v * fb.v11[idx]
                )
        return preds

    def _predict_impl(self, coords: np.ndarray) -> np.ndarray:
        if self._dn_index is None or self._de_index is None:
            raise RuntimeError(
                "AdaptiveQuadtreeRegressor is not initialized. Call load_grids() first."
            )

        n_vals = coords[:, 0]
        e_vals = coords[:, 1]
        pred_dn = self._query_index(self._dn_index, n_vals, e_vals)
        pred_de = self._query_index(self._de_index, n_vals, e_vals)
        return np.column_stack([pred_dn, pred_de])

    def _grid_lookup_bytes(self) -> int:
        total = 0
        if self._dn_index is not None:
            total += self._dn_index.nbytes()
        if self._de_index is not None:
            total += self._de_index.nbytes()
        return total
