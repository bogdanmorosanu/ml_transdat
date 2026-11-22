"""Composite two-stage residual shift models and cell-bilinear baseline."""

from __future__ import annotations

import numpy as np

from pytransdat.spg_reader import SpgReader
from src.models.base import BaseGridShiftModel


class CellBilinearShiftModel(BaseGridShiftModel):
    """4-corner bilinear shift interpolator over the official 11 km SPG lattice.

    Intended primarily as a fast, zero-overshoot Stage-1 macro baseline for
    coarse-to-fine residual models: removes ~98% of the distortion magnitude so
    Stage-2 learners focus strictly on millimeter-scale intra-cell curvature.
    """

    def __init__(self, spg_path: str | None = None) -> None:
        super().__init__()
        self.spg_path = spg_path
        self._min_n: float = 0.0
        self._min_e: float = 0.0
        self._step_n: float = 11000.0
        self._step_e: float = 11000.0
        self._n_rows: int = 0
        self._n_cols: int = 0
        self._dn_grid: np.ndarray | None = None
        self._de_grid: np.ndarray | None = None

    def _fit_impl(self, coords: np.ndarray, shifts: np.ndarray) -> None:
        """Load the official 11 km shift grid from ``SpgReader``."""
        del coords, shifts
        reader = SpgReader(self.spg_path) if self.spg_path is not None else SpgReader()
        min_e, max_e, min_n, max_n, step_e, step_n = reader.shift_bounds
        self._min_n = float(min_n)
        self._min_e = float(min_e)
        self._step_n = float(step_n)
        self._step_e = float(step_e)
        self._n_cols = round((max_e - min_e) / step_e) + 1
        self._n_rows = round((max_n - min_n) / step_n) + 1

        dn = np.asarray(reader.shift_n_flat, dtype=np.float64).reshape(
            (self._n_rows, self._n_cols)
        )
        de = np.asarray(reader.shift_e_flat, dtype=np.float64).reshape(
            (self._n_rows, self._n_cols)
        )
        valid = np.isfinite(dn) & np.isfinite(de) & (np.abs(dn) < 900.0) & (np.abs(de) < 900.0)
        self._dn_grid = np.where(valid, dn, 0.0)
        self._de_grid = np.where(valid, de, 0.0)

    def _predict_impl(self, coords: np.ndarray) -> np.ndarray:
        """Evaluate vectorized 4-corner bilinear interpolation at ``coords`` ``(M, 2)``."""
        if self._dn_grid is None or self._de_grid is None:
            raise RuntimeError("Model is not fitted.")

        n_arr = coords[:, 0]
        e_arr = coords[:, 1]
        fr = (n_arr - self._min_n) / self._step_n
        fc = (e_arr - self._min_e) / self._step_e

        r0 = np.clip(np.floor(fr).astype(np.int64), 0, self._n_rows - 2)
        c0 = np.clip(np.floor(fc).astype(np.int64), 0, self._n_cols - 2)
        u = fr - r0
        v = fc - c0

        w00 = (1.0 - u) * (1.0 - v)
        w01 = (1.0 - u) * v
        w10 = u * (1.0 - v)
        w11 = u * v

        dn_pred = (
            w00 * self._dn_grid[r0, c0]
            + w01 * self._dn_grid[r0, c0 + 1]
            + w10 * self._dn_grid[r0 + 1, c0]
            + w11 * self._dn_grid[r0 + 1, c0 + 1]
        )
        de_pred = (
            w00 * self._de_grid[r0, c0]
            + w01 * self._de_grid[r0, c0 + 1]
            + w10 * self._de_grid[r0 + 1, c0]
            + w11 * self._de_grid[r0 + 1, c0 + 1]
        )
        return np.column_stack([dn_pred, de_pred])

    def _runtime_dependencies(self) -> tuple[str, ...]:
        return ("numpy",)

    def _grid_lookup_bytes(self) -> int:
        if self._dn_grid is None or self._de_grid is None:
            return 0
        return int(self._dn_grid.nbytes + self._de_grid.nbytes)


class CompositeResidualShiftModel(BaseGridShiftModel):
    """Two-stage coarse-to-fine residual shift regressor.

    Combines a macro ``stage1_model`` (either pre-fitted or fitted on ``coords``)
    with a micro ``stage2_model`` trained strictly on ``shifts - stage1.predict(coords)``.
    """

    def __init__(
        self,
        stage1_model: BaseGridShiftModel,
        stage2_model: BaseGridShiftModel,
    ) -> None:
        super().__init__()
        self.stage1_model = stage1_model
        self.stage2_model = stage2_model

    def _fit_impl(self, coords: np.ndarray, shifts: np.ndarray) -> None:
        if not self.stage1_model.is_fitted:
            self.stage1_model.fit(coords, shifts)
        base_pred = self.stage1_model.predict(coords)
        residuals = shifts - base_pred
        self.stage2_model.fit(coords, residuals)

    def _predict_impl(self, coords: np.ndarray) -> np.ndarray:
        return self.stage1_model.predict(coords) + self.stage2_model.predict(coords)

    def _runtime_dependencies(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                (
                    *self.stage1_model._runtime_dependencies(),
                    *self.stage2_model._runtime_dependencies(),
                )
            )
        )

    def _grid_lookup_bytes(self) -> int:
        return self.stage1_model._grid_lookup_bytes() + self.stage2_model._grid_lookup_bytes()
