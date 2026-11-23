"""Reusable parameterized 11 km SPG grid-shift interpolators for benchmarking & ML."""

from __future__ import annotations

from typing import Literal

import numpy as np

from pytransdat.spg_reader import SpgReader
from src.models.base import BaseGridShiftModel

BicubicScheme = Literal[
    "transdatro",
    "catmull_rom",
    "pchip",
    "akima",
    "natural_spline_c2",
    "lagrange_bicubic",
]

_HERMITE_H = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [-3.0, 3.0, -2.0, -1.0],
        [2.0, -2.0, 1.0, 1.0],
    ],
    dtype=np.float64,
)


def _solve_natural_spline_1d_slopes(vals: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Compute C^2 natural cubic spline first derivatives along each valid 1D run.

    Solves the tridiagonal system ``d_{k-1} + 4 d_k + d_{k+1} = 3 (g_{k+1} - g_{k-1})``
    with natural boundary conditions (zero second derivative at run endpoints).

    :param vals: 1D array of nodal values of length ``L``.
    :type vals: np.ndarray
    :param valid: Boolean validity mask of length ``L``.
    :type valid: np.ndarray
    :return: 1D array of nodal first derivatives of length ``L``.
    :rtype: np.ndarray
    """
    n = len(vals)
    slopes = np.zeros(n, dtype=np.float64)
    i = 0
    while i < n:
        if not valid[i]:
            i += 1
            continue
        j = i
        while j < n and valid[j]:
            j += 1
        # Contiguous valid segment [i, j) of length m = j - i
        m = j - i
        if m == 2:
            d = vals[i + 1] - vals[i]
            slopes[i] = d
            slopes[i + 1] = d
        elif m >= 3:
            seg = vals[i:j]
            mat = np.zeros((m, m), dtype=np.float64)
            rhs = np.zeros(m, dtype=np.float64)
            mat[0, 0] = 2.0
            mat[0, 1] = 1.0
            rhs[0] = 3.0 * (seg[1] - seg[0])
            for k in range(1, m - 1):
                mat[k, k - 1] = 1.0
                mat[k, k] = 4.0
                mat[k, k + 1] = 1.0
                rhs[k] = 3.0 * (seg[k + 1] - seg[k - 1])
            mat[m - 1, m - 2] = 1.0
            mat[m - 1, m - 1] = 2.0
            rhs[m - 1] = 3.0 * (seg[m - 1] - seg[m - 2])
            slopes[i:j] = np.linalg.solve(mat, rhs)
        i = j
    return slopes


def _compute_pchip_1d_slopes(vals: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Compute Fritsch-Carlson (1980) monotone PCHIP slopes along a 1D grid line.

    :param vals: 1D array of nodal values of length ``L``.
    :type vals: np.ndarray
    :param valid: Boolean validity mask of length ``L``.
    :type valid: np.ndarray
    :return: 1D array of monotone C^1 nodal slopes of length ``L``.
    :rtype: np.ndarray
    """
    n = len(vals)
    slopes = np.zeros(n, dtype=np.float64)
    for k in range(n):
        if not valid[k]:
            continue
        has_l = k - 1 >= 0 and valid[k - 1]
        has_r = k + 1 < n and valid[k + 1]
        if has_l and has_r:
            dl = vals[k] - vals[k - 1]
            dr = vals[k + 1] - vals[k]
            if dl * dr > 0.0:
                slopes[k] = (2.0 * dl * dr) / (dl + dr)
            else:
                slopes[k] = 0.0
        elif has_r:
            slopes[k] = vals[k + 1] - vals[k]
        elif has_l:
            slopes[k] = vals[k] - vals[k - 1]
    return slopes


def _compute_akima_1d_slopes(vals: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Compute Akima (1970) 5-point outlier-resistant slopes along a 1D grid line.

    :param vals: 1D array of nodal values of length ``L``.
    :type vals: np.ndarray
    :param valid: Boolean validity mask of length ``L``.
    :type valid: np.ndarray
    :return: 1D array of Akima C^1 nodal slopes of length ``L``.
    :rtype: np.ndarray
    """
    n = len(vals)
    slopes = np.zeros(n, dtype=np.float64)
    i = 0
    while i < n:
        if not valid[i]:
            i += 1
            continue
        j = i
        while j < n and valid[j]:
            j += 1
        m = j - i
        if m == 2:
            d = vals[i + 1] - vals[i]
            slopes[i] = d
            slopes[i + 1] = d
        elif m >= 3:
            seg = vals[i:j]
            # Secant slopes m_0 .. m_{m-2}, padded by 2 at each end via linear extrapolation
            sec = np.diff(seg)
            m_pad = np.empty(len(sec) + 4, dtype=np.float64)
            m_pad[2:-2] = sec
            m_pad[1] = 2.0 * sec[0] - sec[1]
            m_pad[0] = 2.0 * m_pad[1] - sec[0]
            m_pad[-2] = 2.0 * sec[-1] - sec[-2]
            m_pad[-1] = 2.0 * m_pad[-2] - sec[-1]
            for k in range(m):
                # Secants around node k are at index k+2 in m_pad: m_{k-2}, m_{k-1}, m_k, m_{k+1}
                mm2, mm1, mp0, mp1 = m_pad[k], m_pad[k + 1], m_pad[k + 2], m_pad[k + 3]
                w1 = abs(mp1 - mp0)
                w2 = abs(mm1 - mm2)
                denom = w1 + w2
                if denom > 1e-14:
                    slopes[i + k] = (w1 * mm1 + w2 * mp0) / denom
                else:
                    slopes[i + k] = 0.5 * (mm1 + mp0)
        i = j
    return slopes


def _compute_catmull_rom_1d_slopes(vals: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Compute symmetric central-difference (Catmull-Rom) slopes along a 1D line."""
    n = len(vals)
    slopes = np.zeros(n, dtype=np.float64)
    for k in range(n):
        if not valid[k]:
            continue
        has_l = k - 1 >= 0 and valid[k - 1]
        has_r = k + 1 < n and valid[k + 1]
        if has_l and has_r:
            slopes[k] = 0.5 * (vals[k + 1] - vals[k - 1])
        elif has_r:
            slopes[k] = vals[k + 1] - vals[k]
        elif has_l:
            slopes[k] = vals[k] - vals[k - 1]
    return slopes


class ParameterizedBicubicGridShiftModel(BaseGridShiftModel):
    """Vectorized 11 km SPG grid-shift interpolator supporting multiple bicubic schemes.

    Precomputes the ``(52, 71, 4, 4)`` polynomial coefficient tensor for every cell
    on the official 11 km SPG lattice using the selected derivative estimation scheme:

    - ``"transdatro"``: Official ANCPI one-sided 3-point stencils (exact match to ``Grid2D``).
    - ``"catmull_rom"``: Symmetric central differences (:math:`C^1` continuous).
    - ``"pchip"``: Fritsch–Carlson monotone harmonic-mean slopes (:math:`C^1`, zero overshoot).
    - ``"akima"``: Akima 5-point outlier-resistant slopes (:math:`C^1`).
    - ``"natural_spline_c2"``: Global tridiagonal natural cubic spline (:math:`C^2` continuous).
    - ``"lagrange_bicubic"``: 16-node cubic Lagrange polynomial (:math:`C^0` continuous).
    """

    def __init__(
        self,
        scheme: BicubicScheme = "catmull_rom",
        spg_path: str | None = None,
    ) -> None:
        super().__init__()
        self.scheme: BicubicScheme = scheme
        self.spg_path = spg_path
        self._min_n: float = 0.0
        self._min_e: float = 0.0
        self._step_n: float = 11000.0
        self._step_e: float = 11000.0
        self._n_rows: int = 0
        self._n_cols: int = 0
        self._dn_coeffs: np.ndarray | None = None
        self._de_coeffs: np.ndarray | None = None

    def _build_cell_coeffs(self, grid: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """Compute ``(n_rows - 1, n_cols - 1, 4, 4)`` bicubic coefficients for one component."""
        n_rows, n_cols = grid.shape
        coeffs = np.zeros((n_rows - 1, n_cols - 1, 4, 4), dtype=np.float64)

        def _get(rr: int, cc: int, fallback: float) -> float:
            if 0 <= rr < n_rows and 0 <= cc < n_cols and valid[rr, cc]:
                return float(grid[rr, cc])
            return fallback

        if self.scheme in ("catmull_rom", "pchip", "akima", "natural_spline_c2"):
            slope_fn = {
                "catmull_rom": _compute_catmull_rom_1d_slopes,
                "pchip": _compute_pchip_1d_slopes,
                "akima": _compute_akima_1d_slopes,
                "natural_spline_c2": _solve_natural_spline_1d_slopes,
            }[self.scheme]

            # Nodal first derivatives along East (cols) and North (rows)
            du = np.zeros_like(grid)
            dv = np.zeros_like(grid)
            for r in range(n_rows):
                dv[r, :] = slope_fn(grid[r, :], valid[r, :])
            for c in range(n_cols):
                du[:, c] = slope_fn(grid[:, c], valid[:, c])

            # Symmetric twist / cross-derivative duv at each valid node
            duv = np.zeros_like(grid)
            for c in range(n_cols):
                duv[:, c] = _compute_catmull_rom_1d_slopes(dv[:, c], valid[:, c])

            for r in range(n_rows - 1):
                for c in range(n_cols - 1):
                    if not (
                        valid[r, c]
                        and valid[r, c + 1]
                        and valid[r + 1, c]
                        and valid[r + 1, c + 1]
                    ):
                        continue
                    f_mat = np.array(
                        [
                            [grid[r, c], grid[r, c + 1], dv[r, c], dv[r, c + 1]],
                            [grid[r + 1, c], grid[r + 1, c + 1], dv[r + 1, c], dv[r + 1, c + 1]],
                            [du[r, c], du[r, c + 1], duv[r, c], duv[r, c + 1]],
                            [du[r + 1, c], du[r + 1, c + 1], duv[r + 1, c], duv[r + 1, c + 1]],
                        ],
                        dtype=np.float64,
                    )
                    coeffs[r, c] = _HERMITE_H @ f_mat @ _HERMITE_H.T
            return coeffs

        # Per-cell schemes ("transdatro" and "lagrange_bicubic")
        for r in range(n_rows - 1):
            for c in range(n_cols - 1):
                if not (
                    valid[r, c]
                    and valid[r, c + 1]
                    and valid[r + 1, c]
                    and valid[r + 1, c + 1]
                ):
                    continue

                # Extract 4x4 subgrid with linear extrapolation fallback at outer domain margins
                g = np.zeros((4, 4), dtype=np.float64)
                for dr in range(4):
                    for dc in range(4):
                        rr, cc = r - 1 + dr, c - 1 + dc
                        if 0 <= rr < n_rows and 0 <= cc < n_cols and valid[rr, cc]:
                            g[dr, dc] = float(grid[rr, cc])
                        else:
                            # Clamp to nearest valid inner cell corner + linear extrapolation
                            r_in = min(max(rr, r), r + 1)
                            c_in = min(max(cc, c), c + 1)
                            base = float(grid[r_in, c_in])
                            slope_r = float(grid[r + 1, c_in] - grid[r, c_in])
                            slope_c = float(grid[r_in, c + 1] - grid[r_in, c])
                            g[dr, dc] = base + (rr - r_in) * slope_r + (cc - c_in) * slope_c

                if self.scheme == "transdatro":
                    dv00 = (-g[1, 3] + 4.0 * g[1, 2] - 3.0 * g[1, 1]) / 2.0
                    dv01 = (3.0 * g[1, 2] - 4.0 * g[1, 1] + g[1, 0]) / 2.0
                    dv10 = (-g[2, 3] + 4.0 * g[2, 2] - 3.0 * g[2, 1]) / 2.0
                    dv11 = (3.0 * g[2, 2] - 4.0 * g[2, 1] + g[2, 0]) / 2.0

                    du00 = (-g[3, 1] + 4.0 * g[2, 1] - 3.0 * g[1, 1]) / 2.0
                    du01 = (-g[3, 2] + 4.0 * g[2, 2] - 3.0 * g[1, 2]) / 2.0
                    du10 = (3.0 * g[2, 1] - 4.0 * g[1, 1] + g[0, 1]) / 2.0
                    du11 = (3.0 * g[2, 2] - 4.0 * g[1, 2] + g[0, 2]) / 2.0
                elif self.scheme == "lagrange_bicubic":
                    dv00 = (-2.0 * g[1, 0] - 3.0 * g[1, 1] + 6.0 * g[1, 2] - g[1, 3]) / 6.0
                    dv01 = (g[1, 0] - 6.0 * g[1, 1] + 3.0 * g[1, 2] + 2.0 * g[1, 3]) / 6.0
                    dv10 = (-2.0 * g[2, 0] - 3.0 * g[2, 1] + 6.0 * g[2, 2] - g[2, 3]) / 6.0
                    dv11 = (g[2, 0] - 6.0 * g[2, 1] + 3.0 * g[2, 2] + 2.0 * g[2, 3]) / 6.0

                    du00 = (-2.0 * g[0, 1] - 3.0 * g[1, 1] + 6.0 * g[2, 1] - g[3, 1]) / 6.0
                    du01 = (-2.0 * g[0, 2] - 3.0 * g[1, 2] + 6.0 * g[2, 2] - g[3, 2]) / 6.0
                    du10 = (g[0, 1] - 6.0 * g[1, 1] + 3.0 * g[2, 1] + 2.0 * g[3, 1]) / 6.0
                    du11 = (g[0, 2] - 6.0 * g[1, 2] + 3.0 * g[2, 2] + 2.0 * g[3, 2]) / 6.0
                else:
                    raise ValueError(f"Unsupported bicubic scheme: {self.scheme}")

                duv00 = ((g[0, 0] + g[2, 2]) - (g[0, 2] + g[2, 0])) / 4.0
                duv01 = ((g[0, 1] + g[2, 3]) - (g[0, 3] + g[2, 1])) / 4.0
                duv10 = ((g[1, 0] + g[3, 2]) - (g[1, 2] + g[3, 0])) / 4.0
                duv11 = ((g[1, 1] + g[3, 3]) - (g[1, 3] + g[3, 1])) / 4.0

                f_mat = np.array(
                    [
                        [g[1, 1], g[1, 2], dv00, dv01],
                        [g[2, 1], g[2, 2], dv10, dv11],
                        [du00, du01, duv00, duv01],
                        [du10, du11, duv10, duv11],
                    ],
                    dtype=np.float64,
                )
                coeffs[r, c] = _HERMITE_H @ f_mat @ _HERMITE_H.T

        return coeffs

    def _fit_impl(self, coords: np.ndarray, shifts: np.ndarray) -> None:
        """Load the official 11 km SPG grid and precompute per-cell bicubic tensors."""
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

        self._dn_coeffs = self._build_cell_coeffs(dn, valid)
        self._de_coeffs = self._build_cell_coeffs(de, valid)

    def _predict_impl(self, coords: np.ndarray) -> np.ndarray:
        """Evaluate vectorized bicubic interpolation at ``coords`` ``(M, 2)``."""
        if self._dn_coeffs is None or self._de_coeffs is None:
            raise RuntimeError("Model is not fitted.")

        n_arr = coords[:, 0]
        e_arr = coords[:, 1]
        fr = (n_arr - self._min_n) / self._step_n
        fc = (e_arr - self._min_e) / self._step_e

        r0 = np.clip(np.floor(fr).astype(np.int64), 0, self._n_rows - 2)
        c0 = np.clip(np.floor(fc).astype(np.int64), 0, self._n_cols - 2)
        u = fr - r0
        v = fc - c0

        u_pow = np.stack([np.ones_like(u), u, u**2, u**3], axis=-1)  # (M, 4)
        v_pow = np.stack([np.ones_like(v), v, v**2, v**3], axis=-1)  # (M, 4)

        dn_c = self._dn_coeffs[r0, c0]  # (M, 4, 4)
        de_c = self._de_coeffs[r0, c0]  # (M, 4, 4)

        dn_pred = np.einsum("mi,mij,mj->m", u_pow, dn_c, v_pow)
        de_pred = np.einsum("mi,mij,mj->m", u_pow, de_c, v_pow)
        return np.column_stack([dn_pred, de_pred])

    def _runtime_dependencies(self) -> tuple[str, ...]:
        return ("numpy",)

    def _grid_lookup_bytes(self) -> int:
        if self._dn_coeffs is None or self._de_coeffs is None:
            return 0
        return int(self._dn_coeffs.nbytes + self._de_coeffs.nbytes)
