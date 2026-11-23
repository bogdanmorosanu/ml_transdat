"""Closed-form 4-parameter 2D conformal Helmert fitting and bidirectional transformation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from pytransdat.proj_stereo import StereoProj
from pytransdat.spg_reader import SpgReader
from src.models.base import BaseCoordinateTransformer, BaseGridShiftModel

ARCSEC_PER_RAD: float = 180.0 * 3600.0 / math.pi
CENTER_N_M: float = 500_000.0
CENTER_E_M: float = 500_000.0


def load_embedded_spg_helmert_params(spg_path: str | None = None) -> Helmert2DParams:
    """Load the official embedded 4-parameter 2D Helmert constants from an ``.spg`` file.

    :param spg_path: Optional path to ``.spg`` binary grid file.
    :type spg_path: str | None
    :return: Embedded :class:`Helmert2DParams` in ``st70 -> os`` convention.
    :rtype: Helmert2DParams
    """
    reader = SpgReader(spg_path) if spg_path is not None else SpgReader()
    return Helmert2DParams(
        tn_m=float(reader.helmert_tn),
        te_m=float(reader.helmert_te),
        ppm=float(reader.helmert_ppm),
        rot_rad=float(reader.helmert_rot_rad),
        rot_arcsec=float(reader.helmert_rot_rad * ARCSEC_PER_RAD),
    )


@dataclass(frozen=True)
class Helmert2DParams:
    """4-parameter 2D conformal Helmert parameters in ``st70 -> os`` convention.

    Matches ``pytransdat.trans_helmert2d.Helmert2D`` forward (``sign = +1``) equation:
        N_os = (1 + dm) * ( cos(rot) * N_st70 + sin(rot) * E_st70) + tn_m
        E_os = (1 + dm) * (-sin(rot) * N_st70 + cos(rot) * E_st70) + te_m
    """

    tn_m: float
    te_m: float
    ppm: float
    rot_rad: float
    rot_arcsec: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize parameters to a JSON-friendly dictionary."""
        return asdict(self)


def fit_helmert2d(
    st70_n: Sequence[float] | np.ndarray,
    st70_e: Sequence[float] | np.ndarray,
    os_n: Sequence[float] | np.ndarray,
    os_e: Sequence[float] | np.ndarray,
) -> Helmert2DParams:
    """Solve for optimal 4-parameter 2D conformal Helmert ``st70 -> os`` via OLS.

    Coordinates are centered at ``(500000, 500000)`` during least-squares assembly
    for numerical conditioning, then converted back to the standard ``(0, 0)`` origin.

    :param st70_n: Target Stereo 70 Northing coordinates (meters), shape ``(N,)``.
    :param st70_e: Target Stereo 70 Easting coordinates (meters), shape ``(N,)``.
    :param os_n: StereoGRS80 Northing coordinates from ``StereoProj.to_grid`` (meters), shape ``(N,)``.
    :param os_e: StereoGRS80 Easting coordinates from ``StereoProj.to_grid`` (meters), shape ``(N,)``.
    :return: Fitted ``Helmert2DParams`` in ``st70 -> os`` convention.
    """
    sn = np.asarray(st70_n, dtype=np.float64) - CENTER_N_M
    se = np.asarray(st70_e, dtype=np.float64) - CENTER_E_M
    dn = np.asarray(os_n, dtype=np.float64) - CENTER_N_M
    de = np.asarray(os_e, dtype=np.float64) - CENTER_E_M

    if sn.shape != se.shape or sn.shape != dn.shape or sn.shape != de.shape:
        raise ValueError("All input coordinate arrays must have identical 1D shapes.")
    if sn.ndim != 1 or len(sn) < 2:
        raise ValueError(
            "At least 2 points are required to fit a 4-parameter 2D Helmert."
        )

    n_pts = len(sn)
    zeros = np.zeros(n_pts, dtype=np.float64)
    ones = np.ones(n_pts, dtype=np.float64)

    # Linear system for unknowns [a, b, tn_centered, te_centered]:
    #   dn =  a * sn + b * se + tn_c
    #   de = -b * sn + a * se + te_c
    design = np.empty((2 * n_pts, 4), dtype=np.float64)
    design[:n_pts, 0] = sn
    design[:n_pts, 1] = se
    design[:n_pts, 2] = ones
    design[:n_pts, 3] = zeros

    design[n_pts:, 0] = se
    design[n_pts:, 1] = -sn
    design[n_pts:, 2] = zeros
    design[n_pts:, 3] = ones

    rhs = np.concatenate([dn, de])
    coeffs, _, _, _ = np.linalg.lstsq(design, rhs, rcond=None)
    a, b, tn_c, te_c = (float(x) for x in coeffs)

    scale = math.hypot(a, b)
    ppm = (scale - 1.0) * 1e6
    rot_rad = math.atan2(b, a)
    rot_arcsec = rot_rad * ARCSEC_PER_RAD

    # Convert centered translations back to (0, 0) origin:
    #   os_n = CENTER_N + a*(st70_n - CENTER_N) + b*(st70_e - CENTER_E) + tn_c
    tn_m = CENTER_N_M + tn_c - a * CENTER_N_M - b * CENTER_E_M
    te_m = CENTER_E_M + te_c + b * CENTER_N_M - a * CENTER_E_M

    return Helmert2DParams(
        tn_m=tn_m,
        te_m=te_m,
        ppm=ppm,
        rot_rad=rot_rad,
        rot_arcsec=rot_arcsec,
    )


class FittedHelmert2D:
    """Exact bidirectional 2D conformal Helmert transformer (analytical inverse)."""

    def __init__(self, params: Helmert2DParams) -> None:
        self.params = params
        scale = 1.0 + params.ppm * 1e-6
        self._a = scale * math.cos(params.rot_rad)
        self._b = scale * math.sin(params.rot_rad)
        self._det = self._a * self._a + self._b * self._b
        self._tn = params.tn_m
        self._te = params.te_m

    def st70_to_os(self, n: np.ndarray, e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Transform Stereo 70 ``(N, E)`` to StereoGRS80 ``(N_os, E_os)``."""
        n_arr = np.asarray(n, dtype=np.float64)
        e_arr = np.asarray(e, dtype=np.float64)
        n_os = self._a * n_arr + self._b * e_arr + self._tn
        e_os = -self._b * n_arr + self._a * e_arr + self._te
        return n_os, e_os

    def os_to_st70(
        self, n_os: np.ndarray, e_os: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Exact analytical inverse from StereoGRS80 ``(N_os, E_os)`` to Stereo 70 ``(N, E)``."""
        dn = np.asarray(n_os, dtype=np.float64) - self._tn
        de = np.asarray(e_os, dtype=np.float64) - self._te
        inv_det = 1.0 / self._det
        n_st70 = inv_det * (self._a * dn - self._b * de)
        e_st70 = inv_det * (self._b * dn + self._a * de)
        return n_st70, e_st70


class TwoStepHelmert2DTransformer(BaseCoordinateTransformer):
    """Grid-free 2-step 2D coordinate transformer: ``StereoProj <-> FittedHelmert2D``."""

    def __init__(self, params: Helmert2DParams) -> None:
        self.proj = StereoProj()
        self.helmert = FittedHelmert2D(params)

    def etrs89_to_st70(
        self,
        lat_rad: Sequence[float] | np.ndarray,
        lon_rad: Sequence[float] | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transform ETRS89 ``(lat_rad, lon_rad)`` to Stereo 70 ``(N, E)`` in meters."""
        n_os, e_os = self.proj.to_grid(
            np.asarray(lat_rad, dtype=np.float64).tolist(),
            np.asarray(lon_rad, dtype=np.float64).tolist(),
        )
        return self.helmert.os_to_st70(np.asarray(n_os), np.asarray(e_os))

    def st70_to_etrs89(
        self, n: Sequence[float] | np.ndarray, e: Sequence[float] | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transform Stereo 70 ``(N, E)`` in meters to ETRS89 ``(lat_rad, lon_rad)``."""
        n_os, e_os = self.helmert.st70_to_os(np.asarray(n), np.asarray(e))
        lat_rad, lon_rad = self.proj.to_geo(n_os.tolist(), e_os.tolist())
        return np.asarray(lat_rad, dtype=np.float64), np.asarray(
            lon_rad, dtype=np.float64
        )


class Helmert2DConformalShiftModel(BaseGridShiftModel):
    """4-parameter 2D conformal Helmert regressor for spatial distortion shifts.

    Fits translation (tn, te), uniform scale factor (a), and rotation (b)
    under the strictly angle-preserving conformal Cauchy-Riemann constraint:
        dn = tn + a * (N - N0) + b * (E - E0)
        de = te - b * (N - N0) + a * (E - E0)
    """

    def __init__(
        self, center_n: float = CENTER_N_M, center_e: float = CENTER_E_M
    ) -> None:
        super().__init__()
        self.center_n = center_n
        self.center_e = center_e
        self.tn_: float = 0.0
        self.te_: float = 0.0
        self.a_: float = 0.0
        self.b_: float = 0.0
        self.ppm_: float = 0.0
        self.rot_rad_: float = 0.0
        self.rot_arcsec_: float = 0.0

    def _fit_impl(self, coords: np.ndarray, shifts: np.ndarray) -> None:
        sn = coords[:, 0] - self.center_n
        se = coords[:, 1] - self.center_e
        n_pts = len(sn)

        design = np.empty((2 * n_pts, 4), dtype=np.float64)
        design[:n_pts, 0] = 1.0
        design[:n_pts, 1] = 0.0
        design[:n_pts, 2] = sn
        design[:n_pts, 3] = se
        design[n_pts:, 0] = 0.0
        design[n_pts:, 1] = 1.0
        design[n_pts:, 2] = se
        design[n_pts:, 3] = -sn
        rhs = np.concatenate([shifts[:, 0], shifts[:, 1]])

        coeffs, _, _, _ = np.linalg.lstsq(design, rhs, rcond=None)
        self.tn_ = float(coeffs[0])
        self.te_ = float(coeffs[1])
        self.a_ = float(coeffs[2])
        self.b_ = float(coeffs[3])

        scale = math.hypot(self.a_, self.b_)
        self.ppm_ = scale * 1e6
        self.rot_rad_ = math.atan2(self.b_, self.a_)
        self.rot_arcsec_ = self.rot_rad_ * ARCSEC_PER_RAD

    def _predict_impl(self, coords: np.ndarray) -> np.ndarray:
        sn = coords[:, 0] - self.center_n
        se = coords[:, 1] - self.center_e
        pred_dn = self.tn_ + self.a_ * sn + self.b_ * se
        pred_de = self.te_ - self.b_ * sn + self.a_ * se
        return np.column_stack([pred_dn, pred_de])

    def _runtime_dependencies(self) -> tuple[str, ...]:
        return ("numpy",)

    @property
    def n_parameters(self) -> int:
        """Return total number of fitted conformal parameters (strictly 4)."""
        return 4

