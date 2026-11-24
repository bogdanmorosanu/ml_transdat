"""Shared SPG 11 km cell-local (u, v) coordinate and cyclic harmonic utilities."""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np

from pytransdat.spg_reader import SpgReader
from src.data.spg_cache import get_default_spg_reader


@functools.lru_cache(maxsize=8)
def get_spg_grid_bounds(
    spg_path: str | Path | None = None,
) -> tuple[float, float, float, float, float, float]:
    """Return cached ``(min_e, max_e, min_n, max_n, step_e, step_n)`` from ``SpgReader``.

    :param spg_path: Optional path to ``.spg`` binary grid file.
    :type spg_path: str | Path | None
    :return: Tuple ``(min_e, max_e, min_n, max_n, step_e, step_n)`` in meters.
    :rtype: tuple[float, float, float, float, float, float]
    """
    reader = SpgReader(str(spg_path)) if spg_path is not None else get_default_spg_reader()
    min_e, max_e, min_n, max_n, step_e, step_n = reader.shift_bounds
    return (
        float(min_e),
        float(max_e),
        float(min_n),
        float(max_n),
        float(step_e),
        float(step_n),
    )


def compute_spg_cell_uv(
    coords: np.ndarray,
    spg_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute normalized intra-cell ``(u, v)`` coordinates in ``[0, 1)``.

    :param coords: Planar Stereo 70 coordinates ``[st70_n, st70_e]`` of shape ``(N, 2)``.
    :type coords: np.ndarray
    :param spg_path: Optional path to ``.spg`` binary grid file.
    :type spg_path: str | Path | None
    :return: Tuple ``(u, v)`` of 1D arrays of shape ``(N,)`` for Northing and Easting.
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    min_e, _, min_n, _, step_e, step_n = get_spg_grid_bounds(spg_path)
    n_arr = np.asarray(coords[:, 0], dtype=np.float64)
    e_arr = np.asarray(coords[:, 1], dtype=np.float64)
    r_idx = np.floor((n_arr - min_n) / step_n)
    c_idx = np.floor((e_arr - min_e) / step_e)
    u = (n_arr - (min_n + r_idx * step_n)) / step_n
    v = (e_arr - (min_e + c_idx * step_e)) / step_e
    return u, v


def encode_cyclic_cell_uv(
    coords: np.ndarray,
    harmonics: int = 1,
    include_cross_terms: bool = False,
    spg_path: str | Path | None = None,
) -> np.ndarray:
    """Encode intra-cell ``(u, v)`` phase using smooth cyclic Fourier harmonics.

    :param coords: Planar Stereo 70 coordinates ``[st70_n, st70_e]`` of shape ``(N, 2)``.
    :type coords: np.ndarray
    :param harmonics: Number of harmonic frequencies ``k = 1..harmonics``.
    :type harmonics: int
    :param include_cross_terms: Whether to append 4 fundamental ``u * v`` product terms.
    :type include_cross_terms: bool
    :param spg_path: Optional path to ``.spg`` binary grid file.
    :type spg_path: str | Path | None
    :return: Encoded feature matrix of shape ``(N, 4 * harmonics [+ 4])``.
    :rtype: np.ndarray
    """
    u, v = compute_spg_cell_uv(coords, spg_path=spg_path)
    cols: list[np.ndarray] = []
    for k in range(1, harmonics + 1):
        ang_u = 2.0 * np.pi * k * u
        ang_v = 2.0 * np.pi * k * v
        cols.extend([np.sin(ang_u), np.cos(ang_u), np.sin(ang_v), np.cos(ang_v)])
    if include_cross_terms and harmonics > 1:
        su, cu = np.sin(2.0 * np.pi * u), np.cos(2.0 * np.pi * u)
        sv, cv = np.sin(2.0 * np.pi * v), np.cos(2.0 * np.pi * v)
        cols.extend([su * sv, su * cv, cu * sv, cu * cv])
    return np.column_stack(cols)
