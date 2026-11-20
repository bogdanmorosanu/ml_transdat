"""Data sampling, lazy chunk streaming, and on-the-fly grid generation engines."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from itertools import pairwise
from typing import Any

import numpy as np
import pandas as pd

from pytransdat import TransRO
from pytransdat.exceptions import PyTransDatError
from pytransdat.trans_grid import Grid2D

DEFAULT_SUBGRID_OFFSETS_4X4: tuple[float, ...] = (0.0, 0.25, 0.50, 0.75)
DEFAULT_SUBGRID_OFFSETS_8X8: tuple[float, ...] = tuple(i / 8.0 for i in range(8))
DEFAULT_SUBGRID_OFFSETS_16X16: tuple[float, ...] = tuple(i / 16.0 for i in range(16))


def build_grid_candidates(
    engine: TransRO,
    offsets: Sequence[float],
    z_normal: float = 0.0,
) -> list[dict[str, Any]]:
    """Build candidate points aligned to the SPG grid cell structure with given subgrid offsets.

    :param engine: Initialized ``TransRO`` ground-truth engine.
    :type engine: TransRO
    :param offsets: Normalized cell offsets in ``[0, 1)`` for both Northing and Easting.
    :type offsets: Sequence[float]
    :param z_normal: Default orthometric height in meters.
    :type z_normal: float
    :return: List of candidate point dictionaries.
    :rtype: list[dict[str, Any]]
    """
    grid2d = engine._t_gr2d
    candidates: list[dict[str, Any]] = []

    for row_idx in range(grid2d.r_count - 1):
        for col_idx in range(grid2d.c_count - 1):
            for u in offsets:
                n_val = grid2d.n_min + (row_idx + u) * grid2d.n_step
                for v in offsets:
                    e_val = grid2d.e_min + (col_idx + v) * grid2d.e_step
                    candidates.append(
                        {
                            "row_idx": row_idx,
                            "col_idx": col_idx,
                            "u": u,
                            "v": v,
                            "is_node": (u == 0.0 and v == 0.0),
                            "st70_n": n_val,
                            "st70_e": e_val,
                            "st70_h": z_normal,
                        }
                    )
    return candidates


def build_3x3_grid_candidates(
    engine: TransRO,
    offsets: Sequence[float] = DEFAULT_SUBGRID_OFFSETS_4X4,
    z_normal: float = 0.0,
) -> list[dict[str, Any]]:
    """Backward-compatible alias for 4x4 subgrid candidates."""
    return build_grid_candidates(engine, offsets=offsets, z_normal=z_normal)


def sample_corridor_candidates(
    waypoints: Sequence[tuple[float, float]],
    half_width_m: float,
    n_candidates: int,
    seed: int = 42,
    z_normal: float = 0.0,
) -> list[dict[str, Any]]:
    """Sample ordered random candidate points uniformly across a multi-segment corridor.

    :param waypoints: Ordered ``(N, E)`` corridor vertices in meters.
    :type waypoints: Sequence[tuple[float, float]]
    :param half_width_m: Half-width of the corridor in meters.
    :type half_width_m: float
    :param n_candidates: Number of candidate points to sample.
    :type n_candidates: int
    :param seed: Random generator seed.
    :type seed: int
    :param z_normal: Default orthometric height in meters.
    :type z_normal: float
    :return: List of candidate point dictionaries.
    :rtype: list[dict[str, Any]]
    """
    rng = np.random.default_rng(seed)

    segments: list[tuple[float, float, float, float, float, float, float, float]] = []
    cum_len = 0.0
    for (n0, e0), (n1, e1) in pairwise(waypoints):
        dn = n1 - n0
        de = e1 - e0
        seg_len = math.hypot(dn, de)
        tn, te = dn / seg_len, de / seg_len
        nn, ne = -te, tn
        segments.append((cum_len, seg_len, n0, e0, tn, te, nn, ne))
        cum_len += seg_len

    s_vals = np.sort(rng.uniform(0.0, cum_len, size=n_candidates))
    w_vals = rng.uniform(-half_width_m, half_width_m, size=n_candidates)

    candidates: list[dict[str, Any]] = []
    seg_idx = 0
    for s_val, w_val in zip(s_vals, w_vals, strict=True):
        while (
            seg_idx < len(segments) - 1
            and s_val > segments[seg_idx][0] + segments[seg_idx][1]
        ):
            seg_idx += 1
        s_start, _, n0, e0, tn, te, nn, ne = segments[seg_idx]
        ds = float(s_val - s_start)
        w = float(w_val)
        n_coord = n0 + ds * tn + w * nn
        e_coord = e0 + ds * te + w * ne
        candidates.append(
            {
                "s_along_m": float(s_val),
                "w_cross_m": float(w),
                "st70_n": float(n_coord),
                "st70_e": float(e_coord),
                "st70_h": z_normal,
            }
        )
    return candidates


def generate_spg_nodes_df(
    spg_path: str = "pytransdat/grids/rom_grid3d_25.09.spg",
) -> pd.DataFrame:
    """Extract all valid 2D nodes directly from the official ``.spg`` grid file.

    :param spg_path: Workspace-relative path to the ``.spg`` binary grid file.
    :type spg_path: str
    :return: DataFrame of valid SPG nodes with shifts and ETRS89 coordinates.
    :rtype: pd.DataFrame
    """
    engine = TransRO()
    spg_data = np.load(spg_path, allow_pickle=True)
    grid_arr = spg_data["grids"]["geodetic_shifts"]["grid"]
    # Band 0 is Easting shift (shift_e), Band 1 is Northing shift (shift_n)
    de_arr = grid_arr[0]
    dn_arr = grid_arr[1]
    grid2d = engine._t_gr2d

    records: list[dict[str, Any]] = []
    node_id = 0
    for r in range(grid2d.r_count):
        for c in range(grid2d.c_count):
            dn_val = float(dn_arr[r, c])
            de_val = float(de_arr[r, c])
            if np.isnan(dn_val) or np.isnan(de_val):
                continue

            n_coord = float(grid2d.n_min + r * grid2d.n_step)
            e_coord = float(grid2d.e_min + c * grid2d.e_step)
            mag_2d = math.hypot(dn_val, de_val)

            etrs_lat_deg = np.nan
            etrs_lon_deg = np.nan
            etrs_lat_rad = np.nan
            etrs_lon_rad = np.nan
            etrs_h = np.nan
            is_transdat_valid = False
            try:
                res = engine.st70_to_etrs89(n_coord, e_coord, 0.0)
                etrs_lat_rad = float(res[0])
                etrs_lon_rad = float(res[1])
                etrs_lat_deg = math.degrees(etrs_lat_rad)
                etrs_lon_deg = math.degrees(etrs_lon_rad)
                etrs_h = float(res[2])
                is_transdat_valid = True
            except PyTransDatError:
                pass

            records.append(
                {
                    "node_id": node_id,
                    "row_idx": r,
                    "col_idx": c,
                    "st70_n": n_coord,
                    "st70_e": e_coord,
                    "st70_h": 0.0,
                    "shift_dn_m": dn_val,
                    "shift_de_m": de_val,
                    "shift_mag_2d_m": mag_2d,
                    "is_transdat_valid": is_transdat_valid,
                    "etrs89_lat_deg": etrs_lat_deg,
                    "etrs89_lon_deg": etrs_lon_deg,
                    "etrs89_lat_rad": etrs_lat_rad,
                    "etrs89_lon_rad": etrs_lon_rad,
                    "etrs89_h": etrs_h,
                }
            )
            node_id += 1

    return pd.DataFrame(records)


def generate_densified_grid_df(
    offsets: Sequence[float],
    engine: TransRO | None = None,
    include_shifts: bool = True,
) -> pd.DataFrame:
    """Generate a bidirectionally valid densified subgrid DataFrame on the fly.

    :param offsets: Normalized cell offsets in ``[0, 1)``.
    :type offsets: Sequence[float]
    :param engine: Optional pre-initialized ``TransRO`` instance.
    :type engine: TransRO | None
    :param include_shifts: Whether to populate ``shift_dn_m`` and ``shift_de_m`` immediately.
    :type include_shifts: bool
    :return: DataFrame with ``st70_*``, ``etrs89_*``, ``roundtrip_err_*``, and optional shift columns.
    :rtype: pd.DataFrame
    """
    active_engine = engine if engine is not None else TransRO()
    candidates = build_grid_candidates(active_engine, offsets=offsets)

    valid_records: list[dict[str, Any]] = []
    pid = 0
    for c in candidates:
        n_in = c["st70_n"]
        e_in = c["st70_e"]
        h_in = c["st70_h"]
        try:
            res_etrs = active_engine.st70_to_etrs89(n_in, e_in, h_in)
            lat_rad = float(res_etrs[0])
            lon_rad = float(res_etrs[1])
            h_etrs = float(res_etrs[2])

            res_st70 = active_engine.etrs89_to_st70(lat_rad, lon_rad, h_etrs)
            n_rt = float(res_st70[0])
            e_rt = float(res_st70[1])
            h_rt = float(res_st70[2])

            is_4x4 = (c["u"] in DEFAULT_SUBGRID_OFFSETS_4X4) and (
                c["v"] in DEFAULT_SUBGRID_OFFSETS_4X4
            )
            valid_records.append(
                {
                    "point_id": pid,
                    "row_idx": c["row_idx"],
                    "col_idx": c["col_idx"],
                    "u": c["u"],
                    "v": c["v"],
                    "is_node": c["is_node"],
                    "is_4x4_point": is_4x4,
                    "st70_n": n_in,
                    "st70_e": e_in,
                    "st70_h": h_in,
                    "etrs89_lat_deg": math.degrees(lat_rad),
                    "etrs89_lon_deg": math.degrees(lon_rad),
                    "etrs89_lat_rad": lat_rad,
                    "etrs89_lon_rad": lon_rad,
                    "etrs89_h": h_etrs,
                    "roundtrip_err_n_m": abs(n_rt - n_in),
                    "roundtrip_err_e_m": abs(e_rt - e_in),
                    "roundtrip_err_h_m": abs(h_rt - h_in),
                }
            )
            pid += 1
        except PyTransDatError:
            continue

    df = pd.DataFrame(valid_records)
    if include_shifts and len(df) > 0:
        dn_list, de_list = active_engine._t_gr2d.interp(
            df["st70_n"].tolist(),
            df["st70_e"].tolist(),
        )
        df["shift_dn_m"] = np.asarray(dn_list, dtype=np.float64)
        df["shift_de_m"] = np.asarray(de_list, dtype=np.float64)
    return df


def iter_densified_grid_chunks(
    offsets: Sequence[float],
    batch_size: int = 8192,
    engine: TransRO | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Lazily yield ``(coords_chunk, shifts_chunk)`` batches cell-by-cell without full DataFrame RAM.

    :param offsets: Normalized cell offsets in ``[0, 1)``.
    :type offsets: Sequence[float]
    :param batch_size: Maximum number of points per yielded chunk.
    :type batch_size: int
    :param engine: Optional pre-initialized ``TransRO`` instance.
    :type engine: TransRO | None
    :yield: Pairs of ``(coords, shifts)`` float64 arrays of shape ``(B, 2)``.
    :rtype: Iterator[tuple[np.ndarray, np.ndarray]]
    :raises ValueError: If ``batch_size`` is not positive.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    active_engine = engine if engine is not None else TransRO()
    grid2d = active_engine._t_gr2d
    coords_buf: list[tuple[float, float]] = []
    shifts_buf: list[tuple[float, float]] = []

    for row_idx in range(grid2d.r_count - 1):
        for col_idx in range(grid2d.c_count - 1):
            for u in offsets:
                n_val = float(grid2d.n_min + (row_idx + u) * grid2d.n_step)
                for v in offsets:
                    e_val = float(grid2d.e_min + (col_idx + v) * grid2d.e_step)
                    try:
                        dn_l, de_l = grid2d.interp([n_val], [e_val])
                    except PyTransDatError:
                        continue
                    coords_buf.append((n_val, e_val))
                    shifts_buf.append((float(dn_l[0]), float(de_l[0])))
                    if len(coords_buf) >= batch_size:
                        yield (
                            np.asarray(coords_buf, dtype=np.float64),
                            np.asarray(shifts_buf, dtype=np.float64),
                        )
                        coords_buf.clear()
                        shifts_buf.clear()

    if coords_buf:
        yield (
            np.asarray(coords_buf, dtype=np.float64),
            np.asarray(shifts_buf, dtype=np.float64),
        )


def generate_single_cell_workload_df(
    n_samples: int = 10_000,
    seed: int = 42,
    row_idx: int = 24,
    col_idx: int = 22,
) -> pd.DataFrame:
    """Generate a spatially localized workload inside one interior 11 x 11 km SPG cell.

    :param n_samples: Number of random points to sample inside the cell.
    :type n_samples: int
    :param seed: Random seed for reproducibility.
    :type seed: int
    :param row_idx: Base SPG grid cell row index.
    :type row_idx: int
    :param col_idx: Base SPG grid cell column index.
    :type col_idx: int
    :return: DataFrame containing ``st70_n``, ``st70_e``, ``st70_h``, ``shift_dn_m``, ``shift_de_m``.
    :rtype: pd.DataFrame
    """
    grid2d = Grid2D()
    n0 = grid2d.n_min + row_idx * grid2d.n_step
    e0 = grid2d.e_min + col_idx * grid2d.e_step

    rng = np.random.default_rng(seed)
    n_coords = rng.uniform(n0, n0 + grid2d.n_step, size=n_samples)
    e_coords = rng.uniform(e0, e0 + grid2d.e_step, size=n_samples)

    dn_list, de_list = grid2d.interp(n_coords.tolist(), e_coords.tolist())
    return pd.DataFrame(
        {
            "st70_n": n_coords,
            "st70_e": e_coords,
            "st70_h": np.zeros(n_samples, dtype=np.float64),
            "shift_dn_m": np.asarray(dn_list, dtype=np.float64),
            "shift_de_m": np.asarray(de_list, dtype=np.float64),
        }
    )
