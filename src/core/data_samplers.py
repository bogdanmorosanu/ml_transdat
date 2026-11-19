"""Data sampling engines for nationwide subgrids and directional corridors."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import pandas as pd

from pytransdat import TransRO

DEFAULT_SUBGRID_OFFSETS_4X4: tuple[float, ...] = (0.0, 0.25, 0.50, 0.75)
DEFAULT_SUBGRID_OFFSETS_8X8: tuple[float, ...] = (
    0.0,
    0.125,
    0.25,
    0.375,
    0.5,
    0.625,
    0.75,
    0.875,
)


def build_grid_candidates(
    engine: TransRO,
    offsets: Sequence[float],
    z_normal: float = 0.0,
) -> list[dict[str, Any]]:
    """Build candidate points aligned to the SPG grid cell structure with given subgrid offsets."""
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
    """Sample ordered random candidate points uniformly across a multi-segment corridor."""
    rng = np.random.default_rng(seed)

    segments: list[tuple[float, float, float, float, float, float, float, float]] = []
    cum_len = 0.0
    for (n0, e0), (n1, e1) in zip(waypoints[:-1], waypoints[1:], strict=True):
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
        while seg_idx < len(segments) - 1 and s_val > segments[seg_idx][0] + segments[seg_idx][1]:
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
