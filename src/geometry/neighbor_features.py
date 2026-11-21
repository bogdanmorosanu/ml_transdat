"""3x3 Grid Cell Neighbor Feature Extraction.

Computes continuous one-sided slopes and diagonal twists at each of the
4 corners of an 11x11 km SPG cell using its 3x3 cell neighborhood (16 nodes),
plus an O(1) spatial lookup extractor for arbitrary (N, E) coordinates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pytransdat.spg_reader import SpgReader

CORNER_OFFSETS: dict[str, tuple[int, int]] = {
    "bl": (0, 0),
    "br": (0, 1),
    "tl": (1, 0),
    "tr": (1, 1),
}

NEIGHBOR_FEATURE_COLS: list[str] = [
    f"{ch}_{corner}_{feat}"
    for ch in ("dn", "de")
    for corner in ("bl", "br", "tl", "tr")
    for feat in ("left_slope", "right_slope", "bottom_slope", "top_slope", "twist")
]


def _compute_corner_features(
    arr: np.ndarray, cr: int, cc: int
) -> tuple[float, float, float, float, float]:
    """Compute one-sided slopes and diagonal twist at grid node (cr, cc)."""
    center = float(arr[cr, cc])
    left_slope = center - float(arr[cr, cc - 1])
    right_slope = float(arr[cr, cc + 1]) - center
    bottom_slope = center - float(arr[cr - 1, cc])
    top_slope = float(arr[cr + 1, cc]) - center
    twist = (
        float(arr[cr + 1, cc + 1])
        - float(arr[cr + 1, cc - 1])
        - float(arr[cr - 1, cc + 1])
        + float(arr[cr - 1, cc - 1])
    ) / 4.0
    return left_slope, right_slope, bottom_slope, top_slope, twist


def extract_cell_neighbor_features(
    spg_path: str | Path | None = None,
) -> pd.DataFrame:
    """Extract 40 continuous 3x3 neighbor features for all valid interior SPG cells.

    A cell (r, c) is included if all 16 nodes in its 3x3 cell neighborhood
    (rows r-1..r+2, cols c-1..c+2) are finite and valid (|shift| < 900 m).
    """
    reader = SpgReader(spg_path) if spg_path is not None else SpgReader()
    grid_min_e, grid_max_e, grid_min_n, grid_max_n, step_e, step_n = reader.shift_bounds
    n_cols = int(round((grid_max_e - grid_min_e) / step_e)) + 1
    n_rows = int(round((grid_max_n - grid_min_n) / step_n)) + 1

    dn = np.asarray(reader.shift_n_flat, dtype=np.float64).reshape((n_rows, n_cols))
    de = np.asarray(reader.shift_e_flat, dtype=np.float64).reshape((n_rows, n_cols))
    valid_mask = np.isfinite(dn) & np.isfinite(de) & (np.abs(dn) < 900.0) & (np.abs(de) < 900.0)

    rows_data: list[dict[str, Any]] = []
    for r in range(1, n_rows - 2):
        for c in range(1, n_cols - 2):
            if not np.all(valid_mask[r - 1 : r + 3, c - 1 : c + 3]):
                continue

            min_n = float(grid_min_n + r * step_n)
            max_n = float(min_n + step_n)
            min_e = float(grid_min_e + c * step_e)
            max_e = float(min_e + step_e)

            record: dict[str, Any] = {
                "row": r,
                "col": c,
                "min_n": min_n,
                "max_n": max_n,
                "min_e": min_e,
                "max_e": max_e,
                "center_n": 0.5 * (min_n + max_n),
                "center_e": 0.5 * (min_e + max_e),
            }

            for ch_name, arr in (("dn", dn), ("de", de)):
                for corner_name, (dr, dc) in CORNER_OFFSETS.items():
                    ls, rs, bs, ts, tw = _compute_corner_features(arr, r + dr, c + dc)
                    prefix = f"{ch_name}_{corner_name}"
                    record[f"{prefix}_left_slope"] = ls
                    record[f"{prefix}_right_slope"] = rs
                    record[f"{prefix}_bottom_slope"] = bs
                    record[f"{prefix}_top_slope"] = ts
                    record[f"{prefix}_twist"] = tw

            rows_data.append(record)

    return pd.DataFrame(rows_data)


class CellNeighborFeatureExtractor:
    """High-speed O(1) spatial lookup for 3x3 cell neighbor features."""

    def __init__(self, parquet_path: str | Path) -> None:
        df = pd.read_parquet(parquet_path)
        self.feature_cols = list(NEIGHBOR_FEATURE_COLS)

        self.min_n = float(df["min_n"].min())
        self.min_e = float(df["min_e"].min())
        self.step_n = float((df["max_n"] - df["min_n"]).iloc[0])
        self.step_e = float((df["max_e"] - df["min_e"]).iloc[0])

        self.row_offset = int(df["row"].min())
        self.col_offset = int(df["col"].min())
        n_rows = int(df["row"].max()) - self.row_offset + 1
        n_cols = int(df["col"].max()) - self.col_offset + 1

        self._grid = np.full(
            (n_rows, n_cols, len(self.feature_cols)), np.nan, dtype=np.float64
        )
        r_idx = df["row"].to_numpy(dtype=np.int64) - self.row_offset
        c_idx = df["col"].to_numpy(dtype=np.int64) - self.col_offset
        self._grid[r_idx, c_idx] = df[self.feature_cols].to_numpy(dtype=np.float64)

        self._origin_n = self.min_n - self.row_offset * self.step_n
        self._origin_e = self.min_e - self.col_offset * self.step_e

    def lookup_points(self, n_coords: np.ndarray, e_coords: np.ndarray) -> np.ndarray:
        """Return (N, 40) neighbor features for arrays of (N, E) coordinates."""
        n_arr = np.asarray(n_coords, dtype=np.float64)
        e_arr = np.asarray(e_coords, dtype=np.float64)

        r_idx = np.floor((n_arr - self._origin_n) / self.step_n).astype(np.int64) - self.row_offset
        c_idx = np.floor((e_arr - self._origin_e) / self.step_e).astype(np.int64) - self.col_offset

        out = np.full((len(n_arr), len(self.feature_cols)), np.nan, dtype=np.float64)
        in_bounds = (
            (r_idx >= 0)
            & (r_idx < self._grid.shape[0])
            & (c_idx >= 0)
            & (c_idx < self._grid.shape[1])
        )
        out[in_bounds] = self._grid[r_idx[in_bounds], c_idx[in_bounds]]
        return out

    def lookup_center_distances(
        self, n_coords: np.ndarray, e_coords: np.ndarray
    ) -> np.ndarray:
        """Return (N, 2) normalized [0, 1] distances from each 11 km cell midpoint.

        Equals 0.0 at the cell center and 1.0 at the cell edges, forming a
        continuous C0 triangle wave across cell boundaries.
        """
        n_arr = np.asarray(n_coords, dtype=np.float64)
        e_arr = np.asarray(e_coords, dtype=np.float64)

        r_idx = np.floor((n_arr - self._origin_n) / self.step_n)
        c_idx = np.floor((e_arr - self._origin_e) / self.step_e)

        u = (n_arr - (self._origin_n + r_idx * self.step_n)) / self.step_n
        v = (e_arr - (self._origin_e + c_idx * self.step_e)) / self.step_e

        dist_n = 2.0 * np.abs(u - 0.5)
        dist_e = 2.0 * np.abs(v - 0.5)
        return np.column_stack([dist_n, dist_e])
