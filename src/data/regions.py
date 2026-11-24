"""SPG cell-aligned regional subgrid extraction and dataset combination helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pytransdat.trans_grid import Grid2D
from src.data.samplers import get_valid_spg_cells
from src.data.spg_cache import get_default_grid2d

if TYPE_CHECKING:
    from src.data.catalog import GridDataset, GridManager


@dataclass(frozen=True)
class CellRegion:
    """SPG cell-aligned rectangular region for spatial subgrid extraction.

    :param row_start: Starting SPG cell row index (0-based).
    :param col_start: Starting SPG cell column index (0-based).
    :param n_rows: Number of SPG cells along Northing.
    :param n_cols: Number of SPG cells along Easting.
    :param name: Optional human-readable identifier for this region.
    """

    row_start: int
    col_start: int
    n_rows: int
    n_cols: int
    name: str = ""

    def __post_init__(self) -> None:
        if self.row_start < 0 or self.col_start < 0:
            raise ValueError(
                f"row_start and col_start must be >= 0, got ({self.row_start}, {self.col_start})"
            )
        if self.n_rows <= 0 or self.n_cols <= 0:
            raise ValueError(
                f"n_rows and n_cols must be > 0, got ({self.n_rows}, {self.n_cols})"
            )

    def bounds_st70(
        self, grid2d: Grid2D | None = None
    ) -> tuple[float, float, float, float]:
        """Return exact SPG-aligned ``(n_min, n_max, e_min, e_max)`` bounds in meters."""
        g = grid2d if grid2d is not None else get_default_grid2d()
        n_min = float(g.n_min + self.row_start * g.n_step)
        n_max = float(g.n_min + (self.row_start + self.n_rows) * g.n_step)
        e_min = float(g.e_min + self.col_start * g.e_step)
        e_max = float(g.e_min + (self.col_start + self.n_cols) * g.e_step)
        return (n_min, n_max, e_min, e_max)

    def padded(
        self, pad_cells: int = 2, grid2d: Grid2D | None = None
    ) -> CellRegion:
        """Return a halo-expanded :class:`CellRegion` clamped to valid SPG grid bounds."""
        if pad_cells < 0:
            raise ValueError(f"pad_cells must be >= 0, got {pad_cells}")
        g = grid2d if grid2d is not None else get_default_grid2d()
        max_rows = g.r_count - 1
        max_cols = g.c_count - 1
        r0 = max(0, self.row_start - pad_cells)
        c0 = max(0, self.col_start - pad_cells)
        r1 = min(max_rows, self.row_start + self.n_rows + pad_cells)
        c1 = min(max_cols, self.col_start + self.n_cols + pad_cells)
        pad_name = f"{self.name}_pad{pad_cells}" if self.name else ""
        return CellRegion(
            row_start=r0,
            col_start=c0,
            n_rows=r1 - r0,
            n_cols=c1 - c0,
            name=pad_name,
        )

    @classmethod
    def from_center(
        cls,
        center_row: int,
        center_col: int,
        n_rows: int,
        n_cols: int,
        *,
        name: str = "",
        grid2d: Grid2D | None = None,
    ) -> CellRegion:
        """Create a :class:`CellRegion` centered at ``(center_row, center_col)``."""
        g = grid2d if grid2d is not None else get_default_grid2d()
        max_rows = g.r_count - 1
        max_cols = g.c_count - 1
        r0 = max(0, min(max_rows - n_rows, center_row - n_rows // 2))
        c0 = max(0, min(max_cols - n_cols, center_col - n_cols // 2))
        return cls(row_start=r0, col_start=c0, n_rows=n_rows, n_cols=n_cols, name=name)


def concat_datasets(datasets: Sequence[GridDataset]) -> GridDataset:
    """Combine multiple :class:`GridDataset` slices, deduplicating by ``(st70_n, st70_e)``.

    :param datasets: Non-empty sequence of :class:`GridDataset` objects.
    :type datasets: Sequence[GridDataset]
    :return: Combined :class:`GridDataset`.
    :rtype: GridDataset
    :raises ValueError: If ``datasets`` is empty.
    """
    from src.data.catalog import GridDataset

    if not datasets:
        raise ValueError("datasets must not be empty.")
    first = datasets[0]
    if len(datasets) == 1:
        return first
    df_cat = pd.concat([ds.df for ds in datasets], ignore_index=True).drop_duplicates(
        subset=["st70_n", "st70_e"]
    )
    return GridDataset(
        grid_id=first.grid_id,
        df=df_cat.reset_index(drop=True),
        metadata=dict(first.metadata),
    )


def find_worst_case_region(
    size_cells: int,
    *,
    min_valid_fraction: float = 0.80,
    pad_cells: int = 2,
    priority_cells: Sequence[tuple[int, int]] | None = None,
    manager: GridManager | None = None,
) -> CellRegion:
    """Find the worst-case ``size_cells x size_cells`` SPG window in Romania.

    Ranks candidate windows (with ``pad_cells`` margin inside the SPG grid) by
    high-curvature ``RO_TEST_ADAPTIVE_1MM_MID`` point density and optional
    ``priority_cells`` coverage, subject to valid-cell fill ratio.
    """
    from src.data.catalog import GridId, get_default_grid_manager

    if size_cells <= 0:
        raise ValueError(f"size_cells must be positive, got {size_cells}")

    gm = manager if manager is not None else get_default_grid_manager()
    grid2d = get_default_grid2d()
    max_rows = grid2d.r_count - 1
    max_cols = grid2d.c_count - 1

    valid_mask = np.zeros((max_rows, max_cols), dtype=np.float64)
    for r, c in get_valid_spg_cells():
        if 0 <= r < max_rows and 0 <= c < max_cols:
            valid_mask[r, c] = 1.0

    adapt_ds = gm.get(GridId.RO_TEST_ADAPTIVE_1MM_MID)
    coords = adapt_ds.coords
    rows = np.floor((coords[:, 0] - grid2d.n_min) / grid2d.n_step).astype(np.int64)
    cols = np.floor((coords[:, 1] - grid2d.e_min) / grid2d.e_step).astype(np.int64)
    in_bounds = (rows >= 0) & (rows < max_rows) & (cols >= 0) & (cols < max_cols)
    density = np.zeros((max_rows, max_cols), dtype=np.float64)
    np.add.at(density, (rows[in_bounds], cols[in_bounds]), 1.0)

    if priority_cells:
        bonus_per_cell = max(float(np.max(density)) * 0.5, 500.0)
        for pr, pc in priority_cells:
            if 0 <= pr < max_rows and 0 <= pc < max_cols:
                density[pr, pc] += bonus_per_cell

    r_lim = max_rows - size_cells - pad_cells
    c_lim = max_cols - size_cells - pad_cells
    r_start_min = min(pad_cells, max(0, r_lim))
    c_start_min = min(pad_cells, max(0, c_lim))

    best_coords: tuple[int, int] | None = None
    best_score = -1.0
    fallback_coords: tuple[int, int] = (r_start_min, c_start_min)
    fallback_valid = -1.0
    fallback_score = -1.0

    n_total = float(size_cells * size_cells)
    for r0 in range(r_start_min, max(r_start_min + 1, r_lim + 1)):
        for c0 in range(c_start_min, max(c_start_min + 1, c_lim + 1)):
            v_frac = float(
                np.sum(valid_mask[r0 : r0 + size_cells, c0 : c0 + size_cells]) / n_total
            )
            w_score = float(
                np.sum(density[r0 : r0 + size_cells, c0 : c0 + size_cells])
            )
            if v_frac >= min_valid_fraction:
                if w_score > best_score:
                    best_score = w_score
                    best_coords = (r0, c0)
            elif v_frac > fallback_valid or (
                abs(v_frac - fallback_valid) < 1e-9 and w_score > fallback_score
            ):
                fallback_valid = v_frac
                fallback_score = w_score
                fallback_coords = (r0, c0)

    chosen_r, chosen_c = best_coords if best_coords is not None else fallback_coords
    return CellRegion(
        row_start=chosen_r,
        col_start=chosen_c,
        n_rows=size_cells,
        n_cols=size_cells,
        name=f"worst_case_{size_cells}x{size_cells}",
    )
