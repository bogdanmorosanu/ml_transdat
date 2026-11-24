"""Unit tests for SPG cell-aligned regional subgrid extraction in src.data."""

from __future__ import annotations

import numpy as np

from pytransdat.trans_grid import Grid2D
from src.data import (
    CellRegion,
    GridId,
    GridManager,
    find_worst_case_region,
    get_valid_spg_cells,
)


def test_cell_region_bounds_and_padding() -> None:
    """Verify exact SPG step alignment and boundary clamping of CellRegion."""
    g = Grid2D()
    region = CellRegion(row_start=10, col_start=15, n_rows=10, n_cols=10, name="test_10")
    n_min, n_max, e_min, e_max = region.bounds_st70(g)

    assert abs((n_max - n_min) - 10 * g.n_step) < 1e-9
    assert abs((e_max - e_min) - 10 * g.e_step) < 1e-9
    assert abs((n_min - g.n_min) - 10 * g.n_step) < 1e-9
    assert abs((e_min - g.e_min) - 15 * g.e_step) < 1e-9

    padded = region.padded(pad_cells=2, grid2d=g)
    assert padded.row_start == 8
    assert padded.col_start == 13
    assert padded.n_rows == 14
    assert padded.n_cols == 14

    edge_region = CellRegion(row_start=0, col_start=0, n_rows=5, n_cols=5)
    edge_padded = edge_region.padded(pad_cells=2, grid2d=g)
    assert edge_padded.row_start == 0
    assert edge_padded.col_start == 0
    assert edge_padded.n_rows == 7
    assert edge_padded.n_cols == 7


def test_slice_region_and_manager_cache() -> None:
    """Verify GridDataset.slice_region preserves exact coordinates and GridManager caches it."""
    gm = GridManager()
    region = CellRegion.from_center(25, 35, 10, 10, name="center_10")

    full_ds = gm.get(GridId.RO_GRID_4X4)
    sub_ds_1 = gm.get_region(GridId.RO_GRID_4X4, region)
    sub_ds_2 = gm.get_region(GridId.RO_GRID_4X4, region)

    assert sub_ds_1 is sub_ds_2
    assert 0 < len(sub_ds_1) < len(full_ds)

    n_min, n_max, e_min, e_max = region.bounds_st70()
    coords = sub_ds_1.coords
    assert np.all(coords[:, 0] >= n_min - 1e-6)
    assert np.all(coords[:, 0] <= n_max + 1e-6)
    assert np.all(coords[:, 1] >= e_min - 1e-6)
    assert np.all(coords[:, 1] <= e_max + 1e-6)
    assert sub_ds_1.metadata["region"]["name"] == "center_10"

    gm.clear_cache()
    sub_ds_3 = gm.get_region(GridId.RO_GRID_4X4, region)
    assert sub_ds_3 is not sub_ds_1


def test_find_worst_case_region() -> None:
    """Verify find_worst_case_region locates a high-density valid 10x10 region."""
    gm = GridManager()
    region = find_worst_case_region(10, min_valid_fraction=0.80, pad_cells=2, manager=gm)
    assert region.n_rows == 10
    assert region.n_cols == 10
    assert region.row_start >= 2
    assert region.col_start >= 2

    valid_set = set(get_valid_spg_cells())
    valid_count = sum(
        1
        for r in range(region.row_start, region.row_start + region.n_rows)
        for c in range(region.col_start, region.col_start + region.n_cols)
        if (r, c) in valid_set
    )
    assert valid_count / 100.0 >= 0.80
