"""Unit tests for spatial boundary loading and zero-duplication GridManager filtering."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import (
    BoundaryId,
    GridDataset,
    GridId,
    GridManager,
    SpatialBoundary,
    get_boundary,
)


class TestSpatialBoundaries(unittest.TestCase):
    """Verify Stereo 70 boundary loading, axis orientation, and GridManager views."""

    def test_romania_boundary_contains_interior_and_excludes_exterior(self) -> None:
        sb = get_boundary(BoundaryId.ROMANIA)
        self.assertIsInstance(sb, SpatialBoundary)
        # Bucharest approx ST70 (N=326000, E=588000); far Black Sea / outside (N=100000, E=950000)
        coords = np.array(
            [
                [326_000.0, 588_000.0],
                [100_000.0, 950_000.0],
                [760_000.0, 120_000.0],
            ],
            dtype=np.float64,
        )
        mask = sb.contains(coords)
        self.assertEqual(mask.tolist(), [True, False, False])

    def test_empty_coords_returns_empty_bool_mask(self) -> None:
        sb = get_boundary(BoundaryId.ROMANIA)
        mask = sb.contains(np.zeros((0, 2), dtype=np.float64))
        self.assertEqual(mask.shape, (0,))
        self.assertEqual(mask.dtype, np.bool_)

    def test_dataset_filter_boundary_preserves_metadata(self) -> None:
        df = pd.DataFrame(
            {
                "st70_n": [326_000.0, 100_000.0],
                "st70_e": [588_000.0, 950_000.0],
                "shift_dn_m": [0.1, 0.2],
                "shift_de_m": [-0.1, -0.2],
            }
        )
        ds = GridDataset(grid_id=GridId.CUSTOM, df=df)
        filtered = ds.filter_boundary(BoundaryId.ROMANIA)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.metadata.get("boundary"), "romania")
        self.assertEqual(filtered.metadata.get("boundary_source_count"), 2)
        self.assertEqual(filtered.metadata.get("boundary_kept_count"), 1)

    def test_grid_manager_virtual_border_alias_matches_explicit_filter(self) -> None:
        gm = GridManager(cache_enabled=True, disk_cache_dir=Path(".cache/grids"))
        full_ds = gm.get(GridId.RO_TEST_UNIFORM_4X4)
        explicit = gm.get(GridId.RO_TEST_UNIFORM_4X4, boundary=BoundaryId.ROMANIA)
        virtual = gm.get(GridId.RO_TEST_UNIFORM_4X4_BORDER)

        self.assertLess(len(explicit), len(full_ds))
        self.assertEqual(len(virtual), len(explicit))
        self.assertEqual(virtual.grid_id, GridId.RO_TEST_UNIFORM_4X4_BORDER)
        self.assertEqual(virtual.metadata.get("boundary"), "romania")
        np.testing.assert_allclose(virtual.coords, explicit.coords)


if __name__ == "__main__":
    unittest.main()
