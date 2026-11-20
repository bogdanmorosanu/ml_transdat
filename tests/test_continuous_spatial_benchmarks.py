"""Unit tests verifying uniform stratified and adaptive midpoint continuous spatial benchmarks."""

from __future__ import annotations

import unittest

import numpy as np

from src.data import (
    GridDataset,
    GridId,
    generate_adaptive_midpoints_df,
    generate_stratified_continuous_df,
    get_default_grid_manager,
    get_valid_spg_cells,
    iter_adaptive_midpoint_chunks,
    iter_stratified_continuous_chunks,
)


class TestContinuousSpatialBenchmarks(unittest.TestCase):
    """Verify contracts, reproducibility, and invariants for continuous spatial benchmarks."""

    def test_valid_spg_cells_count(self) -> None:
        cells = get_valid_spg_cells()
        self.assertIsInstance(cells, list)
        self.assertEqual(len(cells), 2297)

    def test_stratified_continuous_df_4x4(self) -> None:
        df = generate_stratified_continuous_df(strata_per_axis=4, seed=42)
        expected_count = 2297 * 16  # 36,752 points
        self.assertEqual(len(df), expected_count)
        self.assertFalse(
            df.isna().any().any(), "Continuous benchmark must contain zero NaNs"
        )

        # Verify columns
        required_cols = {
            "point_id",
            "row_idx",
            "col_idx",
            "u",
            "v",
            "st70_n",
            "st70_e",
            "st70_h",
            "shift_dn_m",
            "shift_de_m",
        }
        self.assertTrue(required_cols.issubset(set(df.columns)))

        # Verify continuous off-lattice coordinates: u and v are strictly in (0, 1)
        self.assertTrue((df["u"] >= 0.0).all() and (df["u"] <= 1.0).all())
        self.assertTrue((df["v"] >= 0.0).all() and (df["v"] <= 1.0).all())

        # Verify bit-for-bit reproducibility with same seed
        df_repeat = generate_stratified_continuous_df(strata_per_axis=4, seed=42)
        np.testing.assert_array_equal(
            df["st70_n"].to_numpy(), df_repeat["st70_n"].to_numpy()
        )
        np.testing.assert_array_equal(
            df["st70_e"].to_numpy(), df_repeat["st70_e"].to_numpy()
        )
        np.testing.assert_array_equal(
            df["shift_dn_m"].to_numpy(), df_repeat["shift_dn_m"].to_numpy()
        )

    def test_stratified_continuous_chunks_streaming(self) -> None:
        chunks = list(
            iter_stratified_continuous_chunks(
                strata_per_axis=4, seed=42, batch_size=4096
            )
        )
        self.assertGreater(len(chunks), 1)
        total_streamed = sum(coords.shape[0] for coords, _ in chunks)
        self.assertEqual(total_streamed, 2297 * 16)
        for coords, shifts in chunks:
            self.assertEqual(coords.shape[1], 2)
            self.assertEqual(shifts.shape[1], 2)
            self.assertFalse(np.isnan(coords).any())
            self.assertFalse(np.isnan(shifts).any())

    def test_adaptive_midpoints_3mm(self) -> None:
        df = generate_adaptive_midpoints_df(tier="3mm")
        self.assertEqual(len(df), 176350)
        self.assertFalse(
            df.isna().any().any(), "Adaptive midpoints must contain zero NaNs"
        )

        required_cols = {
            "point_id",
            "st70_n",
            "st70_e",
            "st70_h",
            "shift_dn_m",
            "shift_de_m",
            "depth",
        }
        self.assertTrue(required_cols.issubset(set(df.columns)))

    def test_adaptive_midpoints_chunks_streaming(self) -> None:
        chunks = list(iter_adaptive_midpoint_chunks(tier="3mm", batch_size=32768))
        self.assertGreater(len(chunks), 1)
        total_streamed = sum(coords.shape[0] for coords, _ in chunks)
        self.assertEqual(total_streamed, 176350)

    def test_grid_manager_integration(self) -> None:
        gm = get_default_grid_manager()

        # Check Uniform 4x4 resolution
        ds_u4 = gm.get(GridId.RO_TEST_UNIFORM_4X4)
        self.assertIsInstance(ds_u4, GridDataset)
        self.assertEqual(len(ds_u4), 2297 * 16)
        self.assertEqual(ds_u4.coords.shape, (len(ds_u4), 2))
        self.assertEqual(ds_u4.true_shifts.shape, (len(ds_u4), 2))

        # Check Adaptive 3mm resolution
        ds_a3 = gm.get(GridId.RO_TEST_ADAPTIVE_3MM_MID)
        self.assertIsInstance(ds_a3, GridDataset)
        self.assertEqual(len(ds_a3), 176350)
        self.assertEqual(ds_a3.coords.shape, (len(ds_a3), 2))
        self.assertEqual(ds_a3.true_shifts.shape, (len(ds_a3), 2))


if __name__ == "__main__":
    unittest.main()
