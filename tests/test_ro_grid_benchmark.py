"""Unit tests verifying invariants of dynamic grids (via GridManager) and static benchmarks."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.data import GridId, get_default_grid_manager

CORRIDOR_PARQUET = Path("data/benchmarks/ro_corridor_snw.parquet")
CORRIDOR_META = Path("data/benchmarks/ro_corridor_snw.meta.json")


class TestRoGridBenchmark(unittest.TestCase):
    """Verify integrity, schema, and spatial invariants of dynamic and static grids."""

    def test_static_corridor_files_exist(self) -> None:
        for path in (CORRIDOR_PARQUET, CORRIDOR_META):
            self.assertTrue(path.is_file(), f"Missing {path}")

    def test_spg_nodes_invariants(self) -> None:
        ds = get_default_grid_manager().get(GridId.SPG_NODES_VALID)
        df = ds.df
        self.assertEqual(len(df), 2689)
        self.assertEqual(int(df["is_transdat_valid"].sum()), 2259)

    def test_corridor_schema_and_invariants(self) -> None:
        meta = json.loads(CORRIDOR_META.read_text(encoding="utf-8"))
        df = get_default_grid_manager().get(GridId.CORRIDOR_SNW).df

        self.assertEqual(len(df), meta["final_point_count"])
        self.assertEqual(len(df), 36435)
        self.assertFalse(
            df.isna().any().any(), "Corridor benchmark must contain zero NaNs"
        )
        self.assertTrue(df["corridor_s_m"].is_monotonic_increasing)
        self.assertLessEqual(df["corridor_w_m"].abs().max(), 11_000.0)
        self.assertLess(df["roundtrip_err_n_m"].max(), 1e-3)
        self.assertLess(df["roundtrip_err_e_m"].max(), 1e-3)
        self.assertEqual(df["roundtrip_err_h_m"].max(), 0.0)

    def test_schema_and_invariants_4x4(self) -> None:
        ds = get_default_grid_manager().get(GridId.RO_GRID_4X4)
        df = ds.df

        self.assertEqual(len(df), 36435)
        self.assertFalse(df.isna().any().any(), "Benchmark must contain zero NaNs")
        self.assertEqual(sorted(df["u"].unique().tolist()), [0.0, 0.25, 0.5, 0.75])
        self.assertEqual(sorted(df["v"].unique().tolist()), [0.0, 0.25, 0.5, 0.75])
        self.assertLess(df["roundtrip_err_n_m"].max(), 1e-3)
        self.assertLess(df["roundtrip_err_e_m"].max(), 1e-3)
        self.assertEqual(df["roundtrip_err_h_m"].max(), 0.0)


if __name__ == "__main__":
    unittest.main()
