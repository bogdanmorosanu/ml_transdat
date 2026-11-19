"""Unit tests verifying invariants of the canonical ro_grid_4x4 and ro_grid_8x8 benchmarks."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd

GRID_4X4_PARQUET = Path("data/grids/ro_grid_4x4.parquet")
GRID_4X4_META = Path("data/grids/ro_grid_4x4.meta.json")
GRID_8X8_PARQUET = Path("data/grids/ro_grid_8x8.parquet")
GRID_8X8_META = Path("data/grids/ro_grid_8x8.meta.json")
SPG_NODES_PARQUET = Path("data/grids/spg_nodes_valid.parquet")
SPG_NODES_META = Path("data/grids/spg_nodes_valid.meta.json")
CORRIDOR_PARQUET = Path("data/benchmarks/ro_corridor_snw.parquet")
CORRIDOR_META = Path("data/benchmarks/ro_corridor_snw.meta.json")


class TestRoGridBenchmark(unittest.TestCase):
    """Verify integrity, schema, and spatial invariants of canonical grids and benchmarks."""

    def test_files_exist(self) -> None:
        for path in (
            GRID_4X4_PARQUET,
            GRID_4X4_META,
            GRID_8X8_PARQUET,
            GRID_8X8_META,
            SPG_NODES_PARQUET,
            SPG_NODES_META,
            CORRIDOR_PARQUET,
            CORRIDOR_META,
        ):
            self.assertTrue(path.is_file(), f"Missing {path}")

    def test_spg_nodes_invariants(self) -> None:
        meta = json.loads(SPG_NODES_META.read_text(encoding="utf-8"))
        df = pd.read_parquet(SPG_NODES_PARQUET)
        self.assertEqual(len(df), meta["valid_land_nodes_count"])
        self.assertEqual(len(df), 2689)
        self.assertEqual(meta["transdat_valid_nodes_count"], 2259)

    def test_corridor_schema_and_invariants(self) -> None:
        grid_meta = json.loads(GRID_4X4_META.read_text(encoding="utf-8"))
        meta = json.loads(CORRIDOR_META.read_text(encoding="utf-8"))
        df = pd.read_parquet(CORRIDOR_PARQUET)

        self.assertEqual(len(df), meta["final_point_count"])
        self.assertEqual(len(df), grid_meta["bidirectional_valid_count"])
        self.assertFalse(df.isna().any().any(), "Corridor benchmark must contain zero NaNs")
        self.assertTrue(df["corridor_s_m"].is_monotonic_increasing)
        self.assertLessEqual(df["corridor_w_m"].abs().max(), 11_000.0)
        self.assertLess(df["roundtrip_err_n_m"].max(), 1e-3)
        self.assertLess(df["roundtrip_err_e_m"].max(), 1e-3)
        self.assertEqual(df["roundtrip_err_h_m"].max(), 0.0)

    def test_schema_and_invariants_4x4(self) -> None:
        meta = json.loads(GRID_4X4_META.read_text(encoding="utf-8"))
        df = pd.read_parquet(GRID_4X4_PARQUET)

        self.assertEqual(len(df), meta["bidirectional_valid_count"])
        self.assertFalse(df.isna().any().any(), "Benchmark must contain zero NaNs")
        self.assertEqual(int(df["is_node"].sum()), meta["node_count"])
        self.assertEqual(int((~df["is_node"]).sum()), meta["interior_count"])

        # Sub-grid offsets must match {0.0, 0.25, 0.50, 0.75}
        self.assertEqual(sorted(df["u"].unique().tolist()), [0.0, 0.25, 0.5, 0.75])
        self.assertEqual(sorted(df["v"].unique().tolist()), [0.0, 0.25, 0.5, 0.75])

    def test_schema_and_invariants_8x8(self) -> None:
        meta = json.loads(GRID_8X8_META.read_text(encoding="utf-8"))
        df = pd.read_parquet(GRID_8X8_PARQUET)

        self.assertEqual(len(df), meta["bidirectional_valid_count"])
        self.assertFalse(df.isna().any().any(), "Benchmark must contain zero NaNs")
        self.assertEqual(len(df), 146258)

        # Roundtrip errors must stay sub-millimeter
        self.assertLess(df["roundtrip_err_n_m"].max(), 1e-3)
        self.assertLess(df["roundtrip_err_e_m"].max(), 1e-3)
        self.assertEqual(df["roundtrip_err_h_m"].max(), 0.0)


if __name__ == "__main__":
    unittest.main()
