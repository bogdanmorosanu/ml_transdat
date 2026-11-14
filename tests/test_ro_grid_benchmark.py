"""Unit tests verifying invariants of the canonical ro_grid_3x3 benchmark."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd

BENCHMARK_PARQUET = Path("data/benchmarks/ro_grid_3x3.parquet")
BENCHMARK_META = Path("data/benchmarks/ro_grid_3x3.meta.json")
CORRIDOR_PARQUET = Path("data/benchmarks/ro_corridor_snw.parquet")
CORRIDOR_META = Path("data/benchmarks/ro_corridor_snw.meta.json")


class TestRoGridBenchmark(unittest.TestCase):
    """Verify integrity, schema, and spatial invariants of canonical benchmarks."""

    def test_files_exist(self) -> None:
        for path in (BENCHMARK_PARQUET, BENCHMARK_META, CORRIDOR_PARQUET, CORRIDOR_META):
            self.assertTrue(path.is_file(), f"Missing {path}")

    def test_corridor_schema_and_invariants(self) -> None:
        grid_meta = json.loads(BENCHMARK_META.read_text(encoding="utf-8"))
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

    def test_schema_and_invariants(self) -> None:
        meta = json.loads(BENCHMARK_META.read_text(encoding="utf-8"))
        df = pd.read_parquet(BENCHMARK_PARQUET)

        self.assertEqual(len(df), meta["bidirectional_valid_count"])
        self.assertFalse(df.isna().any().any(), "Benchmark must contain zero NaNs")
        self.assertEqual(int(df["is_node"].sum()), meta["node_count"])
        self.assertEqual(int((~df["is_node"]).sum()), meta["interior_count"])

        # Sub-grid offsets must match {0.0, 0.25, 0.50, 0.75}
        self.assertEqual(sorted(df["u"].unique().tolist()), [0.0, 0.25, 0.5, 0.75])
        self.assertEqual(sorted(df["v"].unique().tolist()), [0.0, 0.25, 0.5, 0.75])

        # Roundtrip errors must stay sub-millimeter
        self.assertLess(df["roundtrip_err_n_m"].max(), 1e-3)
        self.assertLess(df["roundtrip_err_e_m"].max(), 1e-3)
        self.assertEqual(df["roundtrip_err_h_m"].max(), 0.0)


if __name__ == "__main__":
    unittest.main()
