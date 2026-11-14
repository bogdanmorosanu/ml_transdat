"""Unit tests for the standardized evaluation metrics and diagnostic suite."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.metrics import compute_errors, summarize_errors
from src.core.viz import generate_standard_diagnostic_suite

BENCHMARK_PATH = Path("data/benchmarks/ro_grid_3x3.parquet")


class TestEvalSuite(unittest.TestCase):
    """Verify metric calculations and diagnostic outputs in src.core."""

    def _sample_ref(self, n: int = 32) -> pd.DataFrame:
        df = pd.read_parquet(BENCHMARK_PATH)
        return df.iloc[:n].copy()

    def test_etrs89_to_st70_known_offset(self) -> None:
        df = self._sample_ref(16)
        pred_n = df["st70_n"].to_numpy() + 0.3
        pred_e = df["st70_e"].to_numpy() - 0.4
        err_df = compute_errors(df, pred_n, pred_e, direction="etrs89_to_st70")
        summary = summarize_errors(err_df)

        self.assertAlmostEqual(summary["mean_2d_m"], 0.5, places=9)
        self.assertAlmostEqual(summary["bias_n_m"], 0.3, places=9)
        self.assertAlmostEqual(summary["bias_e_m"], -0.4, places=9)
        self.assertAlmostEqual(summary["pass_rates_pct"]["lt_1m"], 100.0, places=6)
        self.assertAlmostEqual(summary["pass_rates_pct"]["lt_10cm"], 0.0, places=6)

    def test_st70_to_etrs89_zero_and_meter_offset(self) -> None:
        df = self._sample_ref(16)
        err_zero = compute_errors(
            df,
            df["etrs89_lat_deg"].to_numpy(),
            df["etrs89_lon_deg"].to_numpy(),
            direction="st70_to_etrs89",
            is_degrees=True,
        )
        self.assertAlmostEqual(float(err_zero["err_2d_m"].max()), 0.0, places=8)

        # ~1 arcsecond in latitude is roughly 30.87 m
        err_1s = compute_errors(
            df,
            df["etrs89_lat_deg"].to_numpy() + (1.0 / 3600.0),
            df["etrs89_lon_deg"].to_numpy(),
            direction="st70_to_etrs89",
            is_degrees=True,
        )
        mean_err = float(err_1s["err_2d_m"].mean())
        self.assertGreater(mean_err, 30.0)
        self.assertLess(mean_err, 31.5)

    def test_generate_diagnostic_suite(self) -> None:
        df = self._sample_ref(64)
        rng = np.random.default_rng(0)
        pred_n = df["st70_n"].to_numpy() + rng.normal(0.1, 0.05, size=len(df))
        pred_e = df["st70_e"].to_numpy() + rng.normal(-0.05, 0.05, size=len(df))

        err_df = compute_errors(df, pred_n, pred_e, direction="etrs89_to_st70")
        summary = summarize_errors(err_df)

        with tempfile.TemporaryDirectory() as tmp:
            outputs = generate_standard_diagnostic_suite(
                err_df=err_df,
                summary=summary,
                out_dir=Path(tmp),
                model_name="unit_test_dummy",
                direction="etrs89_to_st70",
                dpi=72,
            )
            for key in ("spatial_error", "hist_ecdf", "signed_scatter", "subcell_heatmap"):
                self.assertIn(key, outputs)
                self.assertTrue(outputs[key].is_file(), f"Missing output {key}")


if __name__ == "__main__":
    unittest.main()
