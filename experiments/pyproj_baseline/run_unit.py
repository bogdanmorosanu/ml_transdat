"""Unit 02: Standard GIS PROJ baseline evaluation against pytransdat ground truth."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyproj

from src.core.base_unit import ResearchUnit
from src.core.metrics import compute_errors, format_scorecard_table, summarize_errors
from src.core.viz import generate_standard_diagnostic_suite

DEFAULT_BENCHMARK = Path("data/grids/ro_grid_4x4.parquet")


class PyprojBaselineUnit(ResearchUnit):
    """Research Unit 02: PyProj (EPSG:3844) standard library baseline."""

    @property
    def unit_id(self) -> str:
        return "02"

    @property
    def unit_name(self) -> str:
        return "pyproj_baseline"

    def run(self) -> dict[str, Any]:
        t0 = time.perf_counter()
        df = pd.read_parquet(DEFAULT_BENCHMARK)

        # 1. Inverse: ETRS89 (EPSG:4258) -> Stereo 70 (EPSG:3844)
        t_inv = pyproj.Transformer.from_crs("EPSG:4258", "EPSG:3844", always_xy=True)
        t_start = time.perf_counter()
        pred_e, pred_n = t_inv.transform(
            df["etrs89_lon_deg"].to_numpy(dtype=np.float64),
            df["etrs89_lat_deg"].to_numpy(dtype=np.float64),
        )
        inv_elapsed = time.perf_counter() - t_start

        inv_err_df = compute_errors(
            df_ref=df,
            pred_c1=np.asarray(pred_n, dtype=np.float64),
            pred_c2=np.asarray(pred_e, dtype=np.float64),
            direction="etrs89_to_st70",
        )
        inv_summary = summarize_errors(inv_err_df)
        inv_plots = generate_standard_diagnostic_suite(
            err_df=inv_err_df,
            summary=inv_summary,
            out_dir=self.plots_dir,
            model_name="pyproj_epsg3844",
            direction="etrs89_to_st70",
            dpi=200,
        )
        (self.results_dir / "pyproj_epsg3844_etrs89_to_st70_metrics.json").write_text(
            json.dumps(inv_summary, indent=2), encoding="utf-8"
        )
        print(format_scorecard_table(inv_summary, "pyproj_epsg3844", "etrs89_to_st70"))

        # 2. Forward: Stereo 70 (EPSG:3844) -> ETRS89 (EPSG:4258)
        t_fwd = pyproj.Transformer.from_crs("EPSG:3844", "EPSG:4258", always_xy=True)
        t_start = time.perf_counter()
        pred_lon, pred_lat = t_fwd.transform(
            df["st70_e"].to_numpy(dtype=np.float64),
            df["st70_n"].to_numpy(dtype=np.float64),
        )
        fwd_elapsed = time.perf_counter() - t_start

        fwd_err_df = compute_errors(
            df_ref=df,
            pred_c1=np.asarray(pred_lat, dtype=np.float64),
            pred_c2=np.asarray(pred_lon, dtype=np.float64),
            direction="st70_to_etrs89",
            is_degrees=True,
        )
        fwd_summary = summarize_errors(fwd_err_df)
        fwd_plots = generate_standard_diagnostic_suite(
            err_df=fwd_err_df,
            summary=fwd_summary,
            out_dir=self.plots_dir,
            model_name="pyproj_epsg3844",
            direction="st70_to_etrs89",
            dpi=200,
        )
        (self.results_dir / "pyproj_epsg3844_st70_to_etrs89_metrics.json").write_text(
            json.dumps(fwd_summary, indent=2), encoding="utf-8"
        )
        print(format_scorecard_table(fwd_summary, "pyproj_epsg3844", "st70_to_etrs89"))

        # Save brief
        elapsed = time.perf_counter() - t0
        brief = {
            "unit_id": self.unit_id,
            "unit_name": self.unit_name,
            "target_crs": "EPSG:3844",
            "reference_crs": "EPSG:4258",
            "pyproj_version": pyproj.__version__,
            "proj_version": pyproj.proj_version_str,
            "sample_count": len(df),
            "etrs89_to_st70": {
                "mean_2d_m": inv_summary["mean_2d_m"],
                "rmse_2d_m": inv_summary["rmse_2d_m"],
                "p50_2d_m": inv_summary["p50_2d_m"],
                "p95_2d_m": inv_summary["p95_2d_m"],
                "p99_2d_m": inv_summary["p99_2d_m"],
                "max_2d_m": inv_summary["max_2d_m"],
                "bias_n_m": inv_summary["bias_n_m"],
                "bias_e_m": inv_summary["bias_e_m"],
                "pass_lt_10cm_pct": inv_summary["pass_rates_pct"]["lt_10cm"],
                "pass_lt_1m_pct": inv_summary["pass_rates_pct"]["lt_1m"],
                "throughput_pts_per_s": round(len(df) / inv_elapsed, 1),
            },
            "st70_to_etrs89": {
                "mean_2d_m": fwd_summary["mean_2d_m"],
                "rmse_2d_m": fwd_summary["rmse_2d_m"],
                "p50_2d_m": fwd_summary["p50_2d_m"],
                "p95_2d_m": fwd_summary["p95_2d_m"],
                "p99_2d_m": fwd_summary["p99_2d_m"],
                "max_2d_m": fwd_summary["max_2d_m"],
                "bias_n_m": fwd_summary["bias_n_m"],
                "bias_e_m": fwd_summary["bias_e_m"],
                "pass_lt_10cm_pct": fwd_summary["pass_rates_pct"]["lt_10cm"],
                "pass_lt_1m_pct": fwd_summary["pass_rates_pct"]["lt_1m"],
                "throughput_pts_per_s": round(len(df) / fwd_elapsed, 1),
            },
            "plots": {
                "etrs89_to_st70": {k: v.name for k, v in inv_plots.items()},
                "st70_to_etrs89": {k: v.name for k, v in fwd_plots.items()},
            },
            "runtime_s": round(elapsed, 3),
        }
        self.save_brief(brief)
        print(f"=== Research Unit 02 [{self.unit_name}] Complete ===")
        return brief


def main() -> None:
    unit = PyprojBaselineUnit(Path("experiments/pyproj_baseline"))
    unit.run()


if __name__ == "__main__":
    main()
