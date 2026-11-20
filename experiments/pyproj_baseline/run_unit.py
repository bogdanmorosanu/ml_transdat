"""Unit 02: Standard GIS PROJ baseline evaluation against pytransdat ground truth."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pyproj

from src.core.base_unit import TransformationExperimentUnit
from src.data import GridId
from src.models import PyprojEpsg3844Transformer
from src.viz import PlotGroup


class PyprojBaselineUnit(TransformationExperimentUnit):
    """Research Unit 02: PyProj (EPSG:3844) standard library baseline."""

    @property
    def unit_id(self) -> str:
        return "02"

    @property
    def unit_name(self) -> str:
        return "pyproj_baseline"

    def run(self) -> dict[str, Any]:
        t0 = time.perf_counter()
        transformer = PyprojEpsg3844Transformer()
        results = self.evaluate_transformer(
            transformer=transformer,
            model_name="pyproj_epsg3844",
            dataset_id=GridId.RO_GRID_4X4,
            directions=("etrs89_to_st70", "st70_to_etrs89"),
            plots=PlotGroup.STANDARD_DIAGNOSTIC,
        )
        self.save_scorecards(results)

        inv_res, fwd_res = results
        (self.results_dir / "pyproj_epsg3844_etrs89_to_st70_metrics.json").write_text(
            json.dumps(inv_res.summary, indent=2), encoding="utf-8"
        )
        (self.results_dir / "pyproj_epsg3844_st70_to_etrs89_metrics.json").write_text(
            json.dumps(fwd_res.summary, indent=2), encoding="utf-8"
        )

        def _compact(res: Any) -> dict[str, Any]:
            s = res.summary
            return {
                "mean_2d_m": s["mean_2d_m"],
                "rmse_2d_m": s["rmse_2d_m"],
                "p50_2d_m": s["p50_2d_m"],
                "p95_2d_m": s["p95_2d_m"],
                "p99_2d_m": s["p99_2d_m"],
                "max_2d_m": s["max_2d_m"],
                "bias_n_m": s["bias_n_m"],
                "bias_e_m": s["bias_e_m"],
                "pass_lt_10cm_pct": s["pass_rates_pct"]["lt_10cm"],
                "pass_lt_1m_pct": s["pass_rates_pct"]["lt_1m"],
                "throughput_pts_per_s": round(res.throughput_pts_per_sec, 1),
            }

        elapsed = time.perf_counter() - t0
        brief = {
            "unit_id": self.unit_id,
            "unit_name": self.unit_name,
            "target_crs": "EPSG:3844",
            "reference_crs": "EPSG:4258",
            "pyproj_version": pyproj.__version__,
            "proj_version": pyproj.proj_version_str,
            "sample_count": inv_res.summary["count"],
            "etrs89_to_st70": _compact(inv_res),
            "st70_to_etrs89": _compact(fwd_res),
            "plots": {
                "etrs89_to_st70": inv_res.extra.get("plots", {}),
                "st70_to_etrs89": fwd_res.extra.get("plots", {}),
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
