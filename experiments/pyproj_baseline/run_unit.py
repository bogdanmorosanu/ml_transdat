"""Unit 02: Standard GIS PROJ baseline evaluation against pytransdat ground truth."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pyproj

from src.core.base_unit import TransformationExperimentUnit, summarize_eval_results
from src.data import GridId
from src.models import PyprojEpsg3844Transformer
from src.viz import PlotGroup


class PyprojBaselineUnit(TransformationExperimentUnit):
    """Transformation Unit: PyProj (EPSG:3844) standard library baseline."""

    @property
    def slug(self) -> str:
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

        elapsed = time.perf_counter() - t0
        plot_list = list(inv_res.extra.get("plots", {}).values()) + list(
            fwd_res.extra.get("plots", {}).values()
        )
        brief = self.build_brief(
            runtime_s=elapsed,
            key_metrics=summarize_eval_results(results),
            artifacts={
                "scorecard_csv": "results/scorecard.csv",
                "scorecard_json": "results/scorecard_summary.json",
                "scorecard_md": "results/summary_table.md",
                "plots": plot_list,
            },
            extra={
                "target_crs": "EPSG:3844",
                "reference_crs": "EPSG:4258",
                "pyproj_version": pyproj.__version__,
                "proj_version": pyproj.proj_version_str,
                "sample_count": inv_res.summary["count"],
            },
        )
        self.save_brief(brief)
        print(f"=== Experiment '{self.slug}' Complete in {elapsed:.2f}s ===")
        return brief


def main() -> None:
    unit = PyprojBaselineUnit(Path("experiments/pyproj_baseline"))
    unit.run()


if __name__ == "__main__":
    main()
