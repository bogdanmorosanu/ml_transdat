"""Unit 02: Standard GIS PROJ baseline evaluation against pytransdat ground truth."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyproj

from src.core.base_unit import TransformationExperimentUnit, TransformationModelSpec
from src.data import GridId
from src.evaluation import EvaluationResult
from src.models import BaseCoordinateTransformer, PyprojEpsg3844Transformer
from src.viz import PlotGroup


class PyprojBaselineUnit(TransformationExperimentUnit):
    """Transformation Unit: PyProj (EPSG:3844) standard library baseline."""

    @property
    def slug(self) -> str:
        return "pyproj_baseline"

    def get_transformer_specs(self) -> list[TransformationModelSpec]:
        return [
            TransformationModelSpec(
                name="pyproj_epsg3844",
                transformer=PyprojEpsg3844Transformer(),
                eval_grid=GridId.RO_GRID_4X4,
                directions=("etrs89_to_st70", "st70_to_etrs89"),
                plots=PlotGroup.STANDARD_DIAGNOSTIC,
            )
        ]

    def after_run(
        self,
        transformers: dict[str, BaseCoordinateTransformer],
        results: list[EvaluationResult],
    ) -> dict[str, Any]:
        del transformers
        for res in results:
            (self.results_dir / f"{res.model_name}_{res.direction}_metrics.json").write_text(
                json.dumps(res.summary, indent=2), encoding="utf-8"
            )
        return {
            "target_crs": "EPSG:3844",
            "reference_crs": "EPSG:4258",
            "pyproj_version": pyproj.__version__,
            "proj_version": pyproj.proj_version_str,
            "sample_count": results[0].summary["count"],
        }


def main() -> None:
    unit = PyprojBaselineUnit(Path("experiments/pyproj_baseline"))
    unit.run()


if __name__ == "__main__":
    main()
