"""Unit 04: K-Nearest Neighbors (KNN) Spatial Regressor on SPG Grid Shifts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from src.core.base_unit import ExperimentModelSpec, GridShiftExperimentUnit
from src.data import GridId
from src.evaluation import EvaluationResult
from src.models.base import BaseGridShiftModel
from src.models.knn_regressor import KNNSpatialRegressor
from src.viz import PlotGroup, plot_transect_66km


class KnnSpatialRegressorUnit(GridShiftExperimentUnit):
    """Grid Shift Unit: K-Nearest Neighbors 2D spatial distortion learning."""

    @property
    def slug(self) -> str:
        return "knn_spatial_regressor"

    def get_model_specs(self) -> list[ExperimentModelSpec]:
        configs: list[tuple[str, int, Literal["uniform", "distance"]]] = [
            ("k1_uniform", 1, "uniform"),
            ("k4_uniform", 4, "uniform"),
            ("k4_distance", 4, "distance"),
            ("k8_uniform", 8, "uniform"),
            ("k8_distance", 8, "distance"),
            ("k16_uniform", 16, "uniform"),
            ("k16_distance", 16, "distance"),
        ]
        benchmark_labels = {
            "k4_distance": "KNN (K=4 IDW, 11km)",
            "k16_distance": "KNN (K=16 IDW, 11km)",
        }
        specs: list[ExperimentModelSpec] = []
        for label, k, weights in configs:
            is_best = label == "k4_distance"
            eval_grids = (
                [GridId.RO_GRID_4X4, GridId.RO_GRID_8X8]
                if is_best
                else [GridId.RO_GRID_4X4]
            )
            plots = (
                PlotGroup.EXTENDED_SHIFT if is_best else PlotGroup.STANDARD_DIAGNOSTIC
            )
            specs.append(
                ExperimentModelSpec(
                    name=label,
                    model=KNNSpatialRegressor(n_neighbors=k, weights=weights),
                    train_grid=GridId.SPG_NODES_VALID,
                    eval_grids=eval_grids,
                    plots=plots,
                    keep_in_memory=label
                    in {"k1_uniform", "k4_uniform", "k4_distance", "k16_distance"},
                    benchmark_candidate=label in benchmark_labels,
                    benchmark_suite="grid_shift_models" if label in benchmark_labels else None,
                    benchmark_label=benchmark_labels.get(label),
                )
            )
        return specs

    def after_run(
        self,
        fitted_models: dict[str, BaseGridShiftModel],
        results: list[EvaluationResult],
    ) -> dict[str, Any]:
        transect_path = self.plots_dir / "cell_transect_66km.png"
        plot_transect_66km(
            models={
                "KNN K=1": fitted_models["k1_uniform"],
                "KNN K=4 Uniform": fitted_models["k4_uniform"],
                "KNN K=4 IDW": fitted_models["k4_distance"],
                "KNN K=16 IDW": fitted_models["k16_distance"],
            },
            out_path=transect_path,
            title="KNN Spatial Regressors vs. True Bicubic Across 6 SPG Cells (66 km)",
        )
        return {"transect_plot": transect_path.name}


def main() -> None:
    unit = KnnSpatialRegressorUnit(Path("experiments/knn_spatial_regressor"))
    unit.run()


if __name__ == "__main__":
    main()
