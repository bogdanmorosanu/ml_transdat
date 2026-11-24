"""Unit 05: Decision Tree Spatial Regressor on SPG Grid Shifts."""

from __future__ import annotations

from pathlib import Path

from src.core.base_unit import ExperimentModelSpec, GridShiftExperimentUnit
from src.data import GridId
from src.models.base import BaseGridShiftModel
from src.models.decision_tree_regressor import DecisionTreeSpatialRegressor
from src.models.knn_regressor import KNNSpatialRegressor
from src.viz import PlotGroup


class DecisionTreeSpatialRegressorUnit(GridShiftExperimentUnit):
    """Grid Shift Unit: Decision Tree 2D spatial distortion learning."""

    @property
    def slug(self) -> str:
        return "decision_tree_regressor"

    def get_model_specs(self) -> list[ExperimentModelSpec]:
        return [
            ExperimentModelSpec(
                name="dt_base_11km",
                model=DecisionTreeSpatialRegressor(),
                train_grid=GridId.SPG_NODES_VALID,
                eval_grids=[GridId.RO_GRID_8X8, GridId.CORRIDOR_SNW],
                plots=PlotGroup.STANDARD_DIAGNOSTIC,
                keep_in_memory=True,
                benchmark_candidate=True,
                benchmark_suite="grid_shift_models",
                benchmark_label="Decision Tree (11km)",
            ),
            ExperimentModelSpec(
                name="dt_4x4_2750m",
                model=DecisionTreeSpatialRegressor(),
                train_grid=GridId.RO_GRID_4X4,
                eval_grids=[GridId.RO_GRID_8X8, GridId.CORRIDOR_SNW],
                plots=PlotGroup.STANDARD_DIAGNOSTIC,
            ),
            ExperimentModelSpec(
                name="dt_8x8_1375m",
                model=DecisionTreeSpatialRegressor(),
                train_grid=GridId.RO_GRID_8X8,
                eval_grids=[GridId.CORRIDOR_SNW],
                plots=PlotGroup.STANDARD_DIAGNOSTIC,
            ),
            ExperimentModelSpec(
                name="dt_8x8_pruned_1e-8",
                model=DecisionTreeSpatialRegressor(min_impurity_decrease=1e-8),
                train_grid=GridId.RO_GRID_8X8,
                eval_grids=[GridId.CORRIDOR_SNW],
                plots=PlotGroup.STANDARD_DIAGNOSTIC,
                benchmark_candidate=True,
                benchmark_suite="grid_shift_models",
                benchmark_label="Decision Tree (8x8, pruned 1e-8)",
            ),
            ExperimentModelSpec(
                name="dt_16x16_688m",
                model=DecisionTreeSpatialRegressor(),
                train_grid=GridId.RO_GRID_16X16,
                eval_grids=[GridId.CORRIDOR_SNW],
                plots=PlotGroup.STANDARD_DIAGNOSTIC,
            ),
        ]

    def get_extreme_scenario_models(
        self,
        fitted_models: dict[str, BaseGridShiftModel],
    ) -> dict[str, BaseGridShiftModel]:
        nodes_ds = self.load_grid(GridId.SPG_NODES_VALID)
        knn_ref = KNNSpatialRegressor(n_neighbors=4, weights="distance").fit(
            nodes_ds.coords, nodes_ds.true_shifts
        )
        return {
            "Decision Tree (11 km nodes)": fitted_models["dt_base_11km"],
            "KNN K=4 IDW (11 km nodes)": knn_ref,
        }


def main() -> None:
    unit = DecisionTreeSpatialRegressorUnit(Path("experiments/decision_tree_regressor"))
    unit.run()


if __name__ == "__main__":
    main()
