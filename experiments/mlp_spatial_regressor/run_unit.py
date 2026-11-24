"""Unit 06: Multilayer Perceptron (MLP) Spatial Regressor on SPG Grid Shifts."""

from __future__ import annotations

from pathlib import Path

from src.core.base_unit import ExperimentModelSpec, GridShiftExperimentUnit
from src.data import GridId
from src.models.base import BaseGridShiftModel
from src.models.knn_regressor import KNNSpatialRegressor
from src.models.mlp import MLPGridShiftModel
from src.viz import PlotGroup


class MLPSpatialRegressorUnit(GridShiftExperimentUnit):
    """Grid Shift Unit: Multilayer Perceptron neural network 2D spatial distortion learning."""

    @property
    def slug(self) -> str:
        return "mlp_spatial_regressor"

    def get_model_specs(self) -> list[ExperimentModelSpec]:
        return [
            # 1. Baseline Raw MLP (11 km nodes, 128 -> 128)
            ExperimentModelSpec(
                name="mlp_raw_11km",
                model=MLPGridShiftModel(
                    hidden_layer_sizes=(128, 128),
                    use_fourier_features=False,
                    solver="lbfgs",
                    max_iter=500,
                    alpha=1e-4,
                ),
                train_grid=GridId.SPG_NODES_VALID,
                eval_grids=[
                    GridId.RO_TEST_UNIFORM_4X4,
                    GridId.RO_TEST_ADAPTIVE_3MM_MID,
                ],
                plots=PlotGroup.STANDARD_DIAGNOSTIC,
                keep_in_memory=True,
            ),
            # 2. Fourier Feature MLP on 11 km nodes (scale=3.0, 32 freqs)
            ExperimentModelSpec(
                name="mlp_fourier_11km",
                model=MLPGridShiftModel(
                    hidden_layer_sizes=(128, 128),
                    use_fourier_features=True,
                    fourier_scale=3.0,
                    num_fourier_frequencies=32,
                    solver="lbfgs",
                    max_iter=500,
                    alpha=1e-4,
                ),
                train_grid=GridId.SPG_NODES_VALID,
                eval_grids=[
                    GridId.RO_TEST_UNIFORM_4X4,
                    GridId.RO_TEST_ADAPTIVE_3MM_MID,
                ],
                plots=PlotGroup.STANDARD_DIAGNOSTIC,
                keep_in_memory=True,
            ),
            # 3. Fourier Feature MLP trained on 4x4 Densified Subgrid (~36.4k pts, 2.7 km)
            ExperimentModelSpec(
                name="mlp_fourier_4x4",
                model=MLPGridShiftModel(
                    hidden_layer_sizes=(256, 128),
                    use_fourier_features=True,
                    fourier_scale=3.0,
                    num_fourier_frequencies=64,
                    solver="adam",
                    batch_size=128,
                    learning_rate_init=0.001,
                    tol=1e-7,
                    n_iter_no_change=25,
                    max_iter=300,
                    alpha=1e-6,
                ),
                train_grid=GridId.RO_GRID_4X4,
                eval_grids=[
                    GridId.RO_TEST_UNIFORM_4X4,
                    GridId.RO_TEST_ADAPTIVE_3MM_MID,
                ],
                plots=PlotGroup.STANDARD_DIAGNOSTIC,
                keep_in_memory=True,
                benchmark_candidate=True,
                benchmark_suite="grid_shift_models",
                benchmark_label="Fourier MLP (4x4, 2.7km)",
            ),
            # 4. Fourier Feature MLP trained on 8x8 Densified Subgrid (~146.2k pts, 1.4 km)
            ExperimentModelSpec(
                name="mlp_fourier_8x8",
                model=MLPGridShiftModel(
                    hidden_layer_sizes=(256, 128),
                    use_fourier_features=True,
                    fourier_scale=3.0,
                    num_fourier_frequencies=64,
                    solver="adam",
                    batch_size=256,
                    learning_rate_init=0.002,
                    tol=1e-6,
                    n_iter_no_change=15,
                    max_iter=150,
                    alpha=1e-6,
                ),
                train_grid=GridId.RO_GRID_8X8,
                eval_grids=[
                    GridId.RO_TEST_UNIFORM_4X4,
                    GridId.RO_TEST_ADAPTIVE_3MM_MID,
                ],
                plots=PlotGroup.EXTENDED_SHIFT,
                keep_in_memory=True,
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
            "KNN K=4 IDW (11 km)": knn_ref,
            "Raw MLP (11 km)": fitted_models["mlp_raw_11km"],
            "Fourier MLP (11 km)": fitted_models["mlp_fourier_11km"],
            "Fourier MLP (4x4, 2.7 km)": fitted_models["mlp_fourier_4x4"],
            "Fourier MLP (8x8, 1.4 km)": fitted_models["mlp_fourier_8x8"],
        }


def main() -> None:
    unit = MLPSpatialRegressorUnit(Path("experiments/mlp_spatial_regressor"))
    unit.run()


if __name__ == "__main__":
    main()
