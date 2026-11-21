"""Experiment Unit: Fourier MLP Spatial Regressor with 5 mm Per-Axis Cell Shape Features."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.core.base_unit import ExperimentModelSpec, GridShiftExperimentUnit
from src.data import GridId
from src.evaluation import EvaluationResult
from src.models.base import BaseGridShiftModel
from src.models.mlp import MLPGridShiftModel
from src.viz import PlotGroup, plot_transect_66km


class MLPShapeFeaturesUnit(GridShiftExperimentUnit):
    """Grid Shift Unit: Controlled ablation of 5 mm per-axis cell shape features on Fourier MLP."""

    @property
    def slug(self) -> str:
        return "mlp_shape_features"

    def get_model_specs(self) -> list[ExperimentModelSpec]:
        return [
            # 1. Champion Baseline: Fourier MLP trained on 4x4 Subgrid (no shape features)
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
                    use_shape_features=False,
                ),
                train_grid=GridId.RO_GRID_4X4,
                eval_grids=[
                    GridId.RO_TEST_UNIFORM_4X4,
                    GridId.RO_TEST_ADAPTIVE_3MM_MID,
                ],
                plots=PlotGroup.STANDARD_DIAGNOSTIC,
                keep_in_memory=True,
            ),
            # 2. Treatment: Fourier MLP + 5 mm Per-Axis One-Hot Shape Features (cat_dn, cat_de)
            ExperimentModelSpec(
                name="mlp_fourier_shape_4x4",
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
                    use_shape_features=True,
                    shape_tol_mm=5.0,
                ),
                train_grid=GridId.RO_GRID_4X4,
                eval_grids=[
                    GridId.RO_TEST_UNIFORM_4X4,
                    GridId.RO_TEST_ADAPTIVE_3MM_MID,
                ],
                plots=PlotGroup.STANDARD_DIAGNOSTIC,
                keep_in_memory=True,
            ),
            # 3. Treatment (3 mm): Fourier MLP + 3 mm Per-Axis One-Hot Shape Features
            ExperimentModelSpec(
                name="mlp_fourier_shape_3mm_4x4",
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
                    use_shape_features=True,
                    shape_tol_mm=3.0,
                ),
                train_grid=GridId.RO_GRID_4X4,
                eval_grids=[
                    GridId.RO_TEST_UNIFORM_4X4,
                    GridId.RO_TEST_ADAPTIVE_3MM_MID,
                ],
                plots=None,
                keep_in_memory=True,
            ),
            # 4. Treatment (10 mm): Fourier MLP + 10 mm Per-Axis One-Hot Shape Features
            ExperimentModelSpec(
                name="mlp_fourier_shape_10mm_4x4",
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
                    use_shape_features=True,
                    shape_tol_mm=10.0,
                ),
                train_grid=GridId.RO_GRID_4X4,
                eval_grids=[
                    GridId.RO_TEST_UNIFORM_4X4,
                    GridId.RO_TEST_ADAPTIVE_3MM_MID,
                ],
                plots=None,
                keep_in_memory=True,
            ),
            # 5. S-Shape Weighting (2x): No shape input features, 2x weight on 5 mm S-Shape cells
            ExperimentModelSpec(
                name="mlp_fourier_sweight2_4x4",
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
                    use_shape_features=False,
                    shape_tol_mm=5.0,
                    s_shape_weight=2,
                ),
                train_grid=GridId.RO_GRID_4X4,
                eval_grids=[
                    GridId.RO_TEST_UNIFORM_4X4,
                    GridId.RO_TEST_ADAPTIVE_3MM_MID,
                ],
                plots=None,
                keep_in_memory=True,
            ),
            # 6. S-Shape Weighting (3x): No shape input features, 3x weight on 5 mm S-Shape cells
            ExperimentModelSpec(
                name="mlp_fourier_sweight3_4x4",
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
                    use_shape_features=False,
                    shape_tol_mm=5.0,
                    s_shape_weight=3,
                ),
                train_grid=GridId.RO_GRID_4X4,
                eval_grids=[
                    GridId.RO_TEST_UNIFORM_4X4,
                    GridId.RO_TEST_ADAPTIVE_3MM_MID,
                ],
                plots=None,
                keep_in_memory=True,
            ),
        ]

    def after_run(
        self,
        fitted_models: dict[str, BaseGridShiftModel],
        results: list[EvaluationResult],
    ) -> dict[str, Any]:
        transect_path = self.plots_dir / "cell_transect_66km_mlp_shape.png"
        plot_transect_66km(
            models={
                "Fourier MLP (4x4, Baseline)": fitted_models["mlp_fourier_4x4"],
                "Fourier MLP + 3mm Shape (4x4)": fitted_models["mlp_fourier_shape_3mm_4x4"],
                "Fourier MLP + 5mm Shape (4x4)": fitted_models["mlp_fourier_shape_4x4"],
                "Fourier MLP + 10mm Shape (4x4)": fitted_models["mlp_fourier_shape_10mm_4x4"],
            },
            out_path=transect_path,
            title="Fourier MLP (4x4): Baseline vs. +5mm Per-Axis Shape Features (66 km Transect)",
        )
        return {"transect_plot": transect_path.name}


def main() -> None:
    unit = MLPShapeFeaturesUnit(Path("experiments/mlp_shape_features"))
    unit.run()


if __name__ == "__main__":
    main()
