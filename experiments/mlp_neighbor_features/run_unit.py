"""Experiment Unit: Fourier MLP Spatial Regressor with 3x3 Cell Neighbor Features."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.core.base_unit import ExperimentModelSpec, GridShiftExperimentUnit
from src.data import GridDataset, GridId
from src.evaluation import EvaluationResult
from src.geometry.neighbor_features import CellNeighborFeatureExtractor
from src.models.base import BaseGridShiftModel
from src.models.mlp import MLPGridShiftModel
from src.viz import PlotGroup, plot_transect_66km

NEIGHBOR_FEATURES_PATH = Path("data/features/cell_neighbor_features.parquet")


class MLPNeighborFeaturesUnit(GridShiftExperimentUnit):
    """Grid Shift Unit: Ablation of 3x3 corner slopes and twists on Fourier MLP."""

    def __init__(self, unit_dir: Path, cache_grids: bool = True) -> None:
        super().__init__(unit_dir, cache_grids=cache_grids)
        self._neighbor_extractor = CellNeighborFeatureExtractor(NEIGHBOR_FEATURES_PATH)

    @property
    def slug(self) -> str:
        return "mlp_neighbor_features"

    def load_grid(
        self, grid_id: GridId | str, force_regenerate: bool = False
    ) -> GridDataset:
        """Load dataset filtered strictly to points inside valid 3x3 neighbor cells."""
        ds = super().load_grid(grid_id, force_regenerate=force_regenerate)
        feats = self._neighbor_extractor.lookup_points(
            ds.df["st70_n"].to_numpy(dtype=np.float64),
            ds.df["st70_e"].to_numpy(dtype=np.float64),
        )
        valid_mask = np.isfinite(feats).all(axis=1)
        filtered_df = ds.df.loc[valid_mask].reset_index(drop=True)
        return GridDataset(
            grid_id=ds.grid_id,
            df=filtered_df,
            metadata={**ds.metadata, "filtered_interior_neighbor_cells": True},
        )

    def get_model_specs(self) -> list[ExperimentModelSpec]:
        base_kwargs: dict[str, Any] = {
            "hidden_layer_sizes": (256, 128),
            "use_fourier_features": True,
            "fourier_scale": 3.0,
            "num_fourier_frequencies": 64,
            "solver": "adam",
            "batch_size": 128,
            "learning_rate_init": 0.001,
            "tol": 1e-7,
            "n_iter_no_change": 25,
            "max_iter": 300,
            "alpha": 1e-6,
        }
        eval_grids = [
            GridId.RO_TEST_UNIFORM_4X4,
            GridId.RO_TEST_ADAPTIVE_3MM_MID,
        ]
        return [
            # 1. Champion Baseline: Fourier MLP trained on 4x4 Subgrid (no neighbor features)
            ExperimentModelSpec(
                name="mlp_fourier_4x4",
                model=MLPGridShiftModel(
                    **base_kwargs,
                    use_neighbor_features=False,
                    use_cell_center_distance=False,
                ),
                train_grid=GridId.RO_GRID_4X4,
                eval_grids=eval_grids,
                plots=PlotGroup.STANDARD_DIAGNOSTIC,
                keep_in_memory=True,
            ),
            # 2. Center Distance Only: Fourier MLP + 2 continuous [0, 1] midpoint distances
            ExperimentModelSpec(
                name="mlp_fourier_center_dist_4x4",
                model=MLPGridShiftModel(
                    **base_kwargs,
                    use_neighbor_features=False,
                    neighbor_features_path=NEIGHBOR_FEATURES_PATH,
                    use_cell_center_distance=True,
                ),
                train_grid=GridId.RO_GRID_4X4,
                eval_grids=eval_grids,
                plots=PlotGroup.STANDARD_DIAGNOSTIC,
                keep_in_memory=True,
            ),
            # 3. Center Distance + Slopes: +2 midpoint distances + 32 standardized slopes
            ExperimentModelSpec(
                name="mlp_fourier_center_dist_slopes_4x4",
                model=MLPGridShiftModel(
                    **base_kwargs,
                    use_neighbor_features=True,
                    neighbor_feature_mode="slopes",
                    neighbor_features_path=NEIGHBOR_FEATURES_PATH,
                    scale_neighbor_features=True,
                    use_cell_center_distance=True,
                ),
                train_grid=GridId.RO_GRID_4X4,
                eval_grids=eval_grids,
                plots=PlotGroup.STANDARD_DIAGNOSTIC,
                keep_in_memory=True,
            ),
            # 4. Center Distance + All: +2 midpoint distances + 40 standardized slopes & twists
            ExperimentModelSpec(
                name="mlp_fourier_center_dist_all_4x4",
                model=MLPGridShiftModel(
                    **base_kwargs,
                    use_neighbor_features=True,
                    neighbor_feature_mode="all",
                    neighbor_features_path=NEIGHBOR_FEATURES_PATH,
                    scale_neighbor_features=True,
                    use_cell_center_distance=True,
                ),
                train_grid=GridId.RO_GRID_4X4,
                eval_grids=eval_grids,
                plots=PlotGroup.STANDARD_DIAGNOSTIC,
                keep_in_memory=True,
            ),
        ]

    def after_run(
        self,
        fitted_models: dict[str, BaseGridShiftModel],
        results: list[EvaluationResult],
    ) -> dict[str, Any]:
        transect_path = self.plots_dir / "cell_transect_66km_mlp_neighbor.png"
        plot_transect_66km(
            models={
                "Fourier MLP (4x4, Baseline)": fitted_models["mlp_fourier_4x4"],
                "Fourier MLP + Center Dist (4x4)": fitted_models[
                    "mlp_fourier_center_dist_4x4"
                ],
                "Fourier MLP + Center Dist + 32 Slopes": fitted_models[
                    "mlp_fourier_center_dist_slopes_4x4"
                ],
                "Fourier MLP + Center Dist + 40 All": fitted_models[
                    "mlp_fourier_center_dist_all_4x4"
                ],
            },
            out_path=transect_path,
            title="Fourier MLP (4x4): Baseline vs. Center Distance + 3x3 Neighbor Features (66 km Transect)",
        )
        return {"transect_plot": transect_path.name}


def main() -> None:
    unit = MLPNeighborFeaturesUnit(Path("experiments/mlp_neighbor_features"))
    unit.run()


if __name__ == "__main__":
    main()
