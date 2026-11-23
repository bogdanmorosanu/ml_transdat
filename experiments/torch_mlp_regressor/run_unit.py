"""Research Unit: Deep PyTorch Residual Neural Network Spatial Regressor.

Evaluates PyTorch residual architectures, activations (ReLU, SiLU, GELU),
smooth cyclic (u, v) cell-position encoding, and 3x3 neighbor slopes/twists
(raw and 2D cubic Hermite-weighted) against the sklearn Fourier MLP baseline.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.core.base_unit import ExperimentModelSpec, GridShiftExperimentUnit
from src.data import GridId
from src.models.torch_mlp import TorchMLPGridShiftModel
from src.viz import PlotGroup

logger = logging.getLogger(__name__)


class TorchMLPRegressorUnit(GridShiftExperimentUnit):
    """Grid Shift Unit: Deep PyTorch Residual Neural Network spatial distortion learning."""

    @property
    def slug(self) -> str:
        return "torch_mlp_regressor"

    def get_model_specs(self) -> list[ExperimentModelSpec]:
        return [
            # 1. Stage 1: PyTorch ResNet ReLU Baseline (130 RFF features)
            ExperimentModelSpec(
                name="torch_res_relu_4x4",
                model=TorchMLPGridShiftModel(
                    hidden_dim=256,
                    num_res_blocks=2,
                    activation="relu",
                    use_fourier_features=True,
                    fourier_scale=3.0,
                    num_fourier_frequencies=64,
                    batch_size=256,
                    lr=1e-3,
                    max_epochs=200,
                    patience=25,
                ),
                train_grid=GridId.RO_GRID_4X4,
                eval_grids=[
                    GridId.RO_TEST_UNIFORM_4X4,
                    GridId.RO_TEST_ADAPTIVE_3MM_MID,
                ],
                plots=PlotGroup.STANDARD_DIAGNOSTIC,
                keep_in_memory=True,
            ),
            # 2. Stage 2 Champion: +4 Smooth Cyclic (u, v) Cell Position (134 inputs, grid-free)
            ExperimentModelSpec(
                name="torch_res_relu_cellpos_4x4",
                model=TorchMLPGridShiftModel(
                    hidden_dim=256,
                    num_res_blocks=2,
                    activation="relu",
                    use_fourier_features=True,
                    fourier_scale=3.0,
                    num_fourier_frequencies=64,
                    use_cyclic_cell_pos=True,
                    batch_size=256,
                    lr=1e-3,
                    max_epochs=200,
                    patience=25,
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
                benchmark_label="PyTorch ResNet + Cyclic CellPos (4x4)",
            ),
            # 3. Stage 3 Representative: +32 Hermite-Weighted Corner Slopes (166 inputs)
            ExperimentModelSpec(
                name="torch_res_relu_cellpos_hermite_slopes_4x4",
                model=TorchMLPGridShiftModel(
                    hidden_dim=256,
                    num_res_blocks=2,
                    activation="relu",
                    use_fourier_features=True,
                    fourier_scale=3.0,
                    num_fourier_frequencies=64,
                    use_cyclic_cell_pos=True,
                    use_neighbor_features=True,
                    neighbor_feature_mode="slopes",
                    use_hermite_weights=True,
                    batch_size=256,
                    lr=1e-3,
                    max_epochs=200,
                    patience=25,
                ),
                train_grid=GridId.RO_GRID_4X4,
                eval_grids=[
                    GridId.RO_TEST_UNIFORM_4X4,
                    GridId.RO_TEST_ADAPTIVE_3MM_MID,
                ],
                plots=PlotGroup.STANDARD_DIAGNOSTIC,
                keep_in_memory=True,
            ),
        ]


if __name__ == "__main__":
    unit = TorchMLPRegressorUnit(Path("experiments/torch_mlp_regressor"))
    unit.run()
