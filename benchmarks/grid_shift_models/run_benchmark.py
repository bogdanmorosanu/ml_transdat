"""Cross-model benchmark comparing all 2D Grid Shift Regressor families."""

from __future__ import annotations

from pathlib import Path

from experiments.adaptive_quadtree_grid.run_unit import AdaptiveQuadtreeGridUnit
from experiments.decision_tree_regressor.run_unit import (
    DecisionTreeSpatialRegressorUnit,
)
from experiments.knn_spatial_regressor.run_unit import KnnSpatialRegressorUnit
from experiments.mlp_spatial_regressor.run_unit import MLPSpatialRegressorUnit
from experiments.rbf_surface_regressor.run_unit import RBFSurfaceRegressorUnit
from experiments.submillimeter_exploration.run_unit import (
    SubmillimeterExplorationUnit,
)
from experiments.torch_mlp_regressor.run_unit import TorchMLPRegressorUnit
from src.core.base_benchmark import (
    BenchmarkModelSpec,
    BenchmarkSuite,
    specs_from_unit,
)


class GridShiftBenchmarkSuite(BenchmarkSuite):
    """Benchmark suite for local spatial partitioning, neural, and RBF/TPS models."""

    @property
    def suite_name(self) -> str:
        return "grid_shift_models"

    @property
    def suite_title(self) -> str:
        return "Cross-Model Grid Shift Benchmark Summary"

    @property
    def suite_description(self) -> str:
        return (
            "Compares validated champions across KNN, Decision Trees, Adaptive Quadtrees, "
            "Fourier MLP, PyTorch ResNet, and Thin Plate Splines with exact source-experiment hyperparameters."
        )

    def get_model_specs(self) -> list[BenchmarkModelSpec]:
        return [
            *specs_from_unit(
                KnnSpatialRegressorUnit,
                suite=self.suite_name,
                unit_dir=Path("experiments/knn_spatial_regressor"),
            ),
            *specs_from_unit(
                DecisionTreeSpatialRegressorUnit,
                suite=self.suite_name,
                unit_dir=Path("experiments/decision_tree_regressor"),
            ),
            *specs_from_unit(
                AdaptiveQuadtreeGridUnit,
                suite=self.suite_name,
                unit_dir=Path("experiments/adaptive_quadtree_grid"),
            ),
            *specs_from_unit(
                MLPSpatialRegressorUnit,
                suite=self.suite_name,
                unit_dir=Path("experiments/mlp_spatial_regressor"),
            ),
            *specs_from_unit(
                RBFSurfaceRegressorUnit,
                suite=self.suite_name,
                unit_dir=Path("experiments/rbf_surface_regressor"),
            ),
            *specs_from_unit(
                TorchMLPRegressorUnit,
                suite=self.suite_name,
                unit_dir=Path("experiments/torch_mlp_regressor"),
            ),
            *specs_from_unit(
                SubmillimeterExplorationUnit,
                suite=self.suite_name,
                unit_dir=Path("experiments/submillimeter_exploration"),
            ),
        ]


def main() -> None:
    suite = GridShiftBenchmarkSuite(Path("benchmarks/grid_shift_models"))
    suite.run()


if __name__ == "__main__":
    main()
