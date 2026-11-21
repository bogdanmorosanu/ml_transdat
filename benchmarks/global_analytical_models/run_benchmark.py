"""Cross-model benchmark comparing Global Analytical & Mathematical Surface models."""

from __future__ import annotations

from pathlib import Path

from experiments.polynomial_surface_regressor.run_unit import (
    PolynomialSurfaceRegressorUnit,
)
from src.core.base_benchmark import (
    BenchmarkModelSpec,
    BenchmarkSuite,
    specs_from_unit,
)
from src.data import GridId
from src.models import Helmert2DConformalShiftModel


class GlobalAnalyticalBenchmarkSuite(BenchmarkSuite):
    """Benchmark suite for closed-form global analytical coordinate shift surfaces."""

    @property
    def suite_name(self) -> str:
        return "global_analytical_models"

    @property
    def suite_title(self) -> str:
        return "Global Analytical Models Benchmark Summary"

    @property
    def suite_description(self) -> str:
        return (
            "Models in this suite evaluate a **single closed-form global equation** "
            "without spatial lookups (0 MB grid lookup memory)."
        )

    def get_model_specs(self) -> list[BenchmarkModelSpec]:
        return [
            BenchmarkModelSpec(
                name="Helmert 2D Conformal (4P)",
                model=Helmert2DConformalShiftModel(),
                train_grid=GridId.SPG_NODES_VALID,
                source_experiment="helmert2d_baseline",
            ),
            *specs_from_unit(
                PolynomialSurfaceRegressorUnit,
                suite=self.suite_name,
                unit_dir=Path("experiments/polynomial_surface_regressor"),
            ),
        ]


def main() -> None:
    suite = GlobalAnalyticalBenchmarkSuite(Path("benchmarks/global_analytical_models"))
    suite.run()


if __name__ == "__main__":
    main()
