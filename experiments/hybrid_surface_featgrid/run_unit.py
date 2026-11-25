"""Grid Shift Unit: Hybrid Stage-1 Smooth Surface + Stage-2 Neural Feature Grid.

Phase 2 Full-Romania evaluation of the Phase-1 subgrid matrix champion
(``cubic_k25_s0`` + ``NeuralFeatureGridShiftModel(hidden_dim=128, loss_type='huber',
cyclic_cell_harmonics=2)``) trained on ``RO_GRID_32X32`` (~2.35M points) and
evaluated on ``RO_TEST_UNIFORM_16X16`` and ``RO_TEST_ADAPTIVE_1MM_MID`` alongside
standalone Stage-1 and Stage-2 references.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.core.base_unit import ExperimentModelSpec, GridShiftExperimentUnit
from src.data import GridId
from src.evaluation import EvaluationResult
from src.evaluation.metrics import dataframe_to_markdown
from src.models import (
    BaseGridShiftModel,
    CompositeResidualShiftModel,
    NeuralFeatureGridShiftModel,
    RBFSurfaceRegressor,
)
from src.viz import PlotGroup

EVAL_GRIDS: list[GridId] = [
    GridId.RO_TEST_UNIFORM_16X16,
    GridId.RO_TEST_ADAPTIVE_1MM_MID,
]


def _make_champion_featgrid_32x32() -> NeuralFeatureGridShiftModel:
    """Create the Phase-1 winning Stage-2 NeuralFeatureGridShiftModel for Full 32x32."""
    return NeuralFeatureGridShiftModel(
        grid_resolutions=((53, 72), (105, 143)),
        feature_dim=8,
        hidden_dim=128,
        cyclic_cell_harmonics=2,
        batch_size=4096,
        lr=3e-3,
        max_epochs=120,
        val_fraction=0.0,
        loss_type="huber",
    )


class HybridSurfaceFeatGridUnit(GridShiftExperimentUnit):
    """Full-Romania evaluation of Stage-1 Surface + Stage-2 Neural Feature Grid."""

    render_extreme_scenarios: bool = True

    @property
    def slug(self) -> str:
        return "hybrid_surface_featgrid"

    def _build_prefitted_stage1_cubic(self) -> RBFSurfaceRegressor:
        """Load or fit the Stage-1 ``cubic_k25_s0`` surface on Full ``RO_GRID_4X4``."""
        cubic = RBFSurfaceRegressor(
            kernel="cubic",
            degree=1,
            smoothing=0.0,
            neighbors=25,
            normalize_coords=True,
        )
        if self._probe_only:
            return cubic
        cubic.load_or_fit(
            lambda: (
                self.load_grid(GridId.RO_GRID_4X4).coords,
                self.load_grid(GridId.RO_GRID_4X4).true_shifts,
            ),
            cache_path=self.models_dir / "_stage1_cubic_k25_s0.pkl",
            train_grid=GridId.RO_GRID_4X4.value,
            label="stage1_cubic_k25_s0",
        )
        return cubic

    def get_model_specs(self) -> list[ExperimentModelSpec]:
        stage1_cubic = self._build_prefitted_stage1_cubic()

        return [
            ExperimentModelSpec(
                name="Stage1_cubic_k25_s0_only",
                model=RBFSurfaceRegressor(
                    kernel="cubic",
                    degree=1,
                    smoothing=0.0,
                    neighbors=25,
                    normalize_coords=True,
                ),
                train_grid=GridId.RO_GRID_4X4,
                eval_grids=EVAL_GRIDS,
                plots=PlotGroup.STANDARD_DIAGNOSTIC,
                keep_in_memory=True,
                extreme_scenario_label="Stage-1 Cubic k25 (4x4 only)",
            ),
            ExperimentModelSpec(
                name="FeatGrid_32x32_only",
                model=_make_champion_featgrid_32x32(),
                train_grid=GridId.RO_GRID_32X32,
                eval_grids=EVAL_GRIDS,
                plots=PlotGroup.STANDARD_DIAGNOSTIC,
                keep_in_memory=True,
                extreme_scenario_label="FeatGrid h128 Huber (32x32 only)",
            ),
            ExperimentModelSpec(
                name="Hybrid_cubic_k25_s0_plus_featgrid_32x32",
                model=CompositeResidualShiftModel(
                    stage1_model=stage1_cubic,
                    stage2_model=_make_champion_featgrid_32x32(),
                ),
                train_grid=GridId.RO_GRID_32X32,
                eval_grids=EVAL_GRIDS,
                plots=PlotGroup.STANDARD_DIAGNOSTIC,
                keep_in_memory=True,
                extreme_scenario_label="Cubic k25 + FeatGrid h128 Huber (32x32)",
                benchmark_candidate=True,
                benchmark_suite="grid_shift_models",
                benchmark_label="Cubic k25 + Neural Feature Grid (32x32, h128, Huber)",
            ),
        ]

    def after_run(
        self,
        fitted_models: dict[str, BaseGridShiftModel],
        results: list[EvaluationResult],
    ) -> dict[str, Any]:
        del fitted_models
        delta_df = self.evaluator.tabulate_delta_vs_baseline(
            results=results,
            baseline_model="Stage1_cubic_k25_s0_only",
            precision_unit="mm",
        )
        delta_csv = self.results_dir / "full_ro_delta_vs_stage1.csv"
        delta_df.to_csv(delta_csv, index=False)

        delta_md = self.results_dir / "full_ro_delta_vs_stage1.md"
        delta_md.write_text(dataframe_to_markdown(delta_df), encoding="utf-8")

        return {
            "delta_csv": str(Path("results") / "full_ro_delta_vs_stage1.csv"),
            "delta_md": str(Path("results") / "full_ro_delta_vs_stage1.md"),
        }


if __name__ == "__main__":
    HybridSurfaceFeatGridUnit(Path("experiments/hybrid_surface_featgrid")).run()
