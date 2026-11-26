"""Grid Shift Unit: Hybrid Stage-1 Smooth Surface + Stage-2 Grid-Free ResNet.

Phase 2 Full-Romania evaluation of the Phase-1 subgrid matrix champion
(``lpoly_deg2_k16_tric`` + ``TorchMLPGridShiftModel(hidden_dim=128, num_res_blocks=3,
activation='silu', loss_type='huber', cyclic_cell_harmonics=2)``) trained on
``RO_GRID_32X32`` (~2.35M points) and evaluated on ``RO_TEST_UNIFORM_16X16`` and
``RO_TEST_ADAPTIVE_1MM_MID`` alongside standalone Stage-1 and Stage-2 references.
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
    LocalPolynomialSurfaceRegressor,
    TorchMLPGridShiftModel,
)
from src.viz import PlotGroup

EVAL_GRIDS: list[GridId] = [
    GridId.RO_TEST_UNIFORM_16X16,
    GridId.RO_TEST_ADAPTIVE_1MM_MID,
]


def _make_champion_resnet_32x32(
    *,
    hidden_dim: int = 128,
    fourier_scales: tuple[float, ...] | None = None,
    num_fourier_frequencies: int = 64,
) -> TorchMLPGridShiftModel:
    """Create a Stage-2 TorchMLPGridShiftModel for Full 32x32."""
    return TorchMLPGridShiftModel(
        hidden_dim=hidden_dim,
        num_res_blocks=3,
        activation="silu",
        use_fourier_features=True,
        fourier_scale=3.0,
        fourier_scales=fourier_scales,
        num_fourier_frequencies=num_fourier_frequencies,
        use_cyclic_cell_pos=True,
        cyclic_cell_harmonics=2,
        batch_size=8192,
        lr=1e-3,
        weight_decay=1e-5,
        max_epochs=150,
        patience=25,
        lr_scheduler="cosine",
        val_fraction=0.0,
        loss_type="huber",
    )


class HybridSurfaceResNetUnit(GridShiftExperimentUnit):
    """Full-Romania evaluation of Stage-1 Surface + Stage-2 Grid-Free ResNet."""

    submillimeter_mode: bool = True
    render_extreme_scenarios: bool = True

    @property
    def slug(self) -> str:
        return "hybrid_surface_resnet"

    def _build_prefitted_stage1_lpoly(self) -> LocalPolynomialSurfaceRegressor:
        """Load or fit the Stage-1 ``lpoly_deg2_k16_tric`` surface on Full ``RO_GRID_4X4``."""
        lpoly = LocalPolynomialSurfaceRegressor(
            degree=2,
            neighbors=16,
            weight="tricube",
        )
        if self._probe_only:
            return lpoly
        lpoly.load_or_fit(
            lambda: (
                self.load_grid(GridId.RO_GRID_4X4).coords,
                self.load_grid(GridId.RO_GRID_4X4).true_shifts,
            ),
            cache_path=self.models_dir / "_stage1_lpoly_deg2_k16_tric.pkl",
            train_grid=GridId.RO_GRID_4X4.value,
            label="stage1_lpoly_deg2_k16_tric",
        )
        return lpoly

    def get_model_specs(self) -> list[ExperimentModelSpec]:
        stage1_lpoly = self._build_prefitted_stage1_lpoly()

        return [
            ExperimentModelSpec(
                name="Stage1_lpoly_deg2_k16_tric_only",
                model=LocalPolynomialSurfaceRegressor(
                    degree=2,
                    neighbors=16,
                    weight="tricube",
                ),
                train_grid=GridId.RO_GRID_4X4,
                eval_grids=EVAL_GRIDS,
                plots=PlotGroup.STANDARD_DIAGNOSTIC,
                keep_in_memory=True,
                extreme_scenario_label="Stage-1 LocalPoly deg2 k16 (4x4 only)",
            ),
            ExperimentModelSpec(
                name="TorchMLP_32x32_only",
                model=_make_champion_resnet_32x32(),
                train_grid=GridId.RO_GRID_32X32,
                eval_grids=EVAL_GRIDS,
                plots=PlotGroup.STANDARD_DIAGNOSTIC,
                keep_in_memory=True,
                extreme_scenario_label="TorchMLP ResNet h128 SiLU Huber (32x32 only)",
            ),
            ExperimentModelSpec(
                name="Hybrid_lpoly_deg2_k16_tric_plus_resnet_32x32",
                model=CompositeResidualShiftModel(
                    stage1_model=stage1_lpoly,
                    stage2_model=_make_champion_resnet_32x32(hidden_dim=128),
                ),
                train_grid=GridId.RO_GRID_32X32,
                eval_grids=EVAL_GRIDS,
                plots=PlotGroup.STANDARD_DIAGNOSTIC,
                keep_in_memory=True,
                extreme_scenario_label="LocalPoly k16 + ResNet h128 SiLU Huber (32x32)",
            ),
            ExperimentModelSpec(
                name="Hybrid_lpoly_deg2_k16_tric_plus_resnet_32x32_h256",
                model=CompositeResidualShiftModel(
                    stage1_model=stage1_lpoly,
                    stage2_model=_make_champion_resnet_32x32(hidden_dim=256),
                ),
                train_grid=GridId.RO_GRID_32X32,
                eval_grids=EVAL_GRIDS,
                plots=PlotGroup.STANDARD_DIAGNOSTIC,
                keep_in_memory=True,
                extreme_scenario_label="LocalPoly k16 + ResNet h256 SiLU Huber (32x32)",
            ),
            ExperimentModelSpec(
                name="Hybrid_lpoly_deg2_k16_tric_plus_resnet_32x32_h512",
                model=CompositeResidualShiftModel(
                    stage1_model=stage1_lpoly,
                    stage2_model=_make_champion_resnet_32x32(hidden_dim=512),
                ),
                train_grid=GridId.RO_GRID_32X32,
                eval_grids=EVAL_GRIDS,
                plots=PlotGroup.STANDARD_DIAGNOSTIC,
                keep_in_memory=True,
                extreme_scenario_label="LocalPoly k16 + ResNet h512 SiLU Huber (32x32)",
            ),
            ExperimentModelSpec(
                name="Hybrid_lpoly_deg2_k16_tric_plus_resnet_32x32_h256_ms_rff",
                model=CompositeResidualShiftModel(
                    stage1_model=stage1_lpoly,
                    stage2_model=_make_champion_resnet_32x32(
                        hidden_dim=256,
                        fourier_scales=(1.0, 3.0, 10.0),
                        num_fourier_frequencies=64,
                    ),
                ),
                train_grid=GridId.RO_GRID_32X32,
                eval_grids=EVAL_GRIDS,
                plots=PlotGroup.STANDARD_DIAGNOSTIC,
                keep_in_memory=True,
                extreme_scenario_label="LocalPoly k16 + ResNet h256 Multi-Scale RFF (32x32)",
                benchmark_candidate=True,
                benchmark_suite="grid_shift_models",
                benchmark_label="LocalPoly k16 + Grid-Free ResNet (32x32, h256, MS-RFF, SiLU, Huber)",
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
            baseline_model="Stage1_lpoly_deg2_k16_tric_only",
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
    HybridSurfaceResNetUnit(Path("experiments/hybrid_surface_resnet")).run()
