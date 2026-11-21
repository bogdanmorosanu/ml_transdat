"""Unit 06: Multi-tier evaluation of Adaptive Quadtree Grids on the corridor benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.core.base_unit import ExperimentModelSpec, GridShiftExperimentUnit
from src.data import GridId
from src.evaluation import EvaluationResult, MetricType
from src.models import AdaptiveQuadtreeRegressor, BaseGridShiftModel
from src.viz import PlotGroup, plot_model_comparison


class AdaptiveQuadtreeGridUnit(GridShiftExperimentUnit):
    """Grid Shift Unit: Curvature-driven 2D adaptive quadtree grid lookup."""

    @property
    def slug(self) -> str:
        return "adaptive_quadtree_grid"

    def get_model_specs(self) -> list[ExperimentModelSpec]:
        tiers = ["2cm", "1cm", "3mm", "1mm"]
        promoted_tiers = {"3mm", "1mm"}
        specs: list[ExperimentModelSpec] = []
        for tier in tiers:
            dn_ds, de_ds = self.grid_manager.get_adaptive_pair(tier)
            model = AdaptiveQuadtreeRegressor(dn_grid=dn_ds.df, de_grid=de_ds.df)
            specs.append(
                ExperimentModelSpec(
                    name=f"quadtree_{tier}",
                    model=model,
                    train_grid=None,
                    eval_grids=[GridId.CORRIDOR_SNW],
                    plots=PlotGroup.MINIMAL if tier == "1mm" else None,
                    benchmark_candidate=tier in promoted_tiers,
                    benchmark_suite="grid_shift_models"
                    if tier in promoted_tiers
                    else None,
                    benchmark_label=f"Adaptive Quadtree ({tier} tier)"
                    if tier in promoted_tiers
                    else None,
                    extra={"tier": tier, "leaf_count": len(dn_ds)},
                )
            )
        return specs

    def after_run(
        self,
        fitted_models: dict[str, BaseGridShiftModel],
        results: list[EvaluationResult],
    ) -> dict[str, Any]:
        comp_path = self.plots_dir / "tier_comparison.png"
        plot_model_comparison(
            results=results,
            out_path=comp_path,
            metrics=[MetricType.MEAN_2D_M, MetricType.P95_2D_M, MetricType.MAX_2D_M],
            title="Adaptive Quadtree Accuracy Across Tolerances (Corridor Benchmark, mm)",
            precision_unit="mm",
            log_scale=True,
        )
        return {
            "tiers": [r.model_name.replace("quadtree_", "") for r in results],
            "tier_comparison_plot": comp_path.name,
        }


def main() -> None:
    unit = AdaptiveQuadtreeGridUnit(Path("experiments/adaptive_quadtree_grid"))
    unit.run()


if __name__ == "__main__":
    main()
