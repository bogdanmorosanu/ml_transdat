"""Grid Shift Unit: Optimal Surface Regressor — Stage 3 Consolidated Family Champions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.core.base_unit import ExperimentModelSpec, GridShiftExperimentUnit
from src.data import GridId
from src.evaluation import EvaluationResult
from src.models import (
    BSplineSurfaceRegressor,
    CellBilinearShiftModel,
    LocalPolynomialSurfaceRegressor,
    ParameterizedBicubicGridShiftModel,
    PolynomialSurfaceRegressor,
    RBFSurfaceRegressor,
)
from src.viz import PlotGroup, plot_surface_learnability_ranking


class OptimalSurfaceRegressorUnit(GridShiftExperimentUnit):
    """Grid Shift Unit: Cross-family Stage-2 Learnability comparison of 10 Family Champions."""

    render_extreme_scenarios: bool = True

    @property
    def slug(self) -> str:
        return "optimal_surface_regressor"

    def get_model_specs(self) -> list[ExperimentModelSpec]:
        eval_grids = [
            GridId.RO_GRID_4X4,
            GridId.RO_TEST_UNIFORM_16X16,
            GridId.RO_TEST_ADAPTIVE_1MM_MID,
        ]
        return [
            # 1. Polyharmonic (Cubic) Champion
            ExperimentModelSpec(
                name="cubic_k25_s0",
                model=RBFSurfaceRegressor(
                    kernel="cubic",
                    degree=1,
                    smoothing=0.0,
                    neighbors=25,
                    normalize_coords=True,
                ),
                train_grid=GridId.RO_GRID_4X4,
                eval_grids=eval_grids,
                plots=PlotGroup.SURFACE_DIAGNOSTIC,
                keep_in_memory=True,
            ),
            # 2. Thin Plate Spline (TPS) Champion
            ExperimentModelSpec(
                name="tps_deg2_k25_s1e-5",
                model=RBFSurfaceRegressor(
                    kernel="thin_plate_spline",
                    degree=2,
                    smoothing=1e-5,
                    neighbors=25,
                    normalize_coords=True,
                ),
                train_grid=GridId.RO_GRID_4X4,
                eval_grids=eval_grids,
                plots=PlotGroup.SURFACE_DIAGNOSTIC,
                keep_in_memory=True,
            ),
            # 3. Local Polynomial (2D LOESS) Champion
            ExperimentModelSpec(
                name="lpoly_deg2_k16_tric",
                model=LocalPolynomialSurfaceRegressor(
                    degree=2,
                    neighbors=16,
                    weight="tricube",
                ),
                train_grid=GridId.RO_GRID_4X4,
                eval_grids=eval_grids,
                plots=PlotGroup.SURFACE_DIAGNOSTIC,
                keep_in_memory=True,
            ),
            # 4. Matérn Champion
            ExperimentModelSpec(
                name="matern32_k16_eps5_s1e-5",
                model=RBFSurfaceRegressor(
                    kernel="matern32",
                    epsilon=5.0,
                    smoothing=1e-5,
                    neighbors=16,
                    normalize_coords=True,
                ),
                train_grid=GridId.RO_GRID_4X4,
                eval_grids=eval_grids,
                plots=PlotGroup.SURFACE_DIAGNOSTIC,
                keep_in_memory=True,
            ),
            # 5. 11 km Smooth Grid Spline Champion
            ExperimentModelSpec(
                name="grid11km_natural_spline_c2",
                model=ParameterizedBicubicGridShiftModel(scheme="natural_spline_c2"),
                train_grid=GridId.SPG_NODES_VALID,
                eval_grids=eval_grids,
                plots=PlotGroup.SURFACE_DIAGNOSTIC,
                keep_in_memory=True,
            ),
            # 6. Inverse Multiquadric Champion
            ExperimentModelSpec(
                name="imq_deg2_k16_eps25_s1e-4",
                model=RBFSurfaceRegressor(
                    kernel="inverse_multiquadric",
                    degree=2,
                    epsilon=25.0,
                    smoothing=1e-4,
                    neighbors=16,
                    normalize_coords=True,
                ),
                train_grid=GridId.RO_GRID_4X4,
                eval_grids=eval_grids,
                plots=PlotGroup.SURFACE_DIAGNOSTIC,
                keep_in_memory=True,
            ),
            # 7. Bilinear Cell Base Reference
            ExperimentModelSpec(
                name="cell_bilinear_11km",
                model=CellBilinearShiftModel(),
                train_grid=GridId.SPG_NODES_VALID,
                eval_grids=eval_grids,
                plots=PlotGroup.SURFACE_DIAGNOSTIC,
                keep_in_memory=True,
            ),
            # 8. Gaussian RBF Champion
            ExperimentModelSpec(
                name="gauss_deg2_k16_eps40_s1e-4",
                model=RBFSurfaceRegressor(
                    kernel="gaussian",
                    degree=2,
                    epsilon=40.0,
                    smoothing=1e-4,
                    neighbors=16,
                    normalize_coords=True,
                ),
                train_grid=GridId.RO_GRID_4X4,
                eval_grids=eval_grids,
                plots=PlotGroup.SURFACE_DIAGNOSTIC,
                keep_in_memory=True,
            ),
            # 9. Global Polynomial Champion (metrics only; >200 mm error omitted from row plots)
            ExperimentModelSpec(
                name="global_poly_deg12",
                model=PolynomialSurfaceRegressor(degree=12, alpha=1e-8),
                train_grid=GridId.RO_GRID_4X4,
                eval_grids=eval_grids,
                plots=None,
                keep_in_memory=False,
            ),
            # 10. Tensor-Product B-Spline Champion (metrics only; >900 mm error omitted from row plots)
            ExperimentModelSpec(
                name="bspline_deg5_s0p05",
                model=BSplineSurfaceRegressor(
                    degree=5,
                    smoothing=0.05,
                    normalize_coords=True,
                ),
                train_grid=GridId.RO_GRID_4X4,
                eval_grids=eval_grids,
                plots=None,
                keep_in_memory=False,
            ),
        ]

    def after_run(
        self,
        fitted_models: dict[str, Any],
        results: list[EvaluationResult],
    ) -> dict[str, Any]:
        ranking_df = self.evaluator.rank_surface_candidates(results)
        ranking_csv = self.results_dir / "learnability_ranking.csv"
        ranking_df.to_csv(ranking_csv, index=False)

        viable_results = [
            r
            for r in results
            if r.model_name not in {"global_poly_deg12", "bspline_deg5_s0p05"}
        ]
        p_u16 = plot_surface_learnability_ranking(
            results=viable_results,
            out_path=self.plots_dir / "surface_learnability_ranking_16x16.png",
            dataset_id=GridId.RO_TEST_UNIFORM_16X16.value,
            title="Cross-Family Stage-2 Learnability Ranking (Uniform 16x16)",
        )
        p_a1 = plot_surface_learnability_ranking(
            results=viable_results,
            out_path=self.plots_dir / "surface_learnability_ranking_adaptive_1mm.png",
            dataset_id=GridId.RO_TEST_ADAPTIVE_1MM_MID.value,
            title="Cross-Family Stage-2 Learnability Ranking (Adaptive 1mm)",
        )
        return {
            "learnability_ranking_csv": ranking_csv.name,
            "ranking_16x16_plot": p_u16.name,
            "ranking_adaptive_1mm_plot": p_a1.name,
        }


def main() -> None:
    unit = OptimalSurfaceRegressorUnit(Path("experiments/optimal_surface_regressor"))
    unit.run()


if __name__ == "__main__":
    main()
