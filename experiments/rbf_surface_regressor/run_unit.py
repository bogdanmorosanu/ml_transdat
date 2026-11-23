"""Grid Shift Unit: 2D Radial Basis Function & Thin Plate Spline Surface Regressor on SPG Grid Shifts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.core.base_unit import ExperimentModelSpec, GridShiftExperimentUnit
from src.data import GridId
from src.evaluation import EvaluationResult, MetricType
from src.models.base import BaseGridShiftModel
from src.models.polynomial_surface import PolynomialSurfaceRegressor
from src.models.rbf_surface import RBFSurfaceRegressor
from src.viz import (
    PlotGroup,
    plot_multi_dataset_comparison,
    plot_surface_comparison_grids,
    plot_transect_66km,
)


class RBFSurfaceRegressorUnit(GridShiftExperimentUnit):
    """Grid Shift Unit: 2D RBF & Thin Plate Spline Localized Continuous Surface Regressors."""

    @property
    def slug(self) -> str:
        return "rbf_surface_regressor"

    def get_model_specs(self) -> list[ExperimentModelSpec]:
        return [
            # 1. TPS with Affine Trend (Deg 1), Exact Pinning (lambda = 0.0) on 2,222 Base Nodes
            ExperimentModelSpec(
                name="tps_base_exact",
                model=RBFSurfaceRegressor(kernel="thin_plate_spline", degree=1, smoothing=0.0),
                train_grid=GridId.SPG_NODES_VALID,
                eval_grids=[
                    GridId.RO_TEST_UNIFORM_4X4,
                    GridId.RO_TEST_ADAPTIVE_3MM_MID,
                ],
                plots=PlotGroup.STANDARD_DIAGNOSTIC,
                keep_in_memory=True,
            ),
            # 2. TPS with Affine Trend (Deg 1), Gentle Smoothing (lambda = 1e-4) on 2,222 Base Nodes
            ExperimentModelSpec(
                name="tps_base_smooth_1e-4",
                model=RBFSurfaceRegressor(kernel="thin_plate_spline", degree=1, smoothing=1e-4),
                train_grid=GridId.SPG_NODES_VALID,
                eval_grids=[
                    GridId.RO_TEST_UNIFORM_4X4,
                    GridId.RO_TEST_ADAPTIVE_3MM_MID,
                ],
                plots=None,
                keep_in_memory=True,
            ),
            # 3. TPS with Affine Trend, Optimal Smoothing (lambda = 2e-4) on 2,222 Base Nodes
            ExperimentModelSpec(
                name="tps_base_optimal_2e-4",
                model=RBFSurfaceRegressor(kernel="thin_plate_spline", degree=1, smoothing=2e-4),
                train_grid=GridId.SPG_NODES_VALID,
                eval_grids=[
                    GridId.RO_TEST_UNIFORM_4X4,
                    GridId.RO_TEST_ADAPTIVE_3MM_MID,
                ],
                plots=PlotGroup.STANDARD_DIAGNOSTIC,
                keep_in_memory=True,
                benchmark_candidate=True,
                benchmark_suite="grid_shift_models",
                benchmark_label="Thin Plate Spline (11km)",
            ),
            # 4. TPS on Densified 4x4 Subgrid (36,435 anchors, neighbors=60)
            ExperimentModelSpec(
                name="tps_densified_4x4",
                model=RBFSurfaceRegressor(kernel="thin_plate_spline", degree=1, smoothing=1e-4, neighbors=60),
                train_grid=GridId.RO_GRID_4X4,
                eval_grids=[
                    GridId.RO_TEST_UNIFORM_4X4,
                    GridId.RO_TEST_ADAPTIVE_3MM_MID,
                ],
                plots=PlotGroup.STANDARD_DIAGNOSTIC,
                keep_in_memory=True,
                benchmark_candidate=True,
                benchmark_suite="grid_shift_models",
                benchmark_label="Thin Plate Spline (4x4, 2.7km)",
            ),
            # 5. Tuned Multiquadric RBF (epsilon = 10.0, lambda = 1e-4)
            ExperimentModelSpec(
                name="mq_tuned_eps10",
                model=RBFSurfaceRegressor(kernel="multiquadric", degree=1, epsilon=10.0, smoothing=1e-4),
                train_grid=GridId.SPG_NODES_VALID,
                eval_grids=[
                    GridId.RO_TEST_UNIFORM_4X4,
                    GridId.RO_TEST_ADAPTIVE_3MM_MID,
                ],
                plots=PlotGroup.STANDARD_DIAGNOSTIC,
                keep_in_memory=True,
            ),
            # 6. Tuned Gaussian RBF (epsilon = 40.0, lambda = 1e-4)
            ExperimentModelSpec(
                name="gaussian_tuned_eps40",
                model=RBFSurfaceRegressor(kernel="gaussian", degree=1, epsilon=40.0, smoothing=1e-4),
                train_grid=GridId.SPG_NODES_VALID,
                eval_grids=[
                    GridId.RO_TEST_UNIFORM_4X4,
                    GridId.RO_TEST_ADAPTIVE_3MM_MID,
                ],
                plots=None,
                keep_in_memory=True,
            ),
            # 7. Scale-Free Linear Kernel (phi(r) = r, lambda = 1e-4)
            ExperimentModelSpec(
                name="linear_base",
                model=RBFSurfaceRegressor(kernel="linear", degree=1, smoothing=1e-4),
                train_grid=GridId.SPG_NODES_VALID,
                eval_grids=[
                    GridId.RO_TEST_UNIFORM_4X4,
                    GridId.RO_TEST_ADAPTIVE_3MM_MID,
                ],
                plots=None,
                keep_in_memory=True,
            ),
            # 8. Scale-Free Cubic Kernel (phi(r) = r^3, lambda = 1e-4)
            ExperimentModelSpec(
                name="cubic_base",
                model=RBFSurfaceRegressor(kernel="cubic", degree=1, smoothing=1e-4),
                train_grid=GridId.SPG_NODES_VALID,
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
        out_plots: dict[str, str] = {}

        # 1. Fit reference models for side-by-side diagnostic visualization
        train_ds = self.load_grid(GridId.SPG_NODES_VALID)
        poly_deg1 = PolynomialSurfaceRegressor(degree=1, alpha=1e-8)
        poly_deg1.fit(train_ds.coords, train_ds.true_shifts)
        poly_deg12 = PolynomialSurfaceRegressor(degree=12, alpha=1e-8)
        poly_deg12.fit(train_ds.coords, train_ds.true_shifts)

        # 2. 66 km Transect Comparison Plot
        transect_path = self.plots_dir / "cell_transect_66km_rbf.png"
        plot_transect_66km(
            models={
                "Flat Plane (Deg 1)": poly_deg1,
                "Poly Deg 12 (11 km)": poly_deg12,
                "TPS Base Exact": fitted_models["tps_base_exact"],
                "TPS Base Smooth (2e-4)": fitted_models["tps_base_optimal_2e-4"],
                "TPS Densified (4x4)": fitted_models["tps_densified_4x4"],
                "Tuned MQ (eps=10)": fitted_models["mq_tuned_eps10"],
            },
            out_path=transect_path,
            title="Level 3 RBF & Thin Plate Spline Progression Across 66 km Transect vs. Ground Truth",
        )
        out_plots["transect_plot"] = transect_path.name

        # 3. 3D Surface & 2D Contour Comparison Grids
        p3d_path = self.plots_dir / "rbf_3d_surfaces_comparison.png"
        p2d_path = self.plots_dir / "rbf_2d_heatmaps_comparison.png"
        self._plot_surface_grids(p3d_path, p2d_path, fitted_models, poly_deg1, poly_deg12)
        out_plots["surfaces_3d_plot"] = p3d_path.name
        out_plots["heatmaps_2d_plot"] = p2d_path.name

        # 4. Foundation & Kernel Comparison Plot (reusing EvaluationResults with zero re-inference)
        comp_path = self.plots_dir / "rbf_foundation_and_smoothing.png"
        plot_multi_dataset_comparison(
            results=results,
            out_path=comp_path,
            dataset_labels={
                GridId.RO_TEST_UNIFORM_4X4.value: "Uniform 4x4 (Nationwide)",
                GridId.RO_TEST_ADAPTIVE_3MM_MID.value: "Adaptive 3mm (High Curvature)",
            },
            model_labels={
                "tps_base_exact": "TPS Base Exact",
                "tps_base_smooth_1e-4": "TPS Base (1e-4)",
                "tps_base_optimal_2e-4": "TPS Base Opt (2e-4)",
                "tps_densified_4x4": "TPS Densified (4x4)",
                "linear_base": "Linear (Scale-Free)",
                "mq_tuned_eps10": "Tuned MQ (eps=10)",
                "gaussian_tuned_eps40": "Tuned Gauss (eps=40)",
                "cubic_base": "Cubic (Overshoot)",
            },
            metric=MetricType.RMSE_2D_M,
            precision_unit="mm",
            reference_lines=[(191.8, "Level 2 Poly Deg 12 Floor (191.8 mm)", "#ea580c")],
            title="Level 3 Comprehensive Kernel Shootout & Density Ladder vs. Level 2 Floor",
        )
        out_plots["foundation_plot"] = comp_path.name

        return out_plots

    def _plot_surface_grids(
        self,
        out_3d_path: Path,
        out_2d_path: Path,
        fitted_models: dict[str, BaseGridShiftModel],
        poly_deg1: BaseGridShiftModel,
        poly_deg12: BaseGridShiftModel,
    ) -> None:
        """Render 2x3 3D perspective surface and 2D contour comparison grids."""
        ds_4x4 = self.load_grid(GridId.RO_GRID_4X4)
        models_info: list[tuple[str, BaseGridShiftModel | None, str]] = [
            ("1. Ground Truth (Bicubic SPG)", None, "Real physical distortion field (Banat dome 2.76m)"),
            ("2. Level 1: Flat Tilted Plane", poly_deg1, "Rigid flat plane (Helmert equivalent, RMSE 42.8cm)"),
            ("3. Level 2: Polynomial Deg 12", poly_deg12, "Macroscopic curved fit (RMSE 19.2cm, edge ripples)"),
            ("4. Level 3: TPS Base Smooth (11km)", fitted_models["tps_base_optimal_2e-4"], "Thin Plate Spline on 2,222 base anchors (RMSE 8.6mm)"),
            ("5. Level 3: TPS Densified (4x4)", fitted_models["tps_densified_4x4"], "Thin Plate Spline on 36,435 anchors (Mean 3.8mm)"),
            ("6. Level 3: Tuned MQ (eps=10)", fitted_models["mq_tuned_eps10"], "Tuned Hardy Multiquadric RBF (RMSE 14.0mm)"),
        ]
        plot_surface_comparison_grids(
            models_info=models_info,
            ref_dataset=ds_4x4,
            out_3d_path=out_3d_path,
            out_2d_path=out_2d_path,
            title_3d="Level 3 Surface Progression: Flat Plane vs. Polynomial vs. Elastic Sheet (RBF / TPS)",
            title_2d="2D Plan-View Contours: Level 1 (Helmert) vs. Level 2 (Polynomial) vs. Level 3 (RBF / TPS)",
        )

def main() -> None:
    unit = RBFSurfaceRegressorUnit(Path("experiments/rbf_surface_regressor"))
    unit.run()


if __name__ == "__main__":
    main()
