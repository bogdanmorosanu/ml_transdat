"""Grid Shift Unit: 2D Radial Basis Function & Thin Plate Spline Surface Regressor on SPG Grid Shifts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.core.base_unit import ExperimentModelSpec, GridShiftExperimentUnit
from src.data import GridId
from src.evaluation import EvaluationResult
from src.models.base import BaseGridShiftModel
from src.models.polynomial_surface import PolynomialSurfaceRegressor
from src.models.rbf_surface import RBFSurfaceRegressor
from src.viz import PlotGroup, plot_surface_comparison_grids, plot_transect_66km


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
                plots=None,
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
                plots=None,
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
                plots=None,
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

        # 4. Foundation & Kernel Comparison Plot
        comp_path = self.plots_dir / "rbf_foundation_and_smoothing.png"
        self._plot_foundation_and_smoothing(comp_path, fitted_models)
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

    def _plot_foundation_and_smoothing(
        self,
        out_path: Path,
        fitted_models: dict[str, BaseGridShiftModel],
    ) -> None:
        """Plot comparison across all evaluated kernel architectures."""
        test_uni = self.load_grid(GridId.RO_TEST_UNIFORM_4X4)
        test_adp = self.load_grid(GridId.RO_TEST_ADAPTIVE_3MM_MID)

        model_keys = [
            ("TPS Base Exact", "tps_base_exact"),
            ("TPS Base (1e-4)", "tps_base_smooth_1e-4"),
            ("TPS Base Opt (2e-4)", "tps_base_optimal_2e-4"),
            ("TPS Densified (4x4)", "tps_densified_4x4"),
            ("Linear (Scale-Free)", "linear_base"),
            ("Tuned MQ (eps=10)", "mq_tuned_eps10"),
            ("Tuned Gauss (eps=40)", "gaussian_tuned_eps40"),
            ("Cubic (Overshoot)", "cubic_base"),
        ]

        names = [k[0] for k in model_keys]
        uni_rmses = []
        adp_rmses = []

        for _, key in model_keys:
            m = fitted_models[key]
            # Uni
            p_u = m.predict(test_uni.coords)
            e_u = np.linalg.norm(p_u - test_uni.true_shifts, axis=1)
            uni_rmses.append(float(np.sqrt(np.mean(e_u**2)) * 1000.0))  # in mm
            # Adp
            p_a = m.predict(test_adp.coords)
            e_a = np.linalg.norm(p_a - test_adp.true_shifts, axis=1)
            adp_rmses.append(float(np.sqrt(np.mean(e_a**2)) * 1000.0))  # in mm

        fig, ax = plt.subplots(figsize=(13, 6))
        x = np.arange(len(names))
        width = 0.38

        rects1 = ax.bar(x - width / 2, uni_rmses, width, label="Uniform 4x4 (Nationwide)", color="#2563eb", alpha=0.88)
        rects2 = ax.bar(x + width / 2, adp_rmses, width, label="Adaptive 3mm (High Curvature)", color="#dc2626", alpha=0.88)

        ax.axhline(191.8, color="#ea580c", linestyle="--", linewidth=1.5, label="Level 2 Poly Deg 12 Floor (191.8 mm)")
        ax.set_ylabel("2D RMSE (millimeters)", fontsize=11, fontweight="bold")
        ax.set_title("Level 3 Comprehensive Kernel Shootout & Density Ladder vs. Level 2 Floor", fontsize=12.5, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9.5, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.5, axis="y")
        ax.legend(loc="upper right", framealpha=0.9)

        # Annotate values
        for rect in rects1:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
        for rect in rects2:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)


def main() -> None:
    unit = RBFSurfaceRegressorUnit(Path("experiments/rbf_surface_regressor"))
    unit.run()


if __name__ == "__main__":
    main()
