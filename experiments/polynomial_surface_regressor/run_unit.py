"""Grid Shift Unit: 2D Polynomial Surface Regressor on SPG Grid Shifts."""

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
from src.viz import PlotGroup, plot_surface_comparison_grids, plot_transect_66km


class PolynomialSurfaceRegressorUnit(GridShiftExperimentUnit):
    """Grid Shift Unit: 2D Bivariate Polynomial Surface Distortion Regressors."""

    @property
    def slug(self) -> str:
        return "polynomial_surface_regressor"

    def get_model_specs(self) -> list[ExperimentModelSpec]:
        return [
            # 1. Degree 1: Flat tilted plane (3 terms, Helmert/Affine equivalent)
            ExperimentModelSpec(
                name="poly_deg1_11km",
                model=PolynomialSurfaceRegressor(degree=1, alpha=1e-8),
                train_grid=GridId.SPG_NODES_VALID,
                eval_grids=[
                    GridId.RO_TEST_UNIFORM_4X4,
                    GridId.RO_TEST_ADAPTIVE_3MM_MID,
                ],
                plots=PlotGroup.STANDARD_DIAGNOSTIC,
                keep_in_memory=True,
                benchmark_candidate=True,
                benchmark_suite="global_analytical_models",
                benchmark_label="Linear Affine Plane (Deg 1, 6T)",
            ),
            # 2. Degree 2: Quadratic bowl/saddle (6 terms)
            ExperimentModelSpec(
                name="poly_deg2_11km",
                model=PolynomialSurfaceRegressor(degree=2, alpha=1e-8),
                train_grid=GridId.SPG_NODES_VALID,
                eval_grids=[
                    GridId.RO_TEST_UNIFORM_4X4,
                    GridId.RO_TEST_ADAPTIVE_3MM_MID,
                ],
                plots=PlotGroup.STANDARD_DIAGNOSTIC,
                keep_in_memory=True,
                benchmark_candidate=True,
                benchmark_suite="global_analytical_models",
                benchmark_label="Quadratic Bowl (Deg 2, 12T)",
            ),
            # 3. Degree 3: Classical geodetic cubic surface (10 terms, inflection/S-curves)
            ExperimentModelSpec(
                name="poly_deg3_11km",
                model=PolynomialSurfaceRegressor(degree=3, alpha=1e-8),
                train_grid=GridId.SPG_NODES_VALID,
                eval_grids=[
                    GridId.RO_TEST_UNIFORM_4X4,
                    GridId.RO_TEST_ADAPTIVE_3MM_MID,
                ],
                plots=None,
                keep_in_memory=True,
                benchmark_candidate=True,
                benchmark_suite="global_analytical_models",
                benchmark_label="Cubic Surface (Deg 3, 20T)",
            ),
            # 4. Degree 5: Classical geodetic quintic surface (21 terms)
            ExperimentModelSpec(
                name="poly_deg5_11km",
                model=PolynomialSurfaceRegressor(degree=5, alpha=1e-8),
                train_grid=GridId.SPG_NODES_VALID,
                eval_grids=[
                    GridId.RO_TEST_UNIFORM_4X4,
                    GridId.RO_TEST_ADAPTIVE_3MM_MID,
                ],
                plots=PlotGroup.STANDARD_DIAGNOSTIC,
                keep_in_memory=True,
                benchmark_candidate=True,
                benchmark_suite="global_analytical_models",
                benchmark_label="Quintic Surface (Deg 5, 42T)",
            ),
            # 5. Degree 8: High-order multi-harmonic surface (45 terms)
            ExperimentModelSpec(
                name="poly_deg8_11km",
                model=PolynomialSurfaceRegressor(degree=8, alpha=1e-8),
                train_grid=GridId.SPG_NODES_VALID,
                eval_grids=[
                    GridId.RO_TEST_UNIFORM_4X4,
                    GridId.RO_TEST_ADAPTIVE_3MM_MID,
                ],
                plots=None,
                keep_in_memory=True,
                benchmark_candidate=True,
                benchmark_suite="global_analytical_models",
                benchmark_label="Multi-Harmonic (Deg 8, 90T)",
            ),
            # 6. Degree 12: High-order polynomial limit on 11 km nodes (91 terms)
            ExperimentModelSpec(
                name="poly_deg12_11km",
                model=PolynomialSurfaceRegressor(degree=12, alpha=1e-8),
                train_grid=GridId.SPG_NODES_VALID,
                eval_grids=[
                    GridId.RO_TEST_UNIFORM_4X4,
                    GridId.RO_TEST_ADAPTIVE_3MM_MID,
                ],
                plots=PlotGroup.STANDARD_DIAGNOSTIC,
                keep_in_memory=True,
                benchmark_candidate=True,
                benchmark_suite="global_analytical_models",
                benchmark_label="High-Order Limit (Deg 12, 182T)",
            ),
            # 7. Degree 5 trained on densified 4x4 subgrid (~36.4k points, 2.7 km)
            ExperimentModelSpec(
                name="poly_deg5_4x4",
                model=PolynomialSurfaceRegressor(degree=5, alpha=1e-8),
                train_grid=GridId.RO_GRID_4X4,
                eval_grids=[
                    GridId.RO_TEST_UNIFORM_4X4,
                    GridId.RO_TEST_ADAPTIVE_3MM_MID,
                ],
                plots=None,
                keep_in_memory=True,
            ),
            # 8. Degree 12 trained on densified 4x4 subgrid (~36.4k points, 2.7 km)
            ExperimentModelSpec(
                name="poly_deg12_4x4",
                model=PolynomialSurfaceRegressor(degree=12, alpha=1e-8),
                train_grid=GridId.RO_GRID_4X4,
                eval_grids=[
                    GridId.RO_TEST_UNIFORM_4X4,
                    GridId.RO_TEST_ADAPTIVE_3MM_MID,
                ],
                plots=None,
                keep_in_memory=True,
            ),
            # 9. Minimalist Sparse Lasso on Degree 5 (13 terms dN, 11 terms dE)
            ExperimentModelSpec(
                name="poly_deg5_sparse_lasso",
                model=PolynomialSurfaceRegressor(degree=5, alpha=5e-4, regressor="lasso"),
                train_grid=GridId.SPG_NODES_VALID,
                eval_grids=[
                    GridId.RO_TEST_UNIFORM_4X4,
                    GridId.RO_TEST_ADAPTIVE_3MM_MID,
                ],
                plots=PlotGroup.STANDARD_DIAGNOSTIC,
                keep_in_memory=True,
                benchmark_candidate=True,
                benchmark_suite="global_analytical_models",
                benchmark_label="Sparse Lasso (Deg 5, 24T)",
            ),
            # 10. Minimalist Sparse Lasso on Degree 12 (21 terms dN, 25 terms dE)
            ExperimentModelSpec(
                name="poly_deg12_sparse_lasso",
                model=PolynomialSurfaceRegressor(degree=12, alpha=1e-4, regressor="lasso"),
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

        # 1. 66 km Transect Comparison Plot
        transect_path = self.plots_dir / "cell_transect_66km_polynomial.png"
        plot_transect_66km(
            models={
                "Flat Plane (Deg 1)": fitted_models["poly_deg1_11km"],
                "Quadratic (Deg 2)": fitted_models["poly_deg2_11km"],
                "Cubic (Deg 3)": fitted_models["poly_deg3_11km"],
                "Quintic (Deg 5)": fitted_models["poly_deg5_11km"],
                "Poly Deg 12 (11 km)": fitted_models["poly_deg12_11km"],
                "Poly Deg 12 (4x4)": fitted_models["poly_deg12_4x4"],
            },
            out_path=transect_path,
            title="2D Polynomial Surface Progression Across 66 km Transect vs. Bicubic Ground Truth",
        )
        out_plots["transect_plot"] = transect_path.name

        # 2. Comprehensive Degree Scaling Plot (Degrees 1 to 14)
        scaling_path = self.plots_dir / "polynomial_degree_scaling.png"
        self._plot_degree_scaling(scaling_path)
        out_plots["degree_scaling_plot"] = scaling_path.name

        # 3. 3D Surface & 2D Contour Comparison Grids
        p3d_path = self.plots_dir / "polynomial_3d_surfaces_comparison.png"
        p2d_path = self.plots_dir / "polynomial_2d_heatmaps_comparison.png"
        self._plot_surface_grids(p3d_path, p2d_path, fitted_models)
        out_plots["surfaces_3d_plot"] = p3d_path.name
        out_plots["heatmaps_2d_plot"] = p2d_path.name

        return out_plots

    def _plot_surface_grids(
        self,
        out_3d_path: Path,
        out_2d_path: Path,
        fitted_models: dict[str, BaseGridShiftModel],
    ) -> None:
        """Render 2x3 3D perspective surface and 2D contour comparison grids."""
        ds_4x4 = self.load_grid(GridId.RO_GRID_4X4)
        models_info: list[tuple[str, BaseGridShiftModel | None, str]] = [
            ("1. Ground Truth (Bicubic SPG)", None, "Real physical distortion field (Banat dome 2.76m)"),
            ("2. Degree 1: Flat Tilted Plane (3 terms)", fitted_models["poly_deg1_11km"], "Rigid flat board (Helmert equivalent, RMSE 42.8cm)"),
            ("3. Degree 2: Quadratic Bowl / Saddle (6 terms)", fitted_models["poly_deg2_11km"], "Bends North-South & East-West + twist (RMSE 33.0cm)"),
            ("4. Degree 3: Cubic Inflection (10 terms)", fitted_models["poly_deg3_11km"], "Introduces S-curves across Carpathians (RMSE 29.5cm)"),
            ("5. Degree 5: Quintic Undulation (21 terms)", fitted_models["poly_deg5_11km"], "Classical geodetic surface; cuts max error by 76cm (RMSE 27.0cm)"),
            ("6. Degree 12: High-Order Polynomial (91 terms)", fitted_models["poly_deg12_11km"], "Detailed macroscopic fit; sub-meter max error (RMSE 19.2cm)"),
        ]
        plot_surface_comparison_grids(
            models_info=models_info,
            ref_dataset=ds_4x4,
            out_3d_path=out_3d_path,
            out_2d_path=out_2d_path,
            title_3d="3D Perspective Surface Progression: How 2D Polynomials Bend from Flat Plane to Wavy Surface",
            title_2d="2D Plan-View Distortion Contours: Progression Across Polynomial Degrees",
        )

    def _plot_degree_scaling(self, out_path: Path) -> None:
        """Evaluate and plot degree scaling curve from D=1 to D=14."""
        train_ds = self.load_grid(GridId.SPG_NODES_VALID)
        test_uni = self.load_grid(GridId.RO_TEST_UNIFORM_4X4)
        test_adp = self.load_grid(GridId.RO_TEST_ADAPTIVE_3MM_MID)

        degrees = list(range(1, 15))
        uni_rmses: list[float] = []
        adp_rmses: list[float] = []
        uni_maxes: list[float] = []
        adp_maxes: list[float] = []
        term_counts: list[int] = []

        for deg in degrees:
            model = PolynomialSurfaceRegressor(degree=deg, alpha=1e-8)
            model.fit(train_ds.coords, train_ds.true_shifts)
            term_counts.append(model.n_terms)

            # Uniform 4x4
            pred_uni = model.predict(test_uni.coords)
            err_uni = np.linalg.norm(pred_uni - test_uni.true_shifts, axis=1)
            uni_rmses.append(float(np.sqrt(np.mean(err_uni**2)) * 100))  # in cm
            uni_maxes.append(float(np.max(err_uni)))  # in m

            # Adaptive 3mm
            pred_adp = model.predict(test_adp.coords)
            err_adp = np.linalg.norm(pred_adp - test_adp.true_shifts, axis=1)
            adp_rmses.append(float(np.sqrt(np.mean(err_adp**2)) * 100))  # in cm
            adp_maxes.append(float(np.max(err_adp)))  # in m

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

        # Panel 1: RMSE Scaling (cm)
        ax1.plot(degrees, uni_rmses, "o-", color="#2563eb", linewidth=2.0, label="Uniform 4x4 (Nationwide)")
        ax1.plot(degrees, adp_rmses, "s--", color="#dc2626", linewidth=2.0, label="Adaptive 3mm (High Curvature)")
        ax1.axhline(44.29, color="#64748b", linestyle=":", label="Helmert 2D Floor (44.3 cm)")
        ax1.set_xlabel("Polynomial Degree (D)", fontsize=11, fontweight="bold")
        ax1.set_ylabel("2D Error RMSE (cm)", fontsize=11, fontweight="bold")
        ax1.set_title("2D Polynomial Surface Accuracy vs. Degree", fontsize=12, fontweight="bold")
        ax1.set_xticks(degrees)
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(loc="upper right", framealpha=0.9)

        # Annotate terms
        for d, u_rmse, terms in zip(degrees, uni_rmses, term_counts, strict=True):
            if d in [1, 2, 3, 5, 8, 12, 14]:
                ax1.annotate(
                    f"{u_rmse:.1f}cm\n({terms}T)",
                    xy=(d, u_rmse),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    fontweight="bold",
                    color="#1e3a8a",
                )

        # Panel 2: Max Error Scaling (m)
        ax2.plot(degrees, uni_maxes, "o-", color="#059669", linewidth=2.0, label="Uniform 4x4 Max Error (m)")
        ax2.plot(degrees, adp_maxes, "s--", color="#ea580c", linewidth=2.0, label="Adaptive 3mm Max Error (m)")
        ax2.axhline(2.68, color="#64748b", linestyle=":", label="Helmert 2D Max Floor (2.68 m)")
        ax2.set_xlabel("Polynomial Degree (D)", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Maximum 2D Error (m)", fontsize=11, fontweight="bold")
        ax2.set_title("Worst-Case Boundary Error vs. Degree", fontsize=12, fontweight="bold")
        ax2.set_xticks(degrees)
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(loc="upper right", framealpha=0.9)

        fig.suptitle(
            "Scaling Laws of 2D Polynomial Surfaces on Romania's SPG Distortion Field",
            fontsize=13.5,
            fontweight="bold",
        )
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)


def main() -> None:
    unit = PolynomialSurfaceRegressorUnit(Path("experiments/polynomial_surface_regressor"))
    unit.run()


if __name__ == "__main__":
    main()
