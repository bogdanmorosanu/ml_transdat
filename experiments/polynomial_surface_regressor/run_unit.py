"""Unit 07: 2D Polynomial Surface Regressor on SPG Grid Shifts."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
from src.viz import PlotGroup, plot_transect_66km


class PolynomialSurfaceRegressorUnit(GridShiftExperimentUnit):
    """Research Unit 07: 2D Bivariate Polynomial Surface Distortion Regressors."""

    @property
    def unit_id(self) -> str:
        return "07"

    @property
    def unit_name(self) -> str:
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
                plots=None,
                keep_in_memory=True,
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
                plots=None,
                keep_in_memory=True,
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
                plots=None,
                keep_in_memory=True,
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
        from matplotlib.colors import Normalize
        from scipy.interpolate import griddata

        ds_4x4 = self.load_grid(GridId.RO_GRID_4X4)

        coords_4x4 = ds_4x4.coords
        mag_4x4 = np.hypot(ds_4x4.true_shifts[:, 0], ds_4x4.true_shifts[:, 1])

        n_min, n_max = coords_4x4[:, 0].min(), coords_4x4[:, 0].max()
        e_min, e_max = coords_4x4[:, 1].min(), coords_4x4[:, 1].max()

        n_vec = np.linspace(n_min, n_max, 80)
        e_vec = np.linspace(e_min, e_max, 100)
        grid_n, grid_e = np.meshgrid(n_vec, e_vec, indexing="ij")

        gt_mag = griddata(coords_4x4, mag_4x4, (grid_n, grid_e), method="linear")
        mask = ~np.isnan(gt_mag)

        x_km = (grid_e - e_min) / 1000.0
        y_km = (grid_n - n_min) / 1000.0
        query_coords = np.column_stack([grid_n.ravel(), grid_e.ravel()])

        models_info = [
            ("1. Ground Truth (Bicubic SPG)", None, gt_mag, "Real physical distortion field (Banat dome 2.76m)"),
            ("2. Degree 1: Flat Tilted Plane (3 terms)", "poly_deg1_11km", None, "Rigid flat board (Helmert equivalent, RMSE 42.8cm)"),
            ("3. Degree 2: Quadratic Bowl / Saddle (6 terms)", "poly_deg2_11km", None, "Bends North-South & East-West + twist (RMSE 33.0cm)"),
            ("4. Degree 3: Cubic Inflection (10 terms)", "poly_deg3_11km", None, "Introduces S-curves across Carpathians (RMSE 29.5cm)"),
            ("5. Degree 5: Quintic Undulation (21 terms)", "poly_deg5_11km", None, "Classical geodetic surface; cuts max error by 76cm (RMSE 27.0cm)"),
            ("6. Degree 12: High-Order Polynomial (91 terms)", "poly_deg12_11km", None, "Detailed macroscopic fit; sub-meter max error (RMSE 19.2cm)"),
        ]

        surfaces = []
        for title, model_key, direct_mag, desc in models_info:
            if direct_mag is not None:
                surfaces.append((title, direct_mag, desc))
            else:
                m = fitted_models[model_key]
                pred = m.predict(query_coords)
                pred_mag = np.hypot(pred[:, 0], pred[:, 1]).reshape(grid_n.shape)
                pred_mag[~mask] = np.nan
                surfaces.append((title, pred_mag, desc))

        norm = Normalize(vmin=0.0, vmax=2.8)
        cmap = plt.cm.turbo

        # 3D Grid
        fig_3d = plt.figure(figsize=(19, 11.5))
        for idx, (title, Z, desc) in enumerate(surfaces):
            ax = fig_3d.add_subplot(2, 3, idx + 1, projection="3d")
            ax.plot_surface(
                x_km, y_km, Z,
                cmap=cmap,
                norm=norm,
                linewidth=0,
                antialiased=True,
                rstride=1,
                cstride=1,
                alpha=0.92,
            )
            ax.contour(
                x_km, y_km, Z,
                zdir="z",
                offset=0.0,
                cmap=cmap,
                norm=norm,
                levels=12,
                alpha=0.35,
                linewidths=0.7,
            )
            ax.set_title(f"{title}\n{desc}", fontsize=10.5, fontweight="bold", pad=12)
            ax.set_xlabel("Easting (km)", fontsize=9, labelpad=6)
            ax.set_ylabel("Northing (km)", fontsize=9, labelpad=6)
            ax.set_zlabel("2D Shift (m)", fontsize=9, labelpad=6)
            ax.set_zlim(0.0, 2.8)
            ax.view_init(elev=28, azim=-55)
            ax.grid(True, linestyle=":", alpha=0.4)

        fig_3d.subplots_adjust(right=0.91, hspace=0.25, wspace=0.18, top=0.90, bottom=0.06)
        cbar_ax_3d = fig_3d.add_axes([0.93, 0.15, 0.015, 0.70])
        cbar_3d = fig_3d.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax_3d)
        cbar_3d.set_label("2D Distortion Shift Magnitude (meters)", fontsize=11, fontweight="bold")
        fig_3d.suptitle(
            "3D Perspective Surface Progression: How 2D Polynomials Bend from Flat Plane to Wavy Surface",
            fontsize=14.5,
            fontweight="bold",
            y=0.97,
        )
        out_3d_path.parent.mkdir(parents=True, exist_ok=True)
        fig_3d.savefig(out_3d_path, dpi=200)
        plt.close(fig_3d)

        # 2D Contour Grid
        fig_2d, axes_2d = plt.subplots(2, 3, figsize=(18, 10.5))
        axes_flat = axes_2d.flatten()
        contour_levels = np.linspace(0.0, 2.8, 29)
        major_levels = [0.25, 0.50, 1.00, 1.50, 2.00, 2.50]

        for idx, (title, Z, desc) in enumerate(surfaces):
            ax = axes_flat[idx]
            ax.contourf(x_km, y_km, Z, levels=contour_levels, cmap=cmap, norm=norm, extend="both")
            cs = ax.contour(x_km, y_km, Z, levels=major_levels, colors="#1e293b", linewidths=0.6, alpha=0.7)
            ax.clabel(cs, inline=True, fontsize=7.5, fmt="%.2fm")
            ax.set_title(f"{title}\n{desc}", fontsize=10.5, fontweight="bold")
            ax.set_xlabel("Easting (km)", fontsize=9)
            ax.set_ylabel("Northing (km)", fontsize=9)
            ax.set_aspect("equal")
            ax.grid(True, linestyle="--", alpha=0.3)

        fig_2d.subplots_adjust(right=0.90, hspace=0.28, wspace=0.20, top=0.90, bottom=0.06)
        cbar_ax_2d = fig_2d.add_axes([0.92, 0.15, 0.015, 0.70])
        cbar_2d = fig_2d.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax_2d)
        cbar_2d.set_label("2D Distortion Shift Magnitude (meters)", fontsize=11, fontweight="bold")
        fig_2d.suptitle(
            "2D Plan-View Distortion Contours: Progression Across Polynomial Degrees",
            fontsize=14.5,
            fontweight="bold",
            y=0.97,
        )
        out_2d_path.parent.mkdir(parents=True, exist_ok=True)
        fig_2d.savefig(out_2d_path, dpi=200)
        plt.close(fig_2d)

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


if __name__ == "__main__":
    unit_dir = Path(__file__).resolve().parent
    unit = PolynomialSurfaceRegressorUnit(unit_dir)
    summary = unit.run()
    brief_path = unit.save_brief(summary)
    print(f"=== Research Unit 07 complete. Brief saved to {brief_path} ===")
