"""Unit 08: 2D Radial Basis Function & Thin Plate Spline Surface Regressor on SPG Grid Shifts."""

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
from src.models.rbf_surface import RBFSurfaceRegressor
from src.viz import PlotGroup, plot_transect_66km


class RBFSurfaceRegressorUnit(GridShiftExperimentUnit):
    """Research Unit 08: 2D RBF & Thin Plate Spline Localized Continuous Surface Regressors."""

    @property
    def unit_id(self) -> str:
        return "08"

    @property
    def unit_name(self) -> str:
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
            ("2. Level 1: Flat Tilted Plane", poly_deg1, None, "Rigid flat plane (Helmert equivalent, RMSE 42.8cm)"),
            ("3. Level 2: Polynomial Deg 12", poly_deg12, None, "Macroscopic curved fit (RMSE 19.2cm, edge ripples)"),
            ("4. Level 3: TPS Base Smooth (11km)", fitted_models["tps_base_optimal_2e-4"], None, "Thin Plate Spline on 2,222 base anchors (RMSE 8.6mm)"),
            ("5. Level 3: TPS Densified (4x4)", fitted_models["tps_densified_4x4"], None, "Thin Plate Spline on 36,435 anchors (Mean 3.8mm)"),
            ("6. Level 3: Tuned MQ (eps=10)", fitted_models["mq_tuned_eps10"], None, "Tuned Hardy Multiquadric RBF (RMSE 14.0mm)"),
        ]

        surfaces = []
        for title, model_obj, direct_mag, desc in models_info:
            if direct_mag is not None:
                surfaces.append((title, direct_mag, desc))
            else:
                pred = model_obj.predict(query_coords)
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
            "Level 3 Surface Progression: Flat Plane vs. Polynomial vs. Elastic Sheet (RBF / TPS)",
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
            "2D Plan-View Contours: Level 1 (Helmert) vs. Level 2 (Polynomial) vs. Level 3 (RBF / TPS)",
            fontsize=14.5,
            fontweight="bold",
            y=0.97,
        )
        out_2d_path.parent.mkdir(parents=True, exist_ok=True)
        fig_2d.savefig(out_2d_path, dpi=200)
        plt.close(fig_2d)

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


if __name__ == "__main__":
    unit_dir = Path(__file__).resolve().parent
    unit = RBFSurfaceRegressorUnit(unit_dir)
    summary = unit.run()
    brief_path = unit.save_brief(summary)
    print(f"=== Research Unit 08 complete. Brief saved to {brief_path} ===")
