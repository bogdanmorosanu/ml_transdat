"""Analysis Unit: Grid Cell Planar & Curvature Shape Classification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

from pytransdat.trans_ro import TransRO
from src.core.base_unit import AnalysisExperimentUnit
from src.geometry.plane import extract_grid_cell_planes
from src.geometry.shape_classifier import (
    CellShapeFeatureExtractor,
    ShapeCategory,
    classify_cells_dataset,
)


class CellPlanarFeaturesUnit(AnalysisExperimentUnit):
    """Exploratory & feature-engineering unit for 11x11 km SPG cell geometry."""

    @property
    def slug(self) -> str:
        return "cell_planar_features"

    def run_analysis(self) -> dict[str, Any]:
        self.data_features_dir.mkdir(parents=True, exist_ok=True)

        # 1. Planar & bilinear twist extraction across all valid cells
        cells = extract_grid_cell_planes()
        df_planes = pd.DataFrame([c.to_dict() for c in cells])
        df_planes["dn_twist_mm"] = df_planes["dn_twist"] * 1000.0
        df_planes["de_twist_mm"] = df_planes["de_twist"] * 1000.0
        df_planes["total_twist_mm"] = df_planes["total_twist_mag"] * 1000.0

        planes_parquet = self.data_features_dir / "cell_planes.parquet"
        df_planes.to_parquet(planes_parquet, index=False)
        df_planes.to_csv(self.results_dir / "cell_planes.csv", index=False)

        self._plot_planar_maps(df_planes)

        # 2. 3-Class Curvature Shape Classification across interior cells
        trans = TransRO()
        metrics_list = classify_cells_dataset(cells, trans._t_gr2d.interp, grid_res=5)
        df_shapes = pd.DataFrame(
            [m.to_dict(default_tol_mm=5.0) for m in metrics_list]
        )

        tolerances = [2.0, 3.0, 5.0, 10.0, 15.0]
        for tol in tolerances:
            tol_tag = f"{int(tol)}mm" if tol == int(tol) else f"{tol:.1f}mm"
            df_shapes[f"cat_dn_{tol_tag}"] = [
                int(m.get_category_dn(tol)) for m in metrics_list
            ]
            df_shapes[f"cat_de_{tol_tag}"] = [
                int(m.get_category_de(tol)) for m in metrics_list
            ]
            df_shapes[f"cat_2d_{tol_tag}"] = [
                int(m.get_category_2d(tol)) for m in metrics_list
            ]

        shapes_parquet = self.data_features_dir / "cell_shape_metrics.parquet"
        df_shapes.to_parquet(shapes_parquet, index=False)
        df_shapes.to_csv(self.results_dir / "cell_shape_metrics.csv", index=False)

        self._plot_shape_maps(df_shapes)

        # Verify O(1) feature extractor against externalized parquet
        extractor = CellShapeFeatureExtractor(shapes_parquet)
        b_test = extractor.lookup_point(335000.0, 585000.0, tol_mm=5.0)

        counts_5mm = df_shapes["cat_2d_5mm"].value_counts(normalize=True).to_dict()
        counts_10mm = df_shapes["cat_2d_10mm"].value_counts(normalize=True).to_dict()

        return {
            "key_metrics": {
                "valid_cells": len(df_planes),
                "interior_cells": len(df_shapes),
                "twist_lt_5mm_pct": round(
                    float((df_planes["total_twist_mm"].abs() < 5.0).mean() * 100.0), 2
                ),
                "twist_lt_10mm_pct": round(
                    float((df_planes["total_twist_mm"].abs() < 10.0).mean() * 100.0), 2
                ),
                "shape_5mm_dome_pct": round(
                    float(counts_5mm.get(int(ShapeCategory.DOME_BOWL), 0.0) * 100.0), 2
                ),
                "shape_5mm_s_shape_pct": round(
                    float(counts_5mm.get(int(ShapeCategory.S_SHAPE), 0.0) * 100.0), 2
                ),
                "shape_10mm_dome_pct": round(
                    float(counts_10mm.get(int(ShapeCategory.DOME_BOWL), 0.0) * 100.0), 2
                ),
                "axis_concordance_5mm_pct": round(
                    float((df_shapes["cat_dn_5mm"] == df_shapes["cat_de_5mm"]).mean() * 100.0),
                    2,
                ),
            },
            "artifacts": {
                "shared_features": [
                    planes_parquet.as_posix(),
                    shapes_parquet.as_posix(),
                ],
                "results": [
                    "results/cell_planes.csv",
                    "results/cell_shape_metrics.csv",
                ],
                "plots": [
                    "national_cell_slopes_map.png",
                    "national_cell_twist_map.png",
                    "national_cell_aspect_distribution.png",
                    "national_shape_map_5mm.png",
                    "national_shape_map_10mm.png",
                ],
            },
            "extra": {
                "bucharest_sample": {
                    "is_valid": bool(b_test["is_valid"]),
                    "cat_2d_5mm": ShapeCategory(int(b_test["cat_2d"])).label,
                }
            },
        }

    def _plot_planar_maps(self, df: pd.DataFrame) -> None:
        # 1. National Planar Slope Maps
        _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
        sc1 = ax1.scatter(
            df["center_e"] / 1000.0,
            df["center_n"] / 1000.0,
            c=df["dn_slope_mag"],
            cmap="viridis",
            s=35,
            marker="s",
            alpha=0.9,
        )
        ax1.set_title(
            "Northing Shift (dN) Planar Slope Magnitude (m / 11 km)",
            fontsize=12,
            fontweight="bold",
        )
        ax1.set_xlabel("Stereo 70 Easting (km)")
        ax1.set_ylabel("Stereo 70 Northing (km)")
        ax1.grid(True, linestyle="--", alpha=0.5)
        plt.colorbar(sc1, ax=ax1, label="Slope Magnitude (m per 11 km cell)")

        sc2 = ax2.scatter(
            df["center_e"] / 1000.0,
            df["center_n"] / 1000.0,
            c=df["de_slope_mag"],
            cmap="plasma",
            s=35,
            marker="s",
            alpha=0.9,
        )
        ax2.set_title(
            "Easting Shift (dE) Planar Slope Magnitude (m / 11 km)",
            fontsize=12,
            fontweight="bold",
        )
        ax2.set_xlabel("Stereo 70 Easting (km)")
        ax2.grid(True, linestyle="--", alpha=0.5)
        plt.colorbar(sc2, ax=ax2, label="Slope Magnitude (m per 11 km cell)")
        plt.tight_layout()
        plt.savefig(self.plots_dir / "national_cell_slopes_map.png", dpi=200)
        plt.close()

        # 2. National Twist Map
        _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
        sc1 = ax1.scatter(
            df["center_e"] / 1000.0,
            df["center_n"] / 1000.0,
            c=df["dn_twist_mm"].abs(),
            cmap="YlOrRd",
            s=35,
            marker="s",
            alpha=0.9,
            vmax=150.0,
        )
        ax1.set_title(
            "Northing Shift (dN) Bilinear Twist |tau| (mm)",
            fontsize=12,
            fontweight="bold",
        )
        ax1.set_xlabel("Stereo 70 Easting (km)")
        ax1.set_ylabel("Stereo 70 Northing (km)")
        ax1.grid(True, linestyle="--", alpha=0.5)
        plt.colorbar(sc1, ax=ax1, label="|Twist| (mm departure from flat plane)")

        sc2 = ax2.scatter(
            df["center_e"] / 1000.0,
            df["center_n"] / 1000.0,
            c=df["de_twist_mm"].abs(),
            cmap="YlOrRd",
            s=35,
            marker="s",
            alpha=0.9,
            vmax=150.0,
        )
        ax2.set_title(
            "Easting Shift (dE) Bilinear Twist |tau| (mm)",
            fontsize=12,
            fontweight="bold",
        )
        ax2.set_xlabel("Stereo 70 Easting (km)")
        ax2.grid(True, linestyle="--", alpha=0.5)
        plt.colorbar(sc2, ax=ax2, label="|Twist| (mm departure from flat plane)")
        plt.tight_layout()
        plt.savefig(self.plots_dir / "national_cell_twist_map.png", dpi=200)
        plt.close()

        # 3. Aspect Distribution
        _fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(14, 6), subplot_kw={"projection": "polar"}
        )
        ax1.hist(
            np.radians(df["dn_aspect_deg"]),
            bins=36,
            color="teal",
            alpha=0.75,
            edgecolor="black",
        )
        ax1.set_title(
            "dN Tilt Aspect Distribution (Direction of Upward Slope)",
            fontsize=11,
            fontweight="bold",
            pad=15,
        )
        ax1.set_theta_zero_location("E")
        ax1.set_theta_direction(1)

        ax2.hist(
            np.radians(df["de_aspect_deg"]),
            bins=36,
            color="coral",
            alpha=0.75,
            edgecolor="black",
        )
        ax2.set_title(
            "dE Tilt Aspect Distribution (Direction of Upward Slope)",
            fontsize=11,
            fontweight="bold",
            pad=15,
        )
        ax2.set_theta_zero_location("E")
        ax2.set_theta_direction(1)
        plt.tight_layout()
        plt.savefig(self.plots_dir / "national_cell_aspect_distribution.png", dpi=200)
        plt.close()

    def _plot_shape_maps(self, df: pd.DataFrame) -> None:
        cmap = ListedColormap(["#2ca02c", "#1f77b4", "#ff7f0e"])
        categories = [ShapeCategory.LINEAR, ShapeCategory.DOME_BOWL, ShapeCategory.S_SHAPE]
        for tol in [5.0, 10.0]:
            tol_tag = f"{int(tol)}mm"
            fig, axes = plt.subplots(1, 3, figsize=(22, 6.5), sharex=True, sharey=True)
            for idx, (field, title) in enumerate(
                [
                    ("dn", f"Northing Shift (dN) Shape [{tol_tag}]"),
                    ("de", f"Easting Shift (dE) Shape [{tol_tag}]"),
                    ("2d", f"Combined 2D Cell Shape [{tol_tag}]"),
                ]
            ):
                ax = axes[idx]
                sc = ax.scatter(
                    df["center_e"] / 1000.0,
                    df["center_n"] / 1000.0,
                    c=df[f"cat_{field}_{tol_tag}"],
                    cmap=cmap,
                    vmin=0.5,
                    vmax=3.5,
                    s=30,
                    marker="s",
                    alpha=0.9,
                )
                ax.set_title(title, fontsize=12, fontweight="bold")
                ax.set_xlabel("Stereo 70 Easting (km)")
                if idx == 0:
                    ax.set_ylabel("Stereo 70 Northing (km)")
                ax.grid(True, linestyle="--", alpha=0.4)

            cbar = fig.colorbar(
                sc,
                ax=axes,
                orientation="horizontal",
                fraction=0.04,
                pad=0.12,
                ticks=[1, 2, 3],
            )
            cbar.ax.set_xticklabels(
                [c.label for c in categories], fontsize=11, fontweight="bold"
            )
            cbar.set_label("Curvature Shape Category", fontsize=11)
            plt.savefig(
                self.plots_dir / f"national_shape_map_{tol_tag}.png",
                dpi=200,
                bbox_inches="tight",
            )
            plt.close()


def main() -> None:
    unit = CellPlanarFeaturesUnit(Path("experiments/cell_planar_features"))
    unit.run()


if __name__ == "__main__":
    main()
