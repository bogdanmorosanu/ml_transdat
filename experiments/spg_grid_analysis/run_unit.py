"""Unit 01: Physical analysis and spatial profiling of the SPG bicubic distortion grid."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from pytransdat.trans_grid import Grid2D
from src.core.base_unit import AnalysisExperimentUnit
from src.data import GridId
from src.evaluation import EvaluationResult, summarize_errors
from src.viz import PlotGroup, plot_vector_quiver


class SpgGridAnalysisUnit(AnalysisExperimentUnit):
    """Analysis Unit: SPG 2D distortion grid profiling."""

    @property
    def slug(self) -> str:
        return "spg_grid_analysis"

    def run_analysis(self) -> dict[str, Any]:
        dataset = self.load_grid(GridId.RO_GRID_4X4)
        df = dataset.df
        grid2d = Grid2D()

        shifts = dataset.true_shifts
        dn = shifts[:, 0]
        de = shifts[:, 1]
        mag = np.hypot(dn, de)

        e_km = df["st70_e"].to_numpy(dtype=np.float64) * 1e-3
        n_km = df["st70_n"].to_numpy(dtype=np.float64) * 1e-3

        # 1. Component heatmaps: dN, dE, Magnitude
        specs = [
            (
                "spg_shift_dn",
                dn,
                "RdBu_r",
                mcolors.TwoSlopeNorm(
                    vmin=float(dn.min()), vcenter=0.0, vmax=float(dn.max())
                ),
                "Northing Shift dN (m)",
                f"SPG Bicubic Grid Shift — Northing dN ({grid2d.file_name})",
            ),
            (
                "spg_shift_de",
                de,
                "RdBu_r",
                mcolors.TwoSlopeNorm(
                    vmin=float(de.min()), vcenter=0.0, vmax=float(de.max())
                ),
                "Easting Shift dE (m)",
                f"SPG Bicubic Grid Shift — Easting dE ({grid2d.file_name})",
            ),
            (
                "spg_shift_mag",
                mag,
                "inferno",
                mcolors.Normalize(vmin=0.0, vmax=float(mag.max())),
                "Combined Shift Magnitude hypot(dN, dE) (m)",
                f"SPG Bicubic Grid Shift — Magnitude ({grid2d.file_name})",
            ),
        ]

        plot_files: dict[str, str] = {}
        for fname, values, cmap, norm, cbar_label, title in specs:
            fig, ax = plt.subplots(figsize=(10, 7.5))
            sc = ax.scatter(
                e_km,
                n_km,
                c=values,
                cmap=cmap,
                norm=norm,
                s=6.5,
                marker="s",
                alpha=0.98,
                edgecolors="none",
                linewidths=0,
            )
            cbar = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
            cbar.set_label(cbar_label)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("Stereo 70 Easting (km)")
            ax.set_ylabel("Stereo 70 Northing (km)")
            ax.set_title(title)
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
            fig.tight_layout()

            out_p = self.plots_dir / f"{fname}.png"
            fig.savefig(out_p, dpi=200)
            plt.close(fig)
            plot_files[fname] = out_p.name

        # 2. Vector field (quiver)
        quiver_p = self.plots_dir / "spg_shift_quiver.png"
        plot_vector_quiver(
            df=df,
            de=de,
            dn=dn,
            mag=mag,
            out_path=quiver_p,
            title=f"SPG Distortion Vector Field (22 km sampled nodes) — {grid2d.file_name}",
            stride=2,
            dpi=200,
        )
        plot_files["quiver"] = quiver_p.name

        # 3. Standard diagnostic suite
        shift_df = df.copy()
        shift_df["err_n_m"] = dn
        shift_df["err_e_m"] = de
        shift_df["err_2d_m"] = mag
        summary = summarize_errors(shift_df)

        diag_res = EvaluationResult(
            model_name="spg_grid_shifts",
            dataset_id=dataset.grid_id.value,
            direction="2d_bicubic",
            err_df=shift_df,
            summary=summary,
            eval_time_sec=0.0,
            throughput_pts_per_sec=0.0,
        )
        diag_paths = self.render_plots(
            diag_res, plots=PlotGroup.STANDARD_DIAGNOSTIC, dpi=200
        )
        for k, v in diag_paths.items():
            plot_files[k.value] = v.name

        stats = {
            "grid_file": grid2d.file_name,
            "sample_count": len(df),
            "dn_min_m": float(dn.min()),
            "dn_max_m": float(dn.max()),
            "dn_mean_m": float(dn.mean()),
            "de_min_m": float(de.min()),
            "de_max_m": float(de.max()),
            "de_mean_m": float(de.mean()),
            "mag_mean_m": float(mag.mean()),
            "mag_p50_m": summary["p50_2d_m"],
            "mag_p95_m": summary["p95_2d_m"],
            "mag_p99_m": summary["p99_2d_m"],
            "mag_max_m": float(mag.max()),
        }
        print(
            f"Shift Magnitude: Mean={stats['mag_mean_m']:.4f} m | Max={stats['mag_max_m']:.4f} m"
        )
        return {
            "key_metrics": stats,
            "artifacts": {"plots": list(plot_files.values())},
            "extra": {"plot_map": plot_files},
        }


def main() -> None:
    unit = SpgGridAnalysisUnit(Path("experiments/spg_grid_analysis"))
    unit.run()


if __name__ == "__main__":
    main()
