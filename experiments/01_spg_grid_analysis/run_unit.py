"""Unit 01: Physical analysis and spatial profiling of the SPG bicubic distortion grid."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pytransdat.trans_grid import Grid2D
from src.core.base_unit import ResearchUnit
from src.core.metrics import summarize_errors
from src.core.viz import generate_standard_diagnostic_suite, plot_vector_quiver

BENCHMARK_PATH = Path("data/benchmarks/ro_grid_3x3.parquet")


class SpgGridAnalysisUnit(ResearchUnit):
    """Research Unit 01: SPG 2D distortion grid profiling."""

    @property
    def unit_id(self) -> str:
        return "01"

    @property
    def unit_name(self) -> str:
        return "spg_grid_analysis"

    def run(self) -> dict[str, Any]:
        t0 = time.perf_counter()
        df = pd.read_parquet(BENCHMARK_PATH)
        grid2d = Grid2D()

        # Extract 2D bicubic shifts on benchmark grid
        dn_list, de_list = grid2d.interp(
            df["st70_n"].tolist(),
            df["st70_e"].tolist(),
        )
        dn = np.asarray(dn_list, dtype=np.float64)
        de = np.asarray(de_list, dtype=np.float64)
        mag = np.hypot(dn, de)

        e_km = df["st70_e"].to_numpy(dtype=np.float64) * 1e-3
        n_km = df["st70_n"].to_numpy(dtype=np.float64) * 1e-3

        # 1. Component heatmaps: dN, dE, Magnitude
        specs = [
            (
                "spg_shift_dn",
                dn,
                "RdBu_r",
                mcolors.TwoSlopeNorm(vmin=float(dn.min()), vcenter=0.0, vmax=float(dn.max())),
                "Northing Shift dN (m)",
                f"SPG Bicubic Grid Shift — Northing dN ({grid2d.file_name})",
            ),
            (
                "spg_shift_de",
                de,
                "RdBu_r",
                mcolors.TwoSlopeNorm(vmin=float(de.min()), vcenter=0.0, vmax=float(de.max())),
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

        diag_paths = generate_standard_diagnostic_suite(
            err_df=shift_df,
            summary=summary,
            out_dir=self.plots_dir,
            model_name="spg_grid_shifts",
            direction="2d_bicubic",
            dpi=200,
        )
        for k, v in diag_paths.items():
            plot_files[k] = v.name

        # Save brief
        elapsed = time.perf_counter() - t0
        brief = {
            "unit_id": self.unit_id,
            "unit_name": self.unit_name,
            "grid_file": grid2d.file_name,
            "sample_count": len(df),
            "statistics": {
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
            },
            "plots": plot_files,
            "runtime_s": round(elapsed, 3),
        }
        self.save_brief(brief)
        print(f"=== Research Unit 01 [{self.unit_name}] Complete ===")
        print(f"Shift Magnitude: Mean={brief['statistics']['mag_mean_m']:.4f} m | Max={brief['statistics']['mag_max_m']:.4f} m")
        return brief


def main() -> None:
    unit = SpgGridAnalysisUnit(Path("experiments/01_spg_grid_analysis"))
    unit.run()


if __name__ == "__main__":
    main()
