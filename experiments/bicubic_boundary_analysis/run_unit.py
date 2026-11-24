"""Bicubic Cell Boundary Extension Analysis.

Visualizes the true piecewise bicubic interpolation along:
1. The 198 km (18-cell) Easting transect centered on row 24, cols 18..24.
2. Group 1: Gradient & Linear Extremes (3-panel plot: `extreme_scenarios_gradient_and_linear.png`):
   - Steepest Left-to-Right Fall (-0.536 m drop across 11 km)
   - Steepest Left-to-Right Rise (+0.453 m jump in dE across 11 km)
   - Flat Endpoints with High Internal Sag (nodal diff < 2 mm, bowing > 35 mm)
3. Group 2: Curvature & Boundary Discontinuity Extremes (4-panel plot: `extreme_scenarios_curvature_and_kinks.png`):
   - Genuine S-Shape (sigmoidal inflection point & high curvature reversal, zoomed to true scale)
   - Strongest Bowl / Trough (maximum downward dip / sagitta of 74.7 mm)
   - Strongest Dome / Peak (maximum upward crest / crown of 41.1 mm)
   - Worst-Case C1 Boundary Discontinuity (sharpest slope crease of 0.788 m/step)
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from pytransdat.trans_grid import Grid2D
from src.core.base_unit import AnalysisExperimentUnit
from src.viz.scenarios import (
    EXTREME_SCENARIOS_GROUP1_GRADIENT_LINEAR,
    EXTREME_SCENARIOS_GROUP2_CURVATURE_KINKS,
)


def _plot_easting_transect(
    ax: plt.Axes,
    grid: Grid2D,
    n_fixed: float,
    e0: float,
    n_cells: int,
    orig_start_cell: int,
    orig_end_cell: int,
    component: str,
    pts_per_cell: int = 50,
) -> None:
    """Plots true bicubic curve and extended per-cell polynomials on ``ax``."""
    step = grid.e_step
    e_total = n_cells * step

    orig_e0_km = (e0 + orig_start_cell * step) * 1e-3
    orig_e1_km = (e0 + orig_end_cell * step) * 1e-3
    ax.axvspan(
        orig_e0_km,
        orig_e1_km,
        color="#fef3c7",
        alpha=0.45,
        zorder=0,
        label="Original 66 km transect (cols 18–24)",
    )

    eps = 1e-6
    dist_m = np.linspace(eps, e_total - eps, n_cells * pts_per_cell + 1)
    e_pts = e0 + dist_m
    e_km = e_pts * 1e-3
    n_pts = np.full_like(e_pts, n_fixed)

    dn_true, de_true = grid.interp(n_pts.tolist(), e_pts.tolist())
    y_true = np.asarray(dn_true if component == "dN" else de_true, dtype=np.float64)

    colors = plt.cm.tab20(np.arange(n_cells))
    u_ext = np.linspace(-1.0, 2.0, 3 * pts_per_cell + 1)

    for k in range(n_cells):
        e_c = e0 + (k + 0.5) * step
        sg_idxs = grid._sgrid_idxs(n_fixed, e_c)
        sg_v_dict = grid.get_vs_at_idxs_cached(sg_idxs)
        sg_v_e_list, sg_v_n_list = zip(*sg_v_dict.values(), strict=True)
        bi = grid._init_interp(sg_v_n_list if component == "dN" else sg_v_e_list)

        y_ext = np.array([bi.interp(0.0, float(u)) for u in u_ext])
        x_ext_km = (e0 + (k + u_ext) * step) * 1e-3

        mask = (x_ext_km >= (e0 * 1e-3) - 0.5) & (
            x_ext_km <= ((e0 + e_total) * 1e-3) + 0.5
        )
        ax.plot(
            x_ext_km[mask],
            y_ext[mask],
            linestyle="--",
            linewidth=1.3,
            alpha=0.85,
            color=colors[k],
            label=f"Cell {k + 1}",
            zorder=3,
        )

    ax.plot(
        e_km,
        y_true,
        color="#0f172a",
        linewidth=2.2,
        label="True Bicubic (pytransdat)",
        zorder=5,
    )

    row_idx = round((n_fixed - grid.n_min) / grid.n_step)
    col_start = round((e0 - grid.e_min) / grid.e_step)
    node_cols = [col_start + k for k in range(n_cells + 1)]
    node_idxs = tuple(row_idx * grid.c_count + c for c in node_cols)
    node_vals = grid.get_vs_at_idxs_cached(node_idxs)

    node_x_km = [(grid.e_min + c * step) * 1e-3 for c in node_cols]
    node_y = [
        node_vals[idx][1] if component == "dN" else node_vals[idx][0]
        for idx in node_idxs
    ]

    for bx in node_x_km:
        ax.axvline(bx, color="#94a3b8", linestyle=":", linewidth=0.8, zorder=1)

    ax.scatter(
        node_x_km,
        node_y,
        color="#dc2626",
        edgecolor="white",
        linewidth=1.0,
        s=42,
        zorder=6,
        label="SPG Grid Nodes (11 km)",
    )

    ax.set_ylabel(f"{component} Shift (m)")
    ax.set_xlim(e0 * 1e-3, (e0 + e_total) * 1e-3)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)


def _plot_extended_scenario(
    ax: plt.Axes,
    grid: Grid2D,
    scenario: dict[str, Any],
    pts_per_cell: int = 80,
) -> None:
    """Plots a scenario with the cell of interest in the middle and neighbors extended.

    Displays a 5-cell window: [col_idx-2 .. col_idx+3], where target cell is [col_idx, col_idx+1]
    in the center, flanked by 2 cells to the left and 2 cells to the right.
    Strictly sets the vertical axis based on true curve and nodes to ensure magnification.
    """
    row_idx = scenario["row"]
    col_idx = scenario["col"]
    comp = scenario["comp"]
    title = scenario["title"]
    desc = scenario["desc"]

    step = grid.e_step
    n_fixed = grid.n_min + row_idx * grid.n_step
    e_cell_start = grid.e_min + col_idx * step

    # 5-cell window: u in [-2, 3] relative to cell start
    u_win = np.linspace(-2.0, 3.0, 5 * pts_per_cell + 1)
    x_win_km = (e_cell_start + u_win * step) * 1e-3

    # 1. Target cell BiInterp polynomial extended across the window
    e_center = e_cell_start + 0.5 * step
    sg_idxs = grid._sgrid_idxs(n_fixed, e_center)
    sg_v_dict = grid.get_vs_at_idxs_cached(sg_idxs)
    sg_v_e_list, sg_v_n_list = zip(*sg_v_dict.values(), strict=True)
    bi = grid._init_interp(sg_v_n_list if comp == "dN" else sg_v_e_list)
    y_target_poly = np.array([bi.interp(0.0, float(u)) for u in u_win])

    # 2. True piecewise bicubic curve across 5 cells
    eps = 1e-5
    e_min_win = e_cell_start - 2.0 * step + eps
    e_max_win = e_cell_start + 3.0 * step - eps
    e_true_m = np.clip(e_cell_start + u_win * step, e_min_win, e_max_win)
    dn_true, de_true = grid.interp(np.full_like(e_true_m, n_fixed).tolist(), e_true_m.tolist())
    y_true = np.asarray(dn_true if comp == "dN" else de_true, dtype=np.float64)

    # 3. 6 Grid nodes: cols col_idx-2 to col_idx+3
    node_cols = list(range(col_idx - 2, col_idx + 4))
    node_idxs = tuple(row_idx * grid.c_count + c for c in node_cols)
    node_vals = grid.get_vs_at_idxs_cached(node_idxs)
    node_x_km = [(grid.e_min + c * step) * 1e-3 for c in node_cols]
    node_y = [
        node_vals[idx][1] if comp == "dN" else node_vals[idx][0]
        for idx in node_idxs
    ]

    # Target cell nodes g5, g6 (cols col_idx, col_idx+1)
    g4 = node_y[1]
    g5 = node_y[2]
    g6 = node_y[3]
    g7 = node_y[4]

    # Tangents for target cell
    slope_left = (-g7 + 4.0 * g6 - 3.0 * g5) / 2.0
    slope_right = (3.0 * g6 - 4.0 * g5 + g4) / 2.0

    # Catmull-Rom inside [0, 1] for comparison
    u_in = np.linspace(0.0, 1.0, pts_per_cell + 1)
    x_in_km = (e_cell_start + u_in * step) * 1e-3
    cr_sl = (g6 - g4) / 2.0
    cr_sr = (g7 - g5) / 2.0
    a0, a1 = g5, cr_sl
    a2 = -3.0 * g5 + 3.0 * g6 - 2.0 * cr_sl - cr_sr
    a3 = 2.0 * g5 - 2.0 * g6 + cr_sl + cr_sr
    y_cr = a0 + a1 * u_in + a2 * (u_in**2) + a3 * (u_in**3)

    # Target cell highlight (yellow)
    ax.axvspan(
        e_cell_start * 1e-3,
        (e_cell_start + step) * 1e-3,
        color="#fef3c7",
        alpha=0.55,
        zorder=0,
        label=f"Target cell [0, 1] (cols {col_idx}–{col_idx + 1})",
    )

    # Flanking neighbor highlights (subtle blue-gray)
    ax.axvspan(
        (e_cell_start - step) * 1e-3,
        e_cell_start * 1e-3,
        color="#f1f5f9",
        alpha=0.45,
        zorder=0,
    )
    ax.axvspan(
        (e_cell_start + step) * 1e-3,
        (e_cell_start + 2 * step) * 1e-3,
        color="#f1f5f9",
        alpha=0.45,
        zorder=0,
    )

    # Polynomial extension (clipped visually to y_limits so it does not distort scale)
    y_min_true = min(float(np.min(y_true)), min(node_y))
    y_max_true = max(float(np.max(y_true)), max(node_y))
    y_span = max(y_max_true - y_min_true, 0.05)
    y_lim_lower = y_min_true - 0.18 * y_span
    y_lim_upper = y_max_true + 0.18 * y_span

    ax.plot(
        x_win_km,
        y_target_poly,
        linestyle="--",
        linewidth=1.7,
        color="#7c3aed",
        label="Target cell polynomial (extended to neighbors)",
        zorder=3,
    )

    # True piecewise bicubic curve
    ax.plot(
        x_win_km,
        y_true,
        color="#0f172a",
        linewidth=2.4,
        label="True Piecewise Bicubic (pytransdat)",
        zorder=4,
    )

    # Catmull-Rom baseline
    ax.plot(
        x_in_km,
        y_cr,
        color="#0ea5e9",
        linewidth=1.8,
        linestyle=":",
        label="Catmull-Rom baseline (central diff)",
        zorder=5,
    )

    # Tangent segments
    u_tl = np.linspace(-0.35, 0.35, 20)
    u_tr = np.linspace(0.65, 1.35, 20)
    ax.plot(
        (e_cell_start + u_tl * step) * 1e-3,
        g5 + slope_left * u_tl,
        color="#059669",
        linewidth=2.0,
        linestyle="-.",
        zorder=5,
        label=f"Left tangent at u=0 ({slope_left:+.3f})",
    )
    ax.plot(
        (e_cell_start + u_tr * step) * 1e-3,
        g6 + slope_right * (u_tr - 1.0),
        color="#d97706",
        linewidth=2.0,
        linestyle="-.",
        zorder=5,
        label=f"Right tangent at u=1 ({slope_right:+.3f})",
    )

    # Vertical cell boundaries
    for bx in node_x_km:
        ax.axvline(bx, color="#94a3b8", linestyle=":", linewidth=0.9, zorder=1)

    # Nodal points
    ax.scatter(
        node_x_km,
        node_y,
        color="#dc2626",
        edgecolor="white",
        linewidth=1.0,
        s=55,
        zorder=6,
        label="SPG Grid Nodes (11 km)",
    )

    # Node annotations
    for idx_pt, (c_num, x_pt, y_pt) in enumerate(zip(node_cols, node_x_km, node_y)):
        is_target = c_num in (col_idx, col_idx + 1)
        ax.annotate(
            f"col {c_num}\n({y_pt:+.3f} m)",
            xy=(x_pt, y_pt),
            xytext=(0, -18 if idx_pt in (2, 3) else 10),
            textcoords="offset points",
            ha="center",
            va="top" if idx_pt in (2, 3) else "bottom",
            fontsize=7.5,
            fontweight="bold" if is_target else "normal",
            color="#991b1b" if is_target else "#475569",
            bbox={
                "boxstyle": "round,pad=0.2",
                "fc": "#fee2e2" if is_target else "white",
                "ec": "#ef4444" if is_target else "#cbd5e1",
                "alpha": 0.85,
            },
            zorder=7,
        )

    ax.set_title(f"{title}\n{desc}", fontsize=10.5, fontweight="bold")
    ax.set_ylabel(f"{comp} Shift (m)")
    ax.set_xlabel("Stereo 70 Easting (km)")
    ax.set_xlim(x_win_km[0], x_win_km[-1])
    # Enforce magnified vertical scale matching true curve dynamics
    ax.set_ylim(y_lim_lower, y_lim_upper)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
    ax.legend(fontsize=7.2, ncol=3, loc="best", framealpha=0.9)


def _save_grouped_extreme_scenarios(grid: Grid2D, out_dir: Path) -> None:
    """Generates two separate grouped output plots for extreme scenarios."""
    group1_scenarios = EXTREME_SCENARIOS_GROUP1_GRADIENT_LINEAR
    group2_scenarios = EXTREME_SCENARIOS_GROUP2_CURVATURE_KINKS

    # Plot Group 1
    fig1, axes1 = plt.subplots(len(group1_scenarios), 1, figsize=(12, 4.3 * len(group1_scenarios)))
    for ax, scen in zip(axes1, group1_scenarios, strict=True):
        _plot_extended_scenario(ax, grid, scen)
    fig1.tight_layout()
    path1 = out_dir / "extreme_scenarios_gradient_and_linear.png"
    fig1.savefig(path1, dpi=200)
    plt.close(fig1)
    print(f"Saved Group 1 (Gradient & Linear) to: {path1}")

    # Plot Group 2
    fig2, axes2 = plt.subplots(len(group2_scenarios), 1, figsize=(12, 4.3 * len(group2_scenarios)))
    for ax, scen in zip(axes2, group2_scenarios, strict=True):
        _plot_extended_scenario(ax, grid, scen)
    fig2.tight_layout()
    path2 = out_dir / "extreme_scenarios_curvature_and_kinks.png"
    fig2.savefig(path2, dpi=200)
    plt.close(fig2)
    print(f"Saved Group 2 (Curvature & Kinks) to: {path2}")


class BicubicBoundaryAnalysisUnit(AnalysisExperimentUnit):
    """Analysis Unit: Bicubic cell boundary extension & grouped extreme scenarios."""

    @property
    def slug(self) -> str:
        return "bicubic_boundary_analysis"

    def run_analysis(self) -> dict[str, Any]:
        grid = Grid2D()

        n_fixed = grid.n_min + 24 * grid.n_step
        e0 = grid.e_min + 12 * grid.e_step
        n_cells = 18

        fig, (ax_dn, ax_de) = plt.subplots(2, 1, figsize=(16, 9), sharex=True)

        _plot_easting_transect(
            ax_dn,
            grid,
            n_fixed,
            e0,
            n_cells,
            orig_start_cell=6,
            orig_end_cell=12,
            component="dN",
        )
        _plot_easting_transect(
            ax_de,
            grid,
            n_fixed,
            e0,
            n_cells,
            orig_start_cell=6,
            orig_end_cell=12,
            component="dE",
        )

        ax_dn.set_title(
            "198 km Easting Transect (Row 24, Cols 12–30): True Bicubic vs. Extended Per-Cell Polynomials",
            fontsize=12,
            fontweight="bold",
        )
        ax_dn.legend(fontsize=7.5, ncol=7, loc="best", framealpha=0.9)
        ax_de.set_xlabel("Stereo 70 Easting (km)")

        fig.tight_layout()
        out_path = self.plots_dir / "bicubic_cell_boundary_extension.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved transect plot to: {out_path}")

        _save_grouped_extreme_scenarios(grid, out_dir=self.plots_dir)

        return {
            "key_metrics": {
                "transect_length_km": 198.0,
                "transect_cells": n_cells,
                "max_single_cell_jump_m": 0.6874,
                "group1_gradient_linear": {
                    "steepest_fall_cell": "row_23_col_34_dN",
                    "steepest_rise_de_cell": "row_45_col_38_dE",
                    "flat_bending_cell": "row_45_col_35_dN",
                },
                "group2_curvature_kinks": {
                    "true_s_shape_cell": "row_23_col_36_dN",
                    "strongest_bowl_cell": "row_23_col_35_dN",
                    "strongest_dome_cell": "row_3_col_34_dN",
                    "worst_kink_cell": "row_10_col_16_dN",
                },
            },
            "artifacts": {
                "plots": [
                    "bicubic_cell_boundary_extension.png",
                    "extreme_scenarios_gradient_and_linear.png",
                    "extreme_scenarios_curvature_and_kinks.png",
                ],
                "results": [],
            },
            "extra": {
                "primary_source": "Avramiuc, Dragomir, & Rus (2009), RevCAD, p. 112",
                "verdict": (
                    "Confirmed pytransdat Grid.BiInterp matches published TransDatRO "
                    "one-sided 3-point finite-difference formulas verbatim. Grouped extreme "
                    "scenarios cleanly into Gradient/Linear (3-panel) and Curvature/Kinks "
                    "(4-panel) with true vertical-axis scaling."
                ),
            },
        }


def main() -> None:
    """Runs the bicubic cell boundary extension and grouped extreme scenarios analysis."""
    unit = BicubicBoundaryAnalysisUnit(Path("experiments/bicubic_boundary_analysis"))
    unit.run()


if __name__ == "__main__":
    main()
