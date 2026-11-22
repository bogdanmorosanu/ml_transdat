"""Bicubic Cell Boundary Extension Analysis.

Visualizes the true piecewise bicubic interpolation along the 198 km (18-cell)
Easting transect centered on the exact 66 km (6-cell) ``plot_transect_66km``
section (row 24, cols 18..24, extended 6 cells before and 6 cells after).
Each cell's individual bicubic polynomial is extended into the adjacent cells
before and after ([-1, 2] in unit-cell coordinates) using distinct dashed colors.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pytransdat.trans_grid import Grid2D


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

    # Highlight the original 66 km plot_transect_66km section
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

    # 1. True piecewise bicubic curve
    eps = 1e-6
    dist_m = np.linspace(eps, e_total - eps, n_cells * pts_per_cell + 1)
    e_pts = e0 + dist_m
    e_km = e_pts * 1e-3
    n_pts = np.full_like(e_pts, n_fixed)

    dn_true, de_true = grid.interp(n_pts.tolist(), e_pts.tolist())
    y_true = np.asarray(dn_true if component == "dN" else de_true, dtype=np.float64)

    # 2. Per-cell polynomial extended to [-1, 2] (1 cell before, cell itself, 1 cell after)
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

    # 3. True curve on top
    ax.plot(
        e_km,
        y_true,
        color="#0f172a",
        linewidth=2.2,
        label="True Bicubic (pytransdat)",
        zorder=5,
    )

    # 4. Vertical 11 km cell boundaries & true SPG nodal shift values
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


def main() -> None:
    """Runs the 198 km (18-cell) bicubic cell boundary extension analysis."""
    grid = Grid2D()

    # Exact row from plot_transect_66km, tripled in Easting (6 before + 6 original + 6 after)
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
    out_dir = Path(__file__).resolve().parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bicubic_cell_boundary_extension.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved plot to: {out_path}")

    # Isolated single-cell zoom (Cell 9: row 24, cols 20–21, 329.783–340.783 km)
    _save_single_cell_zoom(grid, row_idx=24, col_idx=20, out_dir=out_dir)

    # Extreme S-shape and Bowl/Dome scenarios from full-grid scan
    _save_extreme_scenarios(grid, out_dir=out_dir)


def _save_single_cell_zoom(
    grid: Grid2D,
    row_idx: int,
    col_idx: int,
    out_dir: Path,
    pts_per_cell: int = 100,
) -> None:
    """Plots an isolated single-cell bicubic polynomial over [-1, 2] (no neighbors)."""
    n_fixed = grid.n_min + row_idx * grid.n_step
    e_cell_start = grid.e_min + col_idx * grid.e_step
    step = grid.e_step

    # Extract only this cell's 4x4 subgrid and BiInterp polynomial
    e_center = e_cell_start + 0.5 * step
    sg_idxs = grid._sgrid_idxs(n_fixed, e_center)
    sg_v_dict = grid.get_vs_at_idxs_cached(sg_idxs)
    sg_v_e_list, sg_v_n_list = zip(*sg_v_dict.values(), strict=True)

    # 4 SPG nodes along row_idx used by this cell's 4x4 subgrid (cols col_idx-1 .. col_idx+2)
    node_cols = [col_idx - 1, col_idx, col_idx + 1, col_idx + 2]
    node_idxs = tuple(row_idx * grid.c_count + c for c in node_cols)
    node_vals = grid.get_vs_at_idxs_cached(node_idxs)
    node_x_km = [(grid.e_min + c * step) * 1e-3 for c in node_cols]

    u_ext = np.linspace(-1.0, 2.0, 3 * pts_per_cell + 1)
    x_ext_km = (e_cell_start + u_ext * step) * 1e-3

    # True piecewise bicubic curve across [-1, 2] (neighbor before, target cell, neighbor after)
    eps = 1e-6
    e_true_m = np.clip(e_cell_start + u_ext * step, e_cell_start - step + eps, e_cell_start + 2 * step - eps)
    n_true_m = np.full_like(e_true_m, n_fixed)
    dn_true_all, de_true_all = grid.interp(n_true_m.tolist(), e_true_m.tolist())

    fig, (ax_dn, ax_de) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)

    for ax, comp, sg_list, y_true_all, color in (
        (ax_dn, "dN", sg_v_n_list, np.asarray(dn_true_all, dtype=np.float64), "#7c3aed"),
        (ax_de, "dE", sg_v_e_list, np.asarray(de_true_all, dtype=np.float64), "#2563eb"),
    ):
        bi = grid._init_interp(sg_list)
        y_ext = np.array([bi.interp(0.0, float(u)) for u in u_ext])

        ax.axvspan(
            e_cell_start * 1e-3,
            (e_cell_start + step) * 1e-3,
            color="#fef3c7",
            alpha=0.45,
            zorder=0,
            label=f"Target cell [0, 1] (cols {col_idx}–{col_idx + 1})",
        )
        ax.plot(
            x_ext_km,
            y_ext,
            linestyle="--",
            linewidth=1.8,
            color=color,
            label="Isolated cell polynomial (u in [-1, 2])",
            zorder=3,
        )
        ax.plot(
            x_ext_km,
            y_true_all,
            color="#0f172a",
            linewidth=2.4,
            label="True bicubic (pytransdat, u in [-1, 2])",
            zorder=4,
        )
        for bx in node_x_km:
            ax.axvline(bx, color="#94a3b8", linestyle=":", linewidth=0.9, zorder=1)

        node_y = [
            node_vals[idx][1] if comp == "dN" else node_vals[idx][0]
            for idx in node_idxs
        ]
        # Draw the two Hermite tangent lines at u=0 (col_idx) and u=1 (col_idx+1)
        # g4, g5, g6, g7 correspond to cols [col_idx-1, col_idx, col_idx+1, col_idx+2]
        g4, g5, g6, g7 = node_y
        slope_left = (-g7 + 4.0 * g6 - 3.0 * g5) / 2.0
        slope_right = (3.0 * g6 - 4.0 * g5 + g4) / 2.0

        u_tan_l = np.linspace(-0.45, 0.45, 20)
        x_tan_l = (e_cell_start + u_tan_l * step) * 1e-3
        y_tan_l = g5 + slope_left * u_tan_l

        u_tan_r = np.linspace(0.55, 1.45, 20)
        x_tan_r = (e_cell_start + u_tan_r * step) * 1e-3
        y_tan_r = g6 + slope_right * (u_tan_r - 1.0)

        ax.plot(
            x_tan_l,
            y_tan_l,
            color="#059669",
            linewidth=2.0,
            linestyle="-.",
            zorder=5,
            label=f"Left tangent at u=0 (uses cols {col_idx}–{col_idx + 2})",
        )
        ax.plot(
            x_tan_r,
            y_tan_r,
            color="#d97706",
            linewidth=2.0,
            linestyle="-.",
            zorder=5,
            label=f"Right tangent at u=1 (uses cols {col_idx - 1}–{col_idx + 1})",
        )

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
        for idx_pt, (c_num, x_pt, y_pt) in enumerate(zip(node_cols, node_x_km, node_y)):
            dx_off = 6 if idx_pt == 0 else (-6 if idx_pt == len(node_cols) - 1 else 0)
            ha = "left" if idx_pt == 0 else ("right" if idx_pt == len(node_cols) - 1 else "center")
            ax.annotate(
                f"col {c_num}\n({y_pt:+.3f} m)",
                xy=(x_pt, y_pt),
                xytext=(dx_off, -16 if idx_pt in (1, 2) else 10),
                textcoords="offset points",
                ha=ha,
                va="top" if idx_pt in (1, 2) else "bottom",
                fontsize=8,
                fontweight="bold",
                color="#991b1b",
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "#fca5a5", "alpha": 0.85},
                zorder=7,
            )
        ax.set_ylabel(f"{comp} Shift (m)")
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
        ax.legend(loc="best", framealpha=0.9)

    ax_dn.set_title(
        f"Isolated Single Cell (Row {row_idx}, Cols {col_idx}–{col_idx + 1}: "
        f"{e_cell_start * 1e-3:.3f}–{(e_cell_start + step) * 1e-3:.3f} km)",
        fontsize=12,
        fontweight="bold",
    )
    ax_de.set_xlabel("Stereo 70 Easting (km)")
    ax_de.set_xlim(x_ext_km[0], x_ext_km[-1])

    fig.tight_layout()
    zoom_path = out_dir / f"single_cell_zoom_col{col_idx}.png"
    fig.savefig(zoom_path, dpi=200)
    plt.close(fig)
    print(f"Saved single-cell zoom to: {zoom_path}")


def _save_extreme_scenarios(
    grid: Grid2D,
    out_dir: Path,
    pts_per_cell: int = 100,
) -> None:
    """Plots the strongest S-shape and Bowl scenarios in the SPG grid."""
    scenarios = [
        ("Strongest S-Shape (Row 23, Cols 36–37, dN)", 23, 36, "dN"),
        ("Strongest Bowl / Trough (Row 23, Cols 35–36, dN)", 23, 35, "dN"),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.0))

    u_in = np.linspace(0.0, 1.0, pts_per_cell + 1)
    u_ext = np.linspace(-1.0, 2.0, 3 * pts_per_cell + 1)

    for ax, (title, row_idx, col_idx, comp) in zip(axes, scenarios, strict=True):
        n_fixed = grid.n_min + row_idx * grid.n_step
        e_cell_start = grid.e_min + col_idx * grid.e_step
        step = grid.e_step

        e_center = e_cell_start + 0.5 * step
        sg_idxs = grid._sgrid_idxs(n_fixed, e_center)
        sg_v_dict = grid.get_vs_at_idxs_cached(sg_idxs)
        sg_v_e_list, sg_v_n_list = zip(*sg_v_dict.values(), strict=True)
        bi = grid._init_interp(sg_v_n_list if comp == "dN" else sg_v_e_list)

        node_cols = [col_idx - 1, col_idx, col_idx + 1, col_idx + 2]
        node_idxs = tuple(row_idx * grid.c_count + c for c in node_cols)
        node_vals = grid.get_vs_at_idxs_cached(node_idxs)
        node_x_km = [(grid.e_min + c * step) * 1e-3 for c in node_cols]
        node_y = [
            node_vals[idx][1] if comp == "dN" else node_vals[idx][0]
            for idx in node_idxs
        ]
        g4, g5, g6, g7 = node_y

        x_in_km = (e_cell_start + u_in * step) * 1e-3
        x_ext_km = (e_cell_start + u_ext * step) * 1e-3
        y_ext = np.array([bi.interp(0.0, float(u)) for u in u_ext])

        eps = 1e-6
        e_true_m = np.clip(e_cell_start + u_ext * step, e_cell_start - step + eps, e_cell_start + 2 * step - eps)
        dn_true_all, de_true_all = grid.interp(np.full_like(e_true_m, n_fixed).tolist(), e_true_m.tolist())
        y_true_all = np.asarray(dn_true_all if comp == "dN" else de_true_all, dtype=np.float64)

        # Standard central-difference Catmull-Rom inside [0, 1] for comparison
        cr_sl = (g6 - g4) / 2.0
        cr_sr = (g7 - g5) / 2.0
        a0, a1 = g5, cr_sl
        a2 = -3.0 * g5 + 3.0 * g6 - 2.0 * cr_sl - cr_sr
        a3 = 2.0 * g5 - 2.0 * g6 + cr_sl + cr_sr
        y_cr = a0 + a1 * u_in + a2 * (u_in**2) + a3 * (u_in**3)

        ax.axvspan(
            e_cell_start * 1e-3,
            (e_cell_start + step) * 1e-3,
            color="#fef3c7",
            alpha=0.45,
            zorder=0,
            label=f"Target cell [0, 1] (cols {col_idx}–{col_idx + 1})",
        )
        ax.plot(
            x_ext_km,
            y_ext,
            linestyle="--",
            linewidth=1.7,
            color="#7c3aed",
            label="pytransdat polynomial (u in [-1, 2])",
            zorder=3,
        )
        ax.plot(
            x_ext_km,
            y_true_all,
            color="#0f172a",
            linewidth=2.4,
            label="True bicubic (pytransdat, u in [-1, 2])",
            zorder=4,
        )
        ax.plot(
            x_in_km,
            y_cr,
            color="#0ea5e9",
            linewidth=2.0,
            linestyle=":",
            label="Standard Catmull-Rom (central diff.)",
            zorder=5,
        )

        # Tangents
        slope_left = (-g7 + 4.0 * g6 - 3.0 * g5) / 2.0
        slope_right = (3.0 * g6 - 4.0 * g5 + g4) / 2.0
        u_tl = np.linspace(-0.4, 0.4, 20)
        u_tr = np.linspace(0.6, 1.4, 20)
        ax.plot(
            (e_cell_start + u_tl * step) * 1e-3,
            g5 + slope_left * u_tl,
            color="#059669",
            linewidth=2.0,
            linestyle="-.",
            zorder=5,
            label=f"Left tangent (cols {col_idx}–{col_idx + 2})",
        )
        ax.plot(
            (e_cell_start + u_tr * step) * 1e-3,
            g6 + slope_right * (u_tr - 1.0),
            color="#d97706",
            linewidth=2.0,
            linestyle="-.",
            zorder=5,
            label=f"Right tangent (cols {col_idx - 1}–{col_idx + 1})",
        )

        for bx in node_x_km:
            ax.axvline(bx, color="#94a3b8", linestyle=":", linewidth=0.9, zorder=1)

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
        for idx_pt, (c_num, x_pt, y_pt) in enumerate(zip(node_cols, node_x_km, node_y)):
            dx_off = 6 if idx_pt == 0 else (-6 if idx_pt == len(node_cols) - 1 else 0)
            ha = "left" if idx_pt == 0 else ("right" if idx_pt == len(node_cols) - 1 else "center")
            ax.annotate(
                f"col {c_num}\n({y_pt:+.3f} m)",
                xy=(x_pt, y_pt),
                xytext=(dx_off, -14 if idx_pt in (1, 2) else 10),
                textcoords="offset points",
                ha=ha,
                va="top" if idx_pt in (1, 2) else "bottom",
                fontsize=8,
                fontweight="bold",
                color="#991b1b",
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "#fca5a5", "alpha": 0.85},
                zorder=7,
            )

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel(f"{comp} Shift (m)")
        ax.set_xlabel("Stereo 70 Easting (km)")
        ax.set_xlim(x_ext_km[0], x_ext_km[-1])
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
        ax.legend(fontsize=7.5, ncol=2, loc="best", framealpha=0.9)

    fig.tight_layout()
    ext_path = out_dir / "extreme_scenarios_s_and_bowl.png"
    fig.savefig(ext_path, dpi=200)
    plt.close(fig)
    print(f"Saved extreme scenarios to: {ext_path}")


if __name__ == "__main__":
    main()
