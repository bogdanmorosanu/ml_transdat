"""Canonical 9-cell (99 km) extreme scenario definitions, 1D evaluator, and transect plots."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pytransdat import exceptions
from pytransdat.trans_grid import Grid2D
from src.data.spg_cache import get_default_grid2d
from src.models.base import BaseGridShiftModel

EXTREME_SCENARIOS_GROUP1_GRADIENT_LINEAR: tuple[dict[str, Any], ...] = (
    {
        "id": "steepest_fall",
        "title": "Scenario 1A: Steepest Left-to-Right Fall (Row 23, Col 34, dN) — 99 km (9-Cell) Window",
        "row": 23,
        "col": 34,
        "comp": "dN",
        "desc": "Target cell [0, 1] drops -0.536 m across 11 km (4 cells left + target + 4 cells right).",
    },
    {
        "id": "steepest_rise_de",
        "title": "Scenario 1B: Steepest Left-to-Right Rise in Easting (Row 45, Col 38, dE) — 99 km (9-Cell) Window",
        "row": 45,
        "col": 38,
        "comp": "dE",
        "desc": "Target cell [0, 1] rises +0.453 m across 11 km (4 cells left + target + 4 cells right).",
    },
    {
        "id": "flat_bending",
        "title": "Scenario 1C: Flat Endpoints with High Internal Sag (Row 45, Col 35, dN) — 99 km (9-Cell) Window",
        "row": 45,
        "col": 35,
        "comp": "dN",
        "desc": "Bounding nodes differ by 1.5 mm; TransDatRO bows 38.0 mm internally.",
    },
)

EXTREME_SCENARIOS_GROUP2_CURVATURE_KINKS: tuple[dict[str, Any], ...] = (
    {
        "id": "s_shape",
        "title": "Scenario 2A: True S-Shape (Row 23, Col 36, dN) — 99 km (9-Cell) Window",
        "row": 23,
        "col": 36,
        "comp": "dN",
        "desc": "Sigmoidal inflection inside target cell [0, 1] (4 cells left + target + 4 cells right).",
    },
    {
        "id": "bowl",
        "title": "Scenario 2B: Strongest Bowl / Trough (Row 23, Col 35, dN) — 99 km (9-Cell) Window",
        "row": 23,
        "col": 35,
        "comp": "dN",
        "desc": "Deep trough dipping 74.7 mm below chord line inside target cell [0, 1].",
    },
    {
        "id": "dome",
        "title": "Scenario 2C: Strongest Dome / Peak (Row 3, Col 34, dN) — 99 km (9-Cell) Window",
        "row": 3,
        "col": 34,
        "comp": "dN",
        "desc": "Interior crest peaking 41.1 mm above chord line inside target cell [0, 1].",
    },
    {
        "id": "worst_kink",
        "title": "Scenario 2D: Worst-Case C1 Boundary Discontinuity (Row 10, Col 16, dN) — 99 km (9-Cell) Window",
        "row": 10,
        "col": 16,
        "comp": "dN",
        "desc": "Sharpest C1 slope break (0.788 m/step at col 17) across Romanian territory.",
    },
)


def eval_pytransdat_1d_safe(
    grid: Grid2D,
    n_pts: np.ndarray,
    e_pts: np.ndarray,
    comp: str,
) -> np.ndarray:
    """Evaluate pytransdat ``Grid2D`` along a 1D transect, returning NaN at NoData cells."""
    out = np.full(len(e_pts), np.nan, dtype=np.float64)
    for idx, (nv, ev) in enumerate(zip(n_pts, e_pts, strict=True)):
        try:
            dn, de = grid.interp([float(nv)], [float(ev)])
            out[idx] = dn[0] if comp == "dN" else de[0]
        except (exceptions.OutOfGridErr, exceptions.NoDataGridErr):
            continue
    return out


def _draw_9cell_extreme_panel(
    ax: plt.Axes,
    grid: Grid2D,
    models: Mapping[str, BaseGridShiftModel],
    scenario: dict[str, Any],
    pts_per_cell: int = 80,
) -> None:
    """Plot a 99 km (9-cell: 4 left + target [0, 1] + 4 right) extreme scenario comparison."""
    row_idx, col_idx, comp = scenario["row"], scenario["col"], scenario["comp"]
    comp_idx = 0 if comp == "dN" else 1
    step = grid.e_step
    n_fixed = grid.n_min + row_idx * grid.n_step
    e_cell_start = grid.e_min + col_idx * step

    eps = 1e-5
    u_win = np.linspace(-4.0 + eps, 5.0 - eps, 9 * pts_per_cell + 1)
    e_win = e_cell_start + u_win * step
    x_win_km = e_win * 1e-3
    n_win = np.full_like(e_win, n_fixed)
    coords = np.column_stack([n_win, e_win])

    ax.axvspan(
        e_cell_start * 1e-3,
        (e_cell_start + step) * 1e-3,
        color="#fef3c7",
        alpha=0.60,
        zorder=0,
        label=f"Target cell [0, 1] (cols {col_idx}–{col_idx + 1})",
    )

    y_ref = eval_pytransdat_1d_safe(grid, n_win, e_win, comp)
    ax.plot(
        x_win_km,
        y_ref,
        color="#0f172a",
        linewidth=2.4,
        label="True Bicubic (TransDatRO)",
        zorder=6,
    )

    palette = [
        ("#0284c7", "-"),
        ("#dc2626", "--"),
        ("#16a34a", "-."),
        ("#d97706", ":"),
        ("#9333ea", "--"),
        ("#0d9488", "-."),
        ("#e11d48", ":"),
    ]
    in_cell_mask = (u_win >= 0.0) & (u_win <= 1.0) & np.isfinite(y_ref)
    valid_ref = np.isfinite(y_ref)
    diff_lines: list[str] = []

    for idx, (label, model) in enumerate(models.items()):
        color, ls = palette[idx % len(palette)]
        y_vals = model.predict(coords)[:, comp_idx]
        if np.any(valid_ref):
            diff_win_mm = (y_vals[valid_ref] - y_ref[valid_ref]) * 1000.0
            rms_win = float(np.sqrt(np.mean(diff_win_mm**2)))
            max_win = float(np.max(np.abs(diff_win_mm)))
            if np.any(in_cell_mask):
                diff_cell_mm = (y_vals[in_cell_mask] - y_ref[in_cell_mask]) * 1000.0
                rms_cell = float(np.sqrt(np.mean(diff_cell_mm**2)))
                diff_lines.append(
                    f"{label}: 99km RMS {rms_win:.1f}mm (Max {max_win:.1f}) | Cell {rms_cell:.1f}mm"
                )
            else:
                diff_lines.append(f"{label}: 99km RMS {rms_win:.1f}mm (Max {max_win:.1f})")

        ax.plot(
            x_win_km,
            y_vals,
            color=color,
            linestyle=ls,
            linewidth=1.7,
            alpha=0.92,
            label=label,
            zorder=5 - (idx % 3) * 0.2,
        )

    node_cols = list(range(col_idx - 4, col_idx + 6))
    node_idxs = tuple(row_idx * grid.c_count + c for c in node_cols)
    node_vals = grid.get_vs_at_idxs_cached(node_idxs)
    node_x_km = [(grid.e_min + c * step) * 1e-3 for c in node_cols]
    node_y = [
        node_vals[idx][1] if comp == "dN" else node_vals[idx][0]
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
        s=48,
        zorder=7,
        label="SPG Grid Nodes (11 km)",
    )

    for c_num, x_pt, y_pt in zip(node_cols, node_x_km, node_y, strict=True):
        if not np.isfinite(y_pt) or abs(y_pt) > 100.0:
            continue
        is_target = c_num in (col_idx, col_idx + 1)
        ax.annotate(
            f"col {c_num}\n({y_pt:+.3f} m)" if is_target else f"{c_num}",
            xy=(x_pt, y_pt),
            xytext=(0, -16 if is_target else 8),
            textcoords="offset points",
            ha="center",
            va="top" if is_target else "bottom",
            fontsize=7.2 if is_target else 6.8,
            fontweight="bold" if is_target else "normal",
            color="#991b1b" if is_target else "#475569",
            bbox=(
                {
                    "boxstyle": "round,pad=0.2",
                    "fc": "#fee2e2",
                    "ec": "#ef4444",
                    "alpha": 0.85,
                }
                if is_target
                else None
            ),
            zorder=8,
        )

    finite_nodes = [y for y in node_y if np.isfinite(y) and abs(y) < 100.0]
    y_min = min([float(np.nanmin(y_ref))] + finite_nodes) if finite_nodes else float(np.nanmin(y_ref))
    y_max = max([float(np.nanmax(y_ref))] + finite_nodes) if finite_nodes else float(np.nanmax(y_ref))
    y_span = max(y_max - y_min, 0.05)
    ax.set_ylim(y_min - 0.22 * y_span, y_max + 0.24 * y_span)
    ax.set_xlim(x_win_km[0], x_win_km[-1])

    if diff_lines:
        ax.text(
            0.012,
            0.035,
            "Diff vs. TransDatRO:\n" + "\n".join(diff_lines[:6]),
            transform=ax.transAxes,
            fontsize=7.0,
            va="bottom",
            ha="left",
            bbox={
                "boxstyle": "round,pad=0.35",
                "fc": "white",
                "ec": "#cbd5e1",
                "alpha": 0.92,
            },
            zorder=9,
        )

    ax.set_title(f"{scenario['title']}\n{scenario['desc']}", fontsize=10.5, fontweight="bold")
    ax.set_ylabel(f"{comp} Shift (m)")
    ax.set_xlabel("Stereo 70 Easting (km)")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
    ax.legend(fontsize=7.2, ncol=3, loc="upper right", framealpha=0.92)


def plot_extreme_scenarios_9cells(
    models: Mapping[str, BaseGridShiftModel],
    out_dir: Path,
    grid2d: Grid2D | None = None,
    pts_per_cell: int = 80,
    prefix: str = "",
    dpi: int = 200,
) -> tuple[Path, Path]:
    """Render Group 1 (Gradient & Linear) and Group 2 (Curvature & Kinks) 9-cell (99 km) plots.

    :param models: Mapping ``display_label -> fitted BaseGridShiftModel``.
    :type models: Mapping[str, BaseGridShiftModel]
    :param out_dir: Target output directory for the two PNG files.
    :type out_dir: Path
    :param grid2d: Optional ``Grid2D`` instance for true bicubic reference.
    :type grid2d: Grid2D | None
    :param pts_per_cell: Query samples per 11 km cell.
    :type pts_per_cell: int
    :param prefix: Optional filename prefix.
    :type prefix: str
    :param dpi: Output figure DPI.
    :type dpi: int
    :return: Tuple ``(group1_path, group2_path)``.
    :rtype: tuple[Path, Path]
    """
    active_grid = grid2d if grid2d is not None else get_default_grid2d()
    out_dir.mkdir(parents=True, exist_ok=True)

    fig1, axes1 = plt.subplots(
        len(EXTREME_SCENARIOS_GROUP1_GRADIENT_LINEAR),
        1,
        figsize=(13.5, 4.6 * len(EXTREME_SCENARIOS_GROUP1_GRADIENT_LINEAR)),
    )
    for ax, scen in zip(axes1, EXTREME_SCENARIOS_GROUP1_GRADIENT_LINEAR, strict=True):
        _draw_9cell_extreme_panel(ax, active_grid, models, scen, pts_per_cell=pts_per_cell)
    fig1.suptitle(
        "Group 1: Gradient & Linear Extreme Scenarios — 99 km (9-Cell) Window",
        fontsize=13.0,
        fontweight="bold",
        y=0.995,
    )
    fig1.tight_layout()
    path1 = out_dir / f"{prefix}extreme_scenarios_gradient_and_linear_9cells.png"
    fig1.savefig(path1, dpi=dpi)
    plt.close(fig1)

    fig2, axes2 = plt.subplots(
        len(EXTREME_SCENARIOS_GROUP2_CURVATURE_KINKS),
        1,
        figsize=(13.5, 4.6 * len(EXTREME_SCENARIOS_GROUP2_CURVATURE_KINKS)),
    )
    for ax, scen in zip(axes2, EXTREME_SCENARIOS_GROUP2_CURVATURE_KINKS, strict=True):
        _draw_9cell_extreme_panel(ax, active_grid, models, scen, pts_per_cell=pts_per_cell)
    fig2.suptitle(
        "Group 2: Curvature & C1 Boundary Kink Extreme Scenarios — 99 km (9-Cell) Window",
        fontsize=13.0,
        fontweight="bold",
        y=0.995,
    )
    fig2.tight_layout()
    path2 = out_dir / f"{prefix}extreme_scenarios_curvature_and_kinks_9cells.png"
    fig2.savefig(path2, dpi=dpi)
    plt.close(fig2)

    return path1, path2
