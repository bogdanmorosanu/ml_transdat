"""Deep-dive stage-by-stage diagnostic visualizations for hybrid residual models."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pytransdat.trans_grid import Grid2D
from src.data.catalog import GridDataset, GridId
from src.data.spg_cache import get_default_grid2d
from src.models.composite_shift import CompositeResidualShiftModel
from src.viz.registry import PlotGroup, PlotSpec, PlotType, resolve_plot_types
from src.viz.scenarios import (
    EXTREME_SCENARIOS_GROUP1_GRADIENT_LINEAR,
    EXTREME_SCENARIOS_GROUP2_CURVATURE_KINKS,
    _compute_9cell_window,
)


def _draw_hybrid_9cell_panel(
    ax: plt.Axes,
    grid: Grid2D,
    hybrid_model: CompositeResidualShiftModel,
    scenario: dict[str, Any],
    s1_label: str = "Stage 1 error",
    s2_label: str = "Stage 2 NN correction",
    hybrid_label: str = "Hybrid: Stage 1 + Stage 2 error",
    pts_per_cell: int = 80,
    min_ylim_mm: float = 5.0,
) -> None:
    """Plot Stage 1 error, Stage 2 NN correction, and Hybrid error on a 9-cell transect."""
    win = _compute_9cell_window(grid, scenario, pts_per_cell)
    col_idx = win["col_idx"]
    comp = win["comp"]
    comp_idx = win["comp_idx"]
    step = win["step"]
    e_cell_start = win["e_cell_start"]
    x_win_km = win["x_win_km"]
    coords = win["coords"]
    y_ref = win["y_ref"]
    node_x_km = win["node_x_km"]

    ax.axvspan(
        e_cell_start * 1e-3,
        (e_cell_start + step) * 1e-3,
        color="#fef3c7",
        alpha=0.55,
        zorder=0,
        label=f"Target cell [0, 1] (cols {col_idx}–{col_idx + 1})",
    )
    ax.axhspan(
        -1.0,
        1.0,
        color="#dcfce7",
        alpha=0.65,
        zorder=0,
        label="±1 mm target band",
    )
    ax.axhline(
        0.0,
        color="#0f172a",
        linewidth=1.8,
        zorder=5,
        label="True Bicubic (0 mm)",
    )
    for bx in node_x_km:
        ax.axvline(bx, color="#94a3b8", linestyle=":", linewidth=0.8, zorder=1)

    ax2 = ax.twinx()
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    if np.any(np.isfinite(y_ref)):
        y_min = float(np.nanmin(y_ref))
        y_max = float(np.nanmax(y_ref))
        y_span = max(y_max - y_min, 0.05)
        ax2.plot(
            x_win_km,
            y_ref,
            color="#475569",
            linewidth=1.4,
            alpha=0.45,
            zorder=2,
            label=f"True {comp} shape (m, right axis)",
        )
        ax2.set_ylim(y_min - 0.15 * y_span, y_max + 0.15 * y_span)
    ax2.set_ylabel(f"True {comp} Shift (m)", color="#475569", fontsize=9.0)
    ax2.tick_params(axis="y", colors="#475569", labelsize=8.5)

    s1_pred = hybrid_model.stage1_model.predict(coords)[:, comp_idx]
    s2_corr = hybrid_model.stage2_model.predict(coords)[:, comp_idx]
    hyb_pred = s1_pred + s2_corr

    s1_err_mm = (s1_pred - y_ref) * 1000.0
    s2_corr_mm = s2_corr * 1000.0
    hyb_err_mm = (hyb_pred - y_ref) * 1000.0

    ax.plot(
        x_win_km,
        s1_err_mm,
        color="#2563eb",
        linestyle="--",
        linewidth=1.7,
        label=s1_label,
        zorder=6,
    )
    ax.plot(
        x_win_km,
        s2_corr_mm,
        color="#d97706",
        linestyle=":",
        linewidth=1.9,
        label=s2_label,
        zorder=6.2,
    )
    ax.plot(
        x_win_km,
        hyb_err_mm,
        color="#dc2626",
        linestyle="-",
        linewidth=2.0,
        label=hybrid_label,
        zorder=6.5,
    )

    max_abs = float(
        np.nanmax(
            [
                np.nanmax(np.abs(s1_err_mm)),
                np.nanmax(np.abs(s2_corr_mm)),
                np.nanmax(np.abs(hyb_err_mm)),
            ]
        )
    )
    dynamic_ylim = max(min_ylim_mm, float(np.ceil(max_abs * 1.15)))
    ax.set_ylim(-dynamic_ylim, dynamic_ylim)
    ax.set_xlim(x_win_km[0], x_win_km[-1])
    ax.set_title(scenario["title"], fontsize=10.5, fontweight="bold", pad=6)
    ax.set_ylabel("Residual vs. True Bicubic / NN Correction (mm)", fontsize=9.0)
    ax.grid(True, alpha=0.25, linewidth=0.6)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(
        h1 + h2,
        l1 + l2,
        loc="upper right",
        fontsize=7.5,
        framealpha=0.92,
        ncol=2,
    )


def plot_hybrid_extreme_residuals_9cells(
    hybrid_model: CompositeResidualShiftModel,
    out_dir: Path,
    model_name: str = "Hybrid",
    grid2d: Grid2D | None = None,
    pts_per_cell: int = 80,
    prefix: str = "hybrid_",
    dpi: int = 200,
    min_ylim_mm: float = 5.0,
) -> tuple[Path, Path]:
    """Render 9-cell extreme residual plots showing Stage 1, Stage 2, and Hybrid sum.

    :param hybrid_model: Fitted two-stage :class:`CompositeResidualShiftModel`.
    :type hybrid_model: CompositeResidualShiftModel
    :param out_dir: Target output directory for the two PNG files.
    :type out_dir: Path
    :param model_name: Display name for figure titles.
    :type model_name: str
    :param grid2d: Optional ``Grid2D`` instance for True Bicubic reference.
    :type grid2d: Grid2D | None
    :param pts_per_cell: Query samples per 11 km cell.
    :type pts_per_cell: int
    :param prefix: Output filename prefix.
    :type prefix: str
    :param dpi: Output figure DPI.
    :type dpi: int
    :param min_ylim_mm: Minimum symmetric vertical clamp in millimeters.
    :type min_ylim_mm: float
    :return: Tuple ``(group1_path, group2_path)``.
    :rtype: tuple[Path, Path]
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    active_grid = grid2d if grid2d is not None else get_default_grid2d()
    groups = [
        (
            EXTREME_SCENARIOS_GROUP1_GRADIENT_LINEAR,
            out_dir / f"{prefix}extreme_residuals_gradient_and_linear_9cells.png",
            f"{model_name}: Group 1 Gradient & Linear — Stage 1 + Stage 2 + Hybrid",
        ),
        (
            EXTREME_SCENARIOS_GROUP2_CURVATURE_KINKS,
            out_dir / f"{prefix}extreme_residuals_curvature_and_kinks_9cells.png",
            f"{model_name}: Group 2 Curvature & C1 Kinks — Stage 1 + Stage 2 + Hybrid",
        ),
    ]
    paths: list[Path] = []
    for scenarios, out_path, title in groups:
        fig, axes = plt.subplots(
            len(scenarios),
            1,
            figsize=(13.0, 3.6 * len(scenarios)),
            sharex=False,
        )
        if len(scenarios) == 1:
            axes = [axes]
        for ax, scen in zip(axes, scenarios, strict=True):
            _draw_hybrid_9cell_panel(
                ax=ax,
                grid=active_grid,
                hybrid_model=hybrid_model,
                scenario=scen,
                pts_per_cell=pts_per_cell,
                min_ylim_mm=min_ylim_mm,
            )
        axes[-1].set_xlabel("Easting (km)", fontsize=9.5)
        fig.suptitle(title, fontsize=12.5, fontweight="bold", y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)
        paths.append(out_path)
    return paths[0], paths[1]


def plot_hybrid_cancellation_scatter(
    s1_err_mm: np.ndarray,
    s2_corr_mm: np.ndarray,
    out_path: Path,
    title: str,
    dpi: int = 200,
) -> Path:
    """Plot Stage 1 Error (x) vs. Stage 2 NN Correction (y) for ``dN`` and ``dE``.

    :param s1_err_mm: Stage 1 error array of shape ``(N, 2)`` in millimeters.
    :type s1_err_mm: np.ndarray
    :param s2_corr_mm: Stage 2 correction array of shape ``(N, 2)`` in millimeters.
    :type s2_corr_mm: np.ndarray
    :param out_path: Output PNG path.
    :type out_path: Path
    :param title: Figure suptitle.
    :type title: str
    :param dpi: Output figure DPI.
    :type dpi: int
    :return: Saved output path.
    :rtype: Path
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.5), sharex=False, sharey=False)
    comps = [("dN (Northing)", 0), ("dE (Easting)", 1)]

    for ax, (comp_name, idx) in zip(axes, comps, strict=True):
        x = s1_err_mm[:, idx]
        y = s2_corr_mm[:, idx]
        hyb = np.abs(x + y)

        lim = float(np.percentile(np.maximum(np.abs(x), np.abs(y)), 99.5))
        lim = max(5.0, float(np.ceil(lim * 1.1)))

        diag = np.linspace(-lim, lim, 200)
        ax.fill_between(
            diag,
            -diag - 1.0,
            -diag + 1.0,
            color="#dcfce7",
            alpha=0.7,
            label="±1 mm final error band",
            zorder=1,
        )
        ax.plot(
            diag,
            -diag,
            color="#0f172a",
            linestyle="--",
            linewidth=1.5,
            label="Ideal cancellation (y = -x)",
            zorder=2,
        )
        ax.axhline(0.0, color="#94a3b8", linewidth=0.7, zorder=0)
        ax.axvline(0.0, color="#94a3b8", linewidth=0.7, zorder=0)

        sc = ax.scatter(
            x,
            y,
            c=hyb,
            cmap="viridis_r",
            vmin=0.0,
            vmax=2.0,
            s=6,
            alpha=0.55,
            edgecolors="none",
            zorder=3,
        )
        fig.colorbar(
            sc,
            ax=ax,
            label="|Hybrid Final Error| (mm)",
            fraction=0.046,
            pad=0.04,
        )

        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(comp_name, fontsize=11.0, fontweight="bold")
        ax.set_xlabel("Stage 1 Error: stage1 - true (mm)", fontsize=9.5)
        ax.set_ylabel("Stage 2 NN Correction (mm)", fontsize=9.5)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.legend(loc="upper right", fontsize=8.0, framealpha=0.9)

    fig.suptitle(title, fontsize=12.5, fontweight="bold", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_hybrid_spatial_3panel(
    coords: np.ndarray,
    s1_err_mm: np.ndarray,
    s2_corr_mm: np.ndarray,
    hyb_err_mm: np.ndarray,
    out_path: Path,
    title: str,
    dpi: int = 200,
) -> Path:
    """Plot vertical 3-panel spatial map of Stage 1, Stage 2, and Hybrid magnitudes.

    :param coords: Coordinates array of shape ``(N, 2)`` in meters ``(N, E)``.
    :type coords: np.ndarray
    :param s1_err_mm: Stage 1 error array of shape ``(N, 2)`` in millimeters.
    :type s1_err_mm: np.ndarray
    :param s2_corr_mm: Stage 2 correction array of shape ``(N, 2)`` in millimeters.
    :type s2_corr_mm: np.ndarray
    :param hyb_err_mm: Hybrid final error array of shape ``(N, 2)`` in millimeters.
    :type hyb_err_mm: np.ndarray
    :param out_path: Output PNG path.
    :type out_path: Path
    :param title: Figure suptitle.
    :type title: str
    :param dpi: Output figure DPI.
    :type dpi: int
    :return: Saved output path.
    :rtype: Path
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    e_km = coords[:, 1] * 1e-3
    n_km = coords[:, 0] * 1e-3

    s1_mag = np.linalg.norm(s1_err_mm, axis=1)
    s2_mag = np.linalg.norm(s2_corr_mm, axis=1)
    hyb_mag = np.linalg.norm(hyb_err_mm, axis=1)

    panels = [
        ("1. |Stage 1 Error| (mm)", s1_mag),
        ("2. |Stage 2 NN Correction| (mm)", s2_mag),
        ("3. |Hybrid Final Error| (mm)", hyb_mag),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(9.5, 15.5), sharex=True, sharey=True)
    for ax, (ptitle, vals) in zip(axes, panels, strict=True):
        panel_vmax = float(np.percentile(vals, 99.0))
        panel_vmax = max(1.0, float(np.ceil(panel_vmax * 2.0) / 2.0))
        sc = ax.scatter(
            e_km,
            n_km,
            c=vals,
            cmap="inferno_r",
            vmin=0.0,
            vmax=panel_vmax,
            s=1.5,
            alpha=0.85,
            edgecolors="none",
        )
        fig.colorbar(sc, ax=ax, label="Magnitude (mm)", fraction=0.03, pad=0.02)
        ax.set_title(
            f"{ptitle}\n"
            f"(med={np.median(vals):.2f} mm, p95={np.percentile(vals, 95):.2f} mm, "
            f"p99={np.percentile(vals, 99):.2f} mm)",
            fontsize=10.0,
            fontweight="bold",
        )
        ax.set_ylabel("Northing (km)", fontsize=9.5)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.2, linewidth=0.5)

    axes[-1].set_xlabel("Easting (km)", fontsize=9.5)
    fig.suptitle(title, fontsize=12.0, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_hybrid_subcell_phase_heatmap(
    coords: np.ndarray,
    s1_err_mm: np.ndarray,
    s2_corr_mm: np.ndarray,
    hyb_err_mm: np.ndarray,
    out_path: Path,
    title: str,
    bins: int = 16,
    grid2d: Grid2D | None = None,
    dpi: int = 200,
) -> Path:
    """Fold points into normalized ``[0, 1]^2`` cell coordinates for all 3 stages.

    :param coords: Coordinates array of shape ``(N, 2)`` in meters ``(N, E)``.
    :type coords: np.ndarray
    :param s1_err_mm: Stage 1 error array of shape ``(N, 2)`` in millimeters.
    :type s1_err_mm: np.ndarray
    :param s2_corr_mm: Stage 2 correction array of shape ``(N, 2)`` in millimeters.
    :type s2_corr_mm: np.ndarray
    :param hyb_err_mm: Hybrid final error array of shape ``(N, 2)`` in millimeters.
    :type hyb_err_mm: np.ndarray
    :param out_path: Output PNG path.
    :type out_path: Path
    :param title: Figure suptitle.
    :type title: str
    :param bins: Number of sub-cell bins per axis.
    :type bins: int
    :param grid2d: Optional ``Grid2D`` instance for cell origin and spacing.
    :type grid2d: Grid2D | None
    :param dpi: Output figure DPI.
    :type dpi: int
    :return: Saved output path.
    :rtype: Path
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    active_grid = grid2d if grid2d is not None else get_default_grid2d()
    u = ((coords[:, 1] - active_grid.e_min) / active_grid.e_step) % 1.0
    v = ((coords[:, 0] - active_grid.n_min) / active_grid.n_step) % 1.0

    iu = np.clip((u * bins).astype(int), 0, bins - 1)
    iv = np.clip((v * bins).astype(int), 0, bins - 1)

    stages = [
        ("1. |Stage 1 Error|", np.linalg.norm(s1_err_mm, axis=1)),
        ("2. |Stage 2 NN Correction|", np.linalg.norm(s2_corr_mm, axis=1)),
        ("3. |Hybrid Final Error|", np.linalg.norm(hyb_err_mm, axis=1)),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(12.5, 15.0))
    for row_idx, (stage_name, mag) in enumerate(stages):
        mean_grid = np.full((bins, bins), np.nan)
        p95_grid = np.full((bins, bins), np.nan)
        for r in range(bins):
            for c in range(bins):
                vals = mag[(iv == r) & (iu == c)]
                if len(vals) > 0:
                    mean_grid[r, c] = float(np.mean(vals))
                    p95_grid[r, c] = float(np.percentile(vals, 95))

        for col_idx, (stat_name, mat) in enumerate(
            [("Mean", mean_grid), ("95th-Percentile", p95_grid)]
        ):
            ax = axes[row_idx, col_idx]
            im = ax.imshow(
                mat,
                origin="lower",
                extent=(0.0, 1.0, 0.0, 1.0),
                cmap="inferno_r",
                aspect="equal",
            )
            fig.colorbar(im, ax=ax, label="Magnitude (mm)", fraction=0.046, pad=0.04)
            for k in range(1, 4):
                ax.axhline(
                    k / 4.0, color="#38bdf8", linestyle="--", linewidth=0.9, alpha=0.75
                )
                ax.axvline(
                    k / 4.0, color="#38bdf8", linestyle="--", linewidth=0.9, alpha=0.75
                )
            for k in range(1, 8):
                if k % 2 != 0:
                    ax.axhline(
                        k / 8.0, color="#94a3b8", linestyle=":", linewidth=0.6, alpha=0.6
                    )
                    ax.axvline(
                        k / 8.0, color="#94a3b8", linestyle=":", linewidth=0.6, alpha=0.6
                    )
            ax.set_title(
                f"{stage_name} — {stat_name} (mm)", fontsize=10.5, fontweight="bold"
            )
            ax.set_ylabel("Normalized Northing v ∈ [0, 1]", fontsize=9.0)
            if row_idx == len(stages) - 1:
                ax.set_xlabel("Normalized Easting u ∈ [0, 1]", fontsize=9.0)

    fig.suptitle(title, fontsize=12.5, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def select_hybrid_eval_grid(eval_grids: Sequence[GridId | str]) -> GridId | str:
    """Prefer a uniform test/grid dataset when multiple evaluation grids exist."""
    for gid in eval_grids:
        gid_str = gid.value if isinstance(gid, GridId) else str(gid)
        if "uniform" in gid_str.lower():
            return gid
    for gid in eval_grids:
        gid_str = gid.value if isinstance(gid, GridId) else str(gid)
        if "ro_grid_" in gid_str.lower():
            return gid
    return eval_grids[0]


def render_hybrid_deep_dive(
    hybrid_model: CompositeResidualShiftModel,
    model_name: str,
    eval_grids: Sequence[GridId | str],
    load_grid_fn: Callable[[GridId | str], GridDataset],
    out_dir: Path,
    plot_spec: PlotSpec = PlotGroup.HYBRID_DEEP_DIVE,
    dpi: int = 200,
) -> list[Path]:
    """Render all requested :class:`PlotGroup.HYBRID_DEEP_DIVE` plots for a champion hybrid model.

    :param hybrid_model: Fitted :class:`CompositeResidualShiftModel`.
    :type hybrid_model: CompositeResidualShiftModel
    :param model_name: Champion model display name.
    :type model_name: str
    :param eval_grids: Evaluation grid IDs configured for the champion.
    :type eval_grids: Sequence[GridId | str]
    :param load_grid_fn: Callable returning a :class:`GridDataset` for a grid ID.
    :type load_grid_fn: Callable[[GridId | str], GridDataset]
    :param out_dir: Output directory for generated PNGs.
    :type out_dir: Path
    :param plot_spec: Plot specification (defaults to :attr:`PlotGroup.HYBRID_DEEP_DIVE`).
    :type plot_spec: PlotSpec
    :param dpi: Output figure DPI.
    :type dpi: int
    :return: List of generated PNG paths.
    :rtype: list[Path]
    """
    resolved = resolve_plot_types(plot_spec)
    generated: list[Path] = []

    if PlotType.HYBRID_EXTREME_RESIDUALS_9CELLS in resolved:
        g1_path, g2_path = plot_hybrid_extreme_residuals_9cells(
            hybrid_model=hybrid_model,
            out_dir=out_dir,
            model_name=model_name,
            prefix="hybrid_champion_",
            dpi=dpi,
        )
        generated.extend([g1_path, g2_path])

    needs_eval_grid = any(
        pt in resolved
        for pt in (
            PlotType.HYBRID_CANCELLATION_SCATTER,
            PlotType.HYBRID_SPATIAL_3PANEL,
            PlotType.HYBRID_SUBCELL_PHASE_HEATMAP,
        )
    )
    if needs_eval_grid and eval_grids:
        target_gid = select_hybrid_eval_grid(eval_grids)
        gid_label = target_gid.value if isinstance(target_gid, GridId) else str(target_gid)
        ds = load_grid_fn(target_gid)
        coords = ds.coords
        y_true = ds.true_shifts

        s1_pred = hybrid_model.stage1_model.predict(coords)
        s2_corr = hybrid_model.stage2_model.predict(coords)
        hyb_pred = s1_pred + s2_corr

        s1_err_mm = (s1_pred - y_true) * 1000.0
        s2_corr_mm = s2_corr * 1000.0
        hyb_err_mm = (hyb_pred - y_true) * 1000.0

        if PlotType.HYBRID_CANCELLATION_SCATTER in resolved:
            generated.append(
                plot_hybrid_cancellation_scatter(
                    s1_err_mm=s1_err_mm,
                    s2_corr_mm=s2_corr_mm,
                    out_path=out_dir / f"hybrid_champion_{gid_label}_cancellation_scatter.png",
                    title=f"Cancellation Correlation: {model_name} ({gid_label})",
                    dpi=dpi,
                )
            )
        if PlotType.HYBRID_SPATIAL_3PANEL in resolved:
            generated.append(
                plot_hybrid_spatial_3panel(
                    coords=coords,
                    s1_err_mm=s1_err_mm,
                    s2_corr_mm=s2_corr_mm,
                    hyb_err_mm=hyb_err_mm,
                    out_path=out_dir / f"hybrid_champion_{gid_label}_spatial_3panel_stages.png",
                    title=f"Stage-by-Stage Spatial Error Reduction\n({model_name}, {gid_label})",
                    dpi=dpi,
                )
            )
        if PlotType.HYBRID_SUBCELL_PHASE_HEATMAP in resolved:
            generated.append(
                plot_hybrid_subcell_phase_heatmap(
                    coords=coords,
                    s1_err_mm=s1_err_mm,
                    s2_corr_mm=s2_corr_mm,
                    hyb_err_mm=hyb_err_mm,
                    out_path=out_dir / f"hybrid_champion_{gid_label}_subcell_phase_heatmap.png",
                    title=(
                        f"Sub-Cell Phase Error Heatmap — All Stages ({model_name}, {gid_label})\n"
                        "(u, v ∈ [0, 1]², 16×16 bins, 4×4 dashed / 8×8 dotted)"
                    ),
                    dpi=dpi,
                )
            )

    return generated
