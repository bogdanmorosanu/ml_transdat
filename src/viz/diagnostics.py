"""Core diagnostic plots: point distributions, spatial error maps, ECDF, scatter, subcell, and quiver."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pytransdat import TransRO
from src.evaluation.evaluator import EvaluationResult


def plot_point_distribution(
    inputs: Sequence[Path],
    out_path: Path,
    crs: str = "st70",
    background: Path | None = None,
    color_by: str | None = None,
    cmap: str = "viridis",
    grid_lines: bool = False,
    title: str | None = None,
    point_size: float = 1.5,
    dpi: int = 200,
) -> Path:
    """Plot one or more coordinate datasets in Stereo 70 or ETRS89."""
    fig, ax = plt.subplots(figsize=(10, 7.5))

    def _extract_xy(
        df: pd.DataFrame, crs_name: str
    ) -> tuple[pd.Series, pd.Series, str, str]:
        if crs_name == "st70":
            return (
                df["st70_e"] * 1e-3,
                df["st70_n"] * 1e-3,
                "Stereo 70 Easting (km)",
                "Stereo 70 Northing (km)",
            )
        if crs_name == "etrs89":
            return (
                df["etrs89_lon_deg"],
                df["etrs89_lat_deg"],
                "ETRS89 Longitude (deg)",
                "ETRS89 Latitude (deg)",
            )
        raise ValueError(f"Unsupported CRS '{crs_name}'")

    lat_ref_deg: float | None = None
    x_label, y_label = "", ""

    if grid_lines and crs == "st70":
        grid2d = TransRO()._t_gr2d
        e_ticks = [
            (grid2d.e_min + i * grid2d.e_step) * 1e-3 for i in range(grid2d.c_count)
        ]
        n_ticks = [
            (grid2d.n_min + j * grid2d.n_step) * 1e-3 for j in range(grid2d.r_count)
        ]
        for x_g in e_ticks:
            ax.axvline(x_g, color="#e5e7eb", linewidth=0.4, zorder=0)
        for y_g in n_ticks:
            ax.axhline(y_g, color="#e5e7eb", linewidth=0.4, zorder=0)

    if background is not None:
        bg_df = (
            pd.read_parquet(background)
            if background.suffix == ".parquet"
            else pd.read_csv(background)
        )
        bx, by, x_label, y_label = _extract_xy(bg_df, crs)
        if crs == "etrs89":
            lat_ref_deg = float(by.mean())
        ax.scatter(
            bx,
            by,
            s=max(0.8, point_size * 0.7),
            color="#64748b",
            alpha=0.75,
            edgecolors="none",
            label=f"{background.stem} (n={len(bg_df):,})",
            zorder=1,
        )

    palette = ["#2563eb", "#dc2626", "#059669", "#d97706"]
    last_scatter = None
    for idx, path in enumerate(inputs):
        df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        x, y, x_label, y_label = _extract_xy(df, crs)
        if crs == "etrs89" and lat_ref_deg is None:
            lat_ref_deg = float(y.mean())

        if color_by is not None:
            last_scatter = ax.scatter(
                x,
                y,
                c=df[color_by],
                cmap=cmap,
                s=point_size,
                alpha=0.85,
                edgecolors="none",
                label=f"{path.stem} (n={len(df):,})",
                zorder=2 + idx,
            )
        else:
            ax.scatter(
                x,
                y,
                s=point_size,
                color=palette[idx % len(palette)],
                alpha=0.75,
                edgecolors="none",
                label=f"{path.stem} (n={len(df):,})",
                zorder=2 + idx,
            )

    if last_scatter is not None and color_by is not None:
        cbar = fig.colorbar(last_scatter, ax=ax, shrink=0.85, pad=0.02)
        cbar.set_label(color_by)

    if crs == "st70":
        ax.set_aspect("equal", adjustable="box")
    elif lat_ref_deg is not None:
        ax.set_aspect(1.0 / math.cos(math.radians(lat_ref_deg)), adjustable="box")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title or f"Spatial Distribution ({crs.upper()})")
    ax.legend(loc="upper right", markerscale=3.0, framealpha=0.9)
    if not grid_lines:
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _draw_spatial_error_ax(
    fig: plt.Figure,
    ax: plt.Axes,
    err_df: pd.DataFrame,
    summary: dict[str, Any],
    title: str,
    log_scale: bool = False,
    vmin_mm: float = 0.05,
    vmax_mm: float | None = None,
) -> None:
    """Paint a single 2D spatial error heatmap onto ``ax``."""
    e_km = err_df["st70_e"].to_numpy(dtype=np.float64) * 1e-3
    n_km = err_df["st70_n"].to_numpy(dtype=np.float64) * 1e-3
    err = err_df["err_2d_m"].to_numpy(dtype=np.float64)

    if log_scale:
        err_mm = err * 1000.0
        p99_mm = float(summary["p99_2d_m"]) * 1000.0
        max_mm = float(summary["max_2d_m"]) * 1000.0
        vmax_eff = (
            vmax_mm
            if vmax_mm is not None
            else max(p99_mm if max_mm > 2.0 * p99_mm and p99_mm > 0 else max_mm, 10.0)
        )
        sc = ax.scatter(
            e_km,
            n_km,
            c=np.clip(err_mm, vmin_mm, vmax_eff),
            cmap="turbo",
            norm=mcolors.LogNorm(vmin=vmin_mm, vmax=vmax_eff),
            s=6.5,
            marker="s",
            alpha=0.95,
            edgecolors="none",
            linewidths=0,
        )
        cbar = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
        base_ticks = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
        ticks = [t for t in base_ticks if vmin_mm <= t <= vmax_eff]
        if vmax_eff > 12.0:
            ticks.append(vmax_eff)
        if ticks:
            cbar.set_ticks(ticks)
            labels = [
                f"≤{t:g}"
                if i == 0
                else (
                    f"≥{t:.1f}"
                    if i == len(ticks) - 1 and t > 10.0
                    else (f"≥{t:g}" if i == len(ticks) - 1 else f"{t:g}")
                )
                for i, t in enumerate(ticks)
            ]
            cbar.set_ticklabels(labels)
        cbar.set_label(
            f"2D Horizontal Error (mm, log scale) [max={max_mm:.1f} mm]"
        )
    else:
        p99 = summary["p99_2d_m"]
        vmax = p99 if summary["max_2d_m"] > 2.0 * p99 and p99 > 0 else summary["max_2d_m"]

        sc = ax.scatter(
            e_km,
            n_km,
            c=err,
            cmap="inferno",
            vmin=0.0,
            vmax=vmax,
            s=6.5,
            marker="s",
            alpha=0.95,
            edgecolors="none",
            linewidths=0,
        )
        cbar = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
        cbar_label = "2D Horizontal Error (m)"
        if vmax < summary["max_2d_m"]:
            cbar_label += f" [clamped at p99={p99:.3f} m; max={summary['max_2d_m']:.3f} m]"
        cbar.set_label(cbar_label)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Stereo 70 Easting (km)")
    ax.set_ylabel("Stereo 70 Northing (km)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)


def plot_spatial_error(
    err_df: pd.DataFrame,
    summary: dict[str, Any],
    out_path: Path,
    model_name: str,
    direction: str,
    dpi: int = 200,
    log_scale: bool = False,
    vmin_mm: float = 0.05,
    vmax_mm: float | None = None,
) -> Path:
    """Plot borderless square heatmap of 2D horizontal errors across Romania."""
    fig, ax = plt.subplots(figsize=(10, 7.5))
    _draw_spatial_error_ax(
        fig,
        ax,
        err_df,
        summary,
        f"{model_name} ({direction}) — Spatial 2D Error Map",
        log_scale=log_scale,
        vmin_mm=vmin_mm,
        vmax_mm=vmax_mm,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_spatial_error_rows(
    results: Sequence[EvaluationResult],
    out_path: Path,
    dpi: int = 200,
    log_scale: bool = False,
    vmin_mm: float = 0.05,
    vmax_mm: float | None = None,
) -> Path:
    """Plot a stacked 1-row-per-model figure of spatial 2D error maps."""
    if not results:
        raise ValueError("results must contain at least one EvaluationResult.")
    n_models = len(results)
    fig, axes = plt.subplots(
        n_models, 1, figsize=(10.0, 6.2 * n_models), squeeze=False
    )
    for idx, res in enumerate(results):
        s = res.summary
        title = (
            f"{res.model_name} ({res.direction} on {res.dataset_id}) — "
            f"Mean={s['mean_2d_m'] * 1000.0:.2f} mm | "
            f"RMSE={s['rmse_2d_m'] * 1000.0:.2f} mm | "
            f"P95={s['p95_2d_m'] * 1000.0:.2f} mm | "
            f"Max={s['max_2d_m'] * 1000.0:.2f} mm"
        )
        _draw_spatial_error_ax(
            fig,
            axes[idx, 0],
            res.err_df,
            s,
            title,
            log_scale=log_scale,
            vmin_mm=vmin_mm,
            vmax_mm=vmax_mm,
        )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _draw_hist_ecdf_axes(
    ax_hist: plt.Axes,
    ax_ecdf: plt.Axes,
    err_df: pd.DataFrame,
    summary: dict[str, Any],
    row_label: str | None = None,
) -> None:
    """Paint Histogram and ECDF panels onto ``(ax_hist, ax_ecdf)``."""
    err = np.sort(err_df["err_2d_m"].to_numpy(dtype=np.float64))
    ecdf = np.arange(1, len(err) + 1) / len(err) * 100.0

    ax_hist.hist(
        err, bins=60, color="#2563eb", alpha=0.8, edgecolor="#1e40af", linewidth=0.4
    )
    markers = [
        ("p50", summary["p50_2d_m"], "#16a34a", "--"),
        ("p95", summary["p95_2d_m"], "#d97706", "-."),
        ("p99", summary["p99_2d_m"], "#dc2626", ":"),
    ]
    for label, val, color, ls in markers:
        ax_hist.axvline(
            val, color=color, linestyle=ls, linewidth=1.6, label=f"{label}={val:.4f} m"
        )

    prefix = f"{row_label} — " if row_label else ""
    ax_hist.set_xlabel("2D Horizontal Error (m)")
    ax_hist.set_ylabel("Point Count")
    ax_hist.set_title(f"{prefix}Error Distribution (Histogram)")
    ax_hist.legend(framealpha=0.9)
    ax_hist.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    ax_ecdf.plot(err, ecdf, color="#2563eb", linewidth=1.8)
    for label, val, color, ls in markers:
        ax_ecdf.axvline(
            val, color=color, linestyle=ls, linewidth=1.3, label=f"{label}={val:.4f} m"
        )
    ax_ecdf.set_xlabel("2D Horizontal Error (m)")
    ax_ecdf.set_ylabel("Cumulative % of Points")
    ax_ecdf.set_ylim(0.0, 101.0)
    ax_ecdf.set_title(f"{prefix}Cumulative Error Distribution (ECDF)")
    ax_ecdf.legend(loc="lower right", framealpha=0.9)
    ax_ecdf.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)


def plot_hist_ecdf(
    err_df: pd.DataFrame,
    summary: dict[str, Any],
    out_path: Path,
    model_name: str,
    direction: str,
    dpi: int = 200,
) -> Path:
    """Plot dual-panel Histogram and Empirical Cumulative Distribution Function (ECDF)."""
    fig, (ax_hist, ax_ecdf) = plt.subplots(1, 2, figsize=(13, 5))
    _draw_hist_ecdf_axes(ax_hist, ax_ecdf, err_df, summary)
    fig.suptitle(f"{model_name} ({direction}) — Error Distribution", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_hist_ecdf_rows(
    results: Sequence[EvaluationResult],
    out_path: Path,
    dpi: int = 200,
) -> Path:
    """Plot an ``N x 2`` stacked figure of Histogram + ECDF with 1 row per model."""
    if not results:
        raise ValueError("results must contain at least one EvaluationResult.")
    n_models = len(results)
    fig, axes = plt.subplots(
        n_models, 2, figsize=(13.0, 4.3 * n_models), squeeze=False
    )
    for idx, res in enumerate(results):
        _draw_hist_ecdf_axes(
            axes[idx, 0],
            axes[idx, 1],
            res.err_df,
            res.summary,
            row_label=f"{res.model_name} ({res.direction})",
        )
    ds_id = results[0].dataset_id
    fig.suptitle(f"Error Distribution Comparison ({ds_id})", fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _draw_signed_scatter_ax(
    ax: plt.Axes,
    err_df: pd.DataFrame,
    summary: dict[str, Any],
    title: str,
) -> None:
    """Paint a 2D signed residual scatter onto ``ax``."""
    ee = err_df["err_e_m"].to_numpy(dtype=np.float64)
    en = err_df["err_n_m"].to_numpy(dtype=np.float64)

    ax.scatter(
        ee, en, s=1.5, color="#2563eb", alpha=0.35, edgecolors="none", label="Residuals"
    )
    ax.axhline(0.0, color="#64748b", linewidth=0.8)
    ax.axvline(0.0, color="#64748b", linewidth=0.8)

    ax.scatter(
        [summary["bias_e_m"]],
        [summary["bias_n_m"]],
        color="#dc2626",
        s=70,
        marker="x",
        linewidths=2.0,
        label=f"Mean bias ({summary['bias_e_m']:+.3f}, {summary['bias_n_m']:+.3f}) m",
        zorder=5,
    )

    theta = np.linspace(0.0, 2.0 * np.pi, 256)
    for label, radius, color, ls in [
        ("p50", summary["p50_2d_m"], "#16a34a", "--"),
        ("p95", summary["p95_2d_m"], "#d97706", "-."),
    ]:
        ax.plot(
            radius * np.cos(theta),
            radius * np.sin(theta),
            color=color,
            linestyle=ls,
            linewidth=1.4,
            label=f"{label}={radius:.4f} m",
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Signed Easting Error dE (m)")
    ax.set_ylabel("Signed Northing Error dN (m)")
    ax.set_title(title)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)


def plot_signed_scatter(
    err_df: pd.DataFrame,
    summary: dict[str, Any],
    out_path: Path,
    model_name: str,
    direction: str,
    dpi: int = 200,
) -> Path:
    """Plot 2D signed residual scatter (dE vs dN) with bias and tolerance circles."""
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    _draw_signed_scatter_ax(
        ax, err_df, summary, f"{model_name} ({direction}) — Signed Error Vector Scatter"
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_signed_scatter_rows(
    results: Sequence[EvaluationResult],
    out_path: Path,
    dpi: int = 200,
) -> Path:
    """Plot a stacked 1-row-per-model figure of signed residual scatters."""
    if not results:
        raise ValueError("results must contain at least one EvaluationResult.")
    n_models = len(results)
    fig, axes = plt.subplots(
        n_models, 1, figsize=(7.5, 6.8 * n_models), squeeze=False
    )
    for idx, res in enumerate(results):
        _draw_signed_scatter_ax(
            axes[idx, 0],
            res.err_df,
            res.summary,
            f"{res.model_name} ({res.direction} on {res.dataset_id}) — Signed Error Scatter",
        )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _has_discrete_subcell_uv(err_df: pd.DataFrame) -> tuple[list[float], list[float]] | None:
    if "u" not in err_df.columns or "v" not in err_df.columns:
        return None
    u_vals = sorted(err_df["u"].unique())
    v_vals = sorted(err_df["v"].unique())
    if len(u_vals) > 32 or len(v_vals) > 32:
        return None
    return u_vals, v_vals


def _draw_subcell_heatmap_axes(
    fig: plt.Figure,
    ax_mean: plt.Axes,
    ax_p95: plt.Axes,
    err_df: pd.DataFrame,
    u_vals: list[float],
    v_vals: list[float],
    row_label: str | None = None,
) -> None:
    """Paint Mean and p95 sub-cell (u, v) heatmaps onto ``(ax_mean, ax_p95)``."""
    mean_piv = err_df.pivot_table(
        index="u", columns="v", values="err_2d_m", aggfunc="mean"
    ).reindex(index=u_vals, columns=v_vals)
    p95_piv = err_df.pivot_table(
        index="u",
        columns="v",
        values="err_2d_m",
        aggfunc=lambda s: float(np.percentile(s, 95.0)),
    ).reindex(index=u_vals, columns=v_vals)

    prefix = f"{row_label} — " if row_label else ""
    for ax, piv, title in [
        (ax_mean, mean_piv, f"{prefix}Mean 2D Error (m)"),
        (ax_p95, p95_piv, f"{prefix}p95 2D Error (m)"),
    ]:
        data = piv.to_numpy(dtype=np.float64)
        im = ax.imshow(data, origin="lower", cmap="viridis", aspect="equal")
        ax.set_xticks(range(len(v_vals)))
        ax.set_xticklabels([f"{v:.2f}" for v in v_vals])
        ax.set_yticks(range(len(u_vals)))
        ax.set_yticklabels([f"{u:.2f}" for u in u_vals])
        ax.set_xlabel("Sub-cell Easting offset v")
        ax.set_ylabel("Sub-cell Northing offset u")
        ax.set_title(title)

        mid = 0.5 * (np.nanmin(data) + np.nanmax(data))
        for i in range(len(u_vals)):
            for j in range(len(v_vals)):
                val = data[i, j]
                txt_color = "white" if val < mid else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.4f}",
                    ha="center",
                    va="center",
                    color=txt_color,
                    fontsize=8.5,
                )
        fig.colorbar(im, ax=ax, shrink=0.85)


def plot_subcell_heatmap(
    err_df: pd.DataFrame,
    out_path: Path,
    model_name: str,
    direction: str,
    dpi: int = 200,
) -> Path | None:
    """Plot mean and p95 2D error profile across normalized sub-cell (u, v) positions."""
    uv = _has_discrete_subcell_uv(err_df)
    if uv is None:
        return None
    u_vals, v_vals = uv
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    _draw_subcell_heatmap_axes(fig, axes[0], axes[1], err_df, u_vals, v_vals)
    fig.suptitle(
        f"{model_name} ({direction}) — Sub-Cell (u, v) Error Profile", fontsize=12
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_subcell_heatmap_rows(
    results: Sequence[EvaluationResult],
    out_path: Path,
    dpi: int = 200,
) -> Path | None:
    """Plot an ``N x 2`` stacked figure of Mean + p95 sub-cell heatmaps (1 row per model)."""
    valid_items: list[tuple[EvaluationResult, list[float], list[float]]] = []
    for res in results:
        uv = _has_discrete_subcell_uv(res.err_df)
        if uv is not None:
            valid_items.append((res, uv[0], uv[1]))
    if not valid_items:
        return None

    n_models = len(valid_items)
    fig, axes = plt.subplots(
        n_models, 2, figsize=(11.0, 4.4 * n_models), squeeze=False
    )
    for idx, (res, u_vals, v_vals) in enumerate(valid_items):
        _draw_subcell_heatmap_axes(
            fig,
            axes[idx, 0],
            axes[idx, 1],
            res.err_df,
            u_vals,
            v_vals,
            row_label=f"{res.model_name} ({res.direction})",
        )
    ds_id = valid_items[0][0].dataset_id
    fig.suptitle(f"Sub-Cell (u, v) Error Profile ({ds_id})", fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _draw_vector_quiver_ax(
    fig: plt.Figure,
    ax: plt.Axes,
    df: pd.DataFrame,
    de: np.ndarray,
    dn: np.ndarray,
    mag: np.ndarray,
    title: str,
    stride: int = 2,
) -> None:
    """Paint a 2D distortion/residual vector quiver onto ``ax``."""
    if "is_node" in df.columns and "row_idx" in df.columns and "col_idx" in df.columns:
        node_mask = (
            df["is_node"]
            & (df["row_idx"] % stride == 0)
            & (df["col_idx"] % stride == 0)
        )
    else:
        node_mask = np.zeros(len(df), dtype=bool)
        node_mask[:: max(1, stride)] = True

    q_df = df[node_mask]
    e_km = df["st70_e"].to_numpy(dtype=np.float64) * 1e-3
    n_km = df["st70_n"].to_numpy(dtype=np.float64) * 1e-3

    ax.scatter(
        e_km,
        n_km,
        c=mag,
        cmap="Greys",
        vmin=0.0,
        vmax=float(mag.max()),
        s=6.5,
        marker="s",
        alpha=0.25,
        edgecolors="none",
        linewidths=0,
        zorder=1,
    )
    q = ax.quiver(
        q_df["st70_e"].to_numpy(dtype=np.float64) * 1e-3,
        q_df["st70_n"].to_numpy(dtype=np.float64) * 1e-3,
        de[node_mask],
        dn[node_mask],
        mag[node_mask],
        cmap="inferno",
        scale=18.0,
        width=0.0022,
        headwidth=3.5,
        headlength=4.5,
        zorder=2,
    )
    ax.quiverkey(q, 0.88, 0.92, 1.0, "1.0 m shift", labelpos="E", coordinates="axes")
    cbar = fig.colorbar(q, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Shift Magnitude hypot(dN, dE) (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Stereo 70 Easting (km)")
    ax.set_ylabel("Stereo 70 Northing (km)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)


def plot_vector_quiver(
    df: pd.DataFrame,
    de: np.ndarray,
    dn: np.ndarray,
    mag: np.ndarray,
    out_path: Path,
    title: str,
    stride: int = 2,
    dpi: int = 200,
) -> Path:
    """Plot 2D distortion vector field with direction arrows on thinned grid nodes."""
    fig, ax = plt.subplots(figsize=(11, 8))
    _draw_vector_quiver_ax(fig, ax, df, de, dn, mag, title, stride=stride)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_vector_quiver_rows(
    results: Sequence[EvaluationResult],
    out_path: Path,
    stride: int = 2,
    dpi: int = 200,
) -> Path:
    """Plot a stacked 1-row-per-model figure of residual vector quiver maps."""
    if not results:
        raise ValueError("results must contain at least one EvaluationResult.")
    n_models = len(results)
    fig, axes = plt.subplots(
        n_models, 1, figsize=(11.0, 7.2 * n_models), squeeze=False
    )
    for idx, res in enumerate(results):
        ee = res.err_df["err_e_m"].to_numpy(dtype=np.float64)
        en = res.err_df["err_n_m"].to_numpy(dtype=np.float64)
        mag = res.err_df["err_2d_m"].to_numpy(dtype=np.float64)
        _draw_vector_quiver_ax(
            fig,
            axes[idx, 0],
            res.err_df,
            de=ee,
            dn=en,
            mag=mag,
            title=f"{res.model_name} ({res.direction} on {res.dataset_id}) — Residual Quiver",
            stride=stride,
        )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path
