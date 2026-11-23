"""Core visualization functions: spatial maps, diagnostic suites, quivers, and transects."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pytransdat import TransRO
from pytransdat.trans_grid import Grid2D
from src.evaluation.evaluator import EvaluationResult, extract_metric
from src.evaluation.metrics import (
    METRIC_META,
    MetricGroup,
    MetricSpec,
    MetricType,
    format_metric_label,
    resolve_metric_types,
)
from src.models.base import BaseGridShiftModel


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
) -> None:
    """Paint a single 2D spatial error heatmap onto ``ax``."""
    e_km = err_df["st70_e"].to_numpy(dtype=np.float64) * 1e-3
    n_km = err_df["st70_n"].to_numpy(dtype=np.float64) * 1e-3
    err = err_df["err_2d_m"].to_numpy(dtype=np.float64)

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
) -> Path:
    """Plot borderless square heatmap of 2D horizontal errors across Romania."""
    fig, ax = plt.subplots(figsize=(10, 7.5))
    _draw_spatial_error_ax(
        fig, ax, err_df, summary, f"{model_name} ({direction}) — Spatial 2D Error Map"
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
        _draw_spatial_error_ax(fig, axes[idx, 0], res.err_df, s, title)
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


def plot_transect_66km(
    models: dict[str, BaseGridShiftModel],
    out_path: Path,
    grid2d: Grid2D | None = None,
    n_const: float | None = None,
    e_start: float | None = None,
    n_cells: int = 6,
    n_samples: int = 800,
    title: str = "1D Transect Across 6 SPG Grid Cells (66 km)",
    dpi: int = 200,
) -> Path:
    """Plot a 1D cross-section spanning ``n_cells`` SPG grid cells comparing models to bicubic truth.

    :param models: Mapping ``label -> fitted BaseGridShiftModel``.
    :type models: dict[str, BaseGridShiftModel]
    :param out_path: Output PNG file path.
    :type out_path: Path
    :param grid2d: Optional ``Grid2D`` instance for true bicubic curve.
    :type grid2d: Optional[Grid2D]
    :param n_const: Fixed Northing coordinate (defaults to row 24 nodal line).
    :type n_const: Optional[float]
    :param e_start: Starting Easting coordinate (defaults to col 18 nodal line).
    :type e_start: Optional[float]
    :param n_cells: Number of 11 km SPG cells to span.
    :type n_cells: int
    :param n_samples: Number of dense query points along the transect.
    :type n_samples: int
    :param title: Plot title.
    :type title: str
    :param dpi: Output figure DPI.
    :type dpi: int
    :return: Saved plot path.
    :rtype: Path
    """
    active_grid = grid2d if grid2d is not None else Grid2D()
    n_fixed = (
        n_const if n_const is not None else active_grid.n_min + 24 * active_grid.n_step
    )
    e0 = e_start if e_start is not None else active_grid.e_min + 18 * active_grid.e_step
    e1 = e0 + n_cells * active_grid.e_step

    e_line = np.linspace(e0, e1, n_samples)
    n_line = np.full_like(e_line, n_fixed)
    coords_line = np.column_stack([n_line, e_line])
    e_km = e_line * 1e-3

    true_dn_list, true_de_list = active_grid.interp(n_line.tolist(), e_line.tolist())
    true_dn = np.asarray(true_dn_list, dtype=np.float64)
    true_de = np.asarray(true_de_list, dtype=np.float64)

    fig, (ax_dn, ax_de) = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True)

    # Vertical grid cell boundaries
    for idx in range(n_cells + 1):
        xg = (e0 + idx * active_grid.e_step) * 1e-3
        ax_dn.axvline(xg, color="#cbd5e1", linestyle="--", linewidth=0.8, zorder=1)
        ax_de.axvline(xg, color="#cbd5e1", linestyle="--", linewidth=0.8, zorder=1)

    ax_dn.plot(
        e_km,
        true_dn,
        color="#0f172a",
        linewidth=2.2,
        label="True Bicubic (pytransdat)",
        zorder=4,
    )
    ax_de.plot(
        e_km,
        true_de,
        color="#0f172a",
        linewidth=2.2,
        label="True Bicubic (pytransdat)",
        zorder=4,
    )

    palette = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed"]
    for idx, (label, model) in enumerate(models.items()):
        pred = model.predict(coords_line)
        col = palette[idx % len(palette)]
        ax_dn.plot(
            e_km, pred[:, 0], color=col, linewidth=1.4, alpha=0.9, label=label, zorder=3
        )
        ax_de.plot(
            e_km, pred[:, 1], color=col, linewidth=1.4, alpha=0.9, label=label, zorder=3
        )

    ax_dn.set_ylabel("Northing Shift dN (m)")
    ax_dn.set_title(title)
    ax_dn.legend(loc="best", framealpha=0.9)
    ax_dn.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)

    ax_de.set_xlabel("Stereo 70 Easting (km)")
    ax_de.set_ylabel("Easting Shift dE (m)")
    ax_de.legend(loc="best", framealpha=0.9)
    ax_de.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_model_comparison(
    results: Sequence[EvaluationResult],
    out_path: Path,
    metrics: MetricSpec = MetricGroup.PRECISION_CORE,
    title: str | None = None,
    log_scale: bool = False,
    precision_unit: str = "mm",
    dpi: int = 200,
) -> Path:
    """Plot side-by-side multi-model comparison across one or more :class:`MetricType`s.

    Preserves the exact order of ``results`` as supplied by the caller.

    :param results: Ordered sequence of :class:`EvaluationResult` objects.
    :type results: Sequence[EvaluationResult]
    :param out_path: Output PNG file path.
    :type out_path: Path
    :param metrics: :class:`MetricGroup`, :class:`MetricType`, or sequence of metrics to plot.
    :type metrics: MetricSpec
    :param title: Optional figure suptitle.
    :type title: str | None
    :param log_scale: Whether to apply logarithmic y-axis scaling.
    :type log_scale: bool
    :param precision_unit: Display unit for error/distance metrics (``'m'``, ``'cm'``, or ``'mm'``).
    :type precision_unit: str
    :param dpi: Figure DPI.
    :type dpi: int
    :return: Saved plot path.
    :rtype: Path
    :raises ValueError: If ``results`` is empty or ``precision_unit`` is invalid.
    """
    if not results:
        raise ValueError("results must contain at least one EvaluationResult.")

    resolved_metrics = resolve_metric_types(metrics)
    n_metrics = len(resolved_metrics)
    model_names = [r.model_name for r in results]
    x = np.arange(len(model_names))

    palette = ["#2563eb", "#059669", "#d97706", "#dc2626", "#7c3aed", "#0891b2"]
    colors = [palette[i % len(palette)] for i in range(len(model_names))]

    ncols = min(2, n_metrics)
    nrows = math.ceil(n_metrics / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6.2 * ncols, 4.4 * nrows),
        squeeze=False,
    )
    axes_flat = axes.ravel()

    for idx, metric in enumerate(resolved_metrics):
        ax = axes_flat[idx]
        axis_label, scale = format_metric_label(metric, precision_unit=precision_unit)

        vals = [extract_metric(r, metric) * scale for r in results]
        bars = ax.bar(
            x, vals, color=colors, width=0.58, edgecolor="#1e293b", linewidth=0.5
        )

        if log_scale and all(v > 0.0 for v in vals):
            ax.set_yscale("log")

        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=15, ha="right")
        ax.set_ylabel(axis_label)
        ax.set_title(axis_label)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.45)

        max_v = max(vals) if vals else 1.0
        for bar, v in zip(bars, vals):
            if abs(v) >= 100.0:
                label = f"{v:,.1f}"
            elif abs(v) >= 1.0:
                label = f"{v:.2f}"
            elif abs(v) >= 1e-3 or v == 0.0:
                label = f"{v:.4f}"
            else:
                label = f"{v:.2e}"
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + (0.02 * max_v if not log_scale else 0.0),
                label,
                ha="center",
                va="bottom",
                fontsize=8.5,
            )
        if not log_scale and max_v > 0.0:
            ax.set_ylim(0.0, max_v * 1.16)

    for idx in range(n_metrics, len(axes_flat)):
        axes_flat[idx].axis("off")

    dataset_label = results[0].dataset_id
    fig.suptitle(
        title or f"Multi-Model Comparison ({dataset_label})",
        fontsize=12.5,
        fontweight="bold",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_memory_breakdown_stacked(
    results: Sequence[EvaluationResult],
    out_path: Path,
    title: str | None = None,
    unit: str = "MB",
    dpi: int = 200,
) -> Path:
    """Plot a 2-panel stacked bar chart of the 4 memory components.

    Left panel shows **Model + Lookup Footprint** (Code + Model Instance + Grid Lookup)
    so compact models remain legible. Right panel shows **Full Deployment Footprint**
    adding external library dependency sizes.

    :param results: Ordered sequence of :class:`EvaluationResult` objects.
    :type results: Sequence[EvaluationResult]
    :param out_path: Output PNG file path.
    :type out_path: Path
    :param title: Optional figure suptitle.
    :type title: str | None
    :param unit: Display unit (``'KB'`` or ``'MB'``).
    :type unit: str
    :param dpi: Figure DPI.
    :type dpi: int
    :return: Saved plot path.
    :rtype: Path
    """
    if not results:
        raise ValueError("results must contain at least one EvaluationResult.")

    model_names = [r.model_name for r in results]
    scale = 1024.0 if unit.upper() == "KB" else (1024.0 * 1024.0)
    unit_label = unit.upper()

    code_vals: list[float] = []
    deps_vals: list[float] = []
    model_vals: list[float] = []
    grid_vals: list[float] = []

    for r in results:
        mb = r.memory_breakdown
        if mb is not None:
            code_vals.append(mb.code_bytes / scale)
            deps_vals.append(mb.dependencies_bytes / scale)
            model_vals.append(mb.model_instance_bytes / scale)
            grid_vals.append(mb.grid_lookup_bytes / scale)
        else:
            code_vals.append(0.0)
            deps_vals.append(0.0)
            model_vals.append(r.model_size_bytes / scale)
            grid_vals.append(0.0)

    x = np.arange(len(model_names))
    width = 0.56
    fig, (ax_model, ax_full) = plt.subplots(
        1, 2, figsize=(max(13.0, 2.2 * len(model_names)), 5.6)
    )

    # Panel 1: Model + Lookup (Code + Model Instance + Grid Lookup)
    ax_model.bar(x, code_vals, width, label="1. Code Size", color="#3b82f6")
    ax_model.bar(
        x,
        model_vals,
        width,
        bottom=code_vals,
        label="3. Model Instance / Weights",
        color="#059669",
    )
    b_cm = [c + m for c, m in zip(code_vals, model_vals, strict=True)]
    ax_model.bar(
        x, grid_vals, width, bottom=b_cm, label="4. Grid Lookup Data", color="#d97706"
    )
    model_totals = [cm + g for cm, g in zip(b_cm, grid_vals, strict=True)]
    for tot, x_pos in zip(model_totals, x, strict=True):
        lbl = f"{tot:.2f} {unit_label}" if tot < 10.0 else f"{tot:.1f} {unit_label}"
        ax_model.text(
            x_pos,
            tot * 1.02 + 1e-9,
            lbl,
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )
    ax_model.set_xticks(x)
    ax_model.set_xticklabels(model_names, rotation=18, ha="right")
    ax_model.set_ylabel(f"Footprint ({unit_label})")
    ax_model.set_title("Model + Lookup Footprint (excl. Dependencies)")
    ax_model.legend(loc="upper left", framealpha=0.9)
    ax_model.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.45)
    if model_totals and max(model_totals) > 0.0:
        ax_model.set_ylim(0.0, max(model_totals) * 1.18)

    # Panel 2: Full Deployment (all 4 components)
    ax_full.bar(x, code_vals, width, label="1. Code Size", color="#3b82f6")
    ax_full.bar(
        x,
        deps_vals,
        width,
        bottom=code_vals,
        label="2. Library Dependencies",
        color="#64748b",
    )
    b_cd = [c + d for c, d in zip(code_vals, deps_vals, strict=True)]
    ax_full.bar(
        x,
        model_vals,
        width,
        bottom=b_cd,
        label="3. Model Instance / Weights",
        color="#059669",
    )
    b_cdm = [cd + m for cd, m in zip(b_cd, model_vals, strict=True)]
    ax_full.bar(
        x, grid_vals, width, bottom=b_cdm, label="4. Grid Lookup Data", color="#d97706"
    )
    full_totals = [cdm + g for cdm, g in zip(b_cdm, grid_vals, strict=True)]
    for tot, x_pos in zip(full_totals, x, strict=True):
        lbl = f"{tot:.1f} {unit_label}"
        ax_full.text(
            x_pos,
            tot * 1.02 + 1e-9,
            lbl,
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )
    ax_full.set_xticks(x)
    ax_full.set_xticklabels(model_names, rotation=18, ha="right")
    ax_full.set_ylabel(f"Footprint ({unit_label})")
    ax_full.set_title("Full Deployment Footprint (incl. Dependencies)")
    ax_full.legend(loc="upper left", framealpha=0.9)
    ax_full.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.45)
    if full_totals and max(full_totals) > 0.0:
        ax_full.set_ylim(0.0, max(full_totals) * 1.18)

    fig.suptitle(
        title or f"Model Memory & Storage Breakdown (4 Components - {unit_label})",
        fontsize=12.5,
        fontweight="bold",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_speed_workloads(
    results_by_workload: Mapping[str, Sequence[EvaluationResult]],
    out_path: Path,
    metrics: tuple[MetricType, ...] = (
        MetricType.THROUGHPUT_PTS_PER_SEC,
        MetricType.LATENCY_US_PER_PT,
    ),
    title: str | None = None,
    log_scale: bool = True,
    dpi: int = 200,
) -> Path:
    """Plot grouped inference speed comparison across multiple named workloads.

    :param results_by_workload: Ordered mapping ``workload_name -> list[EvaluationResult]``.
    :type results_by_workload: Mapping[str, Sequence[EvaluationResult]]
    :param out_path: Output PNG file path.
    :type out_path: Path
    :param metrics: Speed metrics to plot (one panel per metric).
    :type metrics: tuple[MetricType, ...]
    :param title: Optional figure suptitle.
    :type title: str | None
    :param log_scale: Whether to apply logarithmic y-axis scaling.
    :type log_scale: bool
    :param dpi: Figure DPI.
    :type dpi: int
    :return: Saved plot path.
    :rtype: Path
    :raises ValueError: If ``results_by_workload`` is empty.
    """
    if not results_by_workload:
        raise ValueError("results_by_workload must not be empty.")

    workloads = list(results_by_workload.keys())
    first_results = list(results_by_workload[workloads[0]])
    model_names = [r.model_name for r in first_results]
    n_models = len(model_names)
    n_workloads = len(workloads)

    x = np.arange(n_models)
    bar_width = 0.78 / max(1, n_workloads)
    palette = ["#2563eb", "#059669", "#d97706", "#7c3aed"]

    fig, axes = plt.subplots(
        1, len(metrics), figsize=(6.8 * len(metrics), 5.2), squeeze=False
    )
    axes_flat = axes.ravel()

    for m_idx, metric in enumerate(metrics):
        ax = axes_flat[m_idx]
        axis_label = METRIC_META[metric].label
        for w_idx, w_name in enumerate(workloads):
            w_results = results_by_workload[w_name]
            vals = [extract_metric(r, metric) for r in w_results]
            offset = (w_idx - (n_workloads - 1) / 2.0) * bar_width
            bars = ax.bar(
                x + offset,
                vals,
                bar_width,
                label=w_name,
                color=palette[w_idx % len(palette)],
                edgecolor="#1e293b",
                linewidth=0.4,
            )
            for bar, v in zip(bars, vals, strict=True):
                lbl = (
                    f"{v:,.0f}"
                    if v >= 100.0
                    else (f"{v:.2f}" if v >= 0.1 else f"{v:.3f}")
                )
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() * 1.05,
                    lbl,
                    ha="center",
                    va="bottom",
                    fontsize=7.0,
                    rotation=45 if n_workloads > 1 else 0,
                )

        if log_scale:
            ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=18, ha="right")
        ax.set_ylabel(axis_label)
        ax.set_title(axis_label)
        ax.legend(framealpha=0.9)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.45)

    fig.suptitle(
        title or "Inference Speed Across Named Workloads",
        fontsize=12.5,
        fontweight="bold",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_surface_comparison_grids(
    models_info: Sequence[tuple[str, BaseGridShiftModel | None, str]],
    ref_dataset: Any,
    out_3d_path: Path,
    out_2d_path: Path,
    title_3d: str,
    title_2d: str,
    vmax: float = 2.8,
    dpi: int = 200,
) -> tuple[Path, Path]:
    """Render 2x3 3D perspective surface and 2D contour comparison grids.

    :param models_info: Sequence of ``(panel_title, model_or_none_for_ground_truth, description)``.
    :type models_info: Sequence[tuple[str, BaseGridShiftModel | None, str]]
    :param ref_dataset: Reference :class:`GridDataset` used for ground truth & domain bounds.
    :type ref_dataset: Any
    :param out_3d_path: Output PNG path for the 3D surface grid.
    :type out_3d_path: Path
    :param out_2d_path: Output PNG path for the 2D contour grid.
    :type out_2d_path: Path
    :param title_3d: Suptitle for the 3D surface figure.
    :type title_3d: str
    :param title_2d: Suptitle for the 2D contour figure.
    :type title_2d: str
    :param vmax: Upper color/z-axis limit in meters.
    :type vmax: float
    :param dpi: Output figure DPI.
    :type dpi: int
    :return: Tuple ``(out_3d_path, out_2d_path)``.
    :rtype: tuple[Path, Path]
    """
    from matplotlib.colors import Normalize
    from scipy.interpolate import griddata

    coords = ref_dataset.coords
    mag = np.hypot(ref_dataset.true_shifts[:, 0], ref_dataset.true_shifts[:, 1])

    n_min, n_max = coords[:, 0].min(), coords[:, 0].max()
    e_min, e_max = coords[:, 1].min(), coords[:, 1].max()

    n_vec = np.linspace(n_min, n_max, 80)
    e_vec = np.linspace(e_min, e_max, 100)
    grid_n, grid_e = np.meshgrid(n_vec, e_vec, indexing="ij")

    gt_mag = griddata(coords, mag, (grid_n, grid_e), method="linear")
    mask = ~np.isnan(gt_mag)

    x_km = (grid_e - e_min) / 1000.0
    y_km = (grid_n - n_min) / 1000.0
    query_coords = np.column_stack([grid_n.ravel(), grid_e.ravel()])

    surfaces: list[tuple[str, np.ndarray, str]] = []
    for title, model_obj, desc in models_info:
        if model_obj is None:
            surfaces.append((title, gt_mag, desc))
        else:
            pred = model_obj.predict(query_coords)
            pred_mag = np.hypot(pred[:, 0], pred[:, 1]).reshape(grid_n.shape)
            pred_mag[~mask] = np.nan
            surfaces.append((title, pred_mag, desc))

    norm = Normalize(vmin=0.0, vmax=vmax)
    cmap = plt.cm.turbo

    # 3D Perspective Grid
    fig_3d = plt.figure(figsize=(19, 11.5))
    for idx, (title, z_grid, desc) in enumerate(surfaces):
        ax = fig_3d.add_subplot(2, 3, idx + 1, projection="3d")
        ax.plot_surface(
            x_km,
            y_km,
            z_grid,
            cmap=cmap,
            norm=norm,
            linewidth=0,
            antialiased=True,
            rstride=1,
            cstride=1,
            alpha=0.92,
        )
        ax.contour(
            x_km,
            y_km,
            z_grid,
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
        ax.set_zlim(0.0, vmax)
        ax.view_init(elev=28, azim=-55)
        ax.grid(True, linestyle=":", alpha=0.4)

    fig_3d.subplots_adjust(
        right=0.91, hspace=0.25, wspace=0.18, top=0.90, bottom=0.06
    )
    cbar_ax_3d = fig_3d.add_axes([0.93, 0.15, 0.015, 0.70])
    cbar_3d = fig_3d.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax_3d
    )
    cbar_3d.set_label(
        "2D Distortion Shift Magnitude (meters)", fontsize=11, fontweight="bold"
    )
    fig_3d.suptitle(title_3d, fontsize=14.5, fontweight="bold", y=0.97)
    out_3d_path.parent.mkdir(parents=True, exist_ok=True)
    fig_3d.savefig(out_3d_path, dpi=dpi)
    plt.close(fig_3d)

    # 2D Contour Grid
    fig_2d, axes_2d = plt.subplots(2, 3, figsize=(18, 10.5))
    axes_flat = axes_2d.flatten()
    contour_levels = np.linspace(0.0, vmax, 29)
    major_levels = [0.25, 0.50, 1.00, 1.50, 2.00, 2.50]

    for idx, (title, z_grid, desc) in enumerate(surfaces):
        ax = axes_flat[idx]
        ax.contourf(
            x_km,
            y_km,
            z_grid,
            levels=contour_levels,
            cmap=cmap,
            norm=norm,
            extend="both",
        )
        cs = ax.contour(
            x_km,
            y_km,
            z_grid,
            levels=major_levels,
            colors="#1e293b",
            linewidths=0.6,
            alpha=0.7,
        )
        ax.clabel(cs, inline=True, fontsize=7.5, fmt="%.2fm")
        ax.set_title(f"{title}\n{desc}", fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Easting (km)", fontsize=9)
        ax.set_ylabel("Northing (km)", fontsize=9)
        ax.set_aspect("equal")
        ax.grid(True, linestyle="--", alpha=0.3)

    fig_2d.subplots_adjust(
        right=0.90, hspace=0.28, wspace=0.20, top=0.90, bottom=0.06
    )
    cbar_ax_2d = fig_2d.add_axes([0.92, 0.15, 0.015, 0.70])
    cbar_2d = fig_2d.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax_2d
    )
    cbar_2d.set_label(
        "2D Distortion Shift Magnitude (meters)", fontsize=11, fontweight="bold"
    )
    fig_2d.suptitle(title_2d, fontsize=14.5, fontweight="bold", y=0.97)
    out_2d_path.parent.mkdir(parents=True, exist_ok=True)
    fig_2d.savefig(out_2d_path, dpi=dpi)
    plt.close(fig_2d)

    return out_3d_path, out_2d_path


def plot_multi_dataset_comparison(
    results: Sequence[EvaluationResult],
    out_path: Path,
    dataset_labels: Mapping[str, str] | None = None,
    model_labels: Mapping[str, str] | None = None,
    metric: MetricType = MetricType.RMSE_2D_M,
    precision_unit: str = "mm",
    reference_lines: Sequence[tuple[float, str, str]] | None = None,
    title: str | None = None,
    dpi: int = 200,
) -> Path:
    """Plot a grouped bar chart comparing models across multiple evaluation datasets without re-inference.

    :param results: Pre-computed :class:`EvaluationResult` objects.
    :type results: Sequence[EvaluationResult]
    :param out_path: Output PNG file path.
    :type out_path: Path
    :param dataset_labels: Optional mapping ``dataset_id -> display label``.
    :type dataset_labels: Mapping[str, str] | None
    :param model_labels: Optional mapping ``model_name -> display label`` (also controls order).
    :type model_labels: Mapping[str, str] | None
    :param metric: Scalar metric to compare.
    :type metric: MetricType
    :param precision_unit: Display unit for distance metrics (``'m'``, ``'cm'``, ``'mm'``).
    :type precision_unit: str
    :param reference_lines: Optional sequence of ``(y_value, label, color)`` horizontal lines.
    :type reference_lines: Sequence[tuple[float, str, str]] | None
    :param title: Optional chart title.
    :type title: str | None
    :param dpi: Figure DPI.
    :type dpi: int
    :return: Saved plot path.
    :rtype: Path
    """
    if not results:
        raise ValueError("results must contain at least one EvaluationResult.")

    by_model: dict[str, dict[str, EvaluationResult]] = {}
    ds_order: list[str] = []
    for r in results:
        by_model.setdefault(r.model_name, {})[r.dataset_id] = r
        if r.dataset_id not in ds_order:
            ds_order.append(r.dataset_id)

    if model_labels is not None:
        m_keys = [k for k in model_labels if k in by_model]
        m_names = [model_labels[k] for k in m_keys]
    else:
        m_keys = list(by_model.keys())
        m_names = list(m_keys)

    if dataset_labels is not None:
        ds_keys = [k for k in dataset_labels if k in ds_order]
        ds_names = [dataset_labels[k] for k in ds_keys]
    else:
        ds_keys = ds_order
        ds_names = list(ds_keys)

    axis_label, scale = format_metric_label(metric, precision_unit=precision_unit)
    n_ds = len(ds_keys)
    x = np.arange(len(m_keys))
    width = 0.76 / max(1, n_ds)
    palette = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed"]

    fig, ax = plt.subplots(figsize=(max(11.0, 1.5 * len(m_keys)), 6.0))
    for ds_idx, (ds_k, ds_lbl) in enumerate(zip(ds_keys, ds_names, strict=True)):
        vals = [
            extract_metric(by_model[mk][ds_k], metric) * scale if ds_k in by_model[mk] else 0.0
            for mk in m_keys
        ]
        offset = (ds_idx - (n_ds - 1) / 2.0) * width
        rects = ax.bar(
            x + offset,
            vals,
            width,
            label=ds_lbl,
            color=palette[ds_idx % len(palette)],
            alpha=0.88,
        )
        for rect in rects:
            h = rect.get_height()
            ax.annotate(
                f"{h:.1f}",
                xy=(rect.get_x() + rect.get_width() / 2.0, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
                fontweight="bold",
            )

    if reference_lines:
        for y_val, ref_lbl, ref_col in reference_lines:
            ax.axhline(y_val, color=ref_col, linestyle="--", linewidth=1.5, label=ref_lbl)

    ax.set_ylabel(axis_label, fontsize=11, fontweight="bold")
    ax.set_title(title or f"Multi-Dataset Comparison ({axis_label})", fontsize=12.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(m_names, rotation=25, ha="right", fontsize=9.5, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.5, axis="y")
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path
