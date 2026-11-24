"""Surface comparison, residual roughness variogram, and learnability Pareto ranking plots."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.spg_cache import get_default_grid2d
from src.evaluation.evaluator import EvaluationResult
from src.models.base import BaseGridShiftModel


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
    """Render 2x3 3D perspective surface and 2D contour comparison grids."""
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


def _compute_empirical_variogram(
    err_df: pd.DataFrame,
    max_points: int = 8000,
    max_lag_km: float = 22.0,
    n_bins: int = 18,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute binned RMS vector residual difference vs. spatial separation distance."""
    from scipy.spatial import cKDTree

    n_km = err_df["st70_n"].to_numpy(dtype=np.float64) * 1e-3
    e_km = err_df["st70_e"].to_numpy(dtype=np.float64) * 1e-3
    en_mm = err_df["err_n_m"].to_numpy(dtype=np.float64) * 1000.0
    ee_mm = err_df["err_e_m"].to_numpy(dtype=np.float64) * 1000.0
    valid = np.isfinite(n_km) & np.isfinite(e_km) & np.isfinite(en_mm) & np.isfinite(ee_mm)
    idx_valid = np.flatnonzero(valid)
    if len(idx_valid) < 20:
        return np.array([]), np.array([])

    if len(idx_valid) > max_points:
        rng = np.random.default_rng(seed)
        idx_sub = rng.choice(idx_valid, size=max_points, replace=False)
    else:
        idx_sub = idx_valid

    coords = np.column_stack([n_km[idx_sub], e_km[idx_sub]])
    en_s = en_mm[idx_sub]
    ee_s = ee_mm[idx_sub]

    tree = cKDTree(coords, balanced_tree=False)
    pairs = tree.query_pairs(r=max_lag_km, output_type="ndarray")
    if len(pairs) == 0:
        return np.array([]), np.array([])
    if len(pairs) > 250_000:
        rng = np.random.default_rng(seed)
        pairs = pairs[rng.choice(len(pairs), size=250_000, replace=False)]

    i_idx = pairs[:, 0]
    j_idx = pairs[:, 1]
    d_km = np.hypot(coords[i_idx, 0] - coords[j_idx, 0], coords[i_idx, 1] - coords[j_idx, 1])
    diff_sq = (en_s[i_idx] - en_s[j_idx]) ** 2 + (ee_s[i_idx] - ee_s[j_idx]) ** 2

    bins = np.linspace(0.0, max_lag_km, n_bins + 1)
    bin_ids = np.digitize(d_km, bins) - 1
    centers: list[float] = []
    rms_vals: list[float] = []
    for b in range(n_bins):
        mask = bin_ids == b
        if np.sum(mask) >= 15:
            centers.append(0.5 * (bins[b] + bins[b + 1]))
            rms_vals.append(float(np.sqrt(np.mean(diff_sq[mask]))))
    return np.asarray(centers), np.asarray(rms_vals)


def plot_residual_roughness_rows(
    results: Sequence[EvaluationResult],
    out_path: Path,
    dpi: int = 200,
) -> Path:
    """Plot an ``N x 3`` 1-row-per-model residual roughness & variogram diagnostic suite."""
    if not results:
        raise ValueError("results must contain at least one EvaluationResult.")

    grid2d = get_default_grid2d()
    n_models = len(results)
    fig, axes = plt.subplots(n_models, 3, figsize=(17.5, 4.9 * n_models), squeeze=False)

    for idx, res in enumerate(results):
        df = res.err_df
        s = res.summary
        ax_map, ax_var, ax_wall = axes[idx, 0], axes[idx, 1], axes[idx, 2]

        e_km = df["st70_e"].to_numpy(dtype=np.float64) * 1e-3
        n_km = df["st70_n"].to_numpy(dtype=np.float64) * 1e-3
        grad = (
            df["grad_2d_mm_km"].to_numpy(dtype=np.float64)
            if "grad_2d_mm_km" in df.columns
            else np.zeros(len(df), dtype=np.float64)
        )
        valid_g = np.isfinite(grad)
        vmax_g = float(np.percentile(grad[valid_g], 98.0)) if np.any(valid_g) else 1.0
        vmax_g = max(vmax_g, 1e-3)
        sc = ax_map.scatter(
            e_km[valid_g],
            n_km[valid_g],
            c=grad[valid_g],
            cmap="magma",
            vmin=0.0,
            vmax=vmax_g,
            s=5.5,
            marker="s",
            edgecolors="none",
            alpha=0.92,
        )
        cbar = fig.colorbar(sc, ax=ax_map, shrink=0.84, pad=0.02)
        cbar.set_label("Local Gradient ||∇e|| (mm/km)")
        ax_map.set_aspect("equal", adjustable="box")
        ax_map.set_xlabel("Stereo 70 Easting (km)")
        ax_map.set_ylabel("Stereo 70 Northing (km)")
        ax_map.set_title(
            f"{res.model_name} — Gradient Energy\n"
            f"RoughRMS={s.get('roughness_grad_rms_mm_km', float('nan')):.2f} mm/km | "
            f"Jitter={s.get('local_jitter_ratio', float('nan')):.3f}"
        )
        ax_map.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

        lags, v_rms = _compute_empirical_variogram(df)
        if len(lags) > 0:
            ax_var.plot(lags, v_rms, "o-", color="#2563eb", linewidth=1.8, markersize=4.5)
        ax_var.axvline(2.75, color="#059669", linestyle="--", linewidth=1.2, label="4x4 step (2.75 km)")
        ax_var.axvline(11.0, color="#dc2626", linestyle=":", linewidth=1.4, label="SPG cell (11.0 km)")
        ax_var.set_xlabel("Pair Separation Distance h (km)")
        ax_var.set_ylabel("RMS Vector Difference (mm)")
        ax_var.set_title(
            f"{res.model_name} — Residual Structure Function\n"
            f"cRMSE={s.get('crmse_2d_m', float('nan')) * 1000.0:.2f} mm | "
            f"Score={s.get('learnability_score_mm', float('nan')):.2f} mm"
        )
        ax_var.legend(loc="lower right", fontsize=8.0, framealpha=0.9)
        ax_var.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)

        u_mod = ((df["st70_n"].to_numpy(dtype=np.float64) - grid2d.n_min) / grid2d.n_step) % 1.0
        v_mod = ((df["st70_e"].to_numpy(dtype=np.float64) - grid2d.e_min) / grid2d.e_step) % 1.0
        d_wall_km = np.minimum(
            np.minimum(u_mod, 1.0 - u_mod) * (grid2d.n_step * 1e-3),
            np.minimum(v_mod, 1.0 - v_mod) * (grid2d.e_step * 1e-3),
        )
        e2d_mm = df["err_2d_m"].to_numpy(dtype=np.float64) * 1000.0
        wall_bins = np.linspace(0.0, 5.5, 12)
        w_ids = np.digitize(d_wall_km, wall_bins) - 1
        w_centers: list[float] = []
        w_err: list[float] = []
        w_grad: list[float] = []
        for b in range(len(wall_bins) - 1):
            m = (w_ids == b) & valid_g & np.isfinite(e2d_mm)
            if np.sum(m) >= 10:
                w_centers.append(0.5 * (wall_bins[b] + wall_bins[b + 1]))
                w_err.append(float(np.mean(e2d_mm[m])))
                w_grad.append(float(np.sqrt(np.mean(grad[m] ** 2))))

        ax_wall_r = ax_wall.twinx()
        if w_centers:
            l1 = ax_wall.plot(
                w_centers, w_err, "s-", color="#2563eb", linewidth=1.8, markersize=4.2, label="Mean Error (mm)"
            )
            l2 = ax_wall_r.plot(
                w_centers, w_grad, "o--", color="#dc2626", linewidth=1.6, markersize=4.0, label="RoughRMS (mm/km)"
            )
            lines = l1 + l2
            labels = [str(ln.get_label()) for ln in lines]
            ax_wall.legend(lines, labels, loc="upper right", fontsize=8.0, framealpha=0.9)

        ax_wall.set_xlabel("Distance to Nearest 11 km SPG Cell Wall (km)")
        ax_wall.set_ylabel("Mean 2D Error (mm)", color="#2563eb")
        ax_wall_r.set_ylabel("Roughness Grad RMS (mm/km)", color="#dc2626")
        ax_wall.set_title(f"{res.model_name} — Border vs. Cell-Center Profile")
        ax_wall.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)

    ds_id = results[0].dataset_id
    fig.suptitle(
        f"Stage-2 ML Learnability: Residual Roughness & Structure ({ds_id})",
        fontsize=13.0,
        fontweight="bold",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_surface_learnability_ranking(
    results: Sequence[EvaluationResult],
    out_path: Path,
    dataset_id: str | None = None,
    top_k: int = 15,
    title: str | None = None,
    dpi: int = 200,
) -> Path:
    """Plot 2-panel Pareto frontier (cRMSE vs. Roughness) and Top-K decomposed Learnability Cost."""
    if not results:
        raise ValueError("results must contain at least one EvaluationResult.")

    target_ds = dataset_id
    if target_ds is None:
        non_train = [
            r.dataset_id for r in results if r.dataset_id not in {"ro_grid_4x4", "spg_nodes_valid"}
        ]
        target_ds = non_train[0] if non_train else results[0].dataset_id

    ds_results = [r for r in results if r.dataset_id == target_ds]
    if not ds_results:
        raise ValueError(f"No EvaluationResult entries found for dataset_id='{target_ds}'.")

    ds_results_sorted = sorted(
        ds_results,
        key=lambda r: float(r.summary.get("learnability_score_mm", float("inf"))),
    )

    names = [r.model_name for r in ds_results_sorted]
    crmse = np.array([float(r.summary.get("crmse_2d_m", 0.0)) * 1000.0 for r in ds_results_sorted])
    rough = np.array([float(r.summary.get("roughness_grad_rms_mm_km", 0.0)) for r in ds_results_sorted])
    p95 = np.array([float(r.summary.get("p95_2d_m", 0.0)) * 1000.0 for r in ds_results_sorted])
    p99 = np.array([float(r.summary.get("p99_2d_m", 0.0)) * 1000.0 for r in ds_results_sorted])
    scores = np.array([float(r.summary.get("learnability_score_mm", 0.0)) for r in ds_results_sorted])

    fig, (ax_pareto, ax_bar) = plt.subplots(1, 2, figsize=(16.0, 6.8))

    sizes = np.clip(p99 * 6.0, 45.0, 420.0)
    sc = ax_pareto.scatter(
        crmse,
        rough,
        s=sizes,
        c=scores,
        cmap="viridis_r",
        edgecolors="#0f172a",
        linewidth=0.8,
        alpha=0.88,
        zorder=4,
    )
    cbar = fig.colorbar(sc, ax=ax_pareto, shrink=0.86, pad=0.02)
    cbar.set_label("Learnability Score (mm, lower is better)")

    order_x = np.argsort(crmse)
    pareto_idx: list[int] = []
    best_y = float("inf")
    for idx_pt in order_x:
        if rough[idx_pt] <= best_y:
            best_y = float(rough[idx_pt])
            pareto_idx.append(int(idx_pt))
    if len(pareto_idx) >= 2:
        ax_pareto.plot(
            crmse[pareto_idx],
            rough[pareto_idx],
            color="#dc2626",
            linestyle="--",
            linewidth=1.6,
            label="Pareto Frontier (cRMSE vs. Roughness)",
            zorder=5,
        )

    for i in range(min(len(names), 10)):
        ax_pareto.annotate(
            f"#{i + 1} {names[i]}",
            xy=(crmse[i], rough[i]),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=7.3,
            fontweight="bold" if i < 3 else "normal",
            color="#0f172a",
            zorder=6,
        )

    ax_pareto.set_xlabel("Centered RMSE 2D (mm) [Residual Amplitude]")
    ax_pareto.set_ylabel("Local Roughness Grad RMS (mm/km) [Slope Energy]")
    ax_pareto.set_title(f"Amplitude vs. Roughness Pareto Frontier ({target_ds})\n(Bubble size ∝ P99 Spike)")
    ax_pareto.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    if len(pareto_idx) >= 2:
        ax_pareto.legend(loc="upper right", framealpha=0.9)

    k_show = min(top_k, len(ds_results_sorted))
    y_pos = np.arange(k_show)
    c_part = 0.50 * crmse[:k_show]
    r_part = 0.30 * (2.75 * rough[:k_show])
    p_part = 0.20 * p95[:k_show]

    ax_bar.barh(y_pos, c_part, color="#2563eb", label="0.50 × cRMSE (mm)")
    ax_bar.barh(y_pos, r_part, left=c_part, color="#059669", label="0.30 × 2.75km·RoughRMS")
    ax_bar.barh(y_pos, p_part, left=c_part + r_part, color="#d97706", label="0.20 × P95 (mm)")

    for i in range(k_show):
        tot = scores[i]
        ax_bar.text(
            tot + 0.02 * max(scores[:k_show]),
            y_pos[i],
            f"{tot:.2f} mm",
            va="center",
            ha="left",
            fontsize=8.0,
            fontweight="bold" if i < 3 else "normal",
        )

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels([f"#{i + 1} {names[i]}" for i in range(k_show)], fontsize=8.5)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Composite Learnability Score (mm, lower is better)")
    ax_bar.set_title(f"Top-{k_show} Stage-1 Surface Ranking ({target_ds})")
    ax_bar.set_xlim(0.0, max(scores[:k_show]) * 1.18 if k_show > 0 else 1.0)
    ax_bar.legend(loc="lower right", fontsize=8.2, framealpha=0.9)
    ax_bar.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.45)

    fig.suptitle(
        title or f"Stage-1 Surface Learnability Ranking ({target_ds})",
        fontsize=13.0,
        fontweight="bold",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path
