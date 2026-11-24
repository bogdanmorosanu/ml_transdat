"""Cross-model and cross-dataset benchmark comparison plots."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.evaluation.evaluator import EvaluationResult, extract_metric
from src.evaluation.metrics import (
    METRIC_META,
    MetricGroup,
    MetricSpec,
    MetricType,
    format_metric_label,
    resolve_metric_types,
)


def plot_model_comparison(
    results: Sequence[EvaluationResult],
    out_path: Path,
    metrics: MetricSpec = MetricGroup.PRECISION_CORE,
    title: str | None = None,
    log_scale: bool = False,
    precision_unit: str = "mm",
    dpi: int = 200,
) -> Path:
    """Plot side-by-side multi-model comparison across one or more :class:`MetricType`s."""
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
    """Plot a 2-panel stacked bar chart of the 4 memory components."""
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
    """Plot grouped inference speed comparison across multiple named workloads."""
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
    """Plot a grouped bar chart comparing models across multiple evaluation datasets without re-inference."""
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
