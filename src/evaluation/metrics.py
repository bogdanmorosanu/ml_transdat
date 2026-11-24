"""Standardized error computation, metric metadata, scorecards, and Markdown formatting."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


class MetricType(str, Enum):
    """Canonical identifiers for precision, latency, and size metrics."""

    # Precision
    MEAN_2D_M = "mean_2d_m"
    STD_2D_M = "std_2d_m"
    RMSE_2D_M = "rmse_2d_m"
    CRMSE_2D_M = "crmse_2d_m"
    IQR_2D_M = "iqr_2d_m"
    MIN_2D_M = "min_2d_m"
    MAX_2D_M = "max_2d_m"
    P50_2D_M = "p50_2d_m"
    P90_2D_M = "p90_2d_m"
    P95_2D_M = "p95_2d_m"
    P99_2D_M = "p99_2d_m"
    P99_9_2D_M = "p99_9_2d_m"
    BIAS_N_M = "bias_n_m"
    BIAS_E_M = "bias_e_m"
    RMSE_N_M = "rmse_n_m"
    RMSE_E_M = "rmse_e_m"
    ROUGHNESS_GRAD_RMS_MM_KM = "roughness_grad_rms_mm_km"
    ROUGHNESS_GRAD_P95_MM_KM = "roughness_grad_p95_mm_km"
    LOCAL_JITTER_RATIO = "local_jitter_ratio"
    LEARNABILITY_SCORE_MM = "learnability_score_mm"
    PASS_LT_1MM_PCT = "lt_1mm"
    PASS_LT_1CM_PCT = "lt_1cm"
    PASS_LT_5CM_PCT = "lt_5cm"
    PASS_LT_10CM_PCT = "lt_10cm"
    PASS_LT_1M_PCT = "lt_1m"

    # Time / Latency (Speed)
    EVAL_TIME_SEC = "eval_time_sec"
    FIT_TIME_SEC = "fit_time_sec"
    THROUGHPUT_PTS_PER_SEC = "throughput_pts_per_sec"
    LATENCY_US_PER_PT = "latency_us_per_pt"

    # Memory / Storage Breakdown
    MODEL_SIZE_MB = "model_size_mb"
    SIZE_CODE_BYTES = "size_code_bytes"
    SIZE_DEPENDENCIES_BYTES = "size_dependencies_bytes"
    SIZE_MODEL_BYTES = "size_model_bytes"
    SIZE_GRID_LOOKUP_BYTES = "size_grid_lookup_bytes"
    TOTAL_SIZE_MB = "total_size_mb"


@dataclass(frozen=True)
class MetricMeta:
    """Structured metadata describing display label, physical dimension, and formatting."""

    label: str
    dimension: Literal[
        "distance",
        "percent",
        "time",
        "rate",
        "bytes",
        "megabytes",
        "gradient",
        "ratio",
        "score_mm",
    ]
    is_integer: bool = False


METRIC_META: dict[MetricType, MetricMeta] = {
    MetricType.MEAN_2D_M: MetricMeta("Mean 2D Error (m)", "distance"),
    MetricType.STD_2D_M: MetricMeta("Std Dev 2D (m)", "distance"),
    MetricType.RMSE_2D_M: MetricMeta("RMSE 2D Error (m)", "distance"),
    MetricType.CRMSE_2D_M: MetricMeta("Centered RMSE 2D (m)", "distance"),
    MetricType.IQR_2D_M: MetricMeta("IQR 2D Error (m)", "distance"),
    MetricType.MIN_2D_M: MetricMeta("Min 2D Error (m)", "distance"),
    MetricType.MAX_2D_M: MetricMeta("Max 2D Error (m)", "distance"),
    MetricType.P50_2D_M: MetricMeta("P50 2D Error (m)", "distance"),
    MetricType.P90_2D_M: MetricMeta("P90 2D Error (m)", "distance"),
    MetricType.P95_2D_M: MetricMeta("P95 2D Error (m)", "distance"),
    MetricType.P99_2D_M: MetricMeta("P99 2D Error (m)", "distance"),
    MetricType.P99_9_2D_M: MetricMeta("P99.9 2D Error (m)", "distance"),
    MetricType.BIAS_N_M: MetricMeta("Signed Bias dN (m)", "distance"),
    MetricType.BIAS_E_M: MetricMeta("Signed Bias dE (m)", "distance"),
    MetricType.RMSE_N_M: MetricMeta("RMSE Northing (m)", "distance"),
    MetricType.RMSE_E_M: MetricMeta("RMSE Easting (m)", "distance"),
    MetricType.ROUGHNESS_GRAD_RMS_MM_KM: MetricMeta(
        "Roughness Grad RMS (mm/km)", "gradient"
    ),
    MetricType.ROUGHNESS_GRAD_P95_MM_KM: MetricMeta(
        "Roughness Grad P95 (mm/km)", "gradient"
    ),
    MetricType.LOCAL_JITTER_RATIO: MetricMeta("Local Jitter Ratio", "ratio"),
    MetricType.LEARNABILITY_SCORE_MM: MetricMeta("Learnability Score (mm)", "score_mm"),
    MetricType.PASS_LT_1MM_PCT: MetricMeta("Pass < 1 mm (%)", "percent"),
    MetricType.PASS_LT_1CM_PCT: MetricMeta("Pass < 1 cm (%)", "percent"),
    MetricType.PASS_LT_5CM_PCT: MetricMeta("Pass < 5 cm (%)", "percent"),
    MetricType.PASS_LT_10CM_PCT: MetricMeta("Pass < 10 cm (%)", "percent"),
    MetricType.PASS_LT_1M_PCT: MetricMeta("Pass < 1 m (%)", "percent"),
    MetricType.EVAL_TIME_SEC: MetricMeta("Evaluation Time (s)", "time"),
    MetricType.FIT_TIME_SEC: MetricMeta("Fit Time (s)", "time"),
    MetricType.THROUGHPUT_PTS_PER_SEC: MetricMeta("Throughput (pts/s)", "rate"),
    MetricType.LATENCY_US_PER_PT: MetricMeta("Latency (µs/pt)", "time"),
    MetricType.MODEL_SIZE_MB: MetricMeta("Model In-Memory Size (MB)", "megabytes"),
    MetricType.SIZE_CODE_BYTES: MetricMeta(
        "Code Size (bytes)", "bytes", is_integer=True
    ),
    MetricType.SIZE_DEPENDENCIES_BYTES: MetricMeta(
        "Dependencies Size (bytes)", "bytes", is_integer=True
    ),
    MetricType.SIZE_MODEL_BYTES: MetricMeta(
        "Model Weights/Instance Size (bytes)", "bytes", is_integer=True
    ),
    MetricType.SIZE_GRID_LOOKUP_BYTES: MetricMeta(
        "Grid Lookup Size (bytes)", "bytes", is_integer=True
    ),
    MetricType.TOTAL_SIZE_MB: MetricMeta("Total Footprint (MB)", "megabytes"),
}

METRIC_LABELS: dict[MetricType, str] = {k: v.label for k, v in METRIC_META.items()}


def format_metric_label(
    metric: MetricType, precision_unit: str = "mm"
) -> tuple[str, float]:
    """Return ``(display_label, scale_factor)`` for a metric and target precision unit."""
    unit_scale_map = {"m": 1.0, "cm": 100.0, "mm": 1000.0}
    unit_norm = precision_unit.lower()
    if unit_norm not in unit_scale_map:
        raise ValueError(
            f"Unsupported precision_unit '{precision_unit}'. Expected 'm', 'cm', or 'mm'."
        )

    meta = METRIC_META[metric]
    if meta.dimension == "distance" and unit_norm != "m":
        return meta.label.replace("(m)", f"({unit_norm})"), unit_scale_map[unit_norm]
    return meta.label, 1.0


class MetricGroup(str, Enum):
    """Predefined composable metric suites for multi-model comparison."""

    PRECISION_CORE = "precision_core"
    PRECISION_FULL = "precision_full"
    SURFACE_LEARNABILITY = "surface_learnability"
    PERFORMANCE_SPEED = "performance_speed"
    MEMORY_BREAKDOWN = "memory_breakdown"
    ALL_DIMENSIONS = "all_dimensions"


METRIC_GROUP_MEMBERS: dict[MetricGroup, tuple[MetricType, ...]] = {
    MetricGroup.PRECISION_CORE: (
        MetricType.MEAN_2D_M,
        MetricType.RMSE_2D_M,
        MetricType.P95_2D_M,
        MetricType.MAX_2D_M,
    ),
    MetricGroup.PRECISION_FULL: (
        MetricType.MEAN_2D_M,
        MetricType.STD_2D_M,
        MetricType.RMSE_2D_M,
        MetricType.CRMSE_2D_M,
        MetricType.IQR_2D_M,
        MetricType.MIN_2D_M,
        MetricType.P50_2D_M,
        MetricType.P90_2D_M,
        MetricType.P95_2D_M,
        MetricType.P99_2D_M,
        MetricType.MAX_2D_M,
        MetricType.BIAS_N_M,
        MetricType.BIAS_E_M,
        MetricType.ROUGHNESS_GRAD_RMS_MM_KM,
        MetricType.ROUGHNESS_GRAD_P95_MM_KM,
        MetricType.LOCAL_JITTER_RATIO,
        MetricType.LEARNABILITY_SCORE_MM,
        MetricType.PASS_LT_1MM_PCT,
        MetricType.PASS_LT_1CM_PCT,
        MetricType.PASS_LT_5CM_PCT,
        MetricType.PASS_LT_10CM_PCT,
        MetricType.PASS_LT_1M_PCT,
    ),
    MetricGroup.SURFACE_LEARNABILITY: (
        MetricType.CRMSE_2D_M,
        MetricType.IQR_2D_M,
        MetricType.P95_2D_M,
        MetricType.ROUGHNESS_GRAD_RMS_MM_KM,
        MetricType.ROUGHNESS_GRAD_P95_MM_KM,
        MetricType.LOCAL_JITTER_RATIO,
        MetricType.LEARNABILITY_SCORE_MM,
    ),
    MetricGroup.PERFORMANCE_SPEED: (
        MetricType.EVAL_TIME_SEC,
        MetricType.THROUGHPUT_PTS_PER_SEC,
        MetricType.LATENCY_US_PER_PT,
        MetricType.FIT_TIME_SEC,
    ),
    MetricGroup.MEMORY_BREAKDOWN: (
        MetricType.SIZE_CODE_BYTES,
        MetricType.SIZE_DEPENDENCIES_BYTES,
        MetricType.SIZE_MODEL_BYTES,
        MetricType.SIZE_GRID_LOOKUP_BYTES,
        MetricType.TOTAL_SIZE_MB,
    ),
    MetricGroup.ALL_DIMENSIONS: (
        MetricType.MEAN_2D_M,
        MetricType.RMSE_2D_M,
        MetricType.P95_2D_M,
        MetricType.THROUGHPUT_PTS_PER_SEC,
        MetricType.TOTAL_SIZE_MB,
    ),
}

MetricSpec = MetricGroup | MetricType | Sequence[MetricGroup | MetricType]


def resolve_metric_types(spec: MetricSpec) -> list[MetricType]:
    """Expand a :class:`MetricGroup`, :class:`MetricType`, or mixed sequence into an ordered list."""
    items: Sequence[MetricGroup | MetricType] = (
        [spec] if isinstance(spec, (MetricGroup, MetricType)) else spec
    )

    resolved: list[MetricType] = []
    seen: set[MetricType] = set()
    for item in items:
        if isinstance(item, MetricGroup):
            for m_type in METRIC_GROUP_MEMBERS[item]:
                if m_type not in seen:
                    seen.add(m_type)
                    resolved.append(m_type)
        elif isinstance(item, MetricType):
            if item not in seen:
                seen.add(item)
                resolved.append(item)
        else:
            raise TypeError(
                f"Expected MetricType or MetricGroup, got {type(item).__name__}"
            )
    return resolved


# GRS80 Ellipsoid constants (used for ETRS89 angular -> metric error conversion)
GRS80_A: float = 6_378_137.0
GRS80_F: float = 1.0 / 298.257222101
GRS80_E2: float = 2.0 * GRS80_F - GRS80_F**2

TOLERANCE_THRESHOLDS_M: dict[str, float] = {
    "lt_1mm": 1e-3,
    "lt_1cm": 1e-2,
    "lt_5cm": 5e-2,
    "lt_10cm": 1e-1,
    "lt_1m": 1.0,
}


def grs80_radii(lat_rad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute GRS80 meridian (M) and prime-vertical (N) radii of curvature.

    :param lat_rad: Geodetic latitude array in radians.
    :type lat_rad: np.ndarray
    :return: Tuple ``(M, N)`` in meters.
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    sin_lat = np.sin(lat_rad)
    w2 = 1.0 - GRS80_E2 * (sin_lat**2)
    w = np.sqrt(w2)
    n_radius = GRS80_A / w
    m_radius = GRS80_A * (1.0 - GRS80_E2) / (w2 * w)
    return m_radius, n_radius


def compute_spatial_roughness_stats(
    err_df: pd.DataFrame,
    k_neighbors: int = 4,
    char_step_km: float = 2.75,
) -> tuple[np.ndarray, dict[str, float]]:
    """Compute k-nearest-neighbor local residual gradient energy and jitter ratio.

    :param err_df: Error DataFrame containing ``st70_n``, ``st70_e``, ``err_n_m``, ``err_e_m``.
    :type err_df: pd.DataFrame
    :param k_neighbors: Number of spatial nearest neighbors per point.
    :type k_neighbors: int
    :param char_step_km: Characteristic sub-cell step length in km (default ``2.75`` km).
    :type char_step_km: float
    :return: Tuple ``(grad_2d_mm_km, stats_dict)``.
    :rtype: tuple[np.ndarray, dict[str, float]]
    """
    n_rows = len(err_df)
    nan = float("nan")
    empty_stats: dict[str, float] = {
        "roughness_grad_rms_mm_km": nan,
        "roughness_grad_p95_mm_km": nan,
        "local_jitter_ratio": nan,
    }
    grad_full = np.full(n_rows, np.nan, dtype=np.float64)

    if (
        n_rows < 2
        or "st70_n" not in err_df.columns
        or "st70_e" not in err_df.columns
        or "err_n_m" not in err_df.columns
        or "err_e_m" not in err_df.columns
    ):
        return grad_full, empty_stats

    n_km = err_df["st70_n"].to_numpy(dtype=np.float64) * 1e-3
    e_km = err_df["st70_e"].to_numpy(dtype=np.float64) * 1e-3
    en_m = err_df["err_n_m"].to_numpy(dtype=np.float64)
    ee_m = err_df["err_e_m"].to_numpy(dtype=np.float64)

    valid_mask = (
        np.isfinite(n_km) & np.isfinite(e_km) & np.isfinite(en_m) & np.isfinite(ee_m)
    )
    matched = int(np.sum(valid_mask))
    if matched < 2:
        return grad_full, empty_stats

    coords_km = np.column_stack([n_km[valid_mask], e_km[valid_mask]])
    en = en_m[valid_mask]
    ee = ee_m[valid_mask]

    k_query = min(k_neighbors + 1, matched)
    tree = cKDTree(coords_km, balanced_tree=False)
    dists, idxs = tree.query(coords_km, k=k_query, workers=-1)
    if k_query == 2:
        d_km = np.maximum(dists[:, 1:2], 1e-6)
        nbrs = idxs[:, 1:2]
    else:
        d_km = np.maximum(dists[:, 1:], 1e-6)
        nbrs = idxs[:, 1:]

    diff_n_mm = (en[:, None] - en[nbrs]) * 1000.0
    diff_e_mm = (ee[:, None] - ee[nbrs]) * 1000.0
    diff_2d_mm = np.hypot(diff_n_mm, diff_e_mm)
    grad_per_pt = np.mean(diff_2d_mm / d_km, axis=1)

    mean_nbr_n = np.mean(en[nbrs], axis=1)
    mean_nbr_e = np.mean(ee[nbrs], axis=1)
    jitter_rms_m = float(
        np.sqrt(np.mean((en - mean_nbr_n) ** 2 + (ee - mean_nbr_e) ** 2))
    )
    crmse_2d_m = float(np.sqrt(np.var(en) + np.var(ee)))
    local_jitter_ratio = (
        float(jitter_rms_m / crmse_2d_m) if crmse_2d_m > 1e-12 else 0.0
    )

    grad_full[valid_mask] = grad_per_pt
    del char_step_km
    return grad_full, {
        "roughness_grad_rms_mm_km": float(np.sqrt(np.mean(grad_per_pt**2))),
        "roughness_grad_p95_mm_km": float(np.percentile(grad_per_pt, 95.0)),
        "local_jitter_ratio": local_jitter_ratio,
    }


def compute_shift_errors(
    df_ref: pd.DataFrame,
    pred_shifts: np.ndarray,
    true_shifts: np.ndarray,
) -> pd.DataFrame:
    """Compute per-point metric residuals for 2D grid shift predictions ``(dN, dE)``.

    :param df_ref: Reference grid or benchmark DataFrame.
    :type df_ref: pd.DataFrame
    :param pred_shifts: Predicted shifts ``[dn, de]`` in meters, shape ``(N, 2)``.
    :type pred_shifts: np.ndarray
    :param true_shifts: Ground-truth shifts ``[dn, de]`` in meters, shape ``(N, 2)``.
    :type true_shifts: np.ndarray
    :return: Copy of ``df_ref`` augmented with ``pred_dn_m``, ``pred_de_m``,
        ``true_dn_m``, ``true_de_m``, ``err_n_m``, ``err_e_m``, ``err_2d_m``,
        and ``grad_2d_mm_km``.
    :rtype: pd.DataFrame
    :raises ValueError: If prediction or ground-truth length does not match ``df_ref``.
    """
    pred = np.asarray(pred_shifts, dtype=np.float64)
    true = np.asarray(true_shifts, dtype=np.float64)
    if pred.shape != (len(df_ref), 2) or true.shape != (len(df_ref), 2):
        raise ValueError(
            f"Shift shape mismatch: pred={pred.shape}, true={true.shape}, expected=({len(df_ref)}, 2)."
        )

    err_n = pred[:, 0] - true[:, 0]
    err_e = pred[:, 1] - true[:, 1]

    out = df_ref.copy()
    out["pred_dn_m"] = pred[:, 0]
    out["pred_de_m"] = pred[:, 1]
    out["true_dn_m"] = true[:, 0]
    out["true_de_m"] = true[:, 1]
    out["err_n_m"] = err_n
    out["err_e_m"] = err_e
    out["err_2d_m"] = np.hypot(err_n, err_e)
    grad_full, _ = compute_spatial_roughness_stats(out)
    out["grad_2d_mm_km"] = grad_full
    return out


def compute_coordinate_errors(
    df_ref: pd.DataFrame,
    pred_c1: np.ndarray,
    pred_c2: np.ndarray,
    direction: str,
    pred_h: np.ndarray | None = None,
    is_degrees: bool = True,
) -> pd.DataFrame:
    """Compute per-point metric residuals against ground-truth ``pytransdat`` columns."""
    c1 = np.asarray(pred_c1, dtype=np.float64)
    c2 = np.asarray(pred_c2, dtype=np.float64)
    if len(c1) != len(df_ref) or len(c2) != len(df_ref):
        raise ValueError(
            f"Prediction length ({len(c1)}, {len(c2)}) does not match reference ({len(df_ref)})."
        )

    out = df_ref.copy()

    if direction == "etrs89_to_st70":
        err_n = c1 - df_ref["st70_n"].to_numpy(dtype=np.float64)
        err_e = c2 - df_ref["st70_e"].to_numpy(dtype=np.float64)
        if pred_h is not None:
            out["err_h_m"] = np.asarray(pred_h, dtype=np.float64) - df_ref[
                "st70_h"
            ].to_numpy(dtype=np.float64)
    elif direction == "st70_to_etrs89":
        lat_pred_rad = np.radians(c1) if is_degrees else c1
        lon_pred_rad = np.radians(c2) if is_degrees else c2
        lat_ref_rad = df_ref["etrs89_lat_rad"].to_numpy(dtype=np.float64)
        lon_ref_rad = df_ref["etrs89_lon_rad"].to_numpy(dtype=np.float64)

        d_lat = lat_pred_rad - lat_ref_rad
        d_lon = (lon_pred_rad - lon_ref_rad + np.pi) % (2.0 * np.pi) - np.pi
        m_rad, n_rad = grs80_radii(lat_ref_rad)
        err_n = m_rad * d_lat
        err_e = n_rad * np.cos(lat_ref_rad) * d_lon
        if pred_h is not None:
            out["err_h_m"] = np.asarray(pred_h, dtype=np.float64) - df_ref[
                "etrs89_h"
            ].to_numpy(dtype=np.float64)
    else:
        raise ValueError(
            f"Unknown direction '{direction}'. Expected 'etrs89_to_st70' or 'st70_to_etrs89'."
        )

    out["err_n_m"] = err_n
    out["err_e_m"] = err_e
    out["err_2d_m"] = np.hypot(err_n, err_e)
    grad_full, _ = compute_spatial_roughness_stats(out)
    out["grad_2d_mm_km"] = grad_full
    return out


# Backward-compatible alias
compute_errors = compute_coordinate_errors


def _summarize_slice(err_df: pd.DataFrame) -> dict[str, Any]:
    e2d_raw = err_df["err_2d_m"].to_numpy(dtype=np.float64)
    en_raw = err_df["err_n_m"].to_numpy(dtype=np.float64)
    ee_raw = err_df["err_e_m"].to_numpy(dtype=np.float64)

    total_count = len(err_df)
    valid_mask = np.isfinite(e2d_raw) & np.isfinite(en_raw) & np.isfinite(ee_raw)
    matched_count = int(np.sum(valid_mask))
    unmatched_count = total_count - matched_count
    coverage_pct = (
        float(100.0 * matched_count / total_count) if total_count > 0 else 0.0
    )

    if matched_count == 0:
        nan = float("nan")
        summary: dict[str, Any] = {
            "count": 0,
            "total_count": total_count,
            "unmatched_count": unmatched_count,
            "coverage_pct": coverage_pct,
            "mean_2d_m": nan,
            "std_2d_m": nan,
            "rmse_2d_m": nan,
            "crmse_2d_m": nan,
            "iqr_2d_m": nan,
            "min_2d_m": nan,
            "p50_2d_m": nan,
            "p90_2d_m": nan,
            "p95_2d_m": nan,
            "p99_2d_m": nan,
            "p99_9_2d_m": nan,
            "max_2d_m": nan,
            "bias_n_m": nan,
            "bias_e_m": nan,
            "rmse_n_m": nan,
            "rmse_e_m": nan,
            "roughness_grad_rms_mm_km": nan,
            "roughness_grad_p95_mm_km": nan,
            "local_jitter_ratio": nan,
            "learnability_score_mm": nan,
            "pass_rates_pct": dict.fromkeys(TOLERANCE_THRESHOLDS_M, 0.0),
        }
    else:
        e2d = e2d_raw[valid_mask]
        en = en_raw[valid_mask]
        ee = ee_raw[valid_mask]
        crmse_2d_m = float(np.sqrt(np.var(en) + np.var(ee)))
        p25_2d_m = float(np.percentile(e2d, 25.0))
        p75_2d_m = float(np.percentile(e2d, 75.0))
        iqr_2d_m = p75_2d_m - p25_2d_m
        p95_2d_m = float(np.percentile(e2d, 95.0))

        _, rough_stats = compute_spatial_roughness_stats(err_df)
        grad_rms = rough_stats["roughness_grad_rms_mm_km"]
        grad_p95 = rough_stats["roughness_grad_p95_mm_km"]
        jitter_ratio = rough_stats["local_jitter_ratio"]
        if np.isfinite(grad_rms):
            learnability_score_mm = float(
                0.50 * (crmse_2d_m * 1000.0)
                + 0.30 * (2.75 * grad_rms)
                + 0.20 * (p95_2d_m * 1000.0)
            )
        else:
            learnability_score_mm = float("nan")

        summary = {
            "count": matched_count,
            "total_count": total_count,
            "unmatched_count": unmatched_count,
            "coverage_pct": coverage_pct,
            "mean_2d_m": float(np.mean(e2d)),
            "std_2d_m": float(np.std(e2d)),
            "rmse_2d_m": float(np.sqrt(np.mean(e2d**2))),
            "crmse_2d_m": crmse_2d_m,
            "iqr_2d_m": iqr_2d_m,
            "min_2d_m": float(np.min(e2d)),
            "p50_2d_m": float(np.percentile(e2d, 50.0)),
            "p90_2d_m": float(np.percentile(e2d, 90.0)),
            "p95_2d_m": p95_2d_m,
            "p99_2d_m": float(np.percentile(e2d, 99.0)),
            "p99_9_2d_m": float(np.percentile(e2d, 99.9)),
            "max_2d_m": float(np.max(e2d)),
            "bias_n_m": float(np.mean(en)),
            "bias_e_m": float(np.mean(ee)),
            "rmse_n_m": float(np.sqrt(np.mean(en**2))),
            "rmse_e_m": float(np.sqrt(np.mean(ee**2))),
            "roughness_grad_rms_mm_km": grad_rms,
            "roughness_grad_p95_mm_km": grad_p95,
            "local_jitter_ratio": jitter_ratio,
            "learnability_score_mm": learnability_score_mm,
            "pass_rates_pct": {
                key: float(100.0 * np.mean(e2d < thresh))
                for key, thresh in TOLERANCE_THRESHOLDS_M.items()
            },
        }

    if "err_h_m" in err_df.columns:
        eh_raw = err_df["err_h_m"].to_numpy(dtype=np.float64)
        eh = eh_raw[np.isfinite(eh_raw)]
        if len(eh) > 0:
            summary["vertical"] = {
                "bias_h_m": float(np.mean(eh)),
                "mean_abs_h_m": float(np.mean(np.abs(eh))),
                "rmse_h_m": float(np.sqrt(np.mean(eh**2))),
                "max_abs_h_m": float(np.max(np.abs(eh))),
            }

    return summary


def summarize_errors(err_df: pd.DataFrame) -> dict[str, Any]:
    """Compute overall and node-vs-interior stratified error scorecards.

    :param err_df: DataFrame returned by :func:`compute_shift_errors` or
        :func:`compute_coordinate_errors`.
    :type err_df: pd.DataFrame
    :return: Nested dict of summary metrics.
    :rtype: dict[str, Any]
    """
    result = _summarize_slice(err_df)
    if "is_node" in err_df.columns:
        nodes_df = err_df[err_df["is_node"]]
        interior_df = err_df[~err_df["is_node"]]
        result["by_point_type"] = {
            "nodes": _summarize_slice(nodes_df) if len(nodes_df) > 0 else None,
            "interior": _summarize_slice(interior_df) if len(interior_df) > 0 else None,
        }
    return result


def format_scorecard_table(summary: dict[str, Any], name: str, direction: str) -> str:
    """Format a concise human-readable evaluation scorecard."""
    pr = summary["pass_rates_pct"]
    lines = [
        f"=== Evaluation Scorecard: {name} ({direction}) ===",
        f"Points        : {summary['count']:,}",
        f"2D Mean / SD  : {summary['mean_2d_m']:.4f} m  /  {summary['std_2d_m']:.4f} m",
        f"2D RMSE       : {summary['rmse_2d_m']:.4f} m",
        (
            f"2D Percentiles: p50={summary['p50_2d_m']:.4f} m | p95={summary['p95_2d_m']:.4f} m | "
            f"p99={summary['p99_2d_m']:.4f} m | max={summary['max_2d_m']:.4f} m"
        ),
        f"Signed Bias   : dN={summary['bias_n_m']:+.4f} m | dE={summary['bias_e_m']:+.4f} m",
        (
            f"Learnability  : cRMSE={summary.get('crmse_2d_m', float('nan')) * 1000.0:.2f} mm | "
            f"IQR={summary.get('iqr_2d_m', float('nan')) * 1000.0:.2f} mm | "
            f"RoughRMS={summary.get('roughness_grad_rms_mm_km', float('nan')):.2f} mm/km | "
            f"Jitter={summary.get('local_jitter_ratio', float('nan')):.3f} | "
            f"Score={summary.get('learnability_score_mm', float('nan')):.2f} mm"
        ),
        (
            f"Pass Rates    : <1mm={pr['lt_1mm']:.1f}% | <1cm={pr['lt_1cm']:.1f}% | "
            f"<5cm={pr['lt_5cm']:.1f}% | <10cm={pr['lt_10cm']:.1f}% | <1m={pr['lt_1m']:.1f}%"
        ),
    ]
    if summary.get("unmatched_count", 0) > 0:
        lines.append(
            f"Coverage      : {summary['coverage_pct']:.2f}% "
            f"({summary['unmatched_count']:,} unmatched)"
        )
    if "vertical" in summary:
        v = summary["vertical"]
        lines.append(
            f"Vertical (H)  : bias={v['bias_h_m']:+.4f} m | RMSE={v['rmse_h_m']:.4f} m | "
            f"max|dH|={v['max_abs_h_m']:.4f} m"
        )
    return "\n".join(lines)


def dataframe_to_markdown(df: pd.DataFrame, float_precision: int = 4) -> str:
    """Format a DataFrame as a GitHub-flavored Markdown table without requiring tabulate.

    :param df: Input DataFrame.
    :type df: pd.DataFrame
    :param float_precision: Number of decimal places for float values.
    :type float_precision: int
    :return: Markdown-formatted table string.
    :rtype: str
    """
    headers = [str(c) for c in df.columns]
    numeric_cols = [pd.api.types.is_numeric_dtype(df[c]) for c in df.columns]
    int_like_cols = [
        pd.api.types.is_integer_dtype(df[c]) or "(bytes)" in str(c) or str(c) == "Count"
        for c in df.columns
    ]

    lines = ["| " + " | ".join(headers) + " |"]
    lines.append(
        "| "
        + " | ".join([":---:" if is_num else ":---" for is_num in numeric_cols])
        + " |"
    )
    for _, row in df.iterrows():
        formatted_vals: list[str] = []
        for idx, v in enumerate(row.values):
            if int_like_cols[idx] and isinstance(
                v, (int, float, np.integer, np.floating)
            ):
                formatted_vals.append(
                    f"{round(float(v)):,}" if np.isfinite(v) else "NaN"
                )
            elif isinstance(v, (float, np.floating)):
                fv = float(v)
                if not np.isfinite(fv):
                    formatted_vals.append("NaN")
                elif abs(fv) >= 10_000.0:
                    formatted_vals.append(f"{fv:,.1f}")
                else:
                    formatted_vals.append(f"{fv:.{float_precision}f}")
            elif isinstance(v, (int, np.integer)):
                formatted_vals.append(f"{int(v):,}")
            else:
                formatted_vals.append(str(v))
        lines.append("| " + " | ".join(formatted_vals) + " |")
    return "\n".join(lines)
