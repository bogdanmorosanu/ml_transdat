"""Standardized error computation and scorecard metrics for Stereo70 <-> ETRS89."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

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
    :return: Tuple ``(M, N)`` in meters.
    """
    sin_lat = np.sin(lat_rad)
    w2 = 1.0 - GRS80_E2 * (sin_lat**2)
    w = np.sqrt(w2)
    n_radius = GRS80_A / w
    m_radius = GRS80_A * (1.0 - GRS80_E2) / (w2 * w)
    return m_radius, n_radius


def compute_errors(
    df_ref: pd.DataFrame,
    pred_c1: np.ndarray,
    pred_c2: np.ndarray,
    direction: str,
    pred_h: Optional[np.ndarray] = None,
    is_degrees: bool = True,
) -> pd.DataFrame:
    """Compute per-point metric residuals against ground-truth ``pytransdat`` columns.

    :param df_ref: Reference benchmark/grid DataFrame (e.g., ``ro_grid_4x4.parquet``).
    :param pred_c1: Predicted Northing (m) for ``etrs89_to_st70``, or Latitude for ``st70_to_etrs89``.
    :param pred_c2: Predicted Easting (m) for ``etrs89_to_st70``, or Longitude for ``st70_to_etrs89``.
    :param direction: Transformation direction (``'etrs89_to_st70'`` or ``'st70_to_etrs89'``).
    :param pred_h: Optional predicted height (m) for 3D evaluation.
    :param is_degrees: Whether ``pred_c1``/``pred_c2`` are in degrees when ``direction == 'st70_to_etrs89'``.
    :return: Copy of ``df_ref`` augmented with ``err_n_m``, ``err_e_m``, ``err_2d_m`` (and ``err_h_m``).
    """
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
            out["err_h_m"] = np.asarray(pred_h, dtype=np.float64) - df_ref["st70_h"].to_numpy(
                dtype=np.float64
            )
    elif direction == "st70_to_etrs89":
        lat_pred_rad = np.radians(c1) if is_degrees else c1
        lon_pred_rad = np.radians(c2) if is_degrees else c2
        lat_ref_rad = df_ref["etrs89_lat_rad"].to_numpy(dtype=np.float64)
        lon_ref_rad = df_ref["etrs89_lon_rad"].to_numpy(dtype=np.float64)

        d_lat = lat_pred_rad - lat_ref_rad
        # Wrap longitude difference to [-pi, pi]
        d_lon = (lon_pred_rad - lon_ref_rad + np.pi) % (2.0 * np.pi) - np.pi
        m_rad, n_rad = grs80_radii(lat_ref_rad)
        err_n = m_rad * d_lat
        err_e = n_rad * np.cos(lat_ref_rad) * d_lon
        if pred_h is not None:
            out["err_h_m"] = np.asarray(pred_h, dtype=np.float64) - df_ref["etrs89_h"].to_numpy(
                dtype=np.float64
            )
    else:
        raise ValueError(
            f"Unknown direction '{direction}'. Expected 'etrs89_to_st70' or 'st70_to_etrs89'."
        )

    out["err_n_m"] = err_n
    out["err_e_m"] = err_e
    out["err_2d_m"] = np.hypot(err_n, err_e)
    return out


def _summarize_slice(err_df: pd.DataFrame) -> dict[str, Any]:
    e2d = err_df["err_2d_m"].to_numpy(dtype=np.float64)
    en = err_df["err_n_m"].to_numpy(dtype=np.float64)
    ee = err_df["err_e_m"].to_numpy(dtype=np.float64)

    summary: dict[str, Any] = {
        "count": int(len(err_df)),
        "mean_2d_m": float(np.mean(e2d)),
        "std_2d_m": float(np.std(e2d)),
        "rmse_2d_m": float(np.sqrt(np.mean(e2d**2))),
        "min_2d_m": float(np.min(e2d)),
        "p50_2d_m": float(np.percentile(e2d, 50.0)),
        "p90_2d_m": float(np.percentile(e2d, 90.0)),
        "p95_2d_m": float(np.percentile(e2d, 95.0)),
        "p99_2d_m": float(np.percentile(e2d, 99.0)),
        "p99_9_2d_m": float(np.percentile(e2d, 99.9)),
        "max_2d_m": float(np.max(e2d)),
        "bias_n_m": float(np.mean(en)),
        "bias_e_m": float(np.mean(ee)),
        "rmse_n_m": float(np.sqrt(np.mean(en**2))),
        "rmse_e_m": float(np.sqrt(np.mean(ee**2))),
        "pass_rates_pct": {
            key: float(100.0 * np.mean(e2d < thresh))
            for key, thresh in TOLERANCE_THRESHOLDS_M.items()
        },
    }

    if "err_h_m" in err_df.columns:
        eh = err_df["err_h_m"].to_numpy(dtype=np.float64)
        summary["vertical"] = {
            "bias_h_m": float(np.mean(eh)),
            "mean_abs_h_m": float(np.mean(np.abs(eh))),
            "rmse_h_m": float(np.sqrt(np.mean(eh**2))),
            "max_abs_h_m": float(np.max(np.abs(eh))),
        }

    return summary


def summarize_errors(err_df: pd.DataFrame) -> dict[str, Any]:
    """Compute overall and node-vs-interior stratified error scorecards.

    :param err_df: DataFrame returned by :func:`compute_errors`.
    :return: Nested dict of summary metrics.
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
        f"Points       : {summary['count']:,}",
        f"2D Mean / SD : {summary['mean_2d_m']:.4f} m  /  {summary['std_2d_m']:.4f} m",
        f"2D RMSE      : {summary['rmse_2d_m']:.4f} m",
        f"2D Percentiles: p50={summary['p50_2d_m']:.4f} m | p95={summary['p95_2d_m']:.4f} m | "
        f"p99={summary['p99_2d_m']:.4f} m | max={summary['max_2d_m']:.4f} m",
        f"Signed Bias  : dN={summary['bias_n_m']:+.4f} m | dE={summary['bias_e_m']:+.4f} m",
        f"Pass Rates   : <1mm={pr['lt_1mm']:.1f}% | <1cm={pr['lt_1cm']:.1f}% | "
        f"<5cm={pr['lt_5cm']:.1f}% | <10cm={pr['lt_10cm']:.1f}% | <1m={pr['lt_1m']:.1f}%",
    ]
    if "vertical" in summary:
        v = summary["vertical"]
        lines.append(
            f"Vertical (H) : bias={v['bias_h_m']:+.4f} m | RMSE={v['rmse_h_m']:.4f} m | "
            f"max|dH|={v['max_abs_h_m']:.4f} m"
        )
    return "\n".join(lines)
