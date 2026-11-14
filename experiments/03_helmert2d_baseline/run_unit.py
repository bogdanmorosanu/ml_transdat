"""Unit 03: 2D 4-Parameter Helmert Baseline (without SPG grid) vs pytransdat."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pytransdat.proj_stereo import StereoProj
from pytransdat.spg_reader import SpgReader
from src.core.base_unit import ResearchUnit
from src.core.metrics import compute_errors, format_scorecard_table, summarize_errors
from src.core.viz import generate_standard_diagnostic_suite
from src.models.helmert2d import (
    ARCSEC_PER_RAD,
    Helmert2DParams,
    TwoStepHelmert2DTransformer,
    fit_helmert2d,
)

DEFAULT_BENCHMARK = Path("data/benchmarks/ro_grid_3x3.parquet")


def _load_embedded_params() -> Helmert2DParams:
    """Load the embedded 4-parameter Helmert constants from the active SPG grid."""
    reader = SpgReader()
    return Helmert2DParams(
        tn_m=float(reader.helmert_tn),
        te_m=float(reader.helmert_te),
        ppm=float(reader.helmert_ppm),
        rot_rad=float(reader.helmert_rot_rad),
        rot_arcsec=float(reader.helmert_rot_rad * ARCSEC_PER_RAD),
    )


def _evaluate_transformer(
    transformer: TwoStepHelmert2DTransformer,
    df: pd.DataFrame,
    model_name: str,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame, pd.DataFrame, float, float]:
    """Run bidirectional evaluation and return summaries, error DataFrames, and timings."""
    lat_rad = df["etrs89_lat_rad"].to_numpy(dtype=np.float64)
    lon_rad = df["etrs89_lon_rad"].to_numpy(dtype=np.float64)
    st70_n = df["st70_n"].to_numpy(dtype=np.float64)
    st70_e = df["st70_e"].to_numpy(dtype=np.float64)

    t_start = time.perf_counter()
    pred_n, pred_e = transformer.etrs89_to_st70(lat_rad, lon_rad)
    inv_elapsed = time.perf_counter() - t_start

    inv_err_df = compute_errors(
        df_ref=df,
        pred_c1=pred_n,
        pred_c2=pred_e,
        direction="etrs89_to_st70",
    )
    inv_summary = summarize_errors(inv_err_df)

    t_start = time.perf_counter()
    pred_lat_rad, pred_lon_rad = transformer.st70_to_etrs89(st70_n, st70_e)
    fwd_elapsed = time.perf_counter() - t_start

    fwd_err_df = compute_errors(
        df_ref=df,
        pred_c1=pred_lat_rad,
        pred_c2=pred_lon_rad,
        direction="st70_to_etrs89",
        is_degrees=False,
    )
    fwd_summary = summarize_errors(fwd_err_df)

    print(format_scorecard_table(inv_summary, model_name, "etrs89_to_st70"))
    print(format_scorecard_table(fwd_summary, model_name, "st70_to_etrs89"))

    return inv_summary, fwd_summary, inv_err_df, fwd_err_df, inv_elapsed, fwd_elapsed


def _compact_metrics(summary: dict[str, Any], elapsed_s: float, n_pts: int) -> dict[str, Any]:
    return {
        "mean_2d_m": summary["mean_2d_m"],
        "rmse_2d_m": summary["rmse_2d_m"],
        "p50_2d_m": summary["p50_2d_m"],
        "p95_2d_m": summary["p95_2d_m"],
        "p99_2d_m": summary["p99_2d_m"],
        "max_2d_m": summary["max_2d_m"],
        "bias_n_m": summary["bias_n_m"],
        "bias_e_m": summary["bias_e_m"],
        "pass_lt_10cm_pct": summary["pass_rates_pct"]["lt_10cm"],
        "pass_lt_1m_pct": summary["pass_rates_pct"]["lt_1m"],
        "throughput_pts_per_s": round(n_pts / elapsed_s, 1),
    }


class Helmert2DBaselineUnit(ResearchUnit):
    """Research Unit 03: Refitted 4-parameter 2D Helmert baseline (no SPG grid)."""

    @property
    def unit_id(self) -> str:
        return "03"

    @property
    def unit_name(self) -> str:
        return "helmert2d_baseline"

    def run(self) -> dict[str, Any]:
        t0 = time.perf_counter()
        df = pd.read_parquet(DEFAULT_BENCHMARK)
        df_nodes = df[df["is_node"]].reset_index(drop=True)

        # 1. Fit 4-parameter 2D Helmert on valid SPG grid nodes only
        proj = StereoProj()
        os_n_nodes, os_e_nodes = proj.to_grid(
            df_nodes["etrs89_lat_rad"].to_numpy(dtype=np.float64).tolist(),
            df_nodes["etrs89_lon_rad"].to_numpy(dtype=np.float64).tolist(),
        )
        fitted_params = fit_helmert2d(
            st70_n=df_nodes["st70_n"].to_numpy(dtype=np.float64),
            st70_e=df_nodes["st70_e"].to_numpy(dtype=np.float64),
            os_n=os_n_nodes,
            os_e=os_e_nodes,
        )
        embedded_params = _load_embedded_params()

        param_payload = {
            "fit_point_count": len(df_nodes),
            "convention": "st70_to_os",
            "embedded_spg": embedded_params.to_dict(),
            "fitted_spg_nodes": fitted_params.to_dict(),
            "delta_fitted_minus_embedded": {
                "tn_m": fitted_params.tn_m - embedded_params.tn_m,
                "te_m": fitted_params.te_m - embedded_params.te_m,
                "ppm": fitted_params.ppm - embedded_params.ppm,
                "rot_arcsec": fitted_params.rot_arcsec - embedded_params.rot_arcsec,
            },
        }
        (self.results_dir / "helmert2d_params.json").write_text(
            json.dumps(param_payload, indent=2), encoding="utf-8"
        )

        # 2. Evaluate Embedded Helmert2D (no grid)
        emb_transformer = TwoStepHelmert2DTransformer(embedded_params)
        emb_inv_sum, emb_fwd_sum, _, _, emb_inv_t, emb_fwd_t = _evaluate_transformer(
            emb_transformer, df, "embedded_helmert2d"
        )
        (self.results_dir / "embedded_helmert2d_etrs89_to_st70_metrics.json").write_text(
            json.dumps(emb_inv_sum, indent=2), encoding="utf-8"
        )
        (self.results_dir / "embedded_helmert2d_st70_to_etrs89_metrics.json").write_text(
            json.dumps(emb_fwd_sum, indent=2), encoding="utf-8"
        )

        # 3. Evaluate Refitted Helmert2D (no grid)
        fit_transformer = TwoStepHelmert2DTransformer(fitted_params)
        fit_inv_sum, fit_fwd_sum, fit_inv_err_df, fit_fwd_err_df, fit_inv_t, fit_fwd_t = (
            _evaluate_transformer(fit_transformer, df, "fitted_helmert2d")
        )
        (self.results_dir / "fitted_helmert2d_etrs89_to_st70_metrics.json").write_text(
            json.dumps(fit_inv_sum, indent=2), encoding="utf-8"
        )
        (self.results_dir / "fitted_helmert2d_st70_to_etrs89_metrics.json").write_text(
            json.dumps(fit_fwd_sum, indent=2), encoding="utf-8"
        )

        # 4. Out-of-sample interior-only (34,213 points) check for Fitted Helmert2D
        interior_mask = ~df["is_node"].to_numpy()
        fit_inv_int_sum = summarize_errors(fit_inv_err_df.loc[interior_mask])
        fit_fwd_int_sum = summarize_errors(fit_fwd_err_df.loc[interior_mask])

        # 5. Round-trip verification on full benchmark
        rt_n, rt_e = fit_transformer.etrs89_to_st70(
            df["etrs89_lat_rad"].to_numpy(dtype=np.float64),
            df["etrs89_lon_rad"].to_numpy(dtype=np.float64),
        )
        rt_lat, rt_lon = fit_transformer.st70_to_etrs89(rt_n, rt_e)
        rt_lat2, rt_lon2 = fit_transformer.st70_to_etrs89(
            df["st70_n"].to_numpy(dtype=np.float64),
            df["st70_e"].to_numpy(dtype=np.float64),
        )
        rt_n2, rt_e2 = fit_transformer.etrs89_to_st70(rt_lat2, rt_lon2)
        rt_st70_max_m = float(
            np.max(
                np.hypot(
                    rt_n2 - df["st70_n"].to_numpy(dtype=np.float64),
                    rt_e2 - df["st70_e"].to_numpy(dtype=np.float64),
                )
            )
        )
        rt_geo_max_rad = float(
            np.max(
                np.hypot(
                    rt_lat - df["etrs89_lat_rad"].to_numpy(dtype=np.float64),
                    rt_lon - df["etrs89_lon_rad"].to_numpy(dtype=np.float64),
                )
            )
        )

        # 6. Diagnostic plots for Refitted Helmert2D
        inv_plots = generate_standard_diagnostic_suite(
            err_df=fit_inv_err_df,
            summary=fit_inv_sum,
            out_dir=self.plots_dir,
            model_name="fitted_helmert2d",
            direction="etrs89_to_st70",
            dpi=200,
        )
        fwd_plots = generate_standard_diagnostic_suite(
            err_df=fit_fwd_err_df,
            summary=fit_fwd_sum,
            out_dir=self.plots_dir,
            model_name="fitted_helmert2d",
            direction="st70_to_etrs89",
            dpi=200,
        )

        elapsed = time.perf_counter() - t0
        brief: dict[str, Any] = {
            "unit_id": self.unit_id,
            "unit_name": self.unit_name,
            "fit_node_count": len(df_nodes),
            "sample_count": len(df),
            "interior_count": int(interior_mask.sum()),
            "parameters": param_payload,
            "roundtrip": {
                "st70_to_etrs89_to_st70_max_m": rt_st70_max_m,
                "etrs89_to_st70_to_etrs89_max_rad": rt_geo_max_rad,
            },
            "embedded_helmert2d": {
                "etrs89_to_st70": _compact_metrics(emb_inv_sum, emb_inv_t, len(df)),
                "st70_to_etrs89": _compact_metrics(emb_fwd_sum, emb_fwd_t, len(df)),
            },
            "fitted_helmert2d": {
                "etrs89_to_st70": _compact_metrics(fit_inv_sum, fit_inv_t, len(df)),
                "st70_to_etrs89": _compact_metrics(fit_fwd_sum, fit_fwd_t, len(df)),
                "interior_only_etrs89_to_st70": {
                    "mean_2d_m": fit_inv_int_sum["mean_2d_m"],
                    "rmse_2d_m": fit_inv_int_sum["rmse_2d_m"],
                    "max_2d_m": fit_inv_int_sum["max_2d_m"],
                },
                "interior_only_st70_to_etrs89": {
                    "mean_2d_m": fit_fwd_int_sum["mean_2d_m"],
                    "rmse_2d_m": fit_fwd_int_sum["rmse_2d_m"],
                    "max_2d_m": fit_fwd_int_sum["max_2d_m"],
                },
            },
            "plots": {
                "etrs89_to_st70": {k: v.name for k, v in inv_plots.items()},
                "st70_to_etrs89": {k: v.name for k, v in fwd_plots.items()},
            },
            "runtime_s": round(elapsed, 3),
        }
        self.save_brief(brief)
        print(f"=== Research Unit 03 [{self.unit_name}] Complete ===")
        return brief


def main() -> None:
    unit = Helmert2DBaselineUnit(Path("experiments/03_helmert2d_baseline"))
    unit.run()


if __name__ == "__main__":
    main()
