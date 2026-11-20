"""Unit 03: 2D 4-Parameter Helmert Baseline (without SPG grid) vs pytransdat."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from pytransdat.proj_stereo import StereoProj
from pytransdat.spg_reader import SpgReader
from src.core.base_unit import TransformationExperimentUnit
from src.data import GridId
from src.evaluation import EvaluationResult, summarize_errors
from src.models.helmert2d import (
    ARCSEC_PER_RAD,
    Helmert2DParams,
    TwoStepHelmert2DTransformer,
    fit_helmert2d,
)
from src.viz import PlotGroup


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


def _compact_metrics(res: EvaluationResult) -> dict[str, Any]:
    s = res.summary
    return {
        "mean_2d_m": s["mean_2d_m"],
        "rmse_2d_m": s["rmse_2d_m"],
        "p50_2d_m": s["p50_2d_m"],
        "p95_2d_m": s["p95_2d_m"],
        "p99_2d_m": s["p99_2d_m"],
        "max_2d_m": s["max_2d_m"],
        "bias_n_m": s["bias_n_m"],
        "bias_e_m": s["bias_e_m"],
        "pass_lt_10cm_pct": s["pass_rates_pct"]["lt_10cm"],
        "pass_lt_1m_pct": s["pass_rates_pct"]["lt_1m"],
        "throughput_pts_per_s": round(res.throughput_pts_per_sec, 1),
    }


class Helmert2DBaselineUnit(TransformationExperimentUnit):
    """Research Unit 03: Refitted 4-parameter 2D Helmert baseline (no SPG grid)."""

    @property
    def unit_id(self) -> str:
        return "03"

    @property
    def unit_name(self) -> str:
        return "helmert2d_baseline"

    def run(self) -> dict[str, Any]:
        t0 = time.perf_counter()
        dataset = self.load_grid(GridId.RO_GRID_4X4)
        df = dataset.df
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

        # 2. Evaluate Embedded Helmert2D (no plots needed) and Fitted Helmert2D (with plots)
        emb_results = self.evaluate_transformer(
            TwoStepHelmert2DTransformer(embedded_params),
            model_name="embedded_helmert2d",
            dataset_id=GridId.RO_GRID_4X4,
            plots=None,
        )
        fit_transformer = TwoStepHelmert2DTransformer(fitted_params)
        fit_results = self.evaluate_transformer(
            fit_transformer,
            model_name="fitted_helmert2d",
            dataset_id=GridId.RO_GRID_4X4,
            plots=PlotGroup.STANDARD_DIAGNOSTIC,
        )
        self.save_scorecards([*emb_results, *fit_results])

        emb_inv_res, emb_fwd_res = emb_results
        fit_inv_res, fit_fwd_res = fit_results

        for name, res in [
            ("embedded_helmert2d_etrs89_to_st70_metrics.json", emb_inv_res),
            ("embedded_helmert2d_st70_to_etrs89_metrics.json", emb_fwd_res),
            ("fitted_helmert2d_etrs89_to_st70_metrics.json", fit_inv_res),
            ("fitted_helmert2d_st70_to_etrs89_metrics.json", fit_fwd_res),
        ]:
            (self.results_dir / name).write_text(
                json.dumps(res.summary, indent=2), encoding="utf-8"
            )

        # 3. Out-of-sample interior-only check
        interior_mask = ~df["is_node"].to_numpy()
        fit_inv_int_sum = summarize_errors(fit_inv_res.err_df.loc[interior_mask])
        fit_fwd_int_sum = summarize_errors(fit_fwd_res.err_df.loc[interior_mask])

        # 4. Round-trip verification
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
                    rt_n2 - df["st70_n"].to_numpy(), rt_e2 - df["st70_e"].to_numpy()
                )
            )
        )
        rt_geo_max_rad = float(
            np.max(
                np.hypot(
                    rt_lat - df["etrs89_lat_rad"].to_numpy(),
                    rt_lon - df["etrs89_lon_rad"].to_numpy(),
                )
            )
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
                "etrs89_to_st70": _compact_metrics(emb_inv_res),
                "st70_to_etrs89": _compact_metrics(emb_fwd_res),
            },
            "fitted_helmert2d": {
                "etrs89_to_st70": _compact_metrics(fit_inv_res),
                "st70_to_etrs89": _compact_metrics(fit_fwd_res),
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
                "etrs89_to_st70": fit_inv_res.extra.get("plots", {}),
                "st70_to_etrs89": fit_fwd_res.extra.get("plots", {}),
            },
            "runtime_s": round(elapsed, 3),
        }
        self.save_brief(brief)
        print(f"=== Research Unit 03 [{self.unit_name}] Complete ===")
        return brief


def main() -> None:
    unit = Helmert2DBaselineUnit(Path("experiments/helmert2d_baseline"))
    unit.run()


if __name__ == "__main__":
    main()
