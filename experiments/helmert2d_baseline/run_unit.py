"""Unit 03: 2D 4-Parameter Helmert Baseline (without SPG grid) vs pytransdat."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from pytransdat.proj_stereo import StereoProj
from src.core.base_unit import TransformationExperimentUnit, TransformationModelSpec
from src.data import GridId
from src.evaluation import EvaluationResult, summarize_errors
from src.models import (
    BaseCoordinateTransformer,
    Helmert2DParams,
    TwoStepHelmert2DTransformer,
    fit_helmert2d,
    load_embedded_spg_helmert_params,
)
from src.viz import PlotGroup


class Helmert2DBaselineUnit(TransformationExperimentUnit):
    """Transformation Unit: Refitted 4-parameter 2D Helmert baseline (no SPG grid)."""

    @property
    def slug(self) -> str:
        return "helmert2d_baseline"

    def _fit_and_save_params(self) -> tuple[Helmert2DParams, Helmert2DParams, dict[str, Any]]:
        dataset = self.load_grid(GridId.RO_GRID_4X4)
        df = dataset.df
        df_nodes = df[df["is_node"]].reset_index(drop=True)

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
        embedded_params = load_embedded_spg_helmert_params()

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
        return embedded_params, fitted_params, param_payload

    def get_transformer_specs(self) -> list[TransformationModelSpec]:
        embedded_params, fitted_params, param_payload = self._fit_and_save_params()
        self._param_payload = param_payload
        return [
            TransformationModelSpec(
                name="embedded_helmert2d",
                transformer=TwoStepHelmert2DTransformer(embedded_params),
                eval_grid=GridId.RO_GRID_4X4,
                plots=PlotGroup.STANDARD_DIAGNOSTIC,
            ),
            TransformationModelSpec(
                name="fitted_helmert2d",
                transformer=TwoStepHelmert2DTransformer(fitted_params),
                eval_grid=GridId.RO_GRID_4X4,
                plots=PlotGroup.STANDARD_DIAGNOSTIC,
            ),
        ]

    def after_run(
        self,
        transformers: dict[str, BaseCoordinateTransformer],
        results: list[EvaluationResult],
    ) -> dict[str, Any]:
        for res in results:
            (self.results_dir / f"{res.model_name}_{res.direction}_metrics.json").write_text(
                json.dumps(res.summary, indent=2), encoding="utf-8"
            )

        dataset = self.load_grid(GridId.RO_GRID_4X4)
        df = dataset.df
        interior_mask = ~df["is_node"].to_numpy()
        fit_results = [r for r in results if r.model_name == "fitted_helmert2d"]
        fit_inv_res, fit_fwd_res = fit_results[0], fit_results[1]
        fit_inv_int_sum = summarize_errors(fit_inv_res.err_df.loc[interior_mask])
        fit_fwd_int_sum = summarize_errors(fit_fwd_res.err_df.loc[interior_mask])

        fit_transformer = transformers["fitted_helmert2d"]
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

        return {
            "fit_node_count": int(df["is_node"].sum()),
            "sample_count": len(df),
            "interior_count": int(interior_mask.sum()),
            "parameters": self._param_payload,
            "roundtrip": {
                "st70_to_etrs89_to_st70_max_m": rt_st70_max_m,
                "etrs89_to_st70_to_etrs89_max_rad": rt_geo_max_rad,
            },
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
        }


def main() -> None:
    unit = Helmert2DBaselineUnit(Path("experiments/helmert2d_baseline"))
    unit.run()


if __name__ == "__main__":
    main()
