"""Unified 4-Stage Surface Family Screening & Champion Promotion Pipeline.

Stage 0: Per-family subgrid calibration gate (3 anchors on Full RO vs. 3 subgrids).
Stage 1: Per-family parameter screening on its Stage-0-verified subgrid (no .pkl clutter).
Stage 2: Top-3 promotion per swept family (+ direct-full baselines) evaluated on Full Romania.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.data import CellRegion, GridDataset, GridId, GridManager, concat_datasets
from src.evaluation import ModelEvaluator
from src.models import (
    BaseGridShiftModel,
    BSplineSurfaceRegressor,
    CellBilinearShiftModel,
    LocalPolynomialSurfaceRegressor,
    ParameterizedBicubicGridShiftModel,
    PolynomialSurfaceRegressor,
    RBFSurfaceRegressor,
)


@dataclass(frozen=True)
class CandidateSpec:
    """Declarative specification for one surface regressor configuration."""

    family: str
    name: str
    factory: Callable[[], BaseGridShiftModel]
    params: dict[str, Any]
    train_grid: GridId | None = GridId.RO_GRID_4X4
    direct_full: bool = False


def build_family_catalog() -> dict[str, list[CandidateSpec]]:
    """Build the exhaustive 10-family candidate catalog."""
    catalog: dict[str, list[CandidateSpec]] = {
        "tps": [],
        "matern": [],
        "multiquadric": [],
        "gaussian": [],
        "polyharmonic": [],
        "bspline": [],
        "local_poly": [],
        "grid_bicubic_smooth": [],
        "global_poly": [],
        "bilinear_ref": [],
    }

    # 1. Thin Plate Spline (16 configs)
    for deg in (1, 2):
        for k in (16, 25):
            for s in (0.0, 1e-5, 1e-4, 5e-4):
                s_tag = "0" if s == 0.0 else f"{s:.0e}".replace("e-0", "e-")
                p = {"kernel": "thin_plate_spline", "degree": deg, "neighbors": k, "smoothing": s}
                catalog["tps"].append(
                    CandidateSpec(
                        family="tps",
                        name=f"tps_deg{deg}_k{k}_s{s_tag}",
                        factory=lambda p=p: RBFSurfaceRegressor(normalize_coords=True, **p),
                        params=p,
                    )
                )

    # 2. Matérn C1/C2 (48 configs)
    for kernel in ("matern32", "matern52"):
        for k in (16, 25):
            for eps in (2.0, 5.0, 10.0, 25.0):
                for s in (0.0, 1e-5, 1e-4):
                    s_tag = "0" if s == 0.0 else f"{s:.0e}".replace("e-0", "e-")
                    p = {"kernel": kernel, "neighbors": k, "epsilon": eps, "smoothing": s}
                    catalog["matern"].append(
                        CandidateSpec(
                            family="matern",
                            name=f"{kernel}_k{k}_eps{int(eps)}_s{s_tag}",
                            factory=lambda p=p: RBFSurfaceRegressor(normalize_coords=True, **p),
                            params=p,
                        )
                    )

    # 3. Multiquadric & Inverse Multiquadric (48 configs)
    for kernel in ("multiquadric", "inverse_multiquadric"):
        k_tag = "mq" if kernel == "multiquadric" else "imq"
        for deg in (1, 2):
            for k in (16, 25):
                for eps in (5.0, 10.0, 25.0):
                    for s in (1e-5, 1e-4):
                        s_tag = f"{s:.0e}".replace("e-0", "e-")
                        p = {
                            "kernel": kernel,
                            "degree": deg,
                            "neighbors": k,
                            "epsilon": eps,
                            "smoothing": s,
                        }
                        catalog["multiquadric"].append(
                            CandidateSpec(
                                family="multiquadric",
                                name=f"{k_tag}_deg{deg}_k{k}_eps{int(eps)}_s{s_tag}",
                                factory=lambda p=p: RBFSurfaceRegressor(normalize_coords=True, **p),
                                params=p,
                            )
                        )

    # 4. Gaussian RBF (24 configs)
    for deg in (1, 2):
        for k in (16, 25):
            for eps in (10.0, 25.0, 40.0):
                for s in (1e-5, 1e-4):
                    s_tag = f"{s:.0e}".replace("e-0", "e-")
                    p = {
                        "kernel": "gaussian",
                        "degree": deg,
                        "neighbors": k,
                        "epsilon": eps,
                        "smoothing": s,
                    }
                    catalog["gaussian"].append(
                        CandidateSpec(
                            family="gaussian",
                            name=f"gauss_deg{deg}_k{k}_eps{int(eps)}_s{s_tag}",
                            factory=lambda p=p: RBFSurfaceRegressor(normalize_coords=True, **p),
                            params=p,
                        )
                    )

    # 5. Polyharmonic RBFs: linear, cubic, quintic (18 configs)
    for kernel in ("linear", "cubic", "quintic"):
        for k in (16, 25):
            for s in (0.0, 1e-5, 1e-4):
                s_tag = "0" if s == 0.0 else f"{s:.0e}".replace("e-0", "e-")
                p = {"kernel": kernel, "degree": 1, "neighbors": k, "smoothing": s}
                catalog["polyharmonic"].append(
                    CandidateSpec(
                        family="polyharmonic",
                        name=f"{kernel}_k{k}_s{s_tag}",
                        factory=lambda p=p: RBFSurfaceRegressor(normalize_coords=True, **p),
                        params=p,
                    )
                )

    # 6. Tensor-Product B-Spline (10 configs)
    for deg in (3, 5):
        for s in (1e-4, 1e-3, 1e-2, 5e-2, 1e-1):
            s_tag = str(s).replace(".", "p")
            p = {"degree": deg, "smoothing": s}
            catalog["bspline"].append(
                CandidateSpec(
                    family="bspline",
                    name=f"bspline_deg{deg}_s{s_tag}",
                    factory=lambda p=p: BSplineSurfaceRegressor(normalize_coords=True, **p),
                    params=p,
                )
            )

    # 7. Local Polynomial / 2D LOESS (12 configs)
    for deg in (2, 3):
        for k in (16, 25, 36):
            for w in ("tricube", "gaussian"):
                p = {"degree": deg, "neighbors": k, "weight": w}
                catalog["local_poly"].append(
                    CandidateSpec(
                        family="local_poly",
                        name=f"lpoly_deg{deg}_k{k}_{w[:4]}",
                        factory=lambda p=p: LocalPolynomialSurfaceRegressor(**p),
                        params=p,
                    )
                )

    # 8. 11 km Smooth Grid Splines (4 configs, Direct Full)
    for scheme in ("natural_spline_c2", "catmull_rom", "pchip", "akima"):
        p = {"scheme": scheme}
        catalog["grid_bicubic_smooth"].append(
            CandidateSpec(
                family="grid_bicubic_smooth",
                name=f"grid11km_{scheme}",
                factory=lambda p=p: ParameterizedBicubicGridShiftModel(**p),
                params=p,
                train_grid=None,
                direct_full=True,
            )
        )

    # 9. Global Polynomial Surface (3 configs, Direct Full)
    for deg in (5, 8, 12):
        p = {"degree": deg, "alpha": 1e-8}
        catalog["global_poly"].append(
            CandidateSpec(
                family="global_poly",
                name=f"global_poly_deg{deg}",
                factory=lambda p=p: PolynomialSurfaceRegressor(**p),
                params=p,
                train_grid=GridId.RO_GRID_4X4,
                direct_full=True,
            )
        )

    # 10. Bilinear Cell Base Reference (1 config, Direct Full)
    catalog["bilinear_ref"].append(
        CandidateSpec(
            family="bilinear_ref",
            name="cell_bilinear_11km",
            factory=CellBilinearShiftModel,
            params={},
            train_grid=None,
            direct_full=True,
        )
    )

    return catalog


def _eval_pair(
    spec: CandidateSpec,
    train_coords: np.ndarray,
    train_shifts: np.ndarray,
    ds_u16: GridDataset,
    ds_a1: GridDataset,
    evaluator: ModelEvaluator,
) -> dict[str, Any]:
    """Fit on train_coords/shifts (if trainable) and evaluate Learnability on (u16, a1)."""
    t0 = time.perf_counter()
    try:
        model = spec.factory()
        model.fit(train_coords, train_shifts)
        fit_s = time.perf_counter() - t0
        r_u16 = evaluator.evaluate_shift_model(model, ds_u16, model_name=spec.name)
        r_a1 = evaluator.evaluate_shift_model(model, ds_a1, model_name=spec.name)
        rank_df = evaluator.rank_surface_candidates([r_u16, r_a1])
        row = rank_df.iloc[0]
        return {
            "family": spec.family,
            "name": spec.name,
            "status": "ok",
            "fit_s": round(fit_s, 4),
            "total_s": round(time.perf_counter() - t0, 3),
            "u16_crmse_mm": float(row[f"{GridId.RO_TEST_UNIFORM_16X16.value} cRMSE (mm)"]),
            "u16_rough_rms": float(row[f"{GridId.RO_TEST_UNIFORM_16X16.value} RoughRMS (mm/km)"]),
            "u16_score_mm": float(row[f"{GridId.RO_TEST_UNIFORM_16X16.value} Score (mm)"]),
            "a1_crmse_mm": float(row[f"{GridId.RO_TEST_ADAPTIVE_1MM_MID.value} cRMSE (mm)"]),
            "a1_rough_rms": float(row[f"{GridId.RO_TEST_ADAPTIVE_1MM_MID.value} RoughRMS (mm/km)"]),
            "a1_score_mm": float(row[f"{GridId.RO_TEST_ADAPTIVE_1MM_MID.value} Score (mm)"]),
            "overall_score_mm": float(row["Overall Learnability (mm)"]),
            "params_json": json.dumps(spec.params),
        }
    except (np.linalg.LinAlgError, ValueError, RuntimeError) as exc:
        return {
            "family": spec.family,
            "name": spec.name,
            "status": f"failed: {type(exc).__name__}: {exc}",
            "fit_s": float("nan"),
            "total_s": round(time.perf_counter() - t0, 3),
            "u16_crmse_mm": float("nan"),
            "u16_rough_rms": float("nan"),
            "u16_score_mm": float("nan"),
            "a1_crmse_mm": float("nan"),
            "a1_rough_rms": float("nan"),
            "a1_score_mm": float("nan"),
            "overall_score_mm": float("inf"),
            "params_json": json.dumps(spec.params),
        }


def main() -> None:
    results_dir = Path("experiments/optimal_surface_regressor/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    gm = GridManager()
    evaluator = ModelEvaluator()
    catalog = build_family_catalog()

    ds_train = gm.get(GridId.RO_GRID_4X4)
    train_coords, train_shifts = ds_train.coords, ds_train.true_shifts
    ds_u16_full = gm.get(GridId.RO_TEST_UNIFORM_16X16)
    ds_a1_full = gm.get(GridId.RO_TEST_ADAPTIVE_1MM_MID)

    # Define 3 candidate subgrids for Stage 0 calibration
    box10_int = CellRegion(row_start=19, col_start=28, n_rows=10, n_cols=10, name="box10_interior")
    box10_banat = CellRegion(row_start=10, col_start=14, n_rows=10, n_cols=10, name="box10_banat")
    box20_west = CellRegion(row_start=3, col_start=15, n_rows=20, n_cols=20, name="box20_west")

    subgrids: dict[str, tuple[GridDataset, GridDataset]] = {
        "box10_interior": (
            gm.get_region(GridId.RO_TEST_UNIFORM_16X16, box10_int),
            gm.get_region(GridId.RO_TEST_ADAPTIVE_1MM_MID, box10_int),
        ),
        "box10_dual": (
            concat_datasets([
                gm.get_region(GridId.RO_TEST_UNIFORM_16X16, box10_int),
                gm.get_region(GridId.RO_TEST_UNIFORM_16X16, box10_banat),
            ]),
            concat_datasets([
                gm.get_region(GridId.RO_TEST_ADAPTIVE_1MM_MID, box10_int),
                gm.get_region(GridId.RO_TEST_ADAPTIVE_1MM_MID, box10_banat),
            ]),
        ),
        "box20_west": (
            gm.get_region(GridId.RO_TEST_UNIFORM_16X16, box20_west),
            gm.get_region(GridId.RO_TEST_ADAPTIVE_1MM_MID, box20_west),
        ),
    }

    # --- STAGE 0 & STAGE 1: Per-Family Calibration & Screening ---
    calibration_report: dict[str, Any] = {}
    all_screening_rows: list[dict[str, Any]] = []
    shortlist_specs: list[CandidateSpec] = []

    for family, specs in catalog.items():
        if specs[0].direct_full:
            print(f"\n[Direct Full] Family '{family}' ({len(specs)} configs) -> skipping Stage 0/1.")
            shortlist_specs.extend(specs)
            continue

        print(f"\n=== Stage 0 Calibration: '{family}' ({len(specs)} configs) ===")
        anchor_idxs = [0, len(specs) // 2, len(specs) - 1]
        anchors = [specs[i] for i in anchor_idxs]

        # Evaluate 3 anchors on Full Romania
        full_anchor_scores: dict[str, float] = {}
        for a in anchors:
            res = _eval_pair(a, train_coords, train_shifts, ds_u16_full, ds_a1_full, evaluator)
            full_anchor_scores[a.name] = res["overall_score_mm"]
        full_valid = [a.name for a in anchors if np.isfinite(full_anchor_scores[a.name])]
        full_best = min(full_valid, key=lambda n: full_anchor_scores[n]) if full_valid else anchors[0].name

        best_subgrid = "box10_interior"
        best_metric = (-2.0, -1)  # (spearman_rho, top1_match)
        subgrid_details: dict[str, Any] = {}

        for sg_name, (sg_u16, sg_a1) in subgrids.items():
            sg_scores: dict[str, float] = {}
            for a in anchors:
                res = _eval_pair(a, train_coords, train_shifts, sg_u16, sg_a1, evaluator)
                sg_scores[a.name] = res["overall_score_mm"]
            if len(full_valid) >= 2 and all(np.isfinite(sg_scores[n]) for n in full_valid):
                rho = float(
                    spearmanr(
                        [full_anchor_scores[n] for n in full_valid],
                        [sg_scores[n] for n in full_valid],
                    ).statistic
                )
                if np.isnan(rho):
                    rho = 0.0
                sg_best = min(full_valid, key=lambda n: sg_scores[n])
                top1_match = int(sg_best == full_best)
            else:
                rho = 0.0
                sg_best = anchors[0].name
                top1_match = 0

            subgrid_details[sg_name] = {
                "rho": round(rho, 4),
                "top1_match": bool(top1_match),
                "sg_best": sg_best,
                "scores": sg_scores,
            }
            # Prefer fastest subgrid if tied; box10_interior is checked first
            score_key = (top1_match, round(rho, 4))
            if score_key > best_metric:
                best_metric = score_key
                best_subgrid = sg_name

        calibration_report[family] = {
            "chosen_subgrid": best_subgrid,
            "full_best_anchor": full_best,
            "full_anchor_scores": full_anchor_scores,
            "candidates": subgrid_details,
        }
        print(
            f"  -> Selected '{best_subgrid}' (top1_match={bool(best_metric[0])}, rho={best_metric[1]:.3f})"
        )

        # Stage 1: Screen all configs in family on chosen subgrid
        sg_u16, sg_a1 = subgrids[best_subgrid]
        fam_rows: list[dict[str, Any]] = []
        for idx, spec in enumerate(specs, start=1):
            row = _eval_pair(spec, train_coords, train_shifts, sg_u16, sg_a1, evaluator)
            row["subgrid"] = best_subgrid
            fam_rows.append(row)
            all_screening_rows.append(row)
            if idx % 12 == 0 or idx == len(specs):
                print(f"  Stage 1 [{family}] {idx}/{len(specs)} screened...")

        fam_df = pd.DataFrame(fam_rows).sort_values("overall_score_mm", ascending=True)
        fam_df.to_csv(results_dir / f"screening_{family}.csv", index=False)

        ok_names = list(fam_df[fam_df["status"] == "ok"]["name"].head(3))
        spec_map = {s.name: s for s in specs}
        for top_name in ok_names:
            shortlist_specs.append(spec_map[top_name])
        print(f"  -> Promoted Top 3: {ok_names}")

    (results_dir / "stage0_subgrid_calibration.json").write_text(
        json.dumps(calibration_report, indent=2), encoding="utf-8"
    )
    pd.DataFrame(all_screening_rows).to_csv(
        results_dir / "screening_all_families.csv", index=False
    )

    # --- STAGE 2: Full-Romania Evaluation of Promoted Top-3 + Direct Full ---
    print(f"\n=== Stage 2: Full-Romania Evaluation ({len(shortlist_specs)} shortlisted models) ===")
    stage2_rows: list[dict[str, Any]] = []
    for idx, spec in enumerate(shortlist_specs, start=1):
        row = _eval_pair(spec, train_coords, train_shifts, ds_u16_full, ds_a1_full, evaluator)
        stage2_rows.append(row)
        print(
            f"  [{idx}/{len(shortlist_specs)}] {spec.name:30s} | Overall={row['overall_score_mm']:7.3f} mm "
            f"| 16x16={row['u16_score_mm']:6.3f} | Adapt1mm={row['a1_score_mm']:6.3f} ({row['total_s']:.1f}s)"
        )

    s2_df = pd.DataFrame(stage2_rows).sort_values("overall_score_mm", ascending=True)
    s2_df.to_csv(results_dir / "stage2_full_shortlist.csv", index=False)

    # Pick 1 Champion per family
    champions: dict[str, dict[str, Any]] = {}
    for family in catalog:
        fam_s2 = s2_df[(s2_df["family"] == family) & (s2_df["status"] == "ok")]
        if not fam_s2.empty:
            best_row = fam_s2.iloc[0].to_dict()
            champions[family] = best_row

    (results_dir / "stage2_family_champions.json").write_text(
        json.dumps(champions, indent=2), encoding="utf-8"
    )
    print("\n=== Stage 2 Family Champions ===")
    for fam, info in champions.items():
        print(f"  {fam:20s}: {info['name']:30s} -> {info['overall_score_mm']:.3f} mm")


if __name__ == "__main__":
    main()
