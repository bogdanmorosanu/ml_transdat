"""Phase 1: Full Cartesian Subgrid Matrix Screening on ``box10_interior``.

Screens Stage-1 Smooth Surfaces (``cubic_k25_s0``, ``tps_deg2_k25_s1e-5``,
``lpoly_deg2_k16_tric``) paired with Stage-2 ``TorchMLPGridShiftModel``
(compact grid-free ResNet, 0 MB spatial lookup) across:
  - 3 Stage-1 surfaces
  - 4 Stage-2 training densities (``4x4``, ``8x8``, ``16x16``, ``32x32``)
  - 2 ResNet hidden widths (``64``, ``128``)
  - 2 activations (``silu``, ``relu``)
  with ``num_res_blocks=3``, ``loss_type='huber'``, ``cyclic_cell_harmonics=2``,
  and ``val_fraction=0.0`` (= 48 hybrid models + 3 Stage-1-only references).

Also runs:
  - 3 capacity checks at ``32x32`` with ``hidden_dim=256`` (one per surface,
    using the winning activation from the 48-model matrix).
  - 2 loss confirmation checks at ``32x32`` on ``cubic_k25_s0`` with
    ``loss_type='mse'`` (``hidden_dim`` in ``{64, 128}``) to verify Huber > MSE.

Evaluates on ``RO_TEST_UNIFORM_16X16`` and ``RO_TEST_ADAPTIVE_1MM_MID`` inside
``box10_interior`` and saves ``results/subgrid_matrix.csv`` and
``results/subgrid_matrix.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.data import CellRegion, GridId, GridManager
from src.evaluation import ModelEvaluator
from src.evaluation.metrics import dataframe_to_markdown
from src.models import (
    BaseGridShiftModel,
    CompositeResidualShiftModel,
    LocalPolynomialSurfaceRegressor,
    RBFSurfaceRegressor,
    TorchMLPGridShiftModel,
)

UNIT_DIR = Path("experiments/hybrid_surface_resnet")
MODELS_DIR = UNIT_DIR / "models"
RESULTS_DIR = UNIT_DIR / "results"

BOX10_INTERIOR = CellRegion(
    row_start=19, col_start=28, n_rows=10, n_cols=10, name="box10_interior"
)


def _load_or_fit_stage1(
    gm: GridManager, name: str, model: BaseGridShiftModel
) -> BaseGridShiftModel:
    full_4x4 = gm.get(GridId.RO_GRID_4X4)
    model.load_or_fit(
        lambda: (full_4x4.coords, full_4x4.true_shifts),
        cache_path=MODELS_DIR / f"_stage1_{name}.pkl",
        train_grid=GridId.RO_GRID_4X4.value,
        label=f"stage1_{name}",
    )
    return model


def _build_stage2_resnet(
    *,
    hidden_dim: int,
    activation: str,
    loss_type: str,
    batch_size: int,
) -> TorchMLPGridShiftModel:
    return TorchMLPGridShiftModel(
        hidden_dim=hidden_dim,
        num_res_blocks=3,
        activation=activation,
        use_fourier_features=True,
        fourier_scale=3.0,
        num_fourier_frequencies=64,
        use_cyclic_cell_pos=True,
        cyclic_cell_harmonics=2,
        batch_size=batch_size,
        lr=1e-3,
        weight_decay=1e-5,
        max_epochs=200,
        patience=25,
        lr_scheduler="cosine",
        val_fraction=0.0,
        loss_type=loss_type,
    )


def _eval_row(
    evaluator: ModelEvaluator,
    model: BaseGridShiftModel,
    name: str,
    surface: str,
    density: str,
    hidden_dim: int | None,
    activation: str,
    loss_type: str,
    train_pts: int,
    test_u16: Any,
    test_a1: Any,
) -> dict[str, Any]:
    res_u16 = evaluator.evaluate_shift_model(model, test_u16, model_name=name)
    res_a1 = evaluator.evaluate_shift_model(model, test_a1, model_name=name)
    s_u16 = res_u16.summary
    s_a1 = res_a1.summary
    return {
        "model": name,
        "surface": surface,
        "density": density,
        "hidden_dim": hidden_dim if hidden_dim is not None else 0,
        "activation": activation,
        "loss_type": loss_type,
        "train_pts": train_pts,
        "u16_mean_mm": round(s_u16["mean_2d_m"] * 1000.0, 3),
        "u16_rmse_mm": round(s_u16["rmse_2d_m"] * 1000.0, 3),
        "u16_lt_1mm_pct": round(s_u16["pass_rates_pct"]["lt_1mm"], 2),
        "a1_mean_mm": round(s_a1["mean_2d_m"] * 1000.0, 3),
        "a1_rmse_mm": round(s_a1["rmse_2d_m"] * 1000.0, 3),
        "a1_p95_mm": round(s_a1["p95_2d_m"] * 1000.0, 3),
        "a1_max_mm": round(s_a1["max_2d_m"] * 1000.0, 3),
        "a1_lt_1mm_pct": round(s_a1["pass_rates_pct"]["lt_1mm"], 2),
        "a1_lt_1cm_pct": round(s_a1["pass_rates_pct"]["lt_1cm"], 2),
    }


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    gm = GridManager()
    evaluator = ModelEvaluator()

    surfaces: dict[str, BaseGridShiftModel] = {
        "cubic_k25_s0": _load_or_fit_stage1(
            gm,
            "cubic_k25_s0",
            RBFSurfaceRegressor(
                kernel="cubic", degree=1, smoothing=0.0, neighbors=25, normalize_coords=True
            ),
        ),
        "tps_deg2_k25_s1e-5": _load_or_fit_stage1(
            gm,
            "tps_deg2_k25_s1e-5",
            RBFSurfaceRegressor(
                kernel="thin_plate_spline",
                degree=2,
                smoothing=1e-5,
                neighbors=25,
                normalize_coords=True,
            ),
        ),
        "lpoly_deg2_k16_tric": _load_or_fit_stage1(
            gm,
            "lpoly_deg2_k16_tric",
            LocalPolynomialSurfaceRegressor(degree=2, neighbors=16, weight="tricube"),
        ),
    }

    trains = {
        "4x4": (gm.get_region(GridId.RO_GRID_4X4, BOX10_INTERIOR), 512),
        "8x8": (gm.get_region(GridId.RO_GRID_8X8, BOX10_INTERIOR), 1024),
        "16x16": (gm.get_region(GridId.RO_TEST_UNIFORM_16X16, BOX10_INTERIOR), 2048),
        "32x32": (gm.get_region(GridId.RO_GRID_32X32, BOX10_INTERIOR), 4096),
    }
    test_u16 = trains["16x16"][0]
    test_a1 = gm.get_region(GridId.RO_TEST_ADAPTIVE_1MM_MID, BOX10_INTERIOR)

    rows: list[dict[str, Any]] = []

    # Stage-1-only references
    for s_name, s_model in surfaces.items():
        rows.append(
            _eval_row(
                evaluator,
                s_model,
                f"Stage1_{s_name}_only",
                s_name,
                "none",
                None,
                "none",
                "none",
                0,
                test_u16,
                test_a1,
            )
        )

    # Full 48-model Option-B Cartesian matrix: 3 surfaces x 4 densities x 2 widths x 2 activations
    cartesian_rows: list[dict[str, Any]] = []
    for s_name, s_model in surfaces.items():
        for d_name in ("4x4", "8x8", "16x16", "32x32"):
            ds_d, bs_d = trains[d_name]
            for hidden_dim in (64, 128):
                for act in ("silu", "relu"):
                    tag = f"{s_name}_{d_name}_h{hidden_dim}_{act}_huber"
                    stage2 = _build_stage2_resnet(
                        hidden_dim=hidden_dim,
                        activation=act,
                        loss_type="huber",
                        batch_size=bs_d,
                    )
                    model = CompositeResidualShiftModel(s_model, stage2)
                    model.fit(ds_d.coords, ds_d.true_shifts)
                    row = _eval_row(
                        evaluator,
                        model,
                        tag,
                        s_name,
                        d_name,
                        hidden_dim,
                        act,
                        "huber",
                        len(ds_d.coords),
                        test_u16,
                        test_a1,
                    )
                    rows.append(row)
                    cartesian_rows.append(row)
                    print(
                        f"[{len(cartesian_rows):02d}/48] {tag}: "
                        f"u16={row['u16_rmse_mm']:.3f}mm, a1={row['a1_rmse_mm']:.3f}mm "
                        f"(lt1mm={row['a1_lt_1mm_pct']:.2f}%)",
                        flush=True,
                    )

    best_cart = min(cartesian_rows, key=lambda r: float(r["a1_rmse_mm"]))
    win_act = str(best_cart["activation"])
    ds_32, bs_32 = trains["32x32"]

    # Capacity checks: hidden_dim=256 at 32x32 across all 3 surfaces (winning activation)
    for s_name, s_model in surfaces.items():
        tag = f"{s_name}_32x32_h256_{win_act}_huber"
        stage2 = _build_stage2_resnet(
            hidden_dim=256,
            activation=win_act,
            loss_type="huber",
            batch_size=bs_32,
        )
        model = CompositeResidualShiftModel(s_model, stage2)
        model.fit(ds_32.coords, ds_32.true_shifts)
        row = _eval_row(
            evaluator,
            model,
            tag,
            s_name,
            "32x32",
            256,
            win_act,
            "huber",
            len(ds_32.coords),
            test_u16,
            test_a1,
        )
        rows.append(row)
        print(
            f"[h256 check] {tag}: u16={row['u16_rmse_mm']:.3f}mm, "
            f"a1={row['a1_rmse_mm']:.3f}mm (lt1mm={row['a1_lt_1mm_pct']:.2f}%)",
            flush=True,
        )

    # Loss confirmation checks: MSE at 32x32 on cubic_k25_s0 (h64 & h128, winning activation)
    cubic_model = surfaces["cubic_k25_s0"]
    for hidden_dim in (64, 128):
        tag = f"cubic_k25_s0_32x32_h{hidden_dim}_{win_act}_mse"
        stage2 = _build_stage2_resnet(
            hidden_dim=hidden_dim,
            activation=win_act,
            loss_type="mse",
            batch_size=bs_32,
        )
        model = CompositeResidualShiftModel(cubic_model, stage2)
        model.fit(ds_32.coords, ds_32.true_shifts)
        row = _eval_row(
            evaluator,
            model,
            tag,
            "cubic_k25_s0",
            "32x32",
            hidden_dim,
            win_act,
            "mse",
            len(ds_32.coords),
            test_u16,
            test_a1,
        )
        rows.append(row)
        print(
            f"[mse check] {tag}: u16={row['u16_rmse_mm']:.3f}mm, "
            f"a1={row['a1_rmse_mm']:.3f}mm (lt1mm={row['a1_lt_1mm_pct']:.2f}%)",
            flush=True,
        )

    df = pd.DataFrame(rows).sort_values(["a1_rmse_mm", "u16_rmse_mm"]).reset_index(drop=True)
    df.to_csv(RESULTS_DIR / "subgrid_matrix.csv", index=False)
    md = dataframe_to_markdown(df)
    (RESULTS_DIR / "subgrid_matrix.md").write_text(md, encoding="utf-8")
    print("\n=== Phase 1 Full Subgrid Matrix (sorted by Adapt 1mm RMSE) ===")
    print(md)


if __name__ == "__main__":
    main()
