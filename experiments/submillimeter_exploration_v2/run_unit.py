"""Sub-Millimeter Exploration v2: Exhaustive 12-Track Screening on ``box10_interior``.

Benchmarks all 12 untested candidate model tracks (Wave 1: modern PyTorch INRs &
structured hybrids; Wave 2: zero-PyTorch ML, geostatistics, patch-RBF, and 2D wavelets)
against 4 reference anchors on ``box10_interior`` (10x10 interior SPG cells).

Reports accuracy (Uniform 16x16, Adaptive 1mm, and 11 km cell-wall kink RMSE),
inference latency/throughput, zero-PyTorch status, and projected Full-Romania memory.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import CellRegion, GridId, GridManager
from src.evaluation import ModelEvaluator
from src.evaluation.metrics import dataframe_to_markdown
from src.geometry.cell_uv import compute_spg_cell_uv
from src.models import (
    AdaptiveQuadtreeRegressor,
    BaseGridShiftModel,
    CellBilinearShiftModel,
    CompositeResidualShiftModel,
    KANGridShiftModel,
    KPlanesGridShiftModel,
    LinearLeafTreeShiftModel,
    LocalKrigingShiftModel,
    LocalPolynomialSurfaceRegressor,
    LSSVMGridShiftModel,
    MFNGridShiftModel,
    NeuralBicubicHeadShiftModel,
    NeuralFeatureGridShiftModel,
    ParameterizedBicubicGridShiftModel,
    PartitionOfUnityRBFShiftModel,
    RBFSurfaceRegressor,
    RFFELMGridShiftModel,
    SpatialMoEGridShiftModel,
    TorchMLPGridShiftModel,
    WaveletGridShiftModel,
    WaveletINRGridShiftModel,
    XGBoostGridShiftModel,
)

UNIT_DIR = Path("experiments/submillimeter_exploration_v2")
MODELS_DIR = UNIT_DIR / "models"
RESULTS_DIR = UNIT_DIR / "results"
PLOTS_DIR = UNIT_DIR / "plots"

BOX10_INTERIOR = CellRegion(
    row_start=19, col_start=28, n_rows=10, n_cols=10, name="box10_interior"
)
FULL_RO_SUBGRID_SCALE = 23.0


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


def _wall_mask(coords: np.ndarray, tol: float = 0.05) -> np.ndarray:
    u, v = compute_spg_cell_uv(coords)
    return (u < tol) | (u > 1.0 - tol) | (v < tol) | (v > 1.0 - tol)


def _evaluate_candidate(
    evaluator: ModelEvaluator,
    model: BaseGridShiftModel,
    name: str,
    track: str,
    family: str,
    train_coords: np.ndarray | None,
    train_shifts: np.ndarray | None,
    test_u16: Any,
    test_a1: Any,
    wall_idx: np.ndarray,
    scale_instance_for_ro: bool = False,
    scale_lookup_for_ro: bool = False,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    if train_coords is not None and train_shifts is not None:
        model.fit(train_coords, train_shifts)
    fit_time_s = time.perf_counter() - t0

    res_u16 = evaluator.evaluate_shift_model(model, test_u16, model_name=name)
    inf_time_s = max(res_u16.eval_time_sec, 1e-9)
    n_u16 = len(test_u16.coords)
    latency_us_pt = (inf_time_s * 1e6) / n_u16
    throughput_kpts_s = res_u16.throughput_pts_per_sec / 1000.0

    res_a1 = evaluator.evaluate_shift_model(model, test_a1, model_name=name)
    err_wall = np.hypot(
        res_a1.err_df["err_n_m"].to_numpy()[wall_idx],
        res_a1.err_df["err_e_m"].to_numpy()[wall_idx],
    )
    wall_rmse_mm = float(np.sqrt(np.mean(err_wall**2)) * 1000.0)

    mb = model.memory_breakdown()
    zero_pytorch = "torch" not in model._runtime_dependencies()
    weights_kb = mb.model_instance_bytes / 1024.0
    lookup_kb = mb.grid_lookup_bytes / 1024.0

    est_inst_bytes = mb.model_instance_bytes * (
        FULL_RO_SUBGRID_SCALE if scale_instance_for_ro else 1.0
    )
    est_look_bytes = mb.grid_lookup_bytes * (
        FULL_RO_SUBGRID_SCALE if scale_lookup_for_ro else 1.0
    )
    est_full_ro_mb = (mb.code_bytes + est_inst_bytes + est_look_bytes) / (1024.0 * 1024.0)

    s_u16 = res_u16.summary
    s_a1 = res_a1.summary
    return {
        "model": name,
        "track": track,
        "family": family,
        "zero_pytorch": "Yes" if zero_pytorch else "No",
        "u16_mean_mm": round(s_u16["mean_2d_m"] * 1000.0, 3),
        "u16_rmse_mm": round(s_u16["rmse_2d_m"] * 1000.0, 3),
        "u16_lt_1mm_pct": round(s_u16["pass_rates_pct"]["lt_1mm"], 2),
        "a1_mean_mm": round(s_a1["mean_2d_m"] * 1000.0, 3),
        "a1_rmse_mm": round(s_a1["rmse_2d_m"] * 1000.0, 3),
        "a1_p95_mm": round(s_a1["p95_2d_m"] * 1000.0, 3),
        "a1_max_mm": round(s_a1["max_2d_m"] * 1000.0, 3),
        "a1_lt_1mm_pct": round(s_a1["pass_rates_pct"]["lt_1mm"], 2),
        "a1_wall_rmse_mm": round(wall_rmse_mm, 3),
        "fit_time_s": round(fit_time_s, 2),
        "latency_us_pt": round(latency_us_pt, 3),
        "throughput_kpts_s": round(throughput_kpts_s, 1),
        "weights_kb": round(weights_kb, 1),
        "lookup_kb": round(lookup_kb, 1),
        "est_full_ro_mb": round(est_full_ro_mb, 2),
    }


def _plot_pareto(df: pd.DataFrame) -> None:
    colors = df["zero_pytorch"].map({"Yes": "#1b9e77", "No": "#7570b3"})
    markers = df["track"].apply(lambda t: "*" if str(t).startswith("Anchor") else "o")

    # 1. Accuracy vs Latency
    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, row in df.iterrows():
        ax.scatter(
            row["latency_us_pt"],
            row["a1_rmse_mm"],
            c=colors.iloc[idx],
            marker=markers.iloc[idx],
            s=130 if markers.iloc[idx] == "*" else 80,
            edgecolors="black",
            linewidths=0.6,
            zorder=3,
        )
        ax.annotate(
            row["model"],
            (row["latency_us_pt"], row["a1_rmse_mm"]),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=7.5,
        )
    ax.axhline(1.0, color="#d95f02", linestyle="--", linewidth=1.2, label="1.0 mm Target")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Inference Latency (us / point, log scale)")
    ax.set_ylabel("Adaptive 1mm RMSE (mm, log scale)")
    ax.set_title("v2 Screening Pareto: Accuracy vs. Inference Latency (box10_interior)")
    ax.grid(True, which="both", alpha=0.25)
    # Legend handles
    ax.scatter([], [], c="#7570b3", label="PyTorch Model", s=80, edgecolors="black")
    ax.scatter([], [], c="#1b9e77", label="Zero-PyTorch Model", s=80, edgecolors="black")
    ax.legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "v2_accuracy_vs_latency_pareto.png", dpi=180)
    plt.close(fig)

    # 2. Accuracy vs Projected Full-Romania Memory
    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, row in df.iterrows():
        mem = max(float(row["est_full_ro_mb"]), 0.02)
        ax.scatter(
            mem,
            row["a1_rmse_mm"],
            c=colors.iloc[idx],
            marker=markers.iloc[idx],
            s=130 if markers.iloc[idx] == "*" else 80,
            edgecolors="black",
            linewidths=0.6,
            zorder=3,
        )
        ax.annotate(
            row["model"],
            (mem, row["a1_rmse_mm"]),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=7.5,
        )
    ax.axhline(1.0, color="#d95f02", linestyle="--", linewidth=1.2, label="1.0 mm Target")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Projected Full-Romania Footprint (MB, log scale)")
    ax.set_ylabel("Adaptive 1mm RMSE (mm, log scale)")
    ax.set_title("v2 Screening Pareto: Accuracy vs. Projected Full-Romania Size")
    ax.grid(True, which="both", alpha=0.25)
    ax.scatter([], [], c="#7570b3", label="PyTorch Model", s=80, edgecolors="black")
    ax.scatter([], [], c="#1b9e77", label="Zero-PyTorch Model", s=80, edgecolors="black")
    ax.legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "v2_accuracy_vs_memory_pareto.png", dpi=180)
    plt.close(fig)

    # 3. Horizontal Bar Comparison
    df_s = df.sort_values("a1_rmse_mm", ascending=True).reset_index(drop=True)
    y = np.arange(len(df_s))
    fig, ax = plt.subplots(figsize=(11, max(6.5, 0.36 * len(df_s))))
    ax.barh(y - 0.25, df_s["u16_mean_mm"], height=0.25, label="U16 Mean (mm)", color="#386cb0")
    ax.barh(y, df_s["a1_rmse_mm"], height=0.25, label="Adapt 1mm RMSE (mm)", color="#f0027f")
    ax.barh(
        y + 0.25,
        df_s["a1_wall_rmse_mm"],
        height=0.25,
        label="Wall Kink RMSE (mm)",
        color="#fdc086",
    )
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1.0, label="1.0 mm")
    ax.set_yticks(y)
    ax.set_yticklabels(df_s["model"], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("2D Error (mm, log scale)")
    ax.set_title("v2 Screening Accuracy & Wall-Kink Stress Comparison (box10_interior)")
    ax.grid(True, axis="x", which="both", alpha=0.25)
    ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "v2_family_comparison_bar.png", dpi=180)
    plt.close(fig)


def _write_report(df: pd.DataFrame, total_runtime_s: float) -> None:
    md_table = dataframe_to_markdown(df)
    top_pytorch = df[df["zero_pytorch"] == "No"].head(3)
    top_zero_pt = df[df["zero_pytorch"] == "Yes"].head(4)
    lines = [
        "# Experiment Report: `submillimeter_exploration_v2`",
        "",
        f"- **Status**: `Validated` (Screening Complete in `{total_runtime_s:.1f}s`)",
        "- **Evaluation Domain**: `box10_interior` (`10x10` interior SPG cells, `row_start=19, col_start=28`)",
        "- **Test Benchmarks**: `RO_TEST_UNIFORM_16X16` (`25,600` pts) and `RO_TEST_ADAPTIVE_1MM_MID`",
        "",
        "## 1. Consolidated Screening Scorecard (Sorted by `Adapt 1mm` RMSE)",
        "",
        md_table,
        "",
        "## 2. Top Performers by Deployment Tier",
        "",
        "### PyTorch Candidates (Top 3)",
        dataframe_to_markdown(top_pytorch),
        "",
        "### Zero-PyTorch Candidates (Top 4)",
        dataframe_to_markdown(top_zero_pt),
        "",
        "## 3. Diagnostic Pareto & Stress Plots",
        "- [Accuracy vs. Inference Latency](plots/v2_accuracy_vs_latency_pareto.png)",
        "- [Accuracy vs. Projected Full-Romania Memory](plots/v2_accuracy_vs_memory_pareto.png)",
        "- [Family Accuracy & Wall-Kink Stress Comparison](plots/v2_family_comparison_bar.png)",
    ]
    (UNIT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    t_start = time.perf_counter()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    gm = GridManager()
    evaluator = ModelEvaluator()

    # Stage-1 smooth surfaces fitted on Full RO_GRID_4X4
    lpoly = _load_or_fit_stage1(
        gm,
        "lpoly_deg2_k16_tric",
        LocalPolynomialSurfaceRegressor(degree=2, neighbors=16, weight="tricube"),
    )
    cubic = _load_or_fit_stage1(
        gm,
        "cubic_k25_s0",
        RBFSurfaceRegressor(
            kernel="cubic", degree=1, smoothing=0.0, neighbors=25, normalize_coords=True
        ),
    )
    bilinear = _load_or_fit_stage1(gm, "cell_bilinear", CellBilinearShiftModel())
    catmull = _load_or_fit_stage1(
        gm,
        "catmull_rom",
        ParameterizedBicubicGridShiftModel(scheme="catmull_rom"),
    )

    ds_32 = gm.get_region(GridId.RO_GRID_32X32, BOX10_INTERIOR)
    ds_16 = gm.get_region(GridId.RO_TEST_UNIFORM_16X16, BOX10_INTERIOR)
    ds_8 = gm.get_region(GridId.RO_GRID_8X8, BOX10_INTERIOR)
    test_u16 = ds_16
    test_a1 = gm.get_region(GridId.RO_TEST_ADAPTIVE_1MM_MID, BOX10_INTERIOR)
    wall_idx = _wall_mask(test_a1.coords, tol=0.05)

    dn_qt, de_qt = gm.get_adaptive_pair("1mm")
    quadtree_1mm = AdaptiveQuadtreeRegressor(dn_grid=dn_qt.df, de_grid=de_qt.df)

    # Specs: (name, track, family, model, train_ds, scale_inst_ro, scale_lookup_ro)
    specs: list[tuple[str, str, str, BaseGridShiftModel, Any | None, bool, bool]] = [
        # 4 Anchors
        ("A0_Stage1_lpoly_only", "Anchor", "Stage-1 Classical", lpoly, None, False, False),
        (
            "A1_Champion_FeatGrid_32x32",
            "Anchor",
            "Micro-Grid Hybrid",
            CompositeResidualShiftModel(
                cubic,
                NeuralFeatureGridShiftModel(
                    grid_resolutions=((53, 72), (105, 143)),
                    feature_dim=8,
                    hidden_dim=128,
                    cyclic_cell_harmonics=2,
                    batch_size=1024,
                    lr=3e-3,
                    max_epochs=180,
                    val_fraction=0.0,
                    loss_type="huber",
                ),
            ),
            ds_32,
            False,
            False,
        ),
        (
            "A2_Champion_ResNet_32x32",
            "Anchor",
            "Zero-Lookup ResNet",
            CompositeResidualShiftModel(
                lpoly,
                TorchMLPGridShiftModel(
                    hidden_dim=128,
                    num_res_blocks=3,
                    activation="silu",
                    use_fourier_features=True,
                    fourier_scales=(1.0, 3.0, 10.0),
                    num_fourier_frequencies=64,
                    use_cyclic_cell_pos=True,
                    cyclic_cell_harmonics=2,
                    batch_size=1024,
                    lr=1e-3,
                    max_epochs=150,
                    val_fraction=0.0,
                    loss_type="huber",
                ),
            ),
            ds_32,
            False,
            False,
        ),
        ("A3_Quadtree_1mm", "Anchor", "Geometric Lookup", quadtree_1mm, None, False, False),
        # Wave 1: Modern INRs & Structured Hybrids
        (
            "V2_1a_lpoly_rbf_kan",
            "V2-1",
            "Modern INR (KAN)",
            CompositeResidualShiftModel(
                lpoly, KANGridShiftModel(basis="rbf", hidden_dim=64, num_layers=3, num_basis=8)
            ),
            ds_32,
            False,
            False,
        ),
        (
            "V2_1b_lpoly_fourier_kan",
            "V2-1",
            "Modern INR (KAN)",
            CompositeResidualShiftModel(
                lpoly, KANGridShiftModel(basis="fourier", hidden_dim=64, num_layers=3, num_basis=6)
            ),
            ds_32,
            False,
            False,
        ),
        (
            "V2_2a_lpoly_wire",
            "V2-2",
            "Modern INR (Wavelet)",
            CompositeResidualShiftModel(
                lpoly, WaveletINRGridShiftModel(activation="wire", hidden_dim=96, num_layers=4)
            ),
            ds_32,
            False,
            False,
        ),
        (
            "V2_2b_lpoly_finer",
            "V2-2",
            "Modern INR (Periodic)",
            CompositeResidualShiftModel(
                lpoly, WaveletINRGridShiftModel(activation="finer", hidden_dim=128, num_layers=4)
            ),
            ds_32,
            False,
            False,
        ),
        (
            "V2_3a_lpoly_gabor_mfn",
            "V2-3",
            "Modern INR (MFN)",
            CompositeResidualShiftModel(
                lpoly, MFNGridShiftModel(filter_type="gabor", hidden_dim=128, num_layers=4)
            ),
            ds_32,
            False,
            False,
        ),
        (
            "V2_3b_lpoly_fourier_mfn",
            "V2-3",
            "Modern INR (MFN)",
            CompositeResidualShiftModel(
                lpoly, MFNGridShiftModel(filter_type="fourier", hidden_dim=128, num_layers=4)
            ),
            ds_32,
            False,
            False,
        ),
        (
            "V2_4a_lpoly_spatial_moe",
            "V2-4",
            "Modern INR (MoE)",
            CompositeResidualShiftModel(
                lpoly, SpatialMoEGridShiftModel(mode="moe", num_experts=8, hidden_dim=64)
            ),
            ds_32,
            False,
            False,
        ),
        (
            "V2_4b_lpoly_fbpinn",
            "V2-4",
            "Modern INR (FBPINN)",
            CompositeResidualShiftModel(
                lpoly, SpatialMoEGridShiftModel(mode="fbpinn", fbpinn_grid=(4, 4), hidden_dim=48)
            ),
            ds_32,
            False,
            False,
        ),
        (
            "V2_5a_neural_bicubic_grid",
            "V2-5",
            "Structured Hybrid",
            CompositeResidualShiftModel(
                bilinear, NeuralBicubicHeadShiftModel(backend="grid", max_epochs=180)
            ),
            ds_32,
            False,
            False,
        ),
        (
            "V2_5b_neural_bicubic_mlp",
            "V2-5",
            "Structured Hybrid",
            CompositeResidualShiftModel(
                lpoly, NeuralBicubicHeadShiftModel(backend="mlp", hidden_dim=128, max_epochs=150)
            ),
            ds_32,
            False,
            False,
        ),
        (
            "V2_6a_bilinear_featgrid_3lvl",
            "V2-6",
            "Micro-Grid Hybrid",
            CompositeResidualShiftModel(
                bilinear,
                NeuralFeatureGridShiftModel(
                    grid_resolutions=((53, 72), (105, 143), (209, 285)),
                    feature_dim=6,
                    hidden_dim=128,
                    cyclic_cell_harmonics=2,
                    batch_size=1024,
                    lr=3e-3,
                    max_epochs=180,
                    val_fraction=0.0,
                    loss_type="huber",
                ),
            ),
            ds_32,
            False,
            False,
        ),
        (
            "V2_6b_catmull_featgrid_3lvl",
            "V2-6",
            "Micro-Grid Hybrid",
            CompositeResidualShiftModel(
                catmull,
                NeuralFeatureGridShiftModel(
                    grid_resolutions=((53, 72), (105, 143), (209, 285)),
                    feature_dim=6,
                    hidden_dim=128,
                    cyclic_cell_harmonics=2,
                    batch_size=1024,
                    lr=3e-3,
                    max_epochs=180,
                    val_fraction=0.0,
                    loss_type="huber",
                ),
            ),
            ds_32,
            False,
            False,
        ),
        (
            "V2_7_lpoly_kplanes",
            "V2-7",
            "Structured Hybrid",
            CompositeResidualShiftModel(
                lpoly, KPlanesGridShiftModel(n_bins=128, feature_dim=16, hidden_dim=64)
            ),
            ds_32,
            False,
            False,
        ),
        # Wave 2: Zero-PyTorch ML, Geostatistics & Signal Processing
        (
            "V2_8_lpoly_rff_elm",
            "V2-8",
            "Zero-PT Closed-Form",
            CompositeResidualShiftModel(
                lpoly, RFFELMGridShiftModel(n_features=2048, alpha=1e-4)
            ),
            ds_32,
            False,
            False,
        ),
        (
            "V2_9_lpoly_lssvm_nystroem",
            "V2-9",
            "Zero-PT Kernel ML",
            CompositeResidualShiftModel(
                lpoly, LSSVMGridShiftModel(n_components=1500, gamma=2.0, alpha=1e-4)
            ),
            ds_32,
            False,
            False,
        ),
        (
            "V2_10a_lpoly_xgboost",
            "V2-10",
            "Zero-PT Tree Ensemble",
            CompositeResidualShiftModel(
                lpoly, XGBoostGridShiftModel(n_estimators=600, max_depth=8, learning_rate=0.08)
            ),
            ds_32,
            True,
            False,
        ),
        (
            "V2_10b_lpoly_linear_leaf_tree",
            "V2-10",
            "Zero-PT Model Tree",
            CompositeResidualShiftModel(
                lpoly, LinearLeafTreeShiftModel(max_depth=12, min_samples_leaf=25, leaf_degree=2)
            ),
            ds_32,
            True,
            False,
        ),
        (
            "V2_11_local_kriging_8x8",
            "V2-11",
            "Zero-PT Geostatistics",
            CompositeResidualShiftModel(
                lpoly,
                LocalKrigingShiftModel(
                    neighbors=25, lengthscale_n=2500.0, lengthscale_e=2500.0, nugget=1e-5
                ),
            ),
            ds_8,
            True,
            False,
        ),
        (
            "V2_12a_pu_rbf_8x8",
            "V2-12",
            "Zero-PT Patch-RBF",
            CompositeResidualShiftModel(
                lpoly,
                PartitionOfUnityRBFShiftModel(
                    patch_grid=(10, 10), overlap=1.6, kernel="cubic", smoothing=1e-6
                ),
            ),
            ds_8,
            True,
            False,
        ),
        (
            "V2_12b_loess_deg3_16x16",
            "V2-12",
            "Zero-PT Local Poly",
            LocalPolynomialSurfaceRegressor(degree=3, neighbors=25, weight="tricube"),
            ds_16,
            True,
            False,
        ),
        (
            "V2_12c_wavelet_dwt_16x16",
            "V2-12",
            "Zero-PT Signal Proc",
            CompositeResidualShiftModel(
                bilinear, WaveletGridShiftModel(grid_step_m=687.5, levels=3, threshold_mm=0.2)
            ),
            ds_16,
            False,
            True,
        ),
    ]

    rows: list[dict[str, Any]] = []
    total = len(specs)
    for idx, (name, track, family, model, train_ds, sc_inst, sc_look) in enumerate(specs, 1):
        tr_c = train_ds.coords if train_ds is not None else None
        tr_s = train_ds.true_shifts if train_ds is not None else None
        row = _evaluate_candidate(
            evaluator=evaluator,
            model=model,
            name=name,
            track=track,
            family=family,
            train_coords=tr_c,
            train_shifts=tr_s,
            test_u16=test_u16,
            test_a1=test_a1,
            wall_idx=wall_idx,
            scale_instance_for_ro=sc_inst,
            scale_lookup_for_ro=sc_look,
        )
        rows.append(row)
        print(
            f"[{idx:02d}/{total}] {name}: u16={row['u16_mean_mm']:.3f}mm, "
            f"a1_rmse={row['a1_rmse_mm']:.3f}mm (lt1mm={row['a1_lt_1mm_pct']:.1f}%), "
            f"wall={row['a1_wall_rmse_mm']:.3f}mm, lat={row['latency_us_pt']:.2f}us, "
            f"ro_mb={row['est_full_ro_mb']:.2f}",
            flush=True,
        )

    df = pd.DataFrame(rows).sort_values(["a1_rmse_mm", "u16_mean_mm"]).reset_index(drop=True)
    df.to_csv(RESULTS_DIR / "v2_screening_scorecard.csv", index=False)
    (RESULTS_DIR / "v2_screening_scorecard.md").write_text(
        dataframe_to_markdown(df) + "\n", encoding="utf-8"
    )
    _plot_pareto(df)

    total_runtime_s = time.perf_counter() - t_start
    _write_report(df, total_runtime_s)

    top_row = df.iloc[0].to_dict()
    top_zpt = df[df["zero_pytorch"] == "Yes"].iloc[0].to_dict()
    brief = {
        "schema_version": "2.0",
        "slug": "submillimeter_exploration_v2",
        "experiment_type": "grid_shift",
        "status": "Validated",
        "runtime_s": round(total_runtime_s, 2),
        "key_metrics": {
            "num_models_screened": len(df),
            "best_overall_model": top_row["model"],
            "best_overall_a1_rmse_mm": top_row["a1_rmse_mm"],
            "best_zero_pytorch_model": top_zpt["model"],
            "best_zero_pytorch_a1_rmse_mm": top_zpt["a1_rmse_mm"],
        },
        "artifacts": [
            "experiments/submillimeter_exploration_v2/report.md",
            "experiments/submillimeter_exploration_v2/results/v2_screening_scorecard.csv",
            "experiments/submillimeter_exploration_v2/results/v2_screening_scorecard.md",
            "experiments/submillimeter_exploration_v2/plots/v2_accuracy_vs_latency_pareto.png",
            "experiments/submillimeter_exploration_v2/plots/v2_accuracy_vs_memory_pareto.png",
            "experiments/submillimeter_exploration_v2/plots/v2_family_comparison_bar.png",
        ],
        "benchmark_candidates": [],
    }
    (UNIT_DIR / "unit_brief.json").write_text(json.dumps(brief, indent=2), encoding="utf-8")
    print(f"\n=== Completed v2 screening ({len(df)} models) in {total_runtime_s:.1f}s ===")
    print(dataframe_to_markdown(df))


def run_full_romania_validation() -> None:
    """Run a fast Full-Romania check on the top v2 champions with scaled capacities."""
    t_start = time.perf_counter()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    gm = GridManager()
    evaluator = ModelEvaluator()

    lpoly = _load_or_fit_stage1(
        gm,
        "lpoly_deg2_k16_tric",
        LocalPolynomialSurfaceRegressor(degree=2, neighbors=16, weight="tricube"),
    )
    bilinear = _load_or_fit_stage1(gm, "cell_bilinear", CellBilinearShiftModel())
    catmull = _load_or_fit_stage1(
        gm,
        "catmull_rom",
        ParameterizedBicubicGridShiftModel(scheme="catmull_rom"),
    )

    ds_32 = gm.get(GridId.RO_GRID_32X32)
    ds_16 = gm.get(GridId.RO_GRID_16X16)
    test_u16 = gm.get(GridId.RO_TEST_UNIFORM_16X16_BORDER)
    test_a1 = gm.get(GridId.RO_TEST_ADAPTIVE_1MM_MID_BORDER)
    wall_idx = _wall_mask(test_a1.coords, tol=0.05)

    dn_qt, de_qt = gm.get_adaptive_pair("1mm")
    quadtree_1mm = AdaptiveQuadtreeRegressor(dn_grid=dn_qt.df, de_grid=de_qt.df)

    full_specs: list[tuple[str, str, str, BaseGridShiftModel, Any | None]] = [
        ("A3_Quadtree_1mm", "Anchor", "Geometric Lookup", quadtree_1mm, None),
        (
            "V2_6b_catmull_featgrid_3lvl_FullRO",
            "V2-6",
            "Micro-Grid Hybrid",
            CompositeResidualShiftModel(
                catmull,
                NeuralFeatureGridShiftModel(
                    grid_resolutions=((53, 72), (105, 143), (209, 285)),
                    feature_dim=6,
                    hidden_dim=128,
                    cyclic_cell_harmonics=2,
                    batch_size=8192,
                    lr=3e-3,
                    max_epochs=120,
                    val_fraction=0.0,
                    loss_type="huber",
                ),
            ),
            ds_32,
        ),
        (
            "V2_6a_bilinear_featgrid_3lvl_FullRO",
            "V2-6",
            "Micro-Grid Hybrid",
            CompositeResidualShiftModel(
                bilinear,
                NeuralFeatureGridShiftModel(
                    grid_resolutions=((53, 72), (105, 143), (209, 285)),
                    feature_dim=6,
                    hidden_dim=128,
                    cyclic_cell_harmonics=2,
                    batch_size=8192,
                    lr=3e-3,
                    max_epochs=120,
                    val_fraction=0.0,
                    loss_type="huber",
                ),
            ),
            ds_32,
        ),
        (
            "V2_2a_lpoly_wire_FullRO",
            "V2-2",
            "Modern INR (Wavelet)",
            CompositeResidualShiftModel(
                lpoly,
                WaveletINRGridShiftModel(
                    activation="wire",
                    hidden_dim=192,
                    num_layers=4,
                    batch_size=8192,
                    max_epochs=120,
                ),
            ),
            ds_32,
        ),
        (
            "V2_1a_lpoly_rbf_kan_FullRO",
            "V2-1",
            "Modern INR (KAN)",
            CompositeResidualShiftModel(
                lpoly,
                KANGridShiftModel(
                    basis="rbf",
                    hidden_dim=128,
                    num_layers=3,
                    num_basis=8,
                    num_fourier_frequencies=64,
                    batch_size=8192,
                    max_epochs=120,
                ),
            ),
            ds_32,
        ),
        (
            "V2_4b_lpoly_fbpinn_FullRO",
            "V2-4",
            "Modern INR (FBPINN)",
            CompositeResidualShiftModel(
                lpoly,
                SpatialMoEGridShiftModel(
                    mode="fbpinn",
                    fbpinn_grid=(8, 10),
                    hidden_dim=48,
                    num_fourier_frequencies=64,
                    batch_size=8192,
                    max_epochs=120,
                ),
            ),
            ds_32,
        ),
        (
            "V2_10a_lpoly_xgboost_FullRO",
            "V2-10",
            "Zero-PT Tree Ensemble",
            CompositeResidualShiftModel(
                lpoly,
                XGBoostGridShiftModel(
                    n_estimators=1000,
                    max_depth=10,
                    learning_rate=0.08,
                ),
            ),
            ds_32,
        ),
        (
            "V2_12b_loess_deg3_16x16_FullRO",
            "V2-12",
            "Zero-PT Local Poly",
            LocalPolynomialSurfaceRegressor(degree=3, neighbors=25, weight="tricube"),
            ds_16,
        ),
    ]

    rows: list[dict[str, Any]] = []
    total = len(full_specs)
    for idx, (name, track, family, model, train_ds) in enumerate(full_specs, 1):
        tr_c = train_ds.coords if train_ds is not None else None
        tr_s = train_ds.true_shifts if train_ds is not None else None
        row = _evaluate_candidate(
            evaluator=evaluator,
            model=model,
            name=name,
            track=track,
            family=family,
            train_coords=tr_c,
            train_shifts=tr_s,
            test_u16=test_u16,
            test_a1=test_a1,
            wall_idx=wall_idx,
            scale_instance_for_ro=False,
            scale_lookup_for_ro=False,
        )
        rows.append(row)
        print(
            f"[Full-RO {idx:02d}/{total}] {name}: u16={row['u16_mean_mm']:.3f}mm "
            f"(lt1mm={row['u16_lt_1mm_pct']:.1f}%), a1_rmse={row['a1_rmse_mm']:.3f}mm "
            f"(lt1mm={row['a1_lt_1mm_pct']:.1f}%), lat={row['latency_us_pt']:.2f}us, "
            f"mb={row['est_full_ro_mb']:.2f}",
            flush=True,
        )

    df = pd.DataFrame(rows).sort_values(["a1_rmse_mm", "u16_mean_mm"]).reset_index(drop=True)
    df.to_csv(RESULTS_DIR / "v2_full_ro_validation.csv", index=False)
    md = dataframe_to_markdown(df)
    (RESULTS_DIR / "v2_full_ro_validation.md").write_text(md + "\n", encoding="utf-8")

    total_s = time.perf_counter() - t_start
    print(f"\n=== Completed Full-Romania Validation ({len(df)} models) in {total_s:.1f}s ===")
    print(md)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-ro", action="store_true", help="Run Full-Romania champion check.")
    args = parser.parse_args()
    if args.full_ro:
        run_full_romania_validation()
    else:
        main()
