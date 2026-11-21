"""Cross-model benchmark comparing all 2D Grid Shift Regressor families.

Evaluates representative champions across:
1. Non-parametric Spatial Blending: KNN (K=4 IDW, K=16 IDW)
2. Axis-aligned Partitions: Decision Tree (11 km base nodes, 8x8 pruned 1e-8)
3. Hierarchical Geometric Lookup: Adaptive Quadtree (3 mm, 1 mm tiers)

Produces three distinct views from a single evaluation pass per workload:
- View 1: Comprehensive Precision Scorecard (mm) on ``ro_corridor_snw``
- View 2: Named Speed & Throughput Workloads (``corridor_36k``, ``uniform_4x4``, ``single_cell_10k``)
- View 3: 4-Component Memory & Storage Breakdown (Code, Dependencies, Model Instance, Grid Lookup)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.data import GridId, get_default_grid_manager
from src.evaluation import (
    MetricGroup,
    MetricType,
    ModelEvaluator,
    dataframe_to_markdown,
)
from src.models import (
    AdaptiveQuadtreeRegressor,
    DecisionTreeSpatialRegressor,
    KNNSpatialRegressor,
    MLPGridShiftModel,
)
from src.viz import (
    plot_memory_breakdown_stacked,
    plot_model_comparison,
    plot_speed_workloads,
)


def main() -> None:
    bench_dir = Path("benchmarks/grid_shift_models")
    plots_dir = bench_dir / "plots"
    results_dir = bench_dir / "results"
    plots_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=== Loading Datasets for Grid Shift Models Benchmark ===")
    gm = get_default_grid_manager()
    train_nodes = gm.get(GridId.SPG_NODES_VALID)
    train_8x8 = gm.get(GridId.RO_GRID_8X8)

    workloads = {
        "corridor_36k": gm.get(GridId.CORRIDOR_SNW),
        "uniform_4x4": gm.get(GridId.RO_GRID_4X4),
        "single_cell_10k": gm.get(GridId.SINGLE_CELL_10K),
    }

    print("=== Training / Initializing Models in Defined Evaluation Order ===")
    knn_k4 = KNNSpatialRegressor(n_neighbors=4, weights="distance").fit(
        train_nodes.coords, train_nodes.true_shifts
    )
    knn_k16 = KNNSpatialRegressor(n_neighbors=16, weights="distance").fit(
        train_nodes.coords, train_nodes.true_shifts
    )
    dt_base = DecisionTreeSpatialRegressor().fit(
        train_nodes.coords, train_nodes.true_shifts
    )
    dt_8x8_pruned = DecisionTreeSpatialRegressor(min_impurity_decrease=1e-8).fit(
        train_8x8.coords, train_8x8.true_shifts
    )

    dn_3mm, de_3mm = gm.get_adaptive_pair("3mm")
    quadtree_3mm = AdaptiveQuadtreeRegressor(dn_grid=dn_3mm.df, de_grid=de_3mm.df)

    dn_1mm, de_1mm = gm.get_adaptive_pair("1mm")
    quadtree_1mm = AdaptiveQuadtreeRegressor(dn_grid=dn_1mm.df, de_grid=de_1mm.df)

    train_4x4 = gm.get(GridId.RO_GRID_4X4)
    mlp_fourier_4x4 = MLPGridShiftModel(
        hidden_layer_sizes=(128, 128),
        use_fourier_features=True,
        fourier_scale=3.0,
        num_fourier_frequencies=32,
        solver="adam",
        max_iter=100,
        batch_size=256,
        alpha=1e-5,
    ).fit(train_4x4.coords, train_4x4.true_shifts)

    models: list[tuple[str, Any]] = [
        ("KNN (K=4 IDW, 11km)", knn_k4),
        ("KNN (K=16 IDW, 11km)", knn_k16),
        ("Decision Tree (11km)", dt_base),
        ("Decision Tree (8x8, pruned 1e-8)", dt_8x8_pruned),
        ("Adaptive Quadtree (3mm tier)", quadtree_3mm),
        ("Adaptive Quadtree (1mm tier)", quadtree_1mm),
        ("Fourier MLP (4x4, 2.7km)", mlp_fourier_4x4),
    ]

    evaluator = ModelEvaluator()

    # Single evaluation pass per workload
    results_by_workload = {
        w_name: evaluator.evaluate_models(models=models, dataset=ds)
        for w_name, ds in workloads.items()
    }
    corridor_results = results_by_workload["corridor_36k"]
    corridor_ds = workloads["corridor_36k"]

    # --- VIEW 1: Standard Comprehensive Precision Scorecard (mm) ---
    print("\n=== Generating View 1: Comprehensive Precision Scorecard (mm) ===")
    df_precision = evaluator.tabulate_results(
        results=corridor_results,
        metrics=MetricGroup.PRECISION_FULL,
        precision_unit="mm",
    )
    scorecard_md_path = bench_dir / "summary_scorecard.md"
    lines: list[str] = [
        "# Cross-Model Grid Shift Benchmark Summary\n",
        (
            f"Precision evaluated on canonical corridor benchmark (`{corridor_ds.grid_id.value}`, "
            f"{len(corridor_ds):,} points). All distance/error metrics are reported in **millimeters (mm)**.\n"
        ),
        "## 1. Comprehensive Precision Scorecard (mm)\n",
        dataframe_to_markdown(df_precision, float_precision=2) + "\n",
    ]

    plot_model_comparison(
        results=corridor_results,
        out_path=plots_dir / "01_precision_comparison.png",
        metrics=[
            MetricType.MEAN_2D_M,
            MetricType.RMSE_2D_M,
            MetricType.P95_2D_M,
            MetricType.MAX_2D_M,
        ],
        title=f"Precision Comparison across Grid Shift Models ({corridor_ds.grid_id.value}, mm)",
        precision_unit="mm",
        log_scale=False,
    )

    # --- VIEW 2: Named Speed & Throughput Workloads ---
    print("=== Generating View 2: Performance Speed & Throughput Across Workloads ===")
    lines.append("## 2. Performance Speed & Throughput Across Named Workloads\n")
    for w_name, w_results in results_by_workload.items():
        w_ds = workloads[w_name]
        df_speed = evaluator.tabulate_results(
            results=w_results,
            metrics=MetricGroup.PERFORMANCE_SPEED,
        )
        lines.append(
            f"### Workload: `{w_name}` (`{w_ds.grid_id.value}`, {len(w_ds):,} pts)\n"
        )
        lines.append(dataframe_to_markdown(df_speed) + "\n")

    plot_speed_workloads(
        results_by_workload=results_by_workload,
        out_path=plots_dir / "02_performance_speed.png",
        title="Inference Throughput & Latency Across Named Workloads",
        log_scale=True,
    )

    # --- VIEW 3: 4-Component Memory & Storage Breakdown ---
    print("=== Generating View 3: 4-Component Memory & Storage Breakdown ===")
    df_memory = evaluator.tabulate_results(
        results=corridor_results,
        metrics=MetricGroup.MEMORY_BREAKDOWN,
    )
    lines.extend(
        [
            "## 3. Storage & Memory Breakdown (4 Components)\n",
            (
                "Components:\n"
                "- **Code Size**: Model implementation file on disk.\n"
                "- **Library Dependencies**: Installed package footprint of external runtime dependencies.\n"
                "- **Model Instance / Weights**: In-memory parameters, split trees, or index metadata.\n"
                "- **Grid Lookup Data**: Retained control nodes (KNN) or indexed quadtree cell tables.\n"
            ),
            dataframe_to_markdown(df_memory) + "\n",
        ]
    )
    scorecard_md_path.write_text("\n".join(lines), encoding="utf-8")

    plot_memory_breakdown_stacked(
        results=corridor_results,
        out_path=plots_dir / "03_memory_breakdown_stacked.png",
        title="Model Memory & Storage Breakdown (4 Components - MB)",
        unit="MB",
    )

    summary_json_payload = {
        "precision_dataset_id": corridor_ds.grid_id.value,
        "precision_unit": "mm",
        "workloads": {
            w_name: {
                "dataset_id": workloads[w_name].grid_id.value,
                "sample_count": len(workloads[w_name]),
                "models": [
                    {
                        "name": r.model_name,
                        "summary": r.summary,
                        "eval_time_sec": r.eval_time_sec,
                        "fit_time_sec": r.fit_time_sec,
                        "throughput_pts_per_sec": r.throughput_pts_per_sec,
                        "memory_breakdown": (
                            r.memory_breakdown.to_dict() if r.memory_breakdown else None
                        ),
                    }
                    for r in w_results
                ],
            }
            for w_name, w_results in results_by_workload.items()
        },
    }
    (results_dir / "benchmark_summary.json").write_text(
        json.dumps(summary_json_payload, indent=2), encoding="utf-8"
    )

    print("\n=== Benchmark Complete! ===")
    print(f"Summary Markdown: {scorecard_md_path}")
    print(f"Figures Saved to: {plots_dir}")


if __name__ == "__main__":
    main()
