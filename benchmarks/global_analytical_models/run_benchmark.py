"""Cross-model benchmark comparing Global Analytical & Mathematical Surface models.

Evaluates representative closed-form global mathematical models:
0. Helmert 2D Conformal: Classical 4-parameter conformal transformation baseline (shifts, isotropic scale, rotation)
1. Degree 1: Linear Affine Flat Plane (3 terms per axis = 6 terms, allowing anisotropic scale and shear)
2. Degree 2: Quadratic Bowl & Saddle (6 terms per axis = 12 terms)
3. Degree 3: Classical Geodetic Cubic Surface (10 terms per axis = 20 terms, inflection/asymmetry)
4. Degree 5: Classical Geodetic Quintic Surface (21 terms per axis = 42 terms, multi-wave)
5. Degree 8: Multi-Harmonic Surface (45 terms per axis = 90 terms)
6. Degree 12: High-Order Polynomial Capacity Limit (91 terms per axis = 182 terms)
7. Degree 5 Sparse Lasso: Minimalist closed-form formula (13 terms dN, 11 terms dE = 24 terms)

Produces three distinct views from a single evaluation pass per workload:
- View 1: Comprehensive Precision Scorecard (mm) on canonical corridor benchmark
- View 2: Named Speed & Throughput Workloads (corridor_36k, uniform_4x4, single_cell_10k)
- View 3: 4-Component Memory & Storage Breakdown (zero lookup grid data)
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
from src.models import Helmert2DConformalShiftModel, PolynomialSurfaceRegressor
from src.viz import (
    plot_memory_breakdown_stacked,
    plot_model_comparison,
    plot_speed_workloads,
)


def main() -> None:
    bench_dir = Path("benchmarks/global_analytical_models")
    plots_dir = bench_dir / "plots"
    results_dir = bench_dir / "results"
    plots_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=== Loading Datasets for Global Analytical Models Benchmark ===")
    gm = get_default_grid_manager()
    train_nodes = gm.get(GridId.SPG_NODES_VALID)

    workloads = {
        "corridor_36k": gm.get(GridId.CORRIDOR_SNW),
        "uniform_4x4": gm.get(GridId.RO_GRID_4X4),
        "single_cell_10k": gm.get(GridId.SINGLE_CELL_10K),
    }

    print("=== Training Global Analytical Models on SPG Nodes ===")
    helmert_shift = Helmert2DConformalShiftModel().fit(
        train_nodes.coords, train_nodes.true_shifts
    )
    poly_deg1 = PolynomialSurfaceRegressor(degree=1, alpha=1e-8).fit(
        train_nodes.coords, train_nodes.true_shifts
    )
    poly_deg2 = PolynomialSurfaceRegressor(degree=2, alpha=1e-8).fit(
        train_nodes.coords, train_nodes.true_shifts
    )
    poly_deg3 = PolynomialSurfaceRegressor(degree=3, alpha=1e-8).fit(
        train_nodes.coords, train_nodes.true_shifts
    )
    poly_deg5 = PolynomialSurfaceRegressor(degree=5, alpha=1e-8).fit(
        train_nodes.coords, train_nodes.true_shifts
    )
    poly_deg8 = PolynomialSurfaceRegressor(degree=8, alpha=1e-8).fit(
        train_nodes.coords, train_nodes.true_shifts
    )
    poly_deg12 = PolynomialSurfaceRegressor(degree=12, alpha=1e-8).fit(
        train_nodes.coords, train_nodes.true_shifts
    )
    poly_lasso5 = PolynomialSurfaceRegressor(
        degree=5, alpha=5e-4, regressor="lasso"
    ).fit(train_nodes.coords, train_nodes.true_shifts)

    models: list[tuple[str, Any]] = [
        ("Helmert 2D Conformal (4P)", helmert_shift),
        ("Linear Affine Plane (Deg 1, 6T)", poly_deg1),
        ("Quadratic Bowl (Deg 2, 12T)", poly_deg2),
        ("Cubic Surface (Deg 3, 20T)", poly_deg3),
        ("Quintic Surface (Deg 5, 42T)", poly_deg5),
        ("Multi-Harmonic (Deg 8, 90T)", poly_deg8),
        ("High-Order Limit (Deg 12, 182T)", poly_deg12),
        ("Sparse Lasso (Deg 5, 24T)", poly_lasso5),
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
    print("\n=== Generating View 1: Comprehensive Precision Scorecard ===")
    df_precision = evaluator.tabulate_results(
        results=corridor_results,
        metrics=MetricGroup.PRECISION_FULL,
        precision_unit="mm",
    )

    scorecard_path = bench_dir / "summary_scorecard.md"
    lines: list[str] = [
        "# Global Analytical Models Benchmark Summary\n",
        (
            f"Precision evaluated on canonical corridor benchmark (`{corridor_ds.grid_id.value}`, "
            f"{len(corridor_ds):,} points). Models in this suite evaluate a **single closed-form global equation** "
            "without spatial lookups (0 MB grid lookup memory). All errors are reported in **millimeters (mm)**.\n"
        ),
        "## 1. Comprehensive Precision Scorecard (mm)\n",
        dataframe_to_markdown(df_precision, float_precision=2) + "\n",
        "## 2. Performance Speed & Throughput Across Named Workloads\n",
    ]

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

    # --- VIEW 3: Memory Breakdown ---
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
                "- **Grid Lookup Data**: Retained control nodes or indexed cell tables (0 MB for all analytical models).\n"
            ),
            dataframe_to_markdown(df_memory) + "\n",
        ]
    )

    scorecard_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved comprehensive scorecard to {scorecard_path}")

    # --- VIEW 2: Visualization Plots ---
    print("\n=== Generating Visualization Plots ===")
    p1 = plots_dir / "01_precision_comparison.png"
    plot_model_comparison(
        results=corridor_results,
        out_path=p1,
        metrics=[
            MetricType.MEAN_2D_M,
            MetricType.RMSE_2D_M,
            MetricType.P95_2D_M,
            MetricType.MAX_2D_M,
        ],
        title=f"Precision Comparison across Global Analytical Models ({corridor_ds.grid_id.value}, mm)",
        precision_unit="mm",
        log_scale=False,
    )
    print(f"Saved precision comparison plot to {p1}")

    p2 = plots_dir / "02_performance_speed.png"
    plot_speed_workloads(
        results_by_workload=results_by_workload,
        out_path=p2,
        title="Inference Throughput & Latency Across Named Workloads",
        log_scale=True,
    )
    print(f"Saved performance speed plot to {p2}")

    p3 = plots_dir / "03_memory_breakdown_stacked.png"
    plot_memory_breakdown_stacked(
        results=corridor_results,
        out_path=p3,
        title="Global Analytical Models: 4-Component Memory & Deployment Footprint (MB)",
        unit="MB",
    )
    print(f"Saved memory breakdown plot to {p3}")

    # --- JSON Serialization ---
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
    json_path = results_dir / "benchmark_summary.json"
    json_path.write_text(json.dumps(summary_json_payload, indent=2), encoding="utf-8")
    print(f"Saved benchmark summary JSON to {json_path}")
    print("\n=== Global Analytical Models Benchmark Complete! ===")


if __name__ == "__main__":
    main()
