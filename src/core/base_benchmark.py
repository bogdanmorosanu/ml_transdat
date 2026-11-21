"""Unified cross-model benchmark orchestrator and drift-free experiment promotion."""

from __future__ import annotations

import abc
import json
import time
from dataclasses import dataclass
from pathlib import Path

from src.core.base_unit import GridShiftExperimentUnit
from src.data import GridDataset, GridId, get_default_grid_manager
from src.evaluation import (
    MetricGroup,
    MetricType,
    ModelEvaluator,
    dataframe_to_markdown,
)
from src.models.base import BaseGridShiftModel
from src.viz import (
    plot_memory_breakdown_stacked,
    plot_model_comparison,
    plot_speed_workloads,
)


@dataclass
class BenchmarkModelSpec:
    """Declarative model entry in a :class:`BenchmarkSuite`."""

    name: str
    model: BaseGridShiftModel
    train_grid: GridId | str | None = None
    source_experiment: str | None = None


def specs_from_unit(
    unit_cls: type[GridShiftExperimentUnit],
    suite: str,
    unit_dir: Path | None = None,
) -> list[BenchmarkModelSpec]:
    """Pull benchmark-candidate models directly from a :class:`GridShiftExperimentUnit`.

    Guarantees exact hyperparameter and training-grid parity between the source
    experiment and the benchmark suite.
    """
    probe_dir = unit_dir or Path("experiments") / "__probe__"
    unit = unit_cls(probe_dir)
    out: list[BenchmarkModelSpec] = []
    for spec in unit.get_model_specs():
        if spec.benchmark_candidate and spec.benchmark_suite == suite:
            out.append(
                BenchmarkModelSpec(
                    name=spec.benchmark_label or spec.name,
                    model=spec.model,
                    train_grid=spec.train_grid,
                    source_experiment=unit.slug,
                )
            )
    return out


class BenchmarkSuite(abc.ABC):
    """Template Method orchestrator for standardized 3-view cross-model benchmarks."""

    def __init__(self, bench_dir: Path) -> None:
        self.bench_dir = Path(bench_dir)
        self.plots_dir = self.bench_dir / "plots"
        self.results_dir = self.bench_dir / "results"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.grid_manager = get_default_grid_manager()
        self.evaluator = ModelEvaluator()

    @property
    @abc.abstractmethod
    def suite_name(self) -> str:
        """Directory slug of the benchmark suite."""

    @property
    @abc.abstractmethod
    def suite_title(self) -> str:
        """Markdown heading for ``summary_scorecard.md``."""

    @property
    @abc.abstractmethod
    def suite_description(self) -> str:
        """Short description placed at the top of ``summary_scorecard.md``."""

    @property
    def precision_workload(self) -> str:
        """Primary workload key used for precision scorecard & comparison plots."""
        return "corridor_36k"

    @property
    def precision_plot_log_scale(self) -> bool:
        """Whether the precision comparison bar chart uses a log y-axis."""
        return False

    def get_workloads(self) -> dict[str, GridDataset]:
        """Return ordered mapping of named evaluation workloads."""
        return {
            "corridor_36k": self.grid_manager.get(GridId.CORRIDOR_SNW),
            "uniform_4x4": self.grid_manager.get(GridId.RO_GRID_4X4),
            "single_cell_10k": self.grid_manager.get(GridId.SINGLE_CELL_10K),
        }

    @abc.abstractmethod
    def get_model_specs(self) -> list[BenchmarkModelSpec]:
        """Return ordered list of :class:`BenchmarkModelSpec` entries to benchmark."""

    def run(self) -> dict[str, object]:
        """Fit unfitted models, evaluate all workloads once, and write 3-view artifacts."""
        t0 = time.perf_counter()
        print(f"=== Starting Benchmark Suite: {self.suite_name} ===")

        workloads = self.get_workloads()
        specs = self.get_model_specs()

        train_cache: dict[GridId | str, GridDataset] = {}
        models: list[tuple[str, BaseGridShiftModel]] = []
        provenance: dict[str, str | None] = {}

        for spec in specs:
            model = spec.model
            if not model.is_fitted:
                if spec.train_grid is None:
                    raise ValueError(
                        f"Benchmark model '{spec.name}' is not fitted and has no train_grid."
                    )
                if spec.train_grid not in train_cache:
                    train_cache[spec.train_grid] = self.grid_manager.get(spec.train_grid)
                train_ds = train_cache[spec.train_grid]
                model.fit(train_ds.coords, train_ds.true_shifts)
                print(
                    f"Fitted '{spec.name}' on {train_ds.grid_id.value} "
                    f"in {model.fit_time_sec:.3f}s"
                )
            models.append((spec.name, model))
            provenance[spec.name] = spec.source_experiment

        results_by_workload = {
            w_name: self.evaluator.evaluate_models(models=models, dataset=ds)
            for w_name, ds in workloads.items()
        }
        primary_key = self.precision_workload
        primary_results = results_by_workload[primary_key]
        primary_ds = workloads[primary_key]

        # View 1: Comprehensive Precision Scorecard (mm)
        df_precision = self.evaluator.tabulate_results(
            results=primary_results,
            metrics=MetricGroup.PRECISION_FULL,
            precision_unit="mm",
        )
        scorecard_path = self.bench_dir / "summary_scorecard.md"
        lines: list[str] = [
            f"# {self.suite_title}\n",
            (
                f"Precision evaluated on `{primary_ds.grid_id.value}` ({len(primary_ds):,} points). "
                f"{self.suite_description} All error metrics are reported in **millimeters (mm)**.\n"
            ),
            "## 1. Comprehensive Precision Scorecard (mm)\n",
            dataframe_to_markdown(df_precision, float_precision=2) + "\n",
            "## 2. Performance Speed & Throughput Across Named Workloads\n",
        ]

        # View 2: Speed & Throughput Across Named Workloads
        for w_name, w_results in results_by_workload.items():
            w_ds = workloads[w_name]
            df_speed = self.evaluator.tabulate_results(
                results=w_results,
                metrics=MetricGroup.PERFORMANCE_SPEED,
            )
            lines.append(
                f"### Workload: `{w_name}` (`{w_ds.grid_id.value}`, {len(w_ds):,} pts)\n"
            )
            lines.append(dataframe_to_markdown(df_speed) + "\n")

        # View 3: 4-Component Memory & Storage Breakdown
        df_memory = self.evaluator.tabulate_results(
            results=primary_results,
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
                    "- **Grid Lookup Data**: Retained control nodes or indexed cell tables.\n"
                ),
                dataframe_to_markdown(df_memory) + "\n",
            ]
        )
        scorecard_path.write_text("\n".join(lines), encoding="utf-8")

        # Plots
        plot_model_comparison(
            results=primary_results,
            out_path=self.plots_dir / "01_precision_comparison.png",
            metrics=[
                MetricType.MEAN_2D_M,
                MetricType.RMSE_2D_M,
                MetricType.P95_2D_M,
                MetricType.MAX_2D_M,
            ],
            title=f"Precision Comparison — {self.suite_title} ({primary_ds.grid_id.value}, mm)",
            precision_unit="mm",
            log_scale=self.precision_plot_log_scale,
        )
        plot_speed_workloads(
            results_by_workload=results_by_workload,
            out_path=self.plots_dir / "02_performance_speed.png",
            title="Inference Throughput & Latency Across Named Workloads",
            log_scale=True,
        )
        plot_memory_breakdown_stacked(
            results=primary_results,
            out_path=self.plots_dir / "03_memory_breakdown_stacked.png",
            title=f"{self.suite_title}: 4-Component Memory & Deployment Footprint (MB)",
            unit="MB",
        )

        # JSON summary
        summary_json_payload = {
            "suite_name": self.suite_name,
            "precision_dataset_id": primary_ds.grid_id.value,
            "precision_unit": "mm",
            "workloads": {
                w_name: {
                    "dataset_id": workloads[w_name].grid_id.value,
                    "sample_count": len(workloads[w_name]),
                    "models": [
                        {
                            "name": r.model_name,
                            "source_experiment": provenance.get(r.model_name),
                            "summary": r.summary,
                            "eval_time_sec": r.eval_time_sec,
                            "fit_time_sec": r.fit_time_sec,
                            "throughput_pts_per_sec": r.throughput_pts_per_sec,
                            "memory_breakdown": (
                                r.memory_breakdown.to_dict()
                                if r.memory_breakdown
                                else None
                            ),
                        }
                        for r in w_results
                    ],
                }
                for w_name, w_results in results_by_workload.items()
            },
        }
        json_path = self.results_dir / "benchmark_summary.json"
        json_path.write_text(
            json.dumps(summary_json_payload, indent=2) + "\n", encoding="utf-8"
        )

        elapsed = time.perf_counter() - t0
        print(f"=== Benchmark Suite '{self.suite_name}' Complete in {elapsed:.2f}s ===")
        return summary_json_payload
