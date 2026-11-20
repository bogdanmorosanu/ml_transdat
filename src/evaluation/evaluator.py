"""Unified model evaluation engine for shift regressors and coordinate transformers."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.data.catalog import GridDataset
from src.evaluation.metrics import (
    MetricGroup,
    MetricSpec,
    MetricType,
    compute_coordinate_errors,
    compute_shift_errors,
    format_metric_label,
    resolve_metric_types,
    summarize_errors,
)
from src.models.base import (
    BaseCoordinateTransformer,
    BaseGridShiftModel,
    MemoryBreakdown,
)


@dataclass
class EvaluationResult:
    """Standardized evaluation result returned by :class:`ModelEvaluator`."""

    model_name: str
    dataset_id: str
    direction: str
    err_df: pd.DataFrame
    summary: dict[str, Any]
    eval_time_sec: float
    throughput_pts_per_sec: float
    model_size_bytes: int = 0
    fit_time_sec: float = 0.0
    memory_breakdown: MemoryBreakdown | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def get_metric(self, metric: MetricType) -> float:
        """Extract a scalar metric value from this result by :class:`MetricType`."""
        return extract_metric(self, metric)

    def to_scorecard_row(self, precision_unit: str = "mm") -> dict[str, Any]:
        """Return a flat dictionary suitable for multi-model comparison tables."""
        unit_scale = {"m": 1.0, "cm": 100.0, "mm": 1000.0}.get(precision_unit.lower())
        if unit_scale is None:
            raise ValueError(
                f"Unsupported precision_unit '{precision_unit}'. Expected 'm', 'cm', or 'mm'."
            )
        u = precision_unit.lower()
        pr = self.summary["pass_rates_pct"]
        total_mb = (
            self.memory_breakdown.total_mb
            if self.memory_breakdown is not None
            else self.model_size_bytes / (1024.0 * 1024.0)
        )
        return {
            "Model": self.model_name,
            "Dataset": self.dataset_id,
            "Direction": self.direction,
            "Count": self.summary["count"],
            f"Mean 2D ({u})": self.summary["mean_2d_m"] * unit_scale,
            f"RMSE 2D ({u})": self.summary["rmse_2d_m"] * unit_scale,
            f"P95 ({u})": self.summary["p95_2d_m"] * unit_scale,
            f"P99 ({u})": self.summary["p99_2d_m"] * unit_scale,
            f"Max 2D ({u})": self.summary["max_2d_m"] * unit_scale,
            "<1cm (%)": pr["lt_1cm"],
            "<5cm (%)": pr["lt_5cm"],
            "Time (s)": self.eval_time_sec,
            "Size (MB)": total_mb,
        }


def extract_metric(result: EvaluationResult, metric: MetricType) -> float:
    """Extract a scalar metric from an :class:`EvaluationResult`."""
    if metric == MetricType.EVAL_TIME_SEC:
        return float(result.eval_time_sec)
    if metric == MetricType.FIT_TIME_SEC:
        return float(result.fit_time_sec)
    if metric == MetricType.THROUGHPUT_PTS_PER_SEC:
        return float(result.throughput_pts_per_sec)
    if metric == MetricType.LATENCY_US_PER_PT:
        pts = result.summary.get("count", 0)
        return float((result.eval_time_sec * 1e6) / pts) if pts > 0 else 0.0
    if metric == MetricType.MODEL_SIZE_MB:
        return float(result.model_size_bytes) / (1024.0 * 1024.0)

    mb = result.memory_breakdown
    if metric == MetricType.SIZE_CODE_BYTES:
        return float(mb.code_bytes if mb is not None else 0.0)
    if metric == MetricType.SIZE_DEPENDENCIES_BYTES:
        return float(mb.dependencies_bytes if mb is not None else 0.0)
    if metric == MetricType.SIZE_MODEL_BYTES:
        return float(
            mb.model_instance_bytes if mb is not None else result.model_size_bytes
        )
    if metric == MetricType.SIZE_GRID_LOOKUP_BYTES:
        return float(mb.grid_lookup_bytes if mb is not None else 0.0)
    if metric == MetricType.TOTAL_SIZE_MB:
        return float(
            mb.total_mb
            if mb is not None
            else result.model_size_bytes / (1024.0 * 1024.0)
        )

    if metric.value in result.summary:
        return float(result.summary[metric.value])
    pass_rates = result.summary.get("pass_rates_pct", {})
    if metric.value in pass_rates:
        return float(pass_rates[metric.value])
    raise KeyError(f"Metric '{metric.value}' not found in EvaluationResult.")


class ModelEvaluator:
    """Evaluates :class:`BaseGridShiftModel` and :class:`BaseCoordinateTransformer` uniformly."""

    def evaluate_shift_model(
        self,
        model: BaseGridShiftModel,
        dataset: GridDataset,
        model_name: str,
        direction: str = "grid_shift_2d",
    ) -> EvaluationResult:
        """Evaluate a 2D grid shift regressor against ground-truth SPG shifts."""
        coords = dataset.coords
        true_shifts = dataset.true_shifts

        t0 = time.perf_counter()
        pred_shifts = model.predict(coords)
        elapsed = time.perf_counter() - t0

        err_df = compute_shift_errors(dataset.df, pred_shifts, true_shifts)
        summary = summarize_errors(err_df)
        throughput = len(coords) / elapsed if elapsed > 0.0 else float("inf")

        return EvaluationResult(
            model_name=model_name,
            dataset_id=dataset.grid_id.value,
            direction=direction,
            err_df=err_df,
            summary=summary,
            eval_time_sec=elapsed,
            throughput_pts_per_sec=throughput,
            model_size_bytes=model.memory_footprint_bytes(),
            fit_time_sec=model.fit_time_sec,
            memory_breakdown=model.memory_breakdown(),
        )

    def evaluate_models(
        self,
        models: Sequence[tuple[str, BaseGridShiftModel]]
        | Mapping[str, BaseGridShiftModel],
        dataset: GridDataset,
    ) -> list[EvaluationResult]:
        """Run a single evaluation pass for each model in user-specified order."""
        items = list(models.items()) if isinstance(models, Mapping) else list(models)
        return [
            self.evaluate_shift_model(model=model, dataset=dataset, model_name=name)
            for name, model in items
        ]

    def tabulate_results(
        self,
        results: Sequence[EvaluationResult],
        metrics: MetricSpec = MetricGroup.PRECISION_CORE,
        precision_unit: str = "mm",
    ) -> pd.DataFrame:
        """Project pre-computed :class:`EvaluationResult` objects into a comparison DataFrame."""
        resolved_metrics = resolve_metric_types(metrics)
        rows: list[dict[str, Any]] = []
        for res in results:
            row: dict[str, Any] = {"Model": res.model_name}
            for m in resolved_metrics:
                label, scale = format_metric_label(m, precision_unit=precision_unit)
                row[label] = extract_metric(res, m) * scale
            rows.append(row)
        return pd.DataFrame(rows)

    def compare_models(
        self,
        models: Sequence[tuple[str, BaseGridShiftModel]]
        | Mapping[str, BaseGridShiftModel],
        dataset: GridDataset,
        metrics: MetricSpec = MetricGroup.PRECISION_CORE,
        precision_unit: str = "mm",
    ) -> tuple[list[EvaluationResult], pd.DataFrame]:
        """Evaluate multiple fitted models once and tabulate requested metrics."""
        results = self.evaluate_models(models=models, dataset=dataset)
        table_df = self.tabulate_results(
            results=results,
            metrics=metrics,
            precision_unit=precision_unit,
        )
        return results, table_df

    def evaluate_transformer(
        self,
        transformer: BaseCoordinateTransformer,
        dataset: GridDataset,
        direction: str,
        model_name: str,
    ) -> EvaluationResult:
        """Evaluate a full CRS coordinate transformer on ``etrs89_to_st70`` or ``st70_to_etrs89``."""
        t0 = time.perf_counter()
        if direction == "etrs89_to_st70":
            etrs = dataset.etrs89_rad
            pred_c1, pred_c2 = transformer.etrs89_to_st70(etrs[:, 0], etrs[:, 1])
            elapsed = time.perf_counter() - t0
            err_df = compute_coordinate_errors(
                df_ref=dataset.df,
                pred_c1=pred_c1,
                pred_c2=pred_c2,
                direction=direction,
            )
        elif direction == "st70_to_etrs89":
            coords = dataset.coords
            pred_c1, pred_c2 = transformer.st70_to_etrs89(coords[:, 0], coords[:, 1])
            elapsed = time.perf_counter() - t0
            err_df = compute_coordinate_errors(
                df_ref=dataset.df,
                pred_c1=pred_c1,
                pred_c2=pred_c2,
                direction=direction,
                is_degrees=False,
            )
        else:
            raise ValueError(
                f"Unknown direction '{direction}'. Expected 'etrs89_to_st70' or 'st70_to_etrs89'."
            )

        summary = summarize_errors(err_df)
        throughput = len(dataset) / elapsed if elapsed > 0.0 else float("inf")

        return EvaluationResult(
            model_name=model_name,
            dataset_id=dataset.grid_id.value,
            direction=direction,
            err_df=err_df,
            summary=summary,
            eval_time_sec=elapsed,
            throughput_pts_per_sec=throughput,
            model_size_bytes=transformer.memory_footprint_bytes(),
            memory_breakdown=transformer.memory_breakdown(),
        )
