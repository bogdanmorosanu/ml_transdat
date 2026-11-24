"""Abstract base classes and unified experiment orchestrators for Research Units."""

from __future__ import annotations

import abc
import gc
import inspect
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.catalog import GridDataset, GridId, GridManager
from src.evaluation.evaluator import EvaluationResult, ModelEvaluator
from src.evaluation.metrics import dataframe_to_markdown, format_scorecard_table
from src.models.base import BaseCoordinateTransformer, BaseGridShiftModel
from src.viz.dispatcher import PlotDispatcher
from src.viz.plots import plot_extreme_scenarios_9cells
from src.viz.registry import PlotGroup, PlotSpec, PlotType


class ExperimentType(str, Enum):
    """Canonical experiment category taxonomy."""

    GRID_SHIFT = "grid_shift"
    TRANSFORMATION = "transformation"
    ANALYSIS = "analysis"


def _to_json_safe(val: Any) -> Any:
    """Convert common Python/NumPy/Path objects into JSON-serializable primitives."""
    if val is None or isinstance(val, (bool, int, float, str)):
        return val
    if isinstance(val, Enum):
        return val.value
    if isinstance(val, Path):
        return val.as_posix()
    if isinstance(val, (np.integer, np.floating, np.bool_)):
        return val.item()
    if isinstance(val, (list, tuple)):
        return [_to_json_safe(v) for v in val]
    if isinstance(val, dict):
        return {str(k): _to_json_safe(v) for k, v in val.items()}
    if isinstance(val, pd.DataFrame):
        return f"<DataFrame shape={val.shape}>"
    if isinstance(val, np.ndarray):
        return f"<ndarray shape={val.shape}>"
    return repr(val)


def _extract_model_config(model: Any) -> dict[str, Any]:
    """Inspect a model instance to capture its class path and constructor hyperparameters."""
    cls = type(model)
    sig = inspect.signature(cls.__init__)
    params: dict[str, Any] = {}
    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if hasattr(model, param_name):
            raw_val = getattr(model, param_name)
        elif hasattr(model, f"_{param_name}"):
            raw_val = getattr(model, f"_{param_name}")
        elif param.default is not inspect.Parameter.empty:
            raw_val = param.default
        else:
            raw_val = "<unset>"
        params[param_name] = _to_json_safe(raw_val)

    return {
        "model_class": f"{cls.__module__}.{cls.__qualname__}",
        "params": params,
    }


def summarize_eval_results(results: Sequence[EvaluationResult]) -> list[dict[str, Any]]:
    """Convert a sequence of :class:`EvaluationResult` objects into compact v2 brief rows."""
    rows: list[dict[str, Any]] = []
    for res in results:
        s = res.summary
        pr = s.get("pass_rates_pct", {})
        mb = res.memory_breakdown
        total_mb = (
            mb.total_mb if mb is not None else res.model_size_bytes / (1024.0 * 1024.0)
        )
        rows.append(
            {
                "model": res.model_name,
                "dataset": res.dataset_id,
                "direction": res.direction,
                "count": int(s.get("count", 0)),
                "mean_2d_mm": round(float(s["mean_2d_m"]) * 1000.0, 4),
                "rmse_2d_mm": round(float(s["rmse_2d_m"]) * 1000.0, 4),
                "crmse_2d_mm": round(
                    float(s.get("crmse_2d_m", float("nan"))) * 1000.0, 4
                ),
                "iqr_2d_mm": round(
                    float(s.get("iqr_2d_m", float("nan"))) * 1000.0, 4
                ),
                "p95_2d_mm": round(float(s["p95_2d_m"]) * 1000.0, 4),
                "max_2d_mm": round(float(s["max_2d_m"]) * 1000.0, 4),
                "roughness_grad_rms_mm_km": round(
                    float(s.get("roughness_grad_rms_mm_km", float("nan"))), 4
                ),
                "roughness_grad_p95_mm_km": round(
                    float(s.get("roughness_grad_p95_mm_km", float("nan"))), 4
                ),
                "local_jitter_ratio": round(
                    float(s.get("local_jitter_ratio", float("nan"))), 4
                ),
                "learnability_score_mm": round(
                    float(s.get("learnability_score_mm", float("nan"))), 4
                ),
                "pass_lt_1cm_pct": round(float(pr.get("lt_1cm", 0.0)), 2),
                "total_size_mb": round(float(total_mb), 4),
                "fit_time_s": round(float(res.fit_time_sec), 4),
                "eval_time_s": round(float(res.eval_time_sec), 4),
                "throughput_pts_per_s": round(float(res.throughput_pts_per_sec), 1),
            }
        )
    return rows


class ResearchUnit(abc.ABC):
    """Base interface for all self-contained research units in ``experiments/``."""

    SCHEMA_VERSION: str = "2.0"

    def __init__(self, unit_dir: Path) -> None:
        self.unit_dir = Path(unit_dir)
        self.results_dir = self.unit_dir / "results"
        self.plots_dir = self.unit_dir / "plots"
        self.data_dir = self.unit_dir / "data"
        self.models_dir = self.unit_dir / "models"

        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    @property
    @abc.abstractmethod
    def slug(self) -> str:
        """Descriptive snake_case slug (e.g., ``'spg_grid_analysis'``)."""

    @property
    @abc.abstractmethod
    def experiment_type(self) -> ExperimentType:
        """Experiment category (``grid_shift``, ``transformation``, or ``analysis``)."""

    @property
    def unit_name(self) -> str:
        """Backward-compatible alias for :attr:`slug`."""
        return self.slug

    @abc.abstractmethod
    def run(self) -> dict[str, Any]:
        """Execute the unit workflow, produce artifacts, and return summary dict."""

    def build_brief(
        self,
        *,
        status: str = "completed",
        runtime_s: float = 0.0,
        key_metrics: list[dict[str, Any]] | dict[str, Any] | None = None,
        artifacts: dict[str, Any] | None = None,
        benchmark_candidates: list[dict[str, Any]] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build a standardized Schema v2 ``unit_brief.json`` payload."""
        merged_artifacts: dict[str, Any] = {"report": "report.md"}
        if artifacts:
            merged_artifacts.update(artifacts)
        return {
            "schema_version": self.SCHEMA_VERSION,
            "slug": self.slug,
            "experiment_type": self.experiment_type.value,
            "status": status,
            "runtime_s": round(float(runtime_s), 3),
            "key_metrics": key_metrics if key_metrics is not None else [],
            "artifacts": merged_artifacts,
            "benchmark_candidates": benchmark_candidates or [],
            "extra": _to_json_safe(extra) if extra else {},
        }

    def save_brief(self, payload: dict[str, Any]) -> Path:
        """Save standard machine-readable summary brief for AI agent consumption."""
        brief_path = self.unit_dir / "unit_brief.json"
        brief_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return brief_path


class TransformationExperimentUnit(ResearchUnit):
    """Unified orchestrator for coordinate transformation and grid shift experiments.

    Provides standardized access to :class:`GridManager`, :class:`ModelEvaluator`,
    :class:`PlotDispatcher`, and scorecard serialization.
    """

    def __init__(self, unit_dir: Path, cache_grids: bool = True) -> None:
        super().__init__(unit_dir)
        self.grid_manager = GridManager(cache_enabled=cache_grids)
        self.evaluator = ModelEvaluator()
        self.plotter = PlotDispatcher()

    @property
    def experiment_type(self) -> ExperimentType:
        """Return :attr:`ExperimentType.TRANSFORMATION`."""
        return ExperimentType.TRANSFORMATION

    def load_grid(
        self, grid_id: GridId | str, force_regenerate: bool = False
    ) -> GridDataset:
        """Retrieve a :class:`GridDataset` via the unified :class:`GridManager`."""
        return self.grid_manager.get(grid_id, force_regenerate=force_regenerate)

    def evaluate_shift_model(
        self,
        model: BaseGridShiftModel,
        model_name: str,
        dataset_ids: Sequence[GridId | str],
        plots: PlotSpec | None = PlotGroup.STANDARD_DIAGNOSTIC,
        dpi: int = 200,
        print_scorecard: bool = True,
    ) -> list[EvaluationResult]:
        """Evaluate a fitted :class:`BaseGridShiftModel` across one or more datasets and render plots."""
        results: list[EvaluationResult] = []
        for gid in dataset_ids:
            ds = self.load_grid(gid)
            res = self.evaluator.evaluate_shift_model(
                model=model,
                dataset=ds,
                model_name=model_name,
            )
            if print_scorecard:
                print(format_scorecard_table(res.summary, model_name, res.dataset_id))
            if plots is not None:
                rendered = self.plotter.render(
                    eval_result=res,
                    out_dir=self.plots_dir,
                    plots=plots,
                    dpi=dpi,
                )
                res.extra["plots"] = {k.value: v.name for k, v in rendered.items()}
            results.append(res)
        return results

    def evaluate_transformer(
        self,
        transformer: BaseCoordinateTransformer,
        model_name: str,
        dataset_id: GridId | str,
        directions: Sequence[str] = ("etrs89_to_st70", "st70_to_etrs89"),
        plots: PlotSpec | None = PlotGroup.STANDARD_DIAGNOSTIC,
        dpi: int = 200,
        print_scorecard: bool = True,
    ) -> list[EvaluationResult]:
        """Evaluate a :class:`BaseCoordinateTransformer` across specified directions."""
        ds = self.load_grid(dataset_id)
        results: list[EvaluationResult] = []
        for direction in directions:
            res = self.evaluator.evaluate_transformer(
                transformer=transformer,
                dataset=ds,
                direction=direction,
                model_name=model_name,
            )
            if print_scorecard:
                print(format_scorecard_table(res.summary, model_name, direction))
            if plots is not None:
                rendered = self.plotter.render(
                    eval_result=res,
                    out_dir=self.plots_dir,
                    plots=plots,
                    dpi=dpi,
                )
                res.extra["plots"] = {k.value: v.name for k, v in rendered.items()}
            results.append(res)
        return results

    def save_scorecards(
        self,
        results: Sequence[EvaluationResult],
        csv_name: str = "scorecard.csv",
        json_name: str = "scorecard_summary.json",
        md_name: str = "summary_table.md",
    ) -> pd.DataFrame:
        """Persist standardized CSV, JSON, and Markdown scorecards into ``results/``."""
        rows = [r.to_scorecard_row() for r in results]
        df = pd.DataFrame(rows)
        df.to_csv(self.results_dir / csv_name, index=False)
        (self.results_dir / md_name).write_text(
            dataframe_to_markdown(df), encoding="utf-8"
        )
        json_payload = [
            {
                **r.to_scorecard_row(),
                "summary": r.summary,
                "throughput_pts_per_sec": r.throughput_pts_per_sec,
            }
            for r in results
        ]
        (self.results_dir / json_name).write_text(
            json.dumps(json_payload, indent=2), encoding="utf-8"
        )
        return df

    def render_plots(
        self,
        eval_result: EvaluationResult,
        plots: PlotSpec = PlotGroup.STANDARD_DIAGNOSTIC,
        dpi: int = 200,
        transect_models: dict[str, BaseGridShiftModel] | None = None,
    ) -> dict[PlotType, Path]:
        """Convenience wrapper around :meth:`PlotDispatcher.render`."""
        return self.plotter.render(
            eval_result=eval_result,
            out_dir=self.plots_dir,
            plots=plots,
            dpi=dpi,
            transect_models=transect_models,
        )

    def get_transformer_specs(self) -> list[TransformationModelSpec]:
        """Return ordered list of :class:`TransformationModelSpec` entries (optional)."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement get_transformer_specs() or override run()."
        )

    def after_run(
        self,
        transformers: dict[str, BaseCoordinateTransformer],
        results: list[EvaluationResult],
    ) -> dict[str, Any]:
        """Optional hook executed after all transformation specs are evaluated and plotted."""
        return {}

    def run(self) -> dict[str, Any]:
        """Execute the declarative transformation experiment lifecycle."""
        t0 = time.perf_counter()
        print(f"=== Starting Experiment [{self.experiment_type.value}]: {self.slug} ===")

        specs = self.get_transformer_specs()
        all_results: list[EvaluationResult] = []
        plot_items: list[tuple[EvaluationResult, PlotSpec | None]] = []
        kept_transformers: dict[str, BaseCoordinateTransformer] = {}

        for spec in specs:
            res_list = self.evaluate_transformer(
                transformer=spec.transformer,
                model_name=spec.name,
                dataset_id=spec.eval_grid,
                directions=spec.directions,
                plots=None,
            )
            all_results.extend(res_list)
            for res in res_list:
                plot_items.append((res, spec.plots))
            kept_transformers[spec.name] = spec.transformer

        self.save_scorecards(all_results)
        rendered_paths = self.plotter.render_grouped(plot_items, out_dir=self.plots_dir)
        extra_brief = self.after_run(kept_transformers, all_results)

        plot_files: list[str] = [p.name for p in rendered_paths]
        for v in extra_brief.values():
            if isinstance(v, str) and v.endswith(".png") and v not in plot_files:
                plot_files.append(v)

        elapsed = time.perf_counter() - t0
        brief = self.build_brief(
            runtime_s=elapsed,
            key_metrics=summarize_eval_results(all_results),
            artifacts={
                "scorecard_csv": "results/scorecard.csv",
                "scorecard_json": "results/scorecard_summary.json",
                "scorecard_md": "results/summary_table.md",
                "plots": plot_files,
            },
            benchmark_candidates=[
                s.to_benchmark_candidate_dict()
                for s in specs
                if s.benchmark_candidate
            ],
            extra=extra_brief,
        )
        self.save_brief(brief)
        print(f"=== Experiment '{self.slug}' Complete in {elapsed:.2f}s ===")
        return brief


@dataclass
class TransformationModelSpec:
    """Declarative specification for one transformer in a :class:`TransformationExperimentUnit`."""

    name: str
    transformer: BaseCoordinateTransformer
    eval_grid: GridId | str
    directions: Sequence[str] = ("etrs89_to_st70", "st70_to_etrs89")
    plots: PlotSpec | None = PlotGroup.STANDARD_DIAGNOSTIC
    benchmark_candidate: bool = False
    benchmark_suite: str | None = None
    benchmark_label: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_benchmark_candidate_dict(self) -> dict[str, Any]:
        """Serialize benchmark promotion metadata and exact transformer config."""
        cfg = _extract_model_config(self.transformer)
        eval_val = (
            self.eval_grid.value
            if isinstance(self.eval_grid, GridId)
            else self.eval_grid
        )
        return {
            "name": self.name,
            "label": self.benchmark_label or self.name,
            "suite": self.benchmark_suite,
            "eval_grid": eval_val,
            "model_class": cfg["model_class"],
            "params": cfg["params"],
            "extra": _to_json_safe(self.extra),
        }


@dataclass
class ExperimentModelSpec:
    """Declarative specification for one model variant in a :class:`GridShiftExperimentUnit`."""

    name: str
    model: BaseGridShiftModel
    eval_grids: Sequence[GridId | str]
    train_grid: GridId | str | None = None
    plots: PlotSpec | None = PlotGroup.STANDARD_DIAGNOSTIC
    keep_in_memory: bool = False
    persist_model: bool = True
    benchmark_candidate: bool = False
    benchmark_suite: str | None = None
    benchmark_label: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def model_cache_path(self, unit_dir: Path) -> Path:
        """Return canonical checkpoint path ``<unit_dir>/models/<name>.pkl``."""
        return Path(unit_dir) / "models" / f"{self.name}.pkl"

    def to_benchmark_candidate_dict(self) -> dict[str, Any]:
        """Serialize benchmark promotion metadata and exact model config."""
        cfg = _extract_model_config(self.model)
        train_val = (
            self.train_grid.value
            if isinstance(self.train_grid, GridId)
            else self.train_grid
        )
        return {
            "name": self.name,
            "label": self.benchmark_label or self.name,
            "suite": self.benchmark_suite,
            "train_grid": train_val,
            "model_class": cfg["model_class"],
            "params": cfg["params"],
            "extra": _to_json_safe(self.extra),
        }


class GridShiftExperimentUnit(TransformationExperimentUnit):
    """Template Method orchestrator for 2D grid shift experiments.

    Subclasses declare :meth:`get_model_specs` (and an optional
    :meth:`after_run` hook for custom comparison plots such as transects).
    """

    render_extreme_scenarios: bool = True

    @property
    def experiment_type(self) -> ExperimentType:
        """Return :attr:`ExperimentType.GRID_SHIFT`."""
        return ExperimentType.GRID_SHIFT

    @abc.abstractmethod
    def get_model_specs(self) -> list[ExperimentModelSpec]:
        """Return the ordered list of model variants to train and evaluate."""

    def get_extreme_scenario_models(
        self,
        fitted_models: dict[str, BaseGridShiftModel],
    ) -> dict[str, BaseGridShiftModel]:
        """Return ``display_label -> model`` mapping for 9-cell extreme scenario plots."""
        return dict(fitted_models)

    def after_run(
        self,
        fitted_models: dict[str, BaseGridShiftModel],
        results: list[EvaluationResult],
    ) -> dict[str, Any]:
        """Optional hook executed after all specs are trained, evaluated, and saved."""
        return {}

    def run(self, force_retrain: bool = False) -> dict[str, Any]:
        """Execute the full grid-shift experiment lifecycle automatically."""
        t0 = time.perf_counter()
        print(f"=== Starting Experiment [{self.experiment_type.value}]: {self.slug} ===")

        specs = self.get_model_specs()
        all_results: list[EvaluationResult] = []
        plot_items: list[tuple[EvaluationResult, PlotSpec | None]] = []
        kept_models: dict[str, BaseGridShiftModel] = {}

        for spec in specs:
            model = spec.model
            if not model.is_fitted:
                if spec.train_grid is None:
                    raise ValueError(
                        f"Model '{spec.name}' is not fitted and no train_grid was provided."
                    )
                train_grid_str = (
                    spec.train_grid.value
                    if isinstance(spec.train_grid, GridId)
                    else str(spec.train_grid)
                )
                cache_path = (
                    spec.model_cache_path(self.unit_dir)
                    if spec.persist_model
                    else None
                )
                model.load_or_fit(
                    lambda tg=spec.train_grid: (
                        self.load_grid(tg).coords,
                        self.load_grid(tg).true_shifts,
                    ),
                    cache_path=cache_path,
                    train_grid=train_grid_str,
                    force_retrain=force_retrain,
                    label=spec.name,
                )
                mb = model.memory_breakdown()
                print(
                    f"Ready {spec.name} on {train_grid_str} "
                    f"(fit_time={model.fit_time_sec:.3f}s | total={mb.total_mb:.2f} MB)"
                )

            res_list = self.evaluate_shift_model(
                model=model,
                model_name=spec.name,
                dataset_ids=spec.eval_grids,
                plots=None,
            )
            all_results.extend(res_list)
            for res in res_list:
                plot_items.append((res, spec.plots))

            if spec.keep_in_memory:
                kept_models[spec.name] = model
            else:
                del model
                gc.collect()

        self.save_scorecards(all_results)
        rendered_paths = self.plotter.render_grouped(plot_items, out_dir=self.plots_dir)
        plot_files: list[str] = [p.name for p in rendered_paths]

        if self.render_extreme_scenarios:
            scenario_models = self.get_extreme_scenario_models(kept_models)
            if scenario_models:
                g1_path, g2_path = plot_extreme_scenarios_9cells(
                    models=scenario_models,
                    out_dir=self.plots_dir,
                )
                for p_name in (g1_path.name, g2_path.name):
                    if p_name not in plot_files:
                        plot_files.append(p_name)

        extra_brief = self.after_run(kept_models, all_results)

        for v in extra_brief.values():
            if isinstance(v, str) and v.endswith(".png") and v not in plot_files:
                plot_files.append(v)

        elapsed = time.perf_counter() - t0
        brief = self.build_brief(
            runtime_s=elapsed,
            key_metrics=summarize_eval_results(all_results),
            artifacts={
                "scorecard_csv": "results/scorecard.csv",
                "scorecard_json": "results/scorecard_summary.json",
                "scorecard_md": "results/summary_table.md",
                "plots": plot_files,
            },
            benchmark_candidates=[
                s.to_benchmark_candidate_dict()
                for s in specs
                if s.benchmark_candidate
            ],
            extra=extra_brief,
        )
        self.save_brief(brief)
        print(f"=== Experiment '{self.slug}' Complete in {elapsed:.2f}s ===")
        return brief


class AnalysisExperimentUnit(ResearchUnit):
    """Template Method orchestrator for exploratory, geometric, and feature-engineering units."""

    def __init__(self, unit_dir: Path, cache_grids: bool = True) -> None:
        super().__init__(unit_dir)
        self.grid_manager = GridManager(cache_enabled=cache_grids)
        self.plotter = PlotDispatcher()
        self.data_features_dir = Path("data/features")

    @property
    def experiment_type(self) -> ExperimentType:
        """Return :attr:`ExperimentType.ANALYSIS`."""
        return ExperimentType.ANALYSIS

    def load_grid(
        self, grid_id: GridId | str, force_regenerate: bool = False
    ) -> GridDataset:
        """Retrieve a :class:`GridDataset` via the unified :class:`GridManager`."""
        return self.grid_manager.get(grid_id, force_regenerate=force_regenerate)

    def render_plots(
        self,
        eval_result: EvaluationResult,
        plots: PlotSpec = PlotGroup.STANDARD_DIAGNOSTIC,
        dpi: int = 200,
    ) -> dict[PlotType, Path]:
        """Convenience wrapper around :meth:`PlotDispatcher.render`."""
        return self.plotter.render(
            eval_result=eval_result,
            out_dir=self.plots_dir,
            plots=plots,
            dpi=dpi,
        )

    @abc.abstractmethod
    def run_analysis(self) -> dict[str, Any]:
        """Execute the analysis workflow and return ``key_metrics``, ``artifacts``, and ``extra``."""

    def run(self) -> dict[str, Any]:
        """Execute the analysis unit lifecycle and write standardized ``unit_brief.json`` v2."""
        t0 = time.perf_counter()
        print(f"=== Starting Experiment [{self.experiment_type.value}]: {self.slug} ===")

        out = self.run_analysis()
        elapsed = time.perf_counter() - t0
        brief = self.build_brief(
            runtime_s=elapsed,
            key_metrics=out.get("key_metrics", {}),
            artifacts=out.get("artifacts", {}),
            benchmark_candidates=[],
            extra=out.get("extra", {}),
        )
        self.save_brief(brief)
        print(f"=== Experiment '{self.slug}' Complete in {elapsed:.2f}s ===")
        return brief
