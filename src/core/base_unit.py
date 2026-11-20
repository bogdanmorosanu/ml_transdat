"""Abstract base classes and unified experiment orchestrators for Research Units."""

from __future__ import annotations

import abc
import gc
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.catalog import GridDataset, GridId, GridManager
from src.evaluation.evaluator import EvaluationResult, ModelEvaluator
from src.evaluation.metrics import dataframe_to_markdown, format_scorecard_table
from src.models.base import BaseCoordinateTransformer, BaseGridShiftModel
from src.viz.dispatcher import PlotDispatcher
from src.viz.registry import PlotGroup, PlotSpec, PlotType


class ResearchUnit(abc.ABC):
    """Base interface for all self-contained research units in ``experiments/``."""

    def __init__(self, unit_dir: Path) -> None:
        self.unit_dir = Path(unit_dir)
        self.results_dir = self.unit_dir / "results"
        self.plots_dir = self.unit_dir / "plots"
        self.data_dir = self.unit_dir / "data"

        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    @property
    @abc.abstractmethod
    def unit_id(self) -> str:
        """Sequential unit identifier (e.g., '01', '02')."""

    @property
    @abc.abstractmethod
    def unit_name(self) -> str:
        """Descriptive slug (e.g., 'spg_grid_analysis')."""

    @abc.abstractmethod
    def run(self) -> dict[str, Any]:
        """Execute the unit workflow, produce artifacts, and return summary dict."""

    def save_brief(self, payload: dict[str, Any]) -> Path:
        """Save standard machine-readable summary brief for AI agent consumption."""
        brief_path = self.unit_dir / "unit_brief.json"
        brief_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
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


@dataclass
class ExperimentModelSpec:
    """Declarative specification for one model variant in a :class:`GridShiftExperimentUnit`."""

    name: str
    model: BaseGridShiftModel
    eval_grids: Sequence[GridId | str]
    train_grid: GridId | str | None = None
    plots: PlotSpec | None = PlotGroup.STANDARD_DIAGNOSTIC
    keep_in_memory: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


class GridShiftExperimentUnit(TransformationExperimentUnit):
    """Template Method orchestrator for 2D grid shift experiments.

    Subclasses only declare :meth:`get_model_specs` (and an optional
    :meth:`after_run` hook for custom comparison plots such as transects).
    """

    @abc.abstractmethod
    def get_model_specs(self) -> list[ExperimentModelSpec]:
        """Return the ordered list of model variants to train and evaluate."""

    def after_run(
        self,
        fitted_models: dict[str, BaseGridShiftModel],
        results: list[EvaluationResult],
    ) -> dict[str, Any]:
        """Optional hook executed after all specs are trained, evaluated, and saved."""
        return {}

    def run(self) -> dict[str, Any]:
        """Execute the full grid-shift experiment lifecycle automatically."""
        t0 = time.perf_counter()
        print(f"=== Starting Research Unit {self.unit_id}: {self.unit_name} ===")

        specs = self.get_model_specs()
        all_results: list[EvaluationResult] = []
        kept_models: dict[str, BaseGridShiftModel] = {}

        for spec in specs:
            model = spec.model
            if not model.is_fitted:
                if spec.train_grid is None:
                    raise ValueError(
                        f"Model '{spec.name}' is not fitted and no train_grid was provided."
                    )
                train_ds = self.load_grid(spec.train_grid)
                model.fit(train_ds.coords, train_ds.true_shifts)
                mb = model.memory_breakdown()
                print(
                    f"Fitted {spec.name} on {train_ds.grid_id.value} "
                    f"in {model.fit_time_sec:.3f}s | total={mb.total_mb:.2f} MB"
                )

            res_list = self.evaluate_shift_model(
                model=model,
                model_name=spec.name,
                dataset_ids=spec.eval_grids,
                plots=spec.plots,
            )
            all_results.extend(res_list)

            if spec.keep_in_memory:
                kept_models[spec.name] = model
            else:
                del model
                gc.collect()

        self.save_scorecards(all_results)
        extra_brief = self.after_run(kept_models, all_results)

        elapsed = time.perf_counter() - t0
        brief: dict[str, Any] = {
            "unit_id": self.unit_id,
            "unit_name": self.unit_name,
            "models_evaluated": [f"{r.model_name}@{r.dataset_id}" for r in all_results],
            "runtime_s": round(elapsed, 3),
            **extra_brief,
        }
        self.save_brief(brief)
        print(f"=== Research Unit {self.unit_id} Complete in {elapsed:.2f}s ===")
        return brief
