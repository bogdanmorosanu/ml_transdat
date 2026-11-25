"""Declarative PlotDispatcher rendering unified 1-row-per-model figures per PlotType."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from src.evaluation.evaluator import EvaluationResult
from src.viz.diagnostics import (
    plot_hist_ecdf_rows,
    plot_signed_scatter_rows,
    plot_spatial_error_rows,
    plot_subcell_heatmap_rows,
    plot_vector_quiver_rows,
)
from src.viz.registry import PlotGroup, PlotSpec, PlotType, resolve_plot_types
from src.viz.surfaces import plot_residual_roughness_rows

_GroupedPlotHandler = Callable[
    [Sequence[EvaluationResult], Path, int, dict[str, Any]],
    Path | None,
]


def _handle_spatial_error_rows(
    results: Sequence[EvaluationResult],
    out_path: Path,
    dpi: int,
    _ctx: dict[str, Any],
) -> Path:
    return plot_spatial_error_rows(results, out_path, dpi=dpi)


def _handle_spatial_error_log_rows(
    results: Sequence[EvaluationResult],
    out_path: Path,
    dpi: int,
    _ctx: dict[str, Any],
) -> Path:
    return plot_spatial_error_rows(results, out_path, dpi=dpi, log_scale=True)


def _handle_hist_ecdf_rows(
    results: Sequence[EvaluationResult],
    out_path: Path,
    dpi: int,
    _ctx: dict[str, Any],
) -> Path:
    return plot_hist_ecdf_rows(results, out_path, dpi=dpi)


def _handle_signed_scatter_rows(
    results: Sequence[EvaluationResult],
    out_path: Path,
    dpi: int,
    _ctx: dict[str, Any],
) -> Path:
    return plot_signed_scatter_rows(results, out_path, dpi=dpi)


def _handle_subcell_heatmap_rows(
    results: Sequence[EvaluationResult],
    out_path: Path,
    dpi: int,
    _ctx: dict[str, Any],
) -> Path | None:
    return plot_subcell_heatmap_rows(results, out_path, dpi=dpi)


def _handle_vector_quiver_rows(
    results: Sequence[EvaluationResult],
    out_path: Path,
    dpi: int,
    ctx: dict[str, Any],
) -> Path:
    return plot_vector_quiver_rows(
        results,
        out_path=out_path,
        stride=int(ctx.get("quiver_stride", 2)),
        dpi=dpi,
    )


def _handle_residual_roughness_rows(
    results: Sequence[EvaluationResult],
    out_path: Path,
    dpi: int,
    _ctx: dict[str, Any],
) -> Path:
    return plot_residual_roughness_rows(results, out_path=out_path, dpi=dpi)


_GROUPED_HANDLERS: dict[PlotType, _GroupedPlotHandler] = {
    PlotType.SPATIAL_ERROR: _handle_spatial_error_rows,
    PlotType.SPATIAL_ERROR_LOG: _handle_spatial_error_log_rows,
    PlotType.HIST_ECDF: _handle_hist_ecdf_rows,
    PlotType.SIGNED_SCATTER: _handle_signed_scatter_rows,
    PlotType.SUBCELL_HEATMAP: _handle_subcell_heatmap_rows,
    PlotType.VECTOR_QUIVER: _handle_vector_quiver_rows,
    PlotType.RESIDUAL_ROUGHNESS: _handle_residual_roughness_rows,
}


class PlotDispatcher:
    """Renders requested :class:`PlotType` / :class:`PlotGroup` sets as unified multi-model figures."""

    def render_grouped(
        self,
        items: Sequence[tuple[EvaluationResult, PlotSpec | None]],
        out_dir: Path,
        dpi: int = 200,
        extra_context: dict[str, Any] | None = None,
    ) -> list[Path]:
        """Group evaluation results by ``(dataset_id, direction, plot_type)`` into 1-row-per-model PNGs.

        :param items: Ordered sequence of ``(EvaluationResult, PlotSpec | None)`` pairs.
        :type items: Sequence[tuple[EvaluationResult, PlotSpec | None]]
        :param out_dir: Target directory for output PNG files.
        :type out_dir: Path
        :param dpi: Figure DPI.
        :type dpi: int
        :param extra_context: Optional extra keyword parameters for specific plots.
        :type extra_context: dict[str, Any] | None
        :return: Ordered list of rendered PNG file paths.
        :rtype: list[Path]
        :raises ValueError: If an unsupported :class:`PlotType` is requested.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        ctx = extra_context or {}

        buckets: dict[tuple[str, str, PlotType], list[EvaluationResult]] = {}

        for eval_res, spec in items:
            if spec is None:
                continue
            for p_type in resolve_plot_types(spec):
                if p_type not in _GROUPED_HANDLERS:
                    raise ValueError(
                        f"PlotType.{p_type.name} is not supported by PlotDispatcher.render_grouped()."
                    )
                key = (eval_res.dataset_id, eval_res.direction, p_type)
                buckets.setdefault(key, []).append(eval_res)

        rendered_paths: list[Path] = []
        for (dataset_id, direction, p_type), group_results in buckets.items():
            handler = _GROUPED_HANDLERS[p_type]
            out_path = out_dir / f"{dataset_id}_{direction}_{p_type.value}.png"
            saved_path = handler(group_results, out_path, dpi, ctx)
            if saved_path is not None:
                rendered_paths.append(saved_path)
                for res in group_results:
                    res.extra.setdefault("plots", {})[p_type.value] = saved_path.name

        return rendered_paths

    def render(
        self,
        eval_result: EvaluationResult,
        out_dir: Path,
        plots: PlotSpec = PlotGroup.STANDARD_DIAGNOSTIC,
        dpi: int = 200,
        extra_context: dict[str, Any] | None = None,
        include_dataset_id: bool = True,
    ) -> dict[PlotType, Path]:
        """Render requested plots for a single :class:`EvaluationResult` via :meth:`render_grouped`."""
        del include_dataset_id
        self.render_grouped(
            items=[(eval_result, plots)],
            out_dir=out_dir,
            dpi=dpi,
            extra_context=extra_context,
        )
        out: dict[PlotType, Path] = {}
        for p_key, fname in eval_result.extra.get("plots", {}).items():
            out[PlotType(p_key)] = out_dir / fname
        return out
