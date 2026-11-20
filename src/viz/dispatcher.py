"""Declarative PlotDispatcher mapping PlotType / PlotGroup specs to rendered figures."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation.evaluator import EvaluationResult
from src.models.base import BaseGridShiftModel
from src.viz.plots import (
    plot_hist_ecdf,
    plot_signed_scatter,
    plot_spatial_error,
    plot_subcell_heatmap,
    plot_transect_66km,
    plot_vector_quiver,
)
from src.viz.registry import PlotGroup, PlotSpec, PlotType, resolve_plot_types

_PlotHandler = Callable[
    [
        EvaluationResult,
        Path,
        int,
        dict[str, BaseGridShiftModel] | None,
        dict[str, Any],
    ],
    Path | None,
]


def _handle_spatial_error(
    res: EvaluationResult,
    out_path: Path,
    dpi: int,
    _transect: dict[str, BaseGridShiftModel] | None,
    _ctx: dict[str, Any],
) -> Path:
    return plot_spatial_error(
        res.err_df, res.summary, out_path, res.model_name, res.direction, dpi=dpi
    )


def _handle_hist_ecdf(
    res: EvaluationResult,
    out_path: Path,
    dpi: int,
    _transect: dict[str, BaseGridShiftModel] | None,
    _ctx: dict[str, Any],
) -> Path:
    return plot_hist_ecdf(
        res.err_df, res.summary, out_path, res.model_name, res.direction, dpi=dpi
    )


def _handle_signed_scatter(
    res: EvaluationResult,
    out_path: Path,
    dpi: int,
    _transect: dict[str, BaseGridShiftModel] | None,
    _ctx: dict[str, Any],
) -> Path:
    return plot_signed_scatter(
        res.err_df, res.summary, out_path, res.model_name, res.direction, dpi=dpi
    )


def _handle_subcell_heatmap(
    res: EvaluationResult,
    out_path: Path,
    dpi: int,
    _transect: dict[str, BaseGridShiftModel] | None,
    _ctx: dict[str, Any],
) -> Path | None:
    return plot_subcell_heatmap(
        res.err_df, out_path, res.model_name, res.direction, dpi=dpi
    )


def _handle_vector_quiver(
    res: EvaluationResult,
    out_path: Path,
    dpi: int,
    _transect: dict[str, BaseGridShiftModel] | None,
    ctx: dict[str, Any],
) -> Path:
    ee = res.err_df["err_e_m"].to_numpy(dtype=np.float64)
    en = res.err_df["err_n_m"].to_numpy(dtype=np.float64)
    mag = res.err_df["err_2d_m"].to_numpy(dtype=np.float64)
    return plot_vector_quiver(
        res.err_df,
        de=ee,
        dn=en,
        mag=mag,
        out_path=out_path,
        title=f"{res.model_name} ({res.direction}) — Residual Quiver",
        stride=int(ctx.get("quiver_stride", 2)),
        dpi=dpi,
    )


def _handle_transect_66km(
    _res: EvaluationResult,
    out_path: Path,
    dpi: int,
    transect_models: dict[str, BaseGridShiftModel] | None,
    _ctx: dict[str, Any],
) -> Path:
    if not transect_models:
        raise ValueError("PlotType.TRANSECT_66KM requires non-empty `transect_models`.")
    return plot_transect_66km(models=transect_models, out_path=out_path, dpi=dpi)


_SINGLE_RESULT_HANDLERS: dict[PlotType, _PlotHandler] = {
    PlotType.SPATIAL_ERROR: _handle_spatial_error,
    PlotType.HIST_ECDF: _handle_hist_ecdf,
    PlotType.SIGNED_SCATTER: _handle_signed_scatter,
    PlotType.SUBCELL_HEATMAP: _handle_subcell_heatmap,
    PlotType.VECTOR_QUIVER: _handle_vector_quiver,
    PlotType.TRANSECT_66KM: _handle_transect_66km,
}


class PlotDispatcher:
    """Renders requested :class:`PlotType` / :class:`PlotGroup` sets from an :class:`EvaluationResult`."""

    def render(
        self,
        eval_result: EvaluationResult,
        out_dir: Path,
        plots: PlotSpec = PlotGroup.STANDARD_DIAGNOSTIC,
        dpi: int = 200,
        transect_models: dict[str, BaseGridShiftModel] | None = None,
        extra_context: dict[str, Any] | None = None,
        include_dataset_id: bool = True,
    ) -> dict[PlotType, Path]:
        """Generate all resolved plots for an evaluation result.

        :param eval_result: Evaluation output from :class:`ModelEvaluator`.
        :type eval_result: EvaluationResult
        :param out_dir: Target directory for output PNG files.
        :type out_dir: Path
        :param plots: A :class:`PlotGroup`, :class:`PlotType`, or mixed sequence.
        :type plots: PlotSpec
        :param dpi: Figure DPI.
        :type dpi: int
        :param transect_models: Optional dict of models for :attr:`PlotType.TRANSECT_66KM`.
        :type transect_models: dict[str, BaseGridShiftModel] | None
        :param extra_context: Optional extra keyword parameters for specific plots.
        :type extra_context: dict[str, Any] | None
        :param include_dataset_id: Whether to include ``dataset_id`` in the output filename.
        :type include_dataset_id: bool
        :return: Mapping of rendered :class:`PlotType` to saved file paths.
        :rtype: dict[PlotType, Path]
        :raises ValueError: If a requested :class:`PlotType` is not supported on single results
            or is missing required context.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        resolved = resolve_plot_types(plots)
        prefix = (
            f"{eval_result.model_name}_{eval_result.dataset_id}_{eval_result.direction}"
            if include_dataset_id
            else f"{eval_result.model_name}_{eval_result.direction}"
        )
        rendered: dict[PlotType, Path] = {}
        ctx = extra_context or {}

        for p_type in resolved:
            handler = _SINGLE_RESULT_HANDLERS.get(p_type)
            if handler is None:
                raise ValueError(
                    f"PlotType.{p_type.name} is not supported by PlotDispatcher.render() "
                    "on a single EvaluationResult."
                )
            out_path = out_dir / f"{prefix}_{p_type.value}.png"
            saved_path = handler(eval_result, out_path, dpi, transect_models, ctx)
            if saved_path is not None:
                rendered[p_type] = saved_path

        return rendered
