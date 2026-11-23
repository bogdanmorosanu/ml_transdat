"""Declarative visualization registry, plot groups, and dispatcher."""

from src.viz.dispatcher import PlotDispatcher
from src.viz.plots import (
    plot_hist_ecdf,
    plot_hist_ecdf_rows,
    plot_memory_breakdown_stacked,
    plot_model_comparison,
    plot_multi_dataset_comparison,
    plot_point_distribution,
    plot_signed_scatter,
    plot_signed_scatter_rows,
    plot_spatial_error,
    plot_spatial_error_rows,
    plot_speed_workloads,
    plot_subcell_heatmap,
    plot_subcell_heatmap_rows,
    plot_surface_comparison_grids,
    plot_transect_66km,
    plot_vector_quiver,
    plot_vector_quiver_rows,
)
from src.viz.registry import PlotGroup, PlotSpec, PlotType, resolve_plot_types

__all__ = [
    "PlotDispatcher",
    "PlotGroup",
    "PlotSpec",
    "PlotType",
    "plot_hist_ecdf",
    "plot_hist_ecdf_rows",
    "plot_memory_breakdown_stacked",
    "plot_model_comparison",
    "plot_multi_dataset_comparison",
    "plot_point_distribution",
    "plot_signed_scatter",
    "plot_signed_scatter_rows",
    "plot_spatial_error",
    "plot_spatial_error_rows",
    "plot_speed_workloads",
    "plot_subcell_heatmap",
    "plot_subcell_heatmap_rows",
    "plot_surface_comparison_grids",
    "plot_transect_66km",
    "plot_vector_quiver",
    "plot_vector_quiver_rows",
    "resolve_plot_types",
]
