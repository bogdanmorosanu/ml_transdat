"""Declarative visualization registry, plot groups, and dispatcher."""

from src.viz.benchmarks import (
    plot_memory_breakdown_stacked,
    plot_model_comparison,
    plot_multi_dataset_comparison,
    plot_speed_workloads,
)
from src.viz.diagnostics import (
    plot_hist_ecdf,
    plot_hist_ecdf_rows,
    plot_point_distribution,
    plot_signed_scatter,
    plot_signed_scatter_rows,
    plot_spatial_error,
    plot_spatial_error_rows,
    plot_subcell_heatmap,
    plot_subcell_heatmap_rows,
    plot_vector_quiver,
    plot_vector_quiver_rows,
)
from src.viz.dispatcher import PlotDispatcher
from src.viz.registry import PlotGroup, PlotSpec, PlotType, resolve_plot_types
from src.viz.scenarios import (
    EXTREME_SCENARIOS_GROUP1_GRADIENT_LINEAR,
    EXTREME_SCENARIOS_GROUP2_CURVATURE_KINKS,
    eval_pytransdat_1d_safe,
    plot_extreme_residuals_9cells,
    plot_extreme_scenarios_9cells,
)
from src.viz.surfaces import (
    plot_residual_roughness_rows,
    plot_surface_comparison_grids,
    plot_surface_learnability_ranking,
)

__all__ = [
    "EXTREME_SCENARIOS_GROUP1_GRADIENT_LINEAR",
    "EXTREME_SCENARIOS_GROUP2_CURVATURE_KINKS",
    "PlotDispatcher",
    "PlotGroup",
    "PlotSpec",
    "PlotType",
    "eval_pytransdat_1d_safe",
    "plot_extreme_residuals_9cells",
    "plot_extreme_scenarios_9cells",
    "plot_hist_ecdf",
    "plot_hist_ecdf_rows",
    "plot_memory_breakdown_stacked",
    "plot_model_comparison",
    "plot_multi_dataset_comparison",
    "plot_point_distribution",
    "plot_residual_roughness_rows",
    "plot_signed_scatter",
    "plot_signed_scatter_rows",
    "plot_spatial_error",
    "plot_spatial_error_rows",
    "plot_speed_workloads",
    "plot_subcell_heatmap",
    "plot_subcell_heatmap_rows",
    "plot_surface_comparison_grids",
    "plot_surface_learnability_ranking",
    "plot_vector_quiver",
    "plot_vector_quiver_rows",
    "resolve_plot_types",
]
