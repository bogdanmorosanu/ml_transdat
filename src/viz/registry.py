"""Plot constants, predefined plot groups, and composition resolver."""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum


class PlotType(str, Enum):
    """Canonical identifiers for all supported diagnostic and spatial plots."""

    SPATIAL_ERROR = "spatial_error"
    SPATIAL_ERROR_LOG = "spatial_error_log"
    HIST_ECDF = "hist_ecdf"
    SIGNED_SCATTER = "signed_scatter"
    SUBCELL_HEATMAP = "subcell_heatmap"
    VECTOR_QUIVER = "vector_quiver"
    RESIDUAL_ROUGHNESS = "residual_roughness"
    EXTREME_SCENARIOS_9CELLS = "extreme_scenarios_9cells"
    EXTREME_RESIDUALS_9CELLS = "extreme_residuals_9cells"
    HYBRID_EXTREME_RESIDUALS_9CELLS = "hybrid_extreme_residuals_9cells"
    HYBRID_CANCELLATION_SCATTER = "hybrid_cancellation_scatter"
    HYBRID_SPATIAL_3PANEL = "hybrid_spatial_3panel"
    HYBRID_SUBCELL_PHASE_HEATMAP = "hybrid_subcell_phase_heatmap"
    SURFACE_LEARNABILITY_RANKING = "surface_learnability_ranking"
    POINT_DISTRIBUTION = "point_distribution"
    MODEL_COMPARISON = "model_comparison"


class PlotGroup(str, Enum):
    """Predefined composable plot suites for experiments."""

    MINIMAL = "minimal"
    STANDARD_DIAGNOSTIC = "standard_diagnostic"
    SUBMILLIMETER_DIAGNOSTIC = "submillimeter_diagnostic"
    SURFACE_DIAGNOSTIC = "surface_diagnostic"
    EXTENDED_SHIFT = "extended_shift"
    HYBRID_DEEP_DIVE = "hybrid_deep_dive"


PLOT_GROUP_MEMBERS: dict[PlotGroup, tuple[PlotType, ...]] = {
    PlotGroup.MINIMAL: (
        PlotType.SPATIAL_ERROR,
        PlotType.HIST_ECDF,
    ),
    PlotGroup.STANDARD_DIAGNOSTIC: (
        PlotType.SPATIAL_ERROR,
        PlotType.HIST_ECDF,
        PlotType.SIGNED_SCATTER,
        PlotType.SUBCELL_HEATMAP,
    ),
    PlotGroup.SUBMILLIMETER_DIAGNOSTIC: (
        PlotType.SPATIAL_ERROR_LOG,
        PlotType.HIST_ECDF,
        PlotType.SIGNED_SCATTER,
        PlotType.SUBCELL_HEATMAP,
    ),
    PlotGroup.SURFACE_DIAGNOSTIC: (
        PlotType.SPATIAL_ERROR,
        PlotType.HIST_ECDF,
        PlotType.RESIDUAL_ROUGHNESS,
    ),
    PlotGroup.EXTENDED_SHIFT: (
        PlotType.SPATIAL_ERROR,
        PlotType.HIST_ECDF,
        PlotType.SIGNED_SCATTER,
        PlotType.SUBCELL_HEATMAP,
        PlotType.VECTOR_QUIVER,
    ),
    PlotGroup.HYBRID_DEEP_DIVE: (
        PlotType.HYBRID_EXTREME_RESIDUALS_9CELLS,
        PlotType.HYBRID_CANCELLATION_SCATTER,
        PlotType.HYBRID_SPATIAL_3PANEL,
        PlotType.HYBRID_SUBCELL_PHASE_HEATMAP,
    ),
}

PlotSpec = PlotGroup | PlotType | Sequence[PlotGroup | PlotType]


def resolve_plot_types(spec: PlotSpec) -> list[PlotType]:
    """Expand a :class:`PlotGroup`, :class:`PlotType`, or mixed sequence into an ordered, deduplicated list.

    :param spec: Single plot group/type or a sequence combining groups and custom plot types.
    :type spec: PlotSpec
    :return: Ordered, deduplicated list of :class:`PlotType` values.
    :rtype: list[PlotType]
    """
    items: Sequence[PlotGroup | PlotType] = (
        [spec] if isinstance(spec, (PlotGroup, PlotType)) else spec
    )

    resolved: list[PlotType] = []
    seen: set[PlotType] = set()
    for item in items:
        if isinstance(item, PlotGroup):
            for p_type in PLOT_GROUP_MEMBERS[item]:
                if p_type not in seen:
                    seen.add(p_type)
                    resolved.append(p_type)
        elif isinstance(item, PlotType):
            if item not in seen:
                seen.add(item)
                resolved.append(item)
        else:
            raise TypeError(
                f"Expected PlotType or PlotGroup, got {type(item).__name__}"
            )
    return resolved
