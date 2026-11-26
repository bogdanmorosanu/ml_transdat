"""Data management, grid catalog, and spatial samplers for ``ml_transdat``."""

from src.data.boundaries import (
    BoundaryId,
    SpatialBoundary,
    get_boundary,
)
from src.data.catalog import (
    GridDataset,
    GridId,
    GridManager,
    get_default_grid_manager,
)
from src.data.regions import (
    CellRegion,
    concat_datasets,
    find_worst_case_region,
    generate_region_densified_dataset,
)
from src.data.samplers import (
    DEFAULT_SUBGRID_OFFSETS_4X4,
    DEFAULT_SUBGRID_OFFSETS_8X8,
    DEFAULT_SUBGRID_OFFSETS_16X16,
    DEFAULT_SUBGRID_OFFSETS_32X32,
    build_grid_candidates,
    generate_adaptive_midpoints_df,
    generate_densified_grid_df,
    generate_single_cell_workload_df,
    generate_spg_nodes_df,
    generate_stratified_continuous_df,
    get_valid_spg_cells,
    iter_adaptive_midpoint_chunks,
    iter_densified_grid_chunks,
    iter_stratified_continuous_chunks,
    sample_corridor_candidates,
)
from src.data.spg_cache import get_default_grid2d, get_default_spg_reader

__all__ = [
    "DEFAULT_SUBGRID_OFFSETS_4X4",
    "DEFAULT_SUBGRID_OFFSETS_8X8",
    "DEFAULT_SUBGRID_OFFSETS_16X16",
    "DEFAULT_SUBGRID_OFFSETS_32X32",
    "BoundaryId",
    "CellRegion",
    "GridDataset",
    "GridId",
    "GridManager",
    "SpatialBoundary",
    "build_grid_candidates",
    "concat_datasets",
    "find_worst_case_region",
    "generate_adaptive_midpoints_df",
    "generate_densified_grid_df",
    "generate_region_densified_dataset",
    "generate_single_cell_workload_df",
    "generate_spg_nodes_df",
    "generate_stratified_continuous_df",
    "get_boundary",
    "get_default_grid2d",
    "get_default_grid_manager",
    "get_default_spg_reader",
    "get_valid_spg_cells",
    "iter_adaptive_midpoint_chunks",
    "iter_densified_grid_chunks",
    "iter_stratified_continuous_chunks",
    "sample_corridor_candidates",
]
