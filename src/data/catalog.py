"""Centralized Grid & Benchmark Catalog, Dataset container, and GridManager."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pytransdat.trans_grid import Grid2D
from src.data.regions import CellRegion, concat_datasets, find_worst_case_region
from src.data.samplers import (
    DEFAULT_SUBGRID_OFFSETS_4X4,
    DEFAULT_SUBGRID_OFFSETS_8X8,
    DEFAULT_SUBGRID_OFFSETS_16X16,
    generate_adaptive_hybrid_train_df,
    generate_adaptive_midpoints_df,
    generate_border_enriched_4x4_df,
    generate_densified_grid_df,
    generate_single_cell_workload_df,
    generate_spg_nodes_df,
    generate_stratified_continuous_df,
    iter_adaptive_midpoint_chunks,
    iter_densified_grid_chunks,
    iter_stratified_continuous_chunks,
)


class GridId(str, Enum):
    """Canonical identifiers for grids and spatial benchmarks in ``ml_transdat``."""

    SPG_NODES_VALID = "spg_nodes_valid"
    RO_GRID_4X4 = "ro_grid_4x4"
    RO_GRID_8X8 = "ro_grid_8x8"
    RO_GRID_16X16 = "ro_grid_16x16"
    ADAPTIVE_2CM_DN = "adaptive_2cm_dn"
    ADAPTIVE_2CM_DE = "adaptive_2cm_de"
    ADAPTIVE_1CM_DN = "adaptive_1cm_dn"
    ADAPTIVE_1CM_DE = "adaptive_1cm_de"
    ADAPTIVE_3MM_DN = "adaptive_3mm_dn"
    ADAPTIVE_3MM_DE = "adaptive_3mm_de"
    ADAPTIVE_1MM_DN = "adaptive_1mm_dn"
    ADAPTIVE_1MM_DE = "adaptive_1mm_de"
    CORRIDOR_SNW = "ro_corridor_snw"
    SINGLE_CELL_10K = "single_cell_10k"
    RO_TEST_UNIFORM_4X4 = "ro_test_uniform_4x4"
    RO_TEST_UNIFORM_8X8 = "ro_test_uniform_8x8"
    RO_TEST_UNIFORM_16X16 = "ro_test_uniform_16x16"
    RO_TEST_ADAPTIVE_3MM_MID = "ro_test_adaptive_3mm_mid"
    RO_TEST_ADAPTIVE_1MM_MID = "ro_test_adaptive_1mm_mid"
    RO_TRAIN_ADAPTIVE_HYBRID = "ro_train_adaptive_hybrid"
    RO_TRAIN_BORDER_ENRICHED_4X4 = "ro_train_border_enriched_4x4"
    CUSTOM = "custom"


@dataclass(frozen=True)
class _GridSpec:
    """Internal specification defining how a grid is located or generated."""

    parquet_path: Path | None = None
    meta_path: Path | None = None
    generator: Callable[[], pd.DataFrame] | None = None
    lazy_offsets: tuple[float, ...] | None = None
    lazy_chunk_generator: (
        Callable[[int], Iterator[tuple[np.ndarray, np.ndarray]]] | None
    ) = None


_GRID_REGISTRY: dict[GridId, _GridSpec] = {
    GridId.SPG_NODES_VALID: _GridSpec(
        parquet_path=Path("data/grids/densified/spg_nodes_valid.parquet"),
        meta_path=Path("data/grids/densified/spg_nodes_valid.meta.json"),
        generator=generate_spg_nodes_df,
    ),
    GridId.RO_GRID_4X4: _GridSpec(
        parquet_path=Path("data/grids/densified/ro_grid_4x4.parquet"),
        meta_path=Path("data/grids/densified/ro_grid_4x4.meta.json"),
        generator=lambda: generate_densified_grid_df(DEFAULT_SUBGRID_OFFSETS_4X4),
        lazy_offsets=DEFAULT_SUBGRID_OFFSETS_4X4,
    ),
    GridId.RO_GRID_8X8: _GridSpec(
        parquet_path=Path("data/grids/densified/ro_grid_8x8.parquet"),
        meta_path=Path("data/grids/densified/ro_grid_8x8.meta.json"),
        generator=lambda: generate_densified_grid_df(DEFAULT_SUBGRID_OFFSETS_8X8),
        lazy_offsets=DEFAULT_SUBGRID_OFFSETS_8X8,
    ),
    GridId.RO_GRID_16X16: _GridSpec(
        parquet_path=Path("data/grids/densified/ro_grid_16x16.parquet"),
        meta_path=Path("data/grids/densified/ro_grid_16x16.meta.json"),
        generator=lambda: generate_densified_grid_df(DEFAULT_SUBGRID_OFFSETS_16X16),
        lazy_offsets=DEFAULT_SUBGRID_OFFSETS_16X16,
    ),
    GridId.ADAPTIVE_2CM_DN: _GridSpec(
        parquet_path=Path("data/grids/adaptive/adaptive_2cm_dn.parquet"),
    ),
    GridId.ADAPTIVE_2CM_DE: _GridSpec(
        parquet_path=Path("data/grids/adaptive/adaptive_2cm_de.parquet"),
    ),
    GridId.ADAPTIVE_1CM_DN: _GridSpec(
        parquet_path=Path("data/grids/adaptive/adaptive_1cm_dn.parquet"),
    ),
    GridId.ADAPTIVE_1CM_DE: _GridSpec(
        parquet_path=Path("data/grids/adaptive/adaptive_1cm_de.parquet"),
    ),
    GridId.ADAPTIVE_3MM_DN: _GridSpec(
        parquet_path=Path("data/grids/adaptive/adaptive_3mm_dn.parquet"),
    ),
    GridId.ADAPTIVE_3MM_DE: _GridSpec(
        parquet_path=Path("data/grids/adaptive/adaptive_3mm_de.parquet"),
    ),
    GridId.ADAPTIVE_1MM_DN: _GridSpec(
        parquet_path=Path("data/grids/adaptive/adaptive_1mm_dn.parquet"),
    ),
    GridId.ADAPTIVE_1MM_DE: _GridSpec(
        parquet_path=Path("data/grids/adaptive/adaptive_1mm_de.parquet"),
    ),
    GridId.CORRIDOR_SNW: _GridSpec(
        parquet_path=Path("data/benchmarks/ro_corridor_snw.parquet"),
        meta_path=Path("data/benchmarks/ro_corridor_snw.meta.json"),
    ),
    GridId.SINGLE_CELL_10K: _GridSpec(
        generator=generate_single_cell_workload_df,
    ),
    GridId.RO_TEST_UNIFORM_4X4: _GridSpec(
        generator=lambda: generate_stratified_continuous_df(strata_per_axis=4, seed=42),
        lazy_chunk_generator=lambda b: iter_stratified_continuous_chunks(
            strata_per_axis=4, seed=42, batch_size=b
        ),
    ),
    GridId.RO_TEST_UNIFORM_8X8: _GridSpec(
        generator=lambda: generate_stratified_continuous_df(strata_per_axis=8, seed=42),
        lazy_chunk_generator=lambda b: iter_stratified_continuous_chunks(
            strata_per_axis=8, seed=42, batch_size=b
        ),
    ),
    GridId.RO_TEST_UNIFORM_16X16: _GridSpec(
        generator=lambda: generate_stratified_continuous_df(
            strata_per_axis=16, seed=42
        ),
        lazy_chunk_generator=lambda b: iter_stratified_continuous_chunks(
            strata_per_axis=16, seed=42, batch_size=b
        ),
    ),
    GridId.RO_TEST_ADAPTIVE_3MM_MID: _GridSpec(
        generator=lambda: generate_adaptive_midpoints_df(tier="3mm"),
        lazy_chunk_generator=lambda b: iter_adaptive_midpoint_chunks(
            tier="3mm", batch_size=b
        ),
    ),
    GridId.RO_TEST_ADAPTIVE_1MM_MID: _GridSpec(
        generator=lambda: generate_adaptive_midpoints_df(tier="1mm"),
        lazy_chunk_generator=lambda b: iter_adaptive_midpoint_chunks(
            tier="1mm", batch_size=b
        ),
    ),
    GridId.RO_TRAIN_ADAPTIVE_HYBRID: _GridSpec(
        generator=generate_adaptive_hybrid_train_df,
    ),
    GridId.RO_TRAIN_BORDER_ENRICHED_4X4: _GridSpec(
        generator=generate_border_enriched_4x4_df,
    ),
}


@dataclass
class GridDataset:
    """Unified container for spatial grid and benchmark datasets.

    Provides standardized access to full coordinate/shift arrays as well as
    chunked generator streaming via :meth:`iter_chunks`.
    """

    grid_id: GridId
    df: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)
    _cached_shifts: np.ndarray | None = field(default=None, init=False, repr=False)

    def __len__(self) -> int:
        return len(self.df)

    @property
    def coords(self) -> np.ndarray:
        """Return Stereo 70 planar coordinates of shape ``(N, 2)`` as ``[st70_n, st70_e]``."""
        return self.df[["st70_n", "st70_e"]].to_numpy(dtype=np.float64)

    @property
    def etrs89_rad(self) -> np.ndarray:
        """Return ETRS89 geographic coordinates of shape ``(N, 2)`` as ``[lat_rad, lon_rad]``."""
        return self.df[["etrs89_lat_rad", "etrs89_lon_rad"]].to_numpy(dtype=np.float64)

    @property
    def true_shifts(self) -> np.ndarray:
        """Return ground-truth 2D SPG shifts of shape ``(N, 2)`` as ``[shift_dn_m, shift_de_m]``.

        If shift columns are not pre-computed in the DataFrame, interpolates them
        on demand using the official ``pytransdat.trans_grid.Grid2D`` bicubic engine
        and caches the result.
        """
        if self._cached_shifts is not None:
            return self._cached_shifts

        if "shift_dn_m" in self.df.columns and "shift_de_m" in self.df.columns:
            shifts = self.df[["shift_dn_m", "shift_de_m"]].to_numpy(dtype=np.float64)
        else:
            grid2d = Grid2D()
            dn_list, de_list = grid2d.interp(
                self.df["st70_n"].tolist(),
                self.df["st70_e"].tolist(),
            )
            shifts = np.column_stack(
                [
                    np.asarray(dn_list, dtype=np.float64),
                    np.asarray(de_list, dtype=np.float64),
                ]
            )
        self._cached_shifts = shifts
        return shifts

    def iter_chunks(
        self, batch_size: int = 8192
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(coords_chunk, shifts_chunk)`` pairs as a Python generator.

        :param batch_size: Number of points per yielded chunk.
        :type batch_size: int
        :yield: Tuple of ``(coords, shifts)`` arrays of shape ``(B, 2)``.
        :rtype: Iterator[tuple[np.ndarray, np.ndarray]]
        :raises ValueError: If ``batch_size`` is not positive.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        coords_all = self.coords
        shifts_all = self.true_shifts
        total = len(coords_all)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            yield coords_all[start:end], shifts_all[start:end]

    def slice_region(
        self,
        region: CellRegion,
        *,
        include_upper_boundary: bool = True,
        grid2d: Grid2D | None = None,
    ) -> GridDataset:
        """Extract a cell-aligned spatial subgrid preserving exact grid step and shifts.

        :param region: Target :class:`CellRegion` specifying the SPG cell window.
        :type region: CellRegion
        :param include_upper_boundary: Include points lying on the upper ``n_max``/``e_max`` edges.
        :type include_upper_boundary: bool
        :param grid2d: Optional pre-initialized ``Grid2D`` instance.
        :type grid2d: Grid2D | None
        :return: New :class:`GridDataset` containing only points inside ``region``.
        :rtype: GridDataset
        """
        n_min, n_max, e_min, e_max = region.bounds_st70(grid2d)
        eps = 1e-6
        st_n = self.df["st70_n"]
        st_e = self.df["st70_e"]
        if include_upper_boundary:
            mask = (
                (st_n >= n_min - eps)
                & (st_n <= n_max + eps)
                & (st_e >= e_min - eps)
                & (st_e <= e_max + eps)
            )
        else:
            mask = (
                (st_n >= n_min - eps)
                & (st_n < n_max - eps)
                & (st_e >= e_min - eps)
                & (st_e < e_max - eps)
            )
        sub_df = self.df.loc[mask].reset_index(drop=True)
        meta = dict(self.metadata)
        meta["region"] = {
            "name": region.name,
            "row_start": region.row_start,
            "col_start": region.col_start,
            "n_rows": region.n_rows,
            "n_cols": region.n_cols,
            "bounds_st70": (n_min, n_max, e_min, e_max),
        }
        return GridDataset(grid_id=self.grid_id, df=sub_df, metadata=meta)


class GridManager:
    """Unified registry and loader for static and on-the-fly grids.

    Supports in-memory caching and optional local Parquet disk caching
    (e.g. ``.cache/grids/``) so dynamic grids are generated once per machine
    without polluting version control.
    """

    def __init__(
        self,
        cache_enabled: bool = True,
        disk_cache_dir: Path | None = Path(".cache/grids"),
    ) -> None:
        self._cache_enabled = cache_enabled
        self._disk_cache_dir = disk_cache_dir
        self._cache: dict[GridId, GridDataset] = {}
        self._region_cache: dict[tuple[GridId, CellRegion, bool], GridDataset] = {}

    def clear_cache(self) -> None:
        """Clear all in-memory cached datasets and regional slices."""
        self._cache.clear()
        self._region_cache.clear()

    def get_region(
        self,
        grid_id: GridId | str,
        region: CellRegion,
        *,
        include_upper_boundary: bool = True,
        force_regenerate: bool = False,
    ) -> GridDataset:
        """Retrieve a cell-aligned :class:`GridDataset` slice for ``region``."""
        gid = GridId(grid_id)
        cache_key = (gid, region, include_upper_boundary)
        if (
            self._cache_enabled
            and not force_regenerate
            and cache_key in self._region_cache
        ):
            return self._region_cache[cache_key]

        full_ds = self.get(gid, force_regenerate=force_regenerate)
        sub_ds = full_ds.slice_region(
            region, include_upper_boundary=include_upper_boundary
        )
        if self._cache_enabled:
            self._region_cache[cache_key] = sub_ds
        return sub_ds

    def get(
        self,
        grid_id: GridId | str,
        force_regenerate: bool = False,
    ) -> GridDataset:
        """Retrieve a :class:`GridDataset` by identifier.

        Resolution order (unless ``force_regenerate=True``):
        1. In-memory cache
        2. Static Parquet path (if registered and present on disk)
        3. Local disk cache (``disk_cache_dir/<grid_id>.parquet``)
        4. On-the-fly dynamic generator (and persisted to ``disk_cache_dir`` if enabled)

        :param grid_id: Canonical :class:`GridId` or its string value.
        :type grid_id: GridId | str
        :param force_regenerate: If True and a generator exists, regenerate dynamically.
        :type force_regenerate: bool
        :return: Loaded or generated :class:`GridDataset`.
        :rtype: GridDataset
        :raises KeyError: If ``grid_id`` is not registered.
        :raises FileNotFoundError: If a static-only grid file is missing on disk.
        """
        gid = GridId(grid_id)
        if self._cache_enabled and not force_regenerate and gid in self._cache:
            return self._cache[gid]

        if gid not in _GRID_REGISTRY:
            raise KeyError(f"Unregistered grid id: {gid}")
        spec = _GRID_REGISTRY[gid]

        meta: dict[str, Any] = {}
        if spec.meta_path is not None and spec.meta_path.is_file():
            meta = json.loads(spec.meta_path.read_text(encoding="utf-8"))

        disk_cache_file = (
            self._disk_cache_dir / f"{gid.value}.parquet"
            if self._disk_cache_dir is not None
            else None
        )

        if (
            not force_regenerate
            and spec.parquet_path is not None
            and spec.parquet_path.is_file()
        ):
            df = pd.read_parquet(spec.parquet_path)
            source_mode = "static_parquet"
        elif (
            not force_regenerate
            and disk_cache_file is not None
            and disk_cache_file.is_file()
        ):
            df = pd.read_parquet(disk_cache_file)
            source_mode = "disk_cache_parquet"
        elif spec.generator is not None:
            df = spec.generator()
            source_mode = "generated_on_the_fly"
            if disk_cache_file is not None:
                disk_cache_file.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(disk_cache_file, index=False)
        else:
            raise FileNotFoundError(
                f"Static grid file not found for {gid.value}: {spec.parquet_path}"
            )

        meta.setdefault("grid_id", gid.value)
        meta["source_mode"] = source_mode
        dataset = GridDataset(grid_id=gid, df=df, metadata=meta)

        if self._cache_enabled:
            self._cache[gid] = dataset
        return dataset

    def iter_chunks(
        self,
        grid_id: GridId | str,
        batch_size: int = 8192,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Stream ``(coords, shifts)`` chunks for ``grid_id``.

        Uses true lazy cell-by-cell generation for uncached densified grids, or
        chunks from the cached/static :class:`GridDataset` when already materialized.
        """
        gid = GridId(grid_id)
        if gid not in _GRID_REGISTRY:
            raise KeyError(f"Unregistered grid id: {gid}")
        spec = _GRID_REGISTRY[gid]

        disk_cache_file = (
            self._disk_cache_dir / f"{gid.value}.parquet"
            if self._disk_cache_dir is not None
            else None
        )
        is_materialized = (
            gid in self._cache
            or (spec.parquet_path is not None and spec.parquet_path.is_file())
            or (disk_cache_file is not None and disk_cache_file.is_file())
        )
        if not is_materialized:
            if spec.lazy_chunk_generator is not None:
                return spec.lazy_chunk_generator(batch_size)
            if spec.lazy_offsets is not None:
                return iter_densified_grid_chunks(
                    spec.lazy_offsets, batch_size=batch_size
                )
        return self.get(gid).iter_chunks(batch_size=batch_size)

    def get_adaptive_pair(self, tier: str) -> tuple[GridDataset, GridDataset]:
        """Load the ``(dn, de)`` adaptive quadtree grid pair for a tolerance tier.

        :param tier: Tolerance tier (``'2cm'``, ``'1cm'``, ``'3mm'``, or ``'1mm'``).
        :type tier: str
        :return: Tuple ``(dn_dataset, de_dataset)``.
        :rtype: tuple[GridDataset, GridDataset]
        :raises ValueError: If ``tier`` is not a recognized adaptive tier.
        """
        tier_norm = tier.lower()
        mapping: dict[str, tuple[GridId, GridId]] = {
            "2cm": (GridId.ADAPTIVE_2CM_DN, GridId.ADAPTIVE_2CM_DE),
            "1cm": (GridId.ADAPTIVE_1CM_DN, GridId.ADAPTIVE_1CM_DE),
            "3mm": (GridId.ADAPTIVE_3MM_DN, GridId.ADAPTIVE_3MM_DE),
            "1mm": (GridId.ADAPTIVE_1MM_DN, GridId.ADAPTIVE_1MM_DE),
        }
        if tier_norm not in mapping:
            raise ValueError(
                f"Unknown adaptive tier '{tier}'. Expected one of {list(mapping)}."
            )
        dn_id, de_id = mapping[tier_norm]
        return self.get(dn_id), self.get(de_id)


_DEFAULT_GRID_MANAGER: GridManager | None = None


def get_default_grid_manager() -> GridManager:
    """Return a shared singleton :class:`GridManager` instance."""
    global _DEFAULT_GRID_MANAGER
    if _DEFAULT_GRID_MANAGER is None:
        _DEFAULT_GRID_MANAGER = GridManager()
    return _DEFAULT_GRID_MANAGER


__all__ = [
    "CellRegion",
    "GridDataset",
    "GridId",
    "GridManager",
    "concat_datasets",
    "find_worst_case_region",
    "get_default_grid_manager",
]
