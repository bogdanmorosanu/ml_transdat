"""Cached singleton accessors for immutable ``pytransdat`` SPG grid readers."""

from __future__ import annotations

import functools

from pytransdat.spg_reader import SpgReader
from pytransdat.trans_grid import Grid2D


@functools.lru_cache(maxsize=1)
def get_default_grid2d() -> Grid2D:
    """Return a cached singleton :class:`pytransdat.trans_grid.Grid2D` instance."""
    return Grid2D()


@functools.lru_cache(maxsize=1)
def get_default_spg_reader() -> SpgReader:
    """Return a cached singleton :class:`pytransdat.spg_reader.SpgReader` instance."""
    return SpgReader()
