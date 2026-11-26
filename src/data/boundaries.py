"""Spatial boundary loading and vectorized point-in-polygon filtering."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import shapely
from shapely.geometry.base import BaseGeometry


class BoundaryId(str, Enum):
    """Canonical spatial boundary identifiers in ``ml_transdat``."""

    ROMANIA = "romania"


_BOUNDARY_PATHS: dict[BoundaryId, Path] = {
    BoundaryId.ROMANIA: Path("data/boundaries/romania_border_st70.geojson"),
}


@dataclass
class SpatialBoundary:
    """Prepared planar Stereo 70 polygon mask for fast point-in-polygon checks.

    Coordinates passed to :meth:`contains` follow the project convention
    ``(N, 2)`` in ``[st70_n, st70_e]`` (Northing, Easting) meters, which map
    to planar ``(y, x)`` in ``EPSG:3844``.
    """

    boundary_id: BoundaryId
    geometry: BaseGeometry
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        shapely.prepare(self.geometry)

    @property
    def bounds_st70(self) -> tuple[float, float, float, float]:
        """Return ``(n_min, n_max, e_min, e_max)`` bounding box in meters."""
        e_min, n_min, e_max, n_max = self.geometry.bounds
        return (float(n_min), float(n_max), float(e_min), float(e_max))

    def contains(self, coords: np.ndarray) -> np.ndarray:
        """Return boolean mask of shape ``(N,)`` for points inside the boundary.

        :param coords: Planar Stereo 70 coordinates of shape ``(N, 2)`` as
            ``[st70_n, st70_e]``.
        :type coords: np.ndarray
        :return: Boolean array of shape ``(N,)`` where ``True`` indicates inside
            (or on) the buffered boundary.
        :rtype: np.ndarray
        :raises ValueError: If ``coords`` does not have shape ``(N, 2)``.
        """
        arr = np.asarray(coords, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"Expected coords of shape (N, 2), got {arr.shape}")
        if len(arr) == 0:
            return np.zeros(0, dtype=bool)
        # Stereo 70 coords are [Northing (Y), Easting (X)]
        return np.asarray(
            shapely.contains_xy(self.geometry, arr[:, 1], arr[:, 0]),
            dtype=bool,
        )

    @classmethod
    def from_geojson(
        cls,
        geojson_path: Path,
        boundary_id: BoundaryId = BoundaryId.ROMANIA,
    ) -> SpatialBoundary:
        """Load a pre-projected Stereo 70 GeoJSON file into a :class:`SpatialBoundary`."""
        if not geojson_path.is_file():
            raise FileNotFoundError(f"Boundary GeoJSON not found: {geojson_path}")
        payload = json.loads(geojson_path.read_text(encoding="utf-8"))
        features = payload.get("features", [])
        if not features:
            raise ValueError(f"No features found in boundary GeoJSON: {geojson_path}")
        geoms = [
            shapely.from_geojson(json.dumps(feat["geometry"]))
            for feat in features
            if feat.get("geometry") is not None
        ]
        if not geoms:
            raise ValueError(f"No valid geometries in boundary GeoJSON: {geojson_path}")
        geom = shapely.union_all(geoms) if len(geoms) > 1 else geoms[0]

        meta: dict[str, Any] = {"boundary_id": boundary_id.value}
        meta_path = geojson_path.with_suffix(".meta.json")
        if meta_path.is_file():
            meta.update(json.loads(meta_path.read_text(encoding="utf-8")))
        return cls(boundary_id=boundary_id, geometry=geom, metadata=meta)


_BOUNDARY_CACHE: dict[BoundaryId, SpatialBoundary] = {}


def get_boundary(boundary_id: BoundaryId | str = BoundaryId.ROMANIA) -> SpatialBoundary:
    """Return cached :class:`SpatialBoundary` for ``boundary_id``."""
    bid = BoundaryId(boundary_id)
    if bid not in _BOUNDARY_CACHE:
        path = _BOUNDARY_PATHS.get(bid)
        if path is None:
            raise KeyError(f"Unregistered boundary id: {bid}")
        _BOUNDARY_CACHE[bid] = SpatialBoundary.from_geojson(path, boundary_id=bid)
    return _BOUNDARY_CACHE[bid]
