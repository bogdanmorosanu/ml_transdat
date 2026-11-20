"""PyProj EPSG:4258 <-> EPSG:3844 coordinate transformer baseline."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pyproj

from src.models.base import BaseCoordinateTransformer


class PyprojEpsg3844Transformer(BaseCoordinateTransformer):
    """Wrapper around ``pyproj.Transformer`` for EPSG:4258 (ETRS89) <-> EPSG:3844 (Stereo 70)."""

    def __init__(self) -> None:
        self._to_st70 = pyproj.Transformer.from_crs(
            "EPSG:4258", "EPSG:3844", always_xy=True
        )
        self._to_etrs89 = pyproj.Transformer.from_crs(
            "EPSG:3844", "EPSG:4258", always_xy=True
        )

    def _runtime_dependencies(self) -> tuple[str, ...]:
        return ("pyproj",)

    def etrs89_to_st70(
        self,
        lat_rad: Sequence[float] | np.ndarray,
        lon_rad: Sequence[float] | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transform ETRS89 ``(lat_rad, lon_rad)`` to Stereo 70 ``(N, E)`` in meters."""
        lat_deg = np.degrees(np.asarray(lat_rad, dtype=np.float64))
        lon_deg = np.degrees(np.asarray(lon_rad, dtype=np.float64))
        pred_e, pred_n = self._to_st70.transform(lon_deg, lat_deg)
        return np.asarray(pred_n, dtype=np.float64), np.asarray(
            pred_e, dtype=np.float64
        )

    def st70_to_etrs89(
        self,
        n: Sequence[float] | np.ndarray,
        e: Sequence[float] | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transform Stereo 70 ``(N, E)`` in meters to ETRS89 ``(lat_rad, lon_rad)``."""
        pred_lon_deg, pred_lat_deg = self._to_etrs89.transform(
            np.asarray(e, dtype=np.float64),
            np.asarray(n, dtype=np.float64),
        )
        return (
            np.radians(np.asarray(pred_lat_deg, dtype=np.float64)),
            np.radians(np.asarray(pred_lon_deg, dtype=np.float64)),
        )
