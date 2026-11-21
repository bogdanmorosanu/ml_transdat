"""Planar and Bilinear Surface Modeling for Coordinate Grid Cells.

This module provides reusable primitives to decompose a 2D/3D shift grid
cell into:
1. A zero-reduced base plane with inclination slopes (Northing and Easting).
2. Bilinear twist and planarity deviation indices.
3. Local relative coordinate evaluation (u, v in [0, 1]).
"""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from pytransdat.spg_reader import SpgReader


@dataclass(frozen=True)
class CellPlane:
    """Represents a 3D planar fit and bilinear properties for a single scalar field over a cell.

    Corners are indexed as:
        - z00: South-West (u=0, v=0)
        - z01: South-East (u=1, v=0)
        - z10: North-West (u=0, v=1)
        - z11: North-East (u=1, v=1)
    """

    # Raw corner values
    z00: float
    z01: float
    z10: float
    z11: float

    # Bounding step sizes (in meters)
    step_n: float
    step_e: float

    # Base reference values
    min_val: float
    max_val: float
    span: float
    mean_val: float

    # Planar slopes per normalized cell (shift delta across 1 cell span)
    slope_u: float  # West to East (across step_e)
    slope_v: float  # South to North (across step_n)

    # Physical gradients (m shift / m distance)
    grad_e: float  # slope_u / step_e
    grad_n: float  # slope_v / step_n

    # Inclination magnitude & direction
    slope_mag_cell: float  # hypot(slope_u, slope_v) in meters per cell
    grad_mag: float  # hypot(grad_n, grad_e) in m/m
    aspect_rad: float  # atan2(slope_v, slope_u) in radians [-pi, pi]
    aspect_deg: float  # compass-like aspect in degrees [0, 360)

    # Bilinear twist / non-planarity index
    # tau = (z11 - z10) - (z01 - z00)
    twist: float
    center_plane_deviation: float  # twist / 4.0

    def eval_relative_plane(
        self, u: float | np.ndarray, v: float | np.ndarray
    ) -> float | np.ndarray:
        """Evaluates the zero-reduced best-fit flat plane at normalized coordinates (u, v).

        The minimum corner value is zero:
        z_rel(u, v) = z00_rel + slope_u * u + slope_v * v (adjusted for least squares center)

        :param u: Fractional Easting coordinate in [0, 1]
        :param v: Fractional Northing coordinate in [0, 1]
        :return: Relative plane value (always >= 0 at the minimum corner)
        """
        center_rel = self.mean_val - self.min_val
        return center_rel + self.slope_u * (u - 0.5) + self.slope_v * (v - 0.5)

    def eval_absolute_plane(
        self, u: float | np.ndarray, v: float | np.ndarray
    ) -> float | np.ndarray:
        """Evaluates the absolute best-fit flat plane at normalized coordinates (u, v).

        :param u: Fractional Easting coordinate in [0, 1]
        :param v: Fractional Northing coordinate in [0, 1]
        :return: Absolute plane value in original units
        """
        return self.mean_val + self.slope_u * (u - 0.5) + self.slope_v * (v - 0.5)

    def eval_bilinear(
        self, u: float | np.ndarray, v: float | np.ndarray
    ) -> float | np.ndarray:
        """Evaluates the exact 4-corner bilinear interpolation at (u, v).

        Matches all 4 corners exactly:
        z(u, v) = (1-u)(1-v)*z00 + u(1-v)*z01 + (1-u)v*z10 + uv*z11

        :param u: Fractional Easting coordinate in [0, 1]
        :param v: Fractional Northing coordinate in [0, 1]
        :return: Bilinear interpolated value
        """
        return (
            (1.0 - u) * (1.0 - v) * self.z00
            + u * (1.0 - v) * self.z01
            + (1.0 - u) * v * self.z10
            + u * v * self.z11
        )

    def eval_bilinear_residual(
        self, u: float | np.ndarray, v: float | np.ndarray
    ) -> float | np.ndarray:
        """Evaluates the departure of the bilinear surface from the flat plane.

        residual(u, v) = eval_bilinear(u, v) - eval_absolute_plane(u, v)

        :param u: Fractional Easting coordinate in [0, 1]
        :param v: Fractional Northing coordinate in [0, 1]
        :return: Departure from flatness (meters)
        """
        return self.eval_bilinear(u, v) - self.eval_absolute_plane(u, v)


def fit_cell_plane(
    corners: np.ndarray | list[float],
    step_n: float = 11000.0,
    step_e: float = 11000.0,
) -> CellPlane:
    """Fits planar and bilinear parameters to a 4-corner cell.

    Corners can be passed as:
    - 2x2 numpy array: [[z00, z01], [z10, z11]]
      where row 0 is South, row 1 is North, col 0 is West, col 1 is East.
    - Flat 4-element sequence: [z00, z01, z10, z11].

    :param corners: 4 corner values
    :param step_n: Cell height in meters (default: 11,000 m)
    :param step_e: Cell width in meters (default: 11,000 m)
    :return: Typed CellPlane instance
    """
    arr = np.asarray(corners, dtype=np.float64)
    if arr.shape == (2, 2):
        z00, z01 = float(arr[0, 0]), float(arr[0, 1])
        z10, z11 = float(arr[1, 0]), float(arr[1, 1])
    elif arr.shape == (4,):
        z00, z01, z10, z11 = float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])
    else:
        raise ValueError(f"Corners must be shape (2, 2) or (4,), got {arr.shape}")

    min_val = min(z00, z01, z10, z11)
    max_val = max(z00, z01, z10, z11)
    span = max_val - min_val
    mean_val = (z00 + z01 + z10 + z11) / 4.0

    # Average slope along Easting (col / u)
    slope_u = 0.5 * ((z01 - z00) + (z11 - z10))
    # Average slope along Northing (row / v)
    slope_v = 0.5 * ((z10 - z00) + (z11 - z01))

    grad_e = slope_u / step_e
    grad_n = slope_v / step_n

    slope_mag_cell = math.hypot(slope_u, slope_v)
    grad_mag = math.hypot(grad_n, grad_e)

    aspect_rad = math.atan2(slope_v, slope_u)
    aspect_deg = (math.degrees(aspect_rad) + 360.0) % 360.0

    twist = (z11 - z10) - (z01 - z00)
    center_plane_deviation = twist / 4.0

    return CellPlane(
        z00=z00,
        z01=z01,
        z10=z10,
        z11=z11,
        step_n=step_n,
        step_e=step_e,
        min_val=min_val,
        max_val=max_val,
        span=span,
        mean_val=mean_val,
        slope_u=slope_u,
        slope_v=slope_v,
        grad_e=grad_e,
        grad_n=grad_n,
        slope_mag_cell=slope_mag_cell,
        grad_mag=grad_mag,
        aspect_rad=aspect_rad,
        aspect_deg=aspect_deg,
        twist=twist,
        center_plane_deviation=center_plane_deviation,
    )


@dataclass(frozen=True)
class CellPlane2D:
    """Encapsulates 2D horizontal shift planes (dN and dE) for a single grid cell."""

    row: int
    col: int
    min_n: float
    max_n: float
    min_e: float
    max_e: float
    center_n: float
    center_e: float
    step_n: float
    step_e: float

    plane_dn: CellPlane
    plane_de: CellPlane

    def contains(self, n: float, e: float) -> bool:
        """Checks if a point (n, e) falls within this cell's bounding box."""
        return (self.min_n <= n <= self.max_n) and (self.min_e <= e <= self.max_e)

    def to_fractional_coords(
        self, n: float | np.ndarray, e: float | np.ndarray
    ) -> tuple[float | np.ndarray, float | np.ndarray]:
        """Converts metric (n, e) coordinates to fractional (u, v) within the cell."""
        u = (e - self.min_e) / self.step_e
        v = (n - self.min_n) / self.step_n
        return u, v

    def eval_bilinear(self, n: float, e: float) -> tuple[float, float]:
        """Evaluates 2D bilinear shifts (dn, de) at metric coordinates (n, e)."""
        u, v = self.to_fractional_coords(n, e)
        return float(self.plane_dn.eval_bilinear(u, v)), float(
            self.plane_de.eval_bilinear(u, v)
        )

    def eval_plane(self, n: float, e: float) -> tuple[float, float]:
        """Evaluates 2D flat plane shifts (dn, de) at metric coordinates (n, e)."""
        u, v = self.to_fractional_coords(n, e)
        return float(self.plane_dn.eval_absolute_plane(u, v)), float(
            self.plane_de.eval_absolute_plane(u, v)
        )

    def to_dict(self) -> dict[str, Any]:
        """Serializes cell plane attributes into a flat dictionary."""
        return {
            "row": self.row,
            "col": self.col,
            "min_n": self.min_n,
            "max_n": self.max_n,
            "min_e": self.min_e,
            "max_e": self.max_e,
            "center_n": self.center_n,
            "center_e": self.center_e,
            # dN Plane metrics
            "dn_min": self.plane_dn.min_val,
            "dn_max": self.plane_dn.max_val,
            "dn_span": self.plane_dn.span,
            "dn_mean": self.plane_dn.mean_val,
            "dn_slope_u": self.plane_dn.slope_u,
            "dn_slope_v": self.plane_dn.slope_v,
            "dn_slope_mag": self.plane_dn.slope_mag_cell,
            "dn_grad_mag": self.plane_dn.grad_mag,
            "dn_aspect_deg": self.plane_dn.aspect_deg,
            "dn_twist": self.plane_dn.twist,
            "dn_center_deviation": self.plane_dn.center_plane_deviation,
            # dE Plane metrics
            "de_min": self.plane_de.min_val,
            "de_max": self.plane_de.max_val,
            "de_span": self.plane_de.span,
            "de_mean": self.plane_de.mean_val,
            "de_slope_u": self.plane_de.slope_u,
            "de_slope_v": self.plane_de.slope_v,
            "de_slope_mag": self.plane_de.slope_mag_cell,
            "de_grad_mag": self.plane_de.grad_mag,
            "de_aspect_deg": self.plane_de.aspect_deg,
            "de_twist": self.plane_de.twist,
            "de_center_deviation": self.plane_de.center_plane_deviation,
            # Combined 2D metrics
            "total_slope_mag": math.hypot(
                self.plane_dn.slope_mag_cell, self.plane_de.slope_mag_cell
            ),
            "total_twist_mag": math.hypot(self.plane_dn.twist, self.plane_de.twist),
        }


def extract_grid_cell_planes(
    spg_path: str | None = None,
    nodata_threshold: float = 100.0,
) -> list[CellPlane2D]:
    """Scans all cells in the official SPG grid and fits 2D cell planes for valid cells.

    :param spg_path: Optional path to .spg grid file. If None, uses default :class:`SpgReader`.
    :param nodata_threshold: Magnitude threshold to filter out NoData values (default: 100.0 m)
    :return: List of fitted CellPlane2D objects for all valid cells
    """
    reader = SpgReader(spg_path) if spg_path is not None else SpgReader()
    min_e, max_e, min_n, max_n, step_e, step_n = (
        float(v) for v in reader.shift_bounds
    )
    ncols = round((max_e - min_e) / step_e) + 1
    nrows = round((max_n - min_n) / step_n) + 1

    de_arr = np.asarray(reader.shift_e_flat, dtype=np.float64).reshape((nrows, ncols))
    dn_arr = np.asarray(reader.shift_n_flat, dtype=np.float64).reshape((nrows, ncols))

    valid_cells: list[CellPlane2D] = []

    for r in range(nrows - 1):
        for c in range(ncols - 1):
            corners_dn = dn_arr[r : r + 2, c : c + 2]
            corners_de = de_arr[r : r + 2, c : c + 2]

            # Filter NoData / NaN / offshore
            if (
                np.any(np.abs(corners_dn) > nodata_threshold)
                or np.any(np.abs(corners_de) > nodata_threshold)
                or np.any(np.isnan(corners_dn))
                or np.any(np.isnan(corners_de))
            ):
                continue

            plane_dn = fit_cell_plane(corners_dn, step_n=step_n, step_e=step_e)
            plane_de = fit_cell_plane(corners_de, step_n=step_n, step_e=step_e)

            cell_min_n = min_n + r * step_n
            cell_max_n = cell_min_n + step_n
            cell_min_e = min_e + c * step_e
            cell_max_e = cell_min_e + step_e
            center_n = cell_min_n + 0.5 * step_n
            center_e = cell_min_e + 0.5 * step_e

            valid_cells.append(
                CellPlane2D(
                    row=r,
                    col=c,
                    min_n=cell_min_n,
                    max_n=cell_max_n,
                    min_e=cell_min_e,
                    max_e=cell_max_e,
                    center_n=center_n,
                    center_e=center_e,
                    step_n=step_n,
                    step_e=step_e,
                    plane_dn=plane_dn,
                    plane_de=plane_de,
                )
            )

    return valid_cells
