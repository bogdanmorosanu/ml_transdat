"""Unit tests for CellPlane, fit_cell_plane, and extract_grid_cell_planes."""

import unittest

import numpy as np

from src.geometry.plane import (
    CellPlane,
    CellPlane2D,
    extract_grid_cell_planes,
    fit_cell_plane,
)


class TestCellPlane(unittest.TestCase):
    """Test suite for cell plane fitting and grid extraction."""

    def test_fit_cell_plane_pure_plane(self) -> None:
        """Tests fitting a perfectly planar surface with zero twist."""
        # z = 10.0 + 2.0 * u + 3.0 * v
        # corners:
        # (0, 0) -> 10.0
        # (1, 0) -> 12.0
        # (0, 1) -> 13.0
        # (1, 1) -> 15.0
        corners = np.array([[10.0, 12.0], [13.0, 15.0]])
        plane: CellPlane = fit_cell_plane(corners, step_n=1000.0, step_e=1000.0)

        self.assertAlmostEqual(plane.min_val, 10.0)
        self.assertAlmostEqual(plane.max_val, 15.0)
        self.assertAlmostEqual(plane.span, 5.0)
        self.assertAlmostEqual(plane.mean_val, 12.5)

        self.assertAlmostEqual(plane.slope_u, 2.0)
        self.assertAlmostEqual(plane.slope_v, 3.0)
        self.assertAlmostEqual(plane.twist, 0.0)
        self.assertAlmostEqual(plane.center_plane_deviation, 0.0)

        # Test evaluations
        self.assertAlmostEqual(float(plane.eval_bilinear(0.0, 0.0)), 10.0)
        self.assertAlmostEqual(float(plane.eval_bilinear(1.0, 0.0)), 12.0)
        self.assertAlmostEqual(float(plane.eval_bilinear(0.0, 1.0)), 13.0)
        self.assertAlmostEqual(float(plane.eval_bilinear(1.0, 1.0)), 15.0)
        self.assertAlmostEqual(float(plane.eval_bilinear(0.5, 0.5)), 12.5)

        self.assertAlmostEqual(float(plane.eval_absolute_plane(0.5, 0.5)), 12.5)
        self.assertAlmostEqual(float(plane.eval_relative_plane(0.0, 0.0)), 0.0)

    def test_fit_cell_plane_saddle_twist(self) -> None:
        """Tests fitting a hyperbolic paraboloid (saddle) with non-zero twist."""
        # z = 0.0 at (0, 0), (1, 0), (0, 1), and 4.0 at (1, 1)
        # twist = (4 - 0) - (0 - 0) = 4.0
        corners = np.array([[0.0, 0.0], [0.0, 4.0]])
        plane = fit_cell_plane(corners, step_n=1000.0, step_e=1000.0)

        self.assertAlmostEqual(plane.twist, 4.0)
        self.assertAlmostEqual(plane.center_plane_deviation, 1.0)
        self.assertAlmostEqual(float(plane.eval_bilinear(0.5, 0.5)), 1.0)
        self.assertAlmostEqual(float(plane.eval_absolute_plane(0.5, 0.5)), 1.0)
        self.assertAlmostEqual(float(plane.eval_bilinear(1.0, 1.0)), 4.0)
        self.assertAlmostEqual(float(plane.eval_absolute_plane(1.0, 1.0)), 3.0)
        self.assertAlmostEqual(float(plane.eval_bilinear_residual(1.0, 1.0)), 1.0)

    def test_extract_grid_cell_planes_real_spg(self) -> None:
        """Tests extraction of all valid cell planes from the official SPG grid."""
        cells = extract_grid_cell_planes()
        self.assertEqual(len(cells), 2556)

        first_cell: CellPlane2D = cells[0]
        self.assertEqual(first_cell.step_n, 11000.0)
        self.assertEqual(first_cell.step_e, 11000.0)
        self.assertTrue(first_cell.contains(first_cell.center_n, first_cell.center_e))

        # Check serialization
        d = first_cell.to_dict()
        self.assertIn("dn_slope_mag", d)
        self.assertIn("de_slope_mag", d)
        self.assertIn("dn_twist", d)
        self.assertIn("total_slope_mag", d)


if __name__ == "__main__":
    unittest.main()
