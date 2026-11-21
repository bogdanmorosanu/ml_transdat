"""Unit tests for ShapeCategory, CellShapeMetrics, evaluate_cell_polynomial_errors, and CellShapeFeatureExtractor."""

import unittest

import numpy as np
import pandas as pd

from src.geometry.plane import CellPlane2D, fit_cell_plane
from src.geometry.shape_classifier import (
    CellShapeFeatureExtractor,
    CellShapeMetrics,
    ShapeCategory,
    compute_shape_polynomial_basis,
    evaluate_cell_polynomial_errors,
)


class TestShapeClassifier(unittest.TestCase):
    """Test suite for cell shape classification."""

    def setUp(self) -> None:
        """Set up 25-point coordinate grid and basis matrices."""
        u_vals = np.linspace(0.1, 0.9, 5)
        v_vals = np.linspace(0.1, 0.9, 5)
        u_grid, v_grid = np.meshgrid(u_vals, v_vals)
        self.u_pts = u_grid.flatten()
        self.v_pts = v_grid.flatten()
        self.x_deg2, self.x_deg3 = compute_shape_polynomial_basis(
            self.u_pts, self.v_pts
        )

    def _create_mock_cell(
        self, fn_n, fn_e
    ) -> tuple[CellPlane2D, np.ndarray, np.ndarray]:
        """Creates a mock CellPlane2D with given synthetic scalar fields."""
        corners_n = np.array(
            [
                [fn_n(0.0, 0.0), fn_n(1.0, 0.0)],
                [fn_n(0.0, 1.0), fn_n(1.0, 1.0)],
            ]
        )
        corners_e = np.array(
            [
                [fn_e(0.0, 0.0), fn_e(1.0, 0.0)],
                [fn_e(0.0, 1.0), fn_e(1.0, 1.0)],
            ]
        )
        p_n = fit_cell_plane(corners_n)
        p_e = fit_cell_plane(corners_e)
        cell = CellPlane2D(
            row=0,
            col=0,
            min_n=0.0,
            max_n=1000.0,
            min_e=0.0,
            max_e=1000.0,
            center_n=500.0,
            center_e=500.0,
            step_n=1000.0,
            step_e=1000.0,
            plane_dn=p_n,
            plane_de=p_e,
        )
        dn_true = np.array(
            [fn_n(u, v) for u, v in zip(self.u_pts, self.v_pts, strict=True)]
        )
        de_true = np.array(
            [fn_e(u, v) for u, v in zip(self.u_pts, self.v_pts, strict=True)]
        )
        return cell, dn_true, de_true

    def test_pure_linear_classification(self) -> None:
        """Tests that a linear plane classifies as LINEAR at 1mm tolerance."""
        fn = lambda u, v: 1.0 + 0.5 * u - 0.2 * v
        cell, dn_true, de_true = self._create_mock_cell(fn, fn)

        metrics: CellShapeMetrics = evaluate_cell_polynomial_errors(
            cell=cell,
            dn_true=dn_true,
            de_true=de_true,
            u_pts=self.u_pts,
            v_pts=self.v_pts,
            x_deg2=self.x_deg2,
            x_deg3=self.x_deg3,
        )

        self.assertLess(metrics.err_2d_deg1, 0.01)  # < 0.01 mm
        self.assertEqual(metrics.get_category_2d(tol_mm=1.0), ShapeCategory.LINEAR)

    def test_parabolic_dome_classification(self) -> None:
        """Tests that a pure quadratic dome/bowl classifies as DOME_BOWL."""
        fn = lambda u, v: 0.10 * (u * (1.0 - u) + v * (1.0 - v))
        cell, dn_true, de_true = self._create_mock_cell(fn, fn)

        metrics = evaluate_cell_polynomial_errors(
            cell=cell,
            dn_true=dn_true,
            de_true=de_true,
            u_pts=self.u_pts,
            v_pts=self.v_pts,
            x_deg2=self.x_deg2,
            x_deg3=self.x_deg3,
        )

        self.assertGreater(metrics.err_2d_deg1, 10.0)
        self.assertLess(metrics.err_2d_deg2, 0.01)
        self.assertEqual(metrics.get_category_2d(tol_mm=2.0), ShapeCategory.DOME_BOWL)

    def test_cubic_s_shape_classification(self) -> None:
        """Tests that a cubic inflection wave classifies as S_SHAPE."""
        fn = lambda u, v: 0.20 * ((u - 0.5) ** 3 + (v - 0.5) ** 3)
        cell, dn_true, de_true = self._create_mock_cell(fn, fn)

        metrics = evaluate_cell_polynomial_errors(
            cell=cell,
            dn_true=dn_true,
            de_true=de_true,
            u_pts=self.u_pts,
            v_pts=self.v_pts,
            x_deg2=self.x_deg2,
            x_deg3=self.x_deg3,
        )

        self.assertGreater(metrics.err_2d_deg2, 1.0)
        self.assertLess(metrics.err_2d_deg3, 0.01)
        self.assertEqual(metrics.get_category_2d(tol_mm=0.5), ShapeCategory.S_SHAPE)

    def test_feature_extractor_lookup(self) -> None:
        """Tests CellShapeFeatureExtractor with synthetic cell DataFrame."""
        # Create a mock cell at row=10, col=20
        # N bounds: [213634.564 + 10*11000, ...] -> [323634.564, 334634.564]
        # E bounds: [109783.040 + 20*11000, ...] -> [329783.040, 340783.040]
        df_mock = pd.DataFrame(
            [
                {
                    "row": 10,
                    "col": 20,
                    "center_n": 329134.564,
                    "center_e": 335283.040,
                    "min_n": 323634.564,
                    "max_n": 334634.564,
                    "min_e": 329783.040,
                    "max_e": 340783.040,
                    "base_dn": 0.35,
                    "base_de": -0.15,
                    "err_n_deg1_mm": 12.0,
                    "err_n_deg2_mm": 1.5,
                    "err_n_deg3_mm": 0.1,
                    "err_e_deg1_mm": 18.0,
                    "err_e_deg2_mm": 6.5,
                    "err_e_deg3_mm": 0.8,
                    "err_2d_deg1_mm": 21.6,
                    "err_2d_deg2_mm": 6.7,
                    "err_2d_deg3_mm": 0.8,
                }
            ]
        )

        extractor = CellShapeFeatureExtractor(df_mock, default_tol_mm=5.0)

        # Query point inside the cell: N=329134.564 (center), E=335283.040 (center)
        res = extractor.lookup_point(329134.564, 335283.040, tol_mm=5.0)

        self.assertTrue(res["is_valid"])
        self.assertAlmostEqual(res["u"], 0.5)
        self.assertAlmostEqual(res["v"], 0.5)
        self.assertAlmostEqual(res["base_dn"], 0.35)
        self.assertAlmostEqual(res["base_de"], -0.15)

        # Check category: dN error at deg2 is 1.5mm <= 5mm -> DOME_BOWL
        self.assertEqual(res["cat_dn"], ShapeCategory.DOME_BOWL)
        # dE error at deg2 is 6.5mm > 5mm, deg3 is 0.8mm <= 5mm -> S_SHAPE
        self.assertEqual(res["cat_de"], ShapeCategory.S_SHAPE)
        # 2D error at deg2 is 6.7mm > 5mm -> S_SHAPE
        self.assertEqual(res["cat_2d"], ShapeCategory.S_SHAPE)

        # Check out-of-bounds point returns invalid mask
        out_res = extractor.lookup_point(0.0, 0.0)
        self.assertFalse(out_res["is_valid"])


if __name__ == "__main__":
    unittest.main()
