"""Unit tests for PolynomialSurfaceRegressor."""

import unittest

import numpy as np

from src.models.polynomial_surface import PolynomialSurfaceRegressor


class TestPolynomialSurfaceRegressor(unittest.TestCase):
    """Test suite for PolynomialSurfaceRegressor."""

    def setUp(self) -> None:
        """Create synthetic 2D spatial coordinate test data."""
        np.random.seed(42)
        # 10 distinct coordinates on a regular 2D grid
        self.coords = np.array(
            [
                [100.0, 200.0],
                [100.0, 300.0],
                [200.0, 200.0],
                [200.0, 300.0],
                [300.0, 200.0],
                [300.0, 300.0],
                [400.0, 200.0],
                [400.0, 300.0],
                [500.0, 200.0],
                [500.0, 300.0],
            ],
            dtype=np.float64,
        )

        # Distinct 2D shift values for each point
        self.shifts = np.array(
            [
                [0.10, -0.20],
                [0.15, -0.25],
                [0.20, -0.10],
                [0.25, -0.15],
                [0.30, 0.00],
                [0.35, 0.05],
                [0.40, 0.10],
                [0.45, 0.15],
                [0.50, 0.20],
                [0.55, 0.25],
            ],
            dtype=np.float64,
        )

    def test_fit_and_predict_shape(self) -> None:
        """Fit and predict must produce correctly shaped output arrays."""
        model = PolynomialSurfaceRegressor(degree=2)
        model.fit(self.coords, self.shifts)

        preds = model.predict(self.coords)
        self.assertEqual(preds.shape, (10, 2))
        self.assertEqual(model.n_terms, 6)
        self.assertEqual(len(model.feature_names), 6)
        self.assertEqual(model.coef_.shape, (2, 6))

    def test_term_count_progression(self) -> None:
        """Term count must match (D+1)(D+2)/2."""
        for deg in [1, 2, 3, 4, 5]:
            expected = (deg + 1) * (deg + 2) // 2
            model = PolynomialSurfaceRegressor(degree=deg)
            self.assertEqual(model.n_terms, expected)

    def test_predict_before_fit_raises(self) -> None:
        """Calling predict or inspection properties before fit must raise RuntimeError."""
        model = PolynomialSurfaceRegressor()
        with self.assertRaises(RuntimeError):
            model.predict(self.coords)
        with self.assertRaises(RuntimeError):
            _ = model.feature_names
        with self.assertRaises(RuntimeError):
            _ = model.coef_
        with self.assertRaises(RuntimeError):
            model.get_formula("n")

    def test_formula_formatting(self) -> None:
        """Formula string formatting must produce valid non-empty string."""
        model = PolynomialSurfaceRegressor(degree=2)
        model.fit(self.coords, self.shifts)
        fn = model.get_formula("n")
        fe = model.get_formula("e")
        self.assertTrue(fn.startswith("\\Delta N(u, v) ="))
        self.assertTrue(fe.startswith("\\Delta E(u, v) ="))

    def test_invalid_parameters_raise(self) -> None:
        """Invalid degree, alpha, or regressor must raise ValueError."""
        with self.assertRaises(ValueError):
            PolynomialSurfaceRegressor(degree=0)
        with self.assertRaises(ValueError):
            PolynomialSurfaceRegressor(alpha=-1.0)
        with self.assertRaises(ValueError):
            PolynomialSurfaceRegressor(regressor="unsupported")

    def test_lasso_sparsity(self) -> None:
        """Lasso regressor must produce sparse coefficient vectors."""
        model = PolynomialSurfaceRegressor(degree=3, alpha=0.05, regressor="lasso")
        model.fit(self.coords, self.shifts)
        nz_n, nz_e = model.n_nonzero_terms
        self.assertLess(nz_n, model.n_terms)
        self.assertLess(nz_e, model.n_terms)
        preds = model.predict(self.coords)
        self.assertEqual(preds.shape, (10, 2))


if __name__ == "__main__":
    unittest.main()
