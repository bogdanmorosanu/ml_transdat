"""Unit tests for DecisionTreeSpatialRegressor."""

import unittest

import numpy as np

from src.models.decision_tree_regressor import DecisionTreeSpatialRegressor


class TestDecisionTreeSpatialRegressor(unittest.TestCase):
    """Test suite for DecisionTreeSpatialRegressor."""

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

    def test_unconstrained_exact_memorization(self) -> None:
        """Unconstrained tree must achieve exact zero error on distinct training nodes."""
        model = DecisionTreeSpatialRegressor(max_depth=None, min_samples_leaf=1)
        model.fit(self.coords, self.shifts)

        preds = model.predict(self.coords)
        self.assertEqual(preds.shape, (10, 2))
        np.testing.assert_allclose(preds, self.shifts, atol=1e-12)

        # Exact leaf count should equal sample count
        self.assertEqual(model.n_leaves, 10)
        self.assertGreater(model.depth, 0)

    def test_predict_before_fit_raises(self) -> None:
        """Calling predict or inspection properties before fit must raise RuntimeError."""
        model = DecisionTreeSpatialRegressor()
        with self.assertRaises(RuntimeError):
            model.predict(self.coords)
        with self.assertRaises(RuntimeError):
            _ = model.n_leaves
        with self.assertRaises(RuntimeError):
            _ = model.depth

    def test_invalid_input_shapes_raise(self) -> None:
        """Mismatched or invalid array dimensions must raise ValueError."""
        model = DecisionTreeSpatialRegressor()
        # 1D coords
        with self.assertRaises(ValueError):
            model.fit(np.array([1.0, 2.0]), self.shifts)
        # Sample count mismatch
        with self.assertRaises(ValueError):
            model.fit(self.coords[:5], self.shifts)
        # 3D query coords
        model.fit(self.coords, self.shifts)
        with self.assertRaises(ValueError):
            model.predict(np.zeros((5, 3)))

    def test_constrained_depth(self) -> None:
        """Constraining max_depth must limit the tree depth."""
        model = DecisionTreeSpatialRegressor(max_depth=2)
        model.fit(self.coords, self.shifts)
        self.assertLessEqual(model.depth, 2)
        self.assertLessEqual(model.n_leaves, 4)


if __name__ == "__main__":
    unittest.main()
