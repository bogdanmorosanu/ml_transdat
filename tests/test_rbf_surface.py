"""Unit tests for RBFSurfaceRegressor using Python standard library unittest."""

from __future__ import annotations

import pickle
import unittest

import numpy as np

from src.models.rbf_surface import RBFSurfaceRegressor


class TestRBFSurfaceRegressor(unittest.TestCase):
    """Test suite for RBFSurfaceRegressor."""

    def test_init_validation(self) -> None:
        """Test parameter validation during initialization."""
        with self.assertRaises(ValueError):
            RBFSurfaceRegressor(kernel="invalid_kernel")

        with self.assertRaises(ValueError):
            RBFSurfaceRegressor(smoothing=-1.0)

        with self.assertRaises(ValueError):
            RBFSurfaceRegressor(degree=-2)

        with self.assertRaises(ValueError):
            RBFSurfaceRegressor(neighbors=0)

        with self.assertRaises(ValueError):
            RBFSurfaceRegressor(epsilon=-0.5)

    def test_fit_and_exact_predict(self) -> None:
        """Test exact interpolation property of Thin Plate Splines when smoothing=0."""
        np.random.seed(42)
        # Generate 30 2D control points
        coords = np.random.uniform(300_000, 700_000, size=(30, 2))
        # Simulated 2D shift values
        shifts = np.column_stack([
            0.5 * np.sin(coords[:, 0] / 100_000) + 0.3 * np.cos(coords[:, 1] / 100_000),
            0.2 * np.cos(coords[:, 0] / 80_000) - 0.4 * np.sin(coords[:, 1] / 80_000),
        ])

        model = RBFSurfaceRegressor(kernel="thin_plate_spline", degree=1, smoothing=0.0)
        model.fit(coords, shifts)

        self.assertTrue(model.is_fitted)
        self.assertEqual(model.n_anchors, 30)
        self.assertGreater(model.fit_time_sec, 0.0)

        # Predict at original training coordinates
        preds = model.predict(coords)
        self.assertEqual(preds.shape, shifts.shape)
        # Maximum difference should be effectively zero (< 1e-10 meters)
        np.testing.assert_allclose(preds, shifts, atol=1e-10)

    def test_degree_progression(self) -> None:
        """Test fitting with different trend degrees (-1, 1, 2, 3)."""
        np.random.seed(42)
        coords = np.random.uniform(0, 10, size=(25, 2))
        shifts = np.random.uniform(-1, 1, size=(25, 2))

        for deg in [-1, 1, 2, 3]:
            model = RBFSurfaceRegressor(kernel="thin_plate_spline", degree=deg, smoothing=0.0)
            model.fit(coords, shifts)
            preds = model.predict(coords)
            np.testing.assert_allclose(preds, shifts, atol=1e-8)

    def test_multiquadric_and_gaussian_kernels(self) -> None:
        """Test multiquadric and gaussian kernels with shape parameter epsilon."""
        np.random.seed(42)
        coords = np.random.uniform(0, 10, size=(20, 2))
        shifts = np.random.uniform(-1, 1, size=(20, 2))

        for kernel in ["multiquadric", "inverse_multiquadric", "gaussian"]:
            model = RBFSurfaceRegressor(kernel=kernel, degree=1, epsilon=1.0, smoothing=1e-6)
            model.fit(coords, shifts)
            preds = model.predict(coords)
            self.assertEqual(preds.shape, shifts.shape)
            self.assertFalse(np.isnan(preds).any())

    def test_unfitted_raises(self) -> None:
        """Test predicting with an unfitted model raises RuntimeError."""
        model = RBFSurfaceRegressor()
        with self.assertRaises(RuntimeError):
            model.predict(np.array([[500_000.0, 400_000.0]]))

    def test_pickle_serialization(self) -> None:
        """Test model serialization via pickle and memory footprint reporting."""
        np.random.seed(42)
        coords = np.random.uniform(0, 10, size=(15, 2))
        shifts = np.random.uniform(-1, 1, size=(15, 2))

        model = RBFSurfaceRegressor(kernel="thin_plate_spline", degree=1)
        model.fit(coords, shifts)

        serialized = pickle.dumps(model)
        deserialized: RBFSurfaceRegressor = pickle.loads(serialized)

        preds_orig = model.predict(coords)
        preds_deser = deserialized.predict(coords)
        np.testing.assert_array_equal(preds_orig, preds_deser)

        mb = model.memory_breakdown()
        self.assertGreater(mb.total_bytes, 0)
        self.assertIn("scipy", model._runtime_dependencies())


if __name__ == "__main__":
    unittest.main()
