"""Unit tests for KNNSpatialRegressor."""

from __future__ import annotations

import unittest

import numpy as np

from src.models.knn_regressor import KNNSpatialRegressor


class TestKNNSpatialRegressor(unittest.TestCase):
    """Test suite verifying KNNSpatialRegressor functionality and geodetic invariants."""

    def setUp(self) -> None:
        # Synthetic 3x3 grid points (e.g. 10 km step)
        n_vals = np.array([300_000.0, 310_000.0, 320_000.0])
        e_vals = np.array([500_000.0, 510_000.0, 520_000.0])
        nn, ee = np.meshgrid(n_vals, e_vals, indexing="ij")
        self.n_nodes = nn.ravel()
        self.e_nodes = ee.ravel()

        # Known synthetic shifts
        self.dn_nodes = 0.5 * (self.n_nodes - 300_000.0) / 10_000.0 + 0.1
        self.de_nodes = -0.3 * (self.e_nodes - 500_000.0) / 10_000.0 - 0.2

    def test_unfitted_raises(self) -> None:
        model = KNNSpatialRegressor(n_neighbors=4)
        self.assertFalse(model.is_fitted)
        with self.assertRaises(RuntimeError):
            model.predict([305_000.0], [505_000.0])

    def test_mismatched_shapes_raise(self) -> None:
        model = KNNSpatialRegressor()
        with self.assertRaises(ValueError):
            model.fit(self.n_nodes, self.e_nodes[:-1], self.dn_nodes, self.de_nodes)

    def test_k1_exact_at_nodes(self) -> None:
        model = KNNSpatialRegressor(n_neighbors=1, weights="uniform")
        model.fit(self.n_nodes, self.e_nodes, self.dn_nodes, self.de_nodes)
        self.assertTrue(model.is_fitted)

        pred_dn, pred_de = model.predict(self.n_nodes, self.e_nodes)
        np.testing.assert_allclose(pred_dn, self.dn_nodes, atol=1e-12)
        np.testing.assert_allclose(pred_de, self.de_nodes, atol=1e-12)

    def test_k4_distance_exact_at_nodes(self) -> None:
        model = KNNSpatialRegressor(n_neighbors=4, weights="distance")
        model.fit(self.n_nodes, self.e_nodes, self.dn_nodes, self.de_nodes)

        pred_dn, pred_de = model.predict(self.n_nodes, self.e_nodes)
        np.testing.assert_allclose(pred_dn, self.dn_nodes, atol=1e-12)
        np.testing.assert_allclose(pred_de, self.de_nodes, atol=1e-12)

    def test_k4_interpolation_cell_center(self) -> None:
        model = KNNSpatialRegressor(n_neighbors=4, weights="distance")
        model.fit(self.n_nodes, self.e_nodes, self.dn_nodes, self.de_nodes)

        # Query point exactly in center of cell (305000, 505000)
        # Symmetrical to the 4 corners: (300k, 500k), (300k, 510k), (310k, 500k), (310k, 510k)
        pred_dn, pred_de = model.predict([305_000.0], [505_000.0])
        self.assertEqual(pred_dn.shape, (1,))
        self.assertEqual(pred_de.shape, (1,))

        # At cell center, distances to all 4 corners are identical, so prediction equals arithmetic mean of the 4
        c1 = (self.n_nodes == 300_000.0) & (self.e_nodes == 500_000.0)
        c2 = (self.n_nodes == 300_000.0) & (self.e_nodes == 510_000.0)
        c3 = (self.n_nodes == 310_000.0) & (self.e_nodes == 500_000.0)
        c4 = (self.n_nodes == 310_000.0) & (self.e_nodes == 510_000.0)
        expected_dn = np.mean([self.dn_nodes[c1], self.dn_nodes[c2], self.dn_nodes[c3], self.dn_nodes[c4]])
        expected_de = np.mean([self.de_nodes[c1], self.de_nodes[c2], self.de_nodes[c3], self.de_nodes[c4]])

        self.assertAlmostEqual(float(pred_dn[0]), float(expected_dn), places=9)
        self.assertAlmostEqual(float(pred_de[0]), float(expected_de), places=9)


if __name__ == "__main__":
    unittest.main()
