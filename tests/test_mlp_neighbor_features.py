"""Unit tests for MLPGridShiftModel 3x3 neighbor feature integration."""

from __future__ import annotations

import unittest

import numpy as np

from src.models.mlp import MLPGridShiftModel


class TestMLPNeighborFeatures(unittest.TestCase):
    """Tests for 3x3 neighbor feature extraction, scaling, and validation."""

    def test_neighbor_feature_modes(self) -> None:
        """Verify slopes (32) and all (40) neighbor feature modes fit and predict cleanly."""
        rng = np.random.RandomState(42)
        coords = np.column_stack(
            [
                rng.uniform(450_000.0, 550_000.0, size=24),
                rng.uniform(450_000.0, 550_000.0, size=24),
            ]
        )
        coords[0] = [100_000.0, 100_000.0]
        shifts = rng.randn(24, 2) * 0.1

        for mode, expected_extra_dim in (("slopes", 32), ("all", 40)):
            with self.subTest(mode=mode):
                model = MLPGridShiftModel(
                    hidden_layer_sizes=(16,),
                    use_fourier_features=True,
                    num_fourier_frequencies=8,
                    max_iter=5,
                    use_neighbor_features=True,
                    neighbor_feature_mode=mode,
                    scale_neighbor_features=True,
                )
                model.fit(coords, shifts)

                feats = model._transform_features(coords)
                self.assertEqual(feats.shape, (24, 2 + 16 + expected_extra_dim))
                self.assertTrue(np.all(np.isfinite(feats)))

                preds = model.predict(coords)
                self.assertEqual(preds.shape, (24, 2))
                self.assertTrue(np.all(np.isfinite(preds)))

    def test_invalid_neighbor_feature_mode_raises(self) -> None:
        """Verify invalid neighbor_feature_mode raises ValueError."""
        with self.assertRaisesRegex(ValueError, "Unsupported neighbor_feature_mode"):
            MLPGridShiftModel(
                use_neighbor_features=True, neighbor_feature_mode="twists_only"
            )

    def test_cell_center_distance_feature(self) -> None:
        """Verify center distance is 0 at midpoint, 1 at edge, and appends 2 features."""
        from src.geometry.neighbor_features import CellNeighborFeatureExtractor

        extractor = CellNeighborFeatureExtractor(
            "data/features/cell_neighbor_features.parquet"
        )
        c_n = extractor._origin_n + 20.5 * extractor.step_n
        c_e = extractor._origin_e + 20.5 * extractor.step_e
        e_n = extractor._origin_n + 20.0 * extractor.step_n
        e_e = extractor._origin_e + 20.0 * extractor.step_e

        dists = extractor.lookup_center_distances(
            np.array([c_n, e_n]), np.array([c_e, e_e])
        )
        np.testing.assert_allclose(dists[0], [0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(dists[1], [1.0, 1.0], atol=1e-12)

        rng = np.random.RandomState(42)
        coords = rng.uniform(450_000.0, 550_000.0, size=(16, 2))
        shifts = rng.randn(16, 2) * 0.1
        model = MLPGridShiftModel(
            hidden_layer_sizes=(16,),
            use_fourier_features=True,
            num_fourier_frequencies=8,
            max_iter=5,
            use_neighbor_features=True,
            neighbor_feature_mode="slopes",
            use_cell_center_distance=True,
        )
        model.fit(coords, shifts)
        feats = model._transform_features(coords)
        self.assertEqual(feats.shape, (16, 2 + 16 + 32 + 2))
        self.assertTrue(np.all(np.isfinite(feats)))


if __name__ == "__main__":
    unittest.main()
