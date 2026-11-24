"""Unit tests for shared SpatialFeatureEncoder used by neural grid-shift models."""

from __future__ import annotations

import pickle
import unittest

import numpy as np

from src.models._features import SpatialFeatureEncoder


class TestSpatialFeatureEncoder(unittest.TestCase):
    """Verify coordinate scaling, RFF, cyclic cell (u, v), and pickle safety."""

    def setUp(self) -> None:
        rng = np.random.default_rng(7)
        self.coords = np.column_stack(
            [
                rng.uniform(300_000.0, 700_000.0, size=32),
                rng.uniform(300_000.0, 700_000.0, size=32),
            ]
        )

    def test_single_scale_rff_and_cyclic_uv_dimensions(self) -> None:
        enc = SpatialFeatureEncoder(
            normalize_inputs=True,
            use_fourier_features=True,
            fourier_scale=3.0,
            num_fourier_frequencies=16,
            use_cyclic_cell_pos=True,
            cyclic_cell_harmonics=2,
        )
        feats = enc.fit_transform(self.coords)
        # 2 (norm) + 32 (sin/cos RFF) + 12 (2 harmonics * 4 + 4 cross terms) = 46
        self.assertEqual(feats.shape, (32, 46))
        np.testing.assert_allclose(feats, enc.transform(self.coords), atol=1e-12)

    def test_multiscale_rff_and_pickle_roundtrip(self) -> None:
        enc = SpatialFeatureEncoder(
            normalize_inputs=True,
            use_fourier_features=True,
            fourier_scales=(1.0, 3.0, 8.0),
            num_fourier_frequencies=20,
            use_cyclic_cell_pos=True,
            cyclic_cell_harmonics=1,
        )
        feats_before = enc.fit_transform(self.coords)
        # 2 (norm) + 40 (sin/cos RFF) + 4 (1 harmonic) = 46
        self.assertEqual(feats_before.shape, (32, 46))

        restored: SpatialFeatureEncoder = pickle.loads(pickle.dumps(enc))
        feats_after = restored.transform(self.coords)
        np.testing.assert_allclose(feats_before, feats_after, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
