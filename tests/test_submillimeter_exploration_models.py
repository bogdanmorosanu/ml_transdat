"""Unit tests for models and samplers added in the sub-millimeter exploration."""

from __future__ import annotations

import unittest

import numpy as np

from src.data import GridId, GridManager
from src.models import (
    CellBilinearShiftModel,
    CompositeResidualShiftModel,
    HistGBRGridShiftModel,
    NeuralFeatureGridShiftModel,
    RBFSurfaceRegressor,
    TorchMLPGridShiftModel,
)


def _make_synthetic_data(n_pts: int = 80) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    coords = np.column_stack(
        [
            rng.uniform(350_000.0, 550_000.0, size=n_pts),
            rng.uniform(350_000.0, 650_000.0, size=n_pts),
        ]
    )
    shifts = np.column_stack(
        [
            0.2 * np.sin((coords[:, 0] - 350_000.0) / 50_000.0),
            -0.15 * np.cos((coords[:, 1] - 350_000.0) / 50_000.0),
        ]
    )
    return coords, shifts


class TestSubmillimeterExplorationModels(unittest.TestCase):
    """Tests for new models, options, and datasets in submillimeter exploration."""

    def test_torch_mlp_exploration_options(self) -> None:
        coords, shifts = _make_synthetic_data(64)
        configs = [
            {"cyclic_cell_harmonics": 4, "use_cyclic_cell_pos": True, "val_fraction": 0.0},
            {"fourier_scales": (1.0, 3.0), "loss_type": "huber", "val_fraction": 0.0},
            {"loss_type": "charbonnier", "lbfgs_steps": 3, "val_fraction": 0.0},
            {"activation": "siren", "siren_omega_0": 15.0, "use_fourier_features": False},
        ]
        for kwargs in configs:
            model = TorchMLPGridShiftModel(
                hidden_dim=32,
                num_res_blocks=1,
                max_epochs=4,
                batch_size=32,
                device="cpu",
                **kwargs,
            )
            model.fit(coords, shifts)
            preds = model.predict(coords[:10])
            self.assertEqual(preds.shape, (10, 2))
            self.assertTrue(np.all(np.isfinite(preds)))

    def test_cell_bilinear_and_composite_shift_model(self) -> None:
        coords, shifts = _make_synthetic_data(64)
        stage1 = CellBilinearShiftModel()
        stage2 = TorchMLPGridShiftModel(
            hidden_dim=32,
            num_res_blocks=1,
            max_epochs=3,
            batch_size=32,
            val_fraction=0.0,
            device="cpu",
        )
        comp = CompositeResidualShiftModel(stage1_model=stage1, stage2_model=stage2)
        comp.fit(coords, shifts)
        preds = comp.predict(coords[:8])
        self.assertEqual(preds.shape, (8, 2))
        self.assertTrue(np.all(np.isfinite(preds)))
        mb = comp.memory_breakdown()
        self.assertGreater(mb.grid_lookup_bytes, 0)

    def test_neural_feature_grid_shift_model(self) -> None:
        coords, shifts = _make_synthetic_data(64)
        model = NeuralFeatureGridShiftModel(
            grid_resolutions=((12, 16), (24, 32)),
            feature_dim=4,
            hidden_dim=16,
            max_epochs=4,
            batch_size=32,
            lbfgs_steps=2,
            device="cpu",
        )
        model.fit(coords, shifts)
        preds = model.predict(coords[:8])
        self.assertEqual(preds.shape, (8, 2))
        self.assertTrue(np.all(np.isfinite(preds)))
        self.assertGreater(model._grid_lookup_bytes(), 0)

    def test_matern_rbf_surface_regressor(self) -> None:
        coords, shifts = _make_synthetic_data(64)
        for kernel in ("matern32", "matern52"):
            model = RBFSurfaceRegressor(
                kernel=kernel, smoothing=1e-4, epsilon=10.0, neighbors=20
            )
            model.fit(coords, shifts)
            preds = model.predict(coords[:12])
            self.assertEqual(preds.shape, (12, 2))
            self.assertTrue(np.all(np.isfinite(preds)))

    def test_hist_gbr_grid_shift_model(self) -> None:
        coords, shifts = _make_synthetic_data(64)
        model = HistGBRGridShiftModel(
            max_iter=10, max_leaf_nodes=15, use_cyclic_cell_pos=True
        )
        model.fit(coords, shifts)
        preds = model.predict(coords[:10])
        self.assertEqual(preds.shape, (10, 2))
        self.assertTrue(np.all(np.isfinite(preds)))

    def test_adaptive_hybrid_train_disjoint_from_adaptive_3mm_test(self) -> None:
        gm = GridManager()
        train_ds = gm.get(GridId.RO_TRAIN_ADAPTIVE_HYBRID, force_regenerate=True)
        test_ds = gm.get(GridId.RO_TEST_ADAPTIVE_3MM_MID)

        train_keys = set(
            zip(
                np.round(train_ds.coords[:, 0], 2),
                np.round(train_ds.coords[:, 1], 2),
                strict=True,
            )
        )
        test_keys = set(
            zip(
                np.round(test_ds.coords[:, 0], 2),
                np.round(test_ds.coords[:, 1], 2),
                strict=True,
            )
        )
        self.assertEqual(len(train_keys & test_keys), 0)


if __name__ == "__main__":
    unittest.main()
