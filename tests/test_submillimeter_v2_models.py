"""Unit tests for all 13 submillimeter_exploration_v2 model classes."""

from __future__ import annotations

import unittest

import numpy as np

from src.models import (
    BaseGridShiftModel,
    KANGridShiftModel,
    KPlanesGridShiftModel,
    LinearLeafTreeShiftModel,
    LocalKrigingShiftModel,
    LSSVMGridShiftModel,
    MemoryBreakdown,
    MFNGridShiftModel,
    NeuralBicubicHeadShiftModel,
    PartitionOfUnityRBFShiftModel,
    RFFELMGridShiftModel,
    SpatialMoEGridShiftModel,
    WaveletGridShiftModel,
    WaveletINRGridShiftModel,
    XGBoostGridShiftModel,
)


def _make_synthetic_box(n_side: int = 15) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic 2D coordinates inside Romania's SPG grid bounds."""
    n_vals = np.linspace(380_000.0, 468_000.0, n_side)
    e_vals = np.linspace(440_000.0, 528_000.0, n_side)
    gn, ge = np.meshgrid(n_vals, e_vals, indexing="ij")
    coords = np.column_stack([gn.ravel(), ge.ravel()])
    un = (coords[:, 0] - 380_000.0) / 88_000.0
    ue = (coords[:, 1] - 440_000.0) / 88_000.0
    dn = 0.015 * np.sin(2.0 * np.pi * un) + 0.005 * (ue - 0.5) ** 2
    de = 0.012 * np.cos(2.0 * np.pi * ue) - 0.004 * (un - 0.5) * (ue - 0.5)
    shifts = np.column_stack([dn, de])
    return coords, shifts


class TestSubmillimeterV2Models(unittest.TestCase):
    """Smoke, contract, memory, and zero-PyTorch checks for v2 candidate models."""

    def test_all_models_fit_predict_and_memory(self) -> None:
        coords, shifts = _make_synthetic_box(15)
        models: list[tuple[BaseGridShiftModel, bool]] = [
            (KANGridShiftModel(basis="rbf", hidden_dim=16, num_layers=2, num_basis=4, max_epochs=2, device="cpu"), False),
            (KANGridShiftModel(basis="fourier", hidden_dim=16, num_layers=2, num_basis=4, max_epochs=2, device="cpu"), False),
            (WaveletINRGridShiftModel(activation="wire", hidden_dim=16, num_layers=2, max_epochs=2, device="cpu"), False),
            (WaveletINRGridShiftModel(activation="finer", hidden_dim=16, num_layers=2, max_epochs=2, device="cpu"), False),
            (MFNGridShiftModel(filter_type="gabor", hidden_dim=16, num_layers=2, max_epochs=2, device="cpu"), False),
            (SpatialMoEGridShiftModel(mode="moe", num_experts=4, hidden_dim=16, max_epochs=2, device="cpu"), False),
            (SpatialMoEGridShiftModel(mode="fbpinn", fbpinn_grid=(2, 2), hidden_dim=16, max_epochs=2, device="cpu"), False),
            (NeuralBicubicHeadShiftModel(backend="grid", max_epochs=2, device="cpu"), False),
            (NeuralBicubicHeadShiftModel(backend="mlp", hidden_dim=16, num_res_blocks=1, max_epochs=2, device="cpu"), False),
            (KPlanesGridShiftModel(n_bins=16, feature_dim=4, hidden_dim=16, max_epochs=2, device="cpu"), False),
            (RFFELMGridShiftModel(n_features=64), True),
            (LSSVMGridShiftModel(n_components=64), True),
            (XGBoostGridShiftModel(n_estimators=10, max_depth=3, n_jobs=1), True),
            (LinearLeafTreeShiftModel(max_depth=4, min_samples_leaf=10, leaf_degree=2), True),
            (LocalKrigingShiftModel(neighbors=12), True),
            (PartitionOfUnityRBFShiftModel(patch_grid=(3, 3), overlap=1.8), True),
            (WaveletGridShiftModel(grid_step_m=11_000.0, levels=2, threshold_mm=0.1), True),
        ]

        for model, expect_zero_torch in models:
            with self.subTest(model=type(model).__name__):
                cfg = model.get_config_dict()
                self.assertIsInstance(cfg, dict)
                fp = model.compute_sync_fingerprint("ro_grid_4x4")
                self.assertEqual(len(fp), 64)

                model.fit(coords, shifts)
                pred = model.predict(coords[:25])
                self.assertEqual(pred.shape, (25, 2))
                self.assertTrue(np.all(np.isfinite(pred)))

                mb = model.memory_breakdown()
                self.assertIsInstance(mb, MemoryBreakdown)
                deps = model._runtime_dependencies()
                if expect_zero_torch:
                    self.assertNotIn("torch", deps)
                else:
                    self.assertIn("torch", deps)

    def test_local_kriging_predict_with_std(self) -> None:
        coords, shifts = _make_synthetic_box(12)
        krig = LocalKrigingShiftModel(neighbors=10)
        krig.fit(coords, shifts)
        mean, std = krig.predict_with_std(coords[:15])
        self.assertEqual(mean.shape, (15, 2))
        self.assertEqual(std.shape, (15, 2))
        self.assertTrue(np.all(np.isfinite(mean)))
        self.assertTrue(np.all(np.isfinite(std)))
        self.assertTrue(np.all(std >= 0.0))


if __name__ == "__main__":
    unittest.main()
