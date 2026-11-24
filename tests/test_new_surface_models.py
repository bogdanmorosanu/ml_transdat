"""Unit tests for BSplineSurfaceRegressor and LocalPolynomialSurfaceRegressor."""

from __future__ import annotations

import numpy as np

from src.data import CellRegion, GridId, GridManager
from src.models import BSplineSurfaceRegressor, LocalPolynomialSurfaceRegressor


def test_bspline_surface() -> None:
    """Verify BSplineSurfaceRegressor fits RO_GRID_4X4 and predicts finite sub-decimeter shifts."""
    gm = GridManager()
    train_ds = gm.get(GridId.RO_GRID_4X4)
    region = CellRegion(row_start=19, col_start=28, n_rows=10, n_cols=10)
    test_ds = gm.get_region(GridId.RO_TEST_UNIFORM_4X4, region)

    model = BSplineSurfaceRegressor(degree=3, smoothing=1e-2)
    model.fit(train_ds.coords, train_ds.true_shifts)
    preds = model.predict(test_ds.coords)

    assert preds.shape == test_ds.true_shifts.shape
    assert np.all(np.isfinite(preds))
    rmse_m = float(np.sqrt(np.mean(np.sum((preds - test_ds.true_shifts) ** 2, axis=1))))
    assert rmse_m < 0.15, f"Unexpectedly high B-Spline RMSE: {rmse_m:.4f} m"


def test_local_poly_surface() -> None:
    """Verify LocalPolynomialSurfaceRegressor fits RO_GRID_4X4 and predicts sub-centimeter shifts."""
    gm = GridManager()
    train_ds = gm.get(GridId.RO_GRID_4X4)
    region = CellRegion(row_start=19, col_start=28, n_rows=10, n_cols=10)
    test_ds = gm.get_region(GridId.RO_TEST_UNIFORM_4X4, region)

    model = LocalPolynomialSurfaceRegressor(degree=2, neighbors=25, weight="tricube")
    model.fit(train_ds.coords, train_ds.true_shifts)
    preds = model.predict(test_ds.coords)

    assert preds.shape == test_ds.true_shifts.shape
    assert np.all(np.isfinite(preds))
    rmse_m = float(np.sqrt(np.mean(np.sum((preds - test_ds.true_shifts) ** 2, axis=1))))
    assert rmse_m < 0.02, f"Unexpectedly high LocalPoly RMSE: {rmse_m:.4f} m"
