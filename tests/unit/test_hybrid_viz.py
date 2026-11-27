"""Unit tests for hybrid deep-dive visualizations and champion selection."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.base_unit import ExperimentModelSpec, GridShiftExperimentUnit
from src.data.catalog import GridDataset, GridId
from src.evaluation.evaluator import EvaluationResult
from src.models.base import BaseGridShiftModel
from src.models.composite_shift import CompositeResidualShiftModel
from src.viz import (
    PlotGroup,
    PlotType,
    render_hybrid_deep_dive,
    resolve_plot_types,
    select_hybrid_eval_grid,
)


class _ConstShiftModel(BaseGridShiftModel):
    def __init__(self, offset: tuple[float, float]) -> None:
        super().__init__()
        self.offset = np.asarray(offset, dtype=np.float64)
        self._is_fitted = True

    def _fit_impl(self, coords: np.ndarray, shifts: np.ndarray) -> None:
        del coords, shifts

    def _predict_impl(self, coords: np.ndarray) -> np.ndarray:
        return np.tile(self.offset, (len(coords), 1))


class _DummyGridUnit(GridShiftExperimentUnit):
    @property
    def slug(self) -> str:
        return "dummy_hybrid_unit"

    def get_model_specs(self) -> list[ExperimentModelSpec]:
        return []


class TestHybridViz(unittest.TestCase):
    """Verify PlotGroup.HYBRID_DEEP_DIVE, rendering, and champion selection."""

    def test_plot_group_hybrid_deep_dive_members(self) -> None:
        resolved = resolve_plot_types(PlotGroup.HYBRID_DEEP_DIVE)
        self.assertEqual(
            resolved,
            [
                PlotType.HYBRID_EXTREME_RESIDUALS_9CELLS,
                PlotType.HYBRID_CANCELLATION_SCATTER,
                PlotType.HYBRID_SPATIAL_3PANEL,
                PlotType.HYBRID_SUBCELL_PHASE_HEATMAP,
            ],
        )

    def test_select_hybrid_eval_grid_prefers_uniform(self) -> None:
        picked = select_hybrid_eval_grid(
            [GridId.RO_TEST_ADAPTIVE_1MM_MID, GridId.RO_TEST_UNIFORM_16X16]
        )
        self.assertEqual(picked, GridId.RO_TEST_UNIFORM_16X16)

        fallback = select_hybrid_eval_grid([GridId.RO_TEST_ADAPTIVE_1MM_MID])
        self.assertEqual(fallback, GridId.RO_TEST_ADAPTIVE_1MM_MID)

    def test_render_hybrid_deep_dive_creates_all_pngs(self) -> None:
        hybrid = CompositeResidualShiftModel(
            stage1_model=_ConstShiftModel((100.0, 200.0)),
            stage2_model=_ConstShiftModel((0.002, -0.001)),
        )
        hybrid._is_fitted = True

        rng = np.random.default_rng(42)
        n_pts = 400
        coords = np.column_stack(
            [
                rng.uniform(400_000.0, 500_000.0, size=n_pts),
                rng.uniform(400_000.0, 550_000.0, size=n_pts),
            ]
        )
        true_shifts = np.tile([100.002, 199.999], (n_pts, 1)) + rng.normal(
            0.0, 0.0005, size=(n_pts, 2)
        )
        ds = GridDataset(
            grid_id="ro_test_uniform_16x16",
            df=pd.DataFrame(
                {
                    "st70_n": coords[:, 0],
                    "st70_e": coords[:, 1],
                    "true_dn": true_shifts[:, 0],
                    "true_de": true_shifts[:, 1],
                }
            ),
        )

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            paths = render_hybrid_deep_dive(
                hybrid_model=hybrid,
                model_name="MockHybrid",
                eval_grids=[GridId.RO_TEST_UNIFORM_16X16],
                load_grid_fn=lambda _gid: ds,
                out_dir=out_dir,
                dpi=100,
            )
            self.assertEqual(len(paths), 5)
            for p in paths:
                self.assertTrue(p.is_file(), f"Missing output plot: {p}")
                self.assertGreater(p.stat().st_size, 1000)

    def test_select_champion_hybrid_prefers_benchmark_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            unit = _DummyGridUnit(Path(tmp))
            h1 = CompositeResidualShiftModel(_ConstShiftModel((0.0, 0.0)), _ConstShiftModel((0.0, 0.0)))
            h2 = CompositeResidualShiftModel(_ConstShiftModel((0.0, 0.0)), _ConstShiftModel((0.0, 0.0)))
            specs = [
                ExperimentModelSpec(
                    name="h1", model=h1, eval_grids=[], benchmark_candidate=False
                ),
                ExperimentModelSpec(
                    name="h2", model=h2, eval_grids=[], benchmark_candidate=True
                ),
            ]
            kept = {"h1": h1, "h2": h2}
            results = [
                EvaluationResult(
                    model_name="h1",
                    dataset_id="d1",
                    direction="shift",
                    err_df=pd.DataFrame(),
                    summary={"mean_2d_m": 0.0002},
                    eval_time_sec=0.1,
                    throughput_pts_per_sec=100.0,
                ),
                EvaluationResult(
                    model_name="h2",
                    dataset_id="d1",
                    direction="shift",
                    err_df=pd.DataFrame(),
                    summary={"mean_2d_m": 0.0003},
                    eval_time_sec=0.1,
                    throughput_pts_per_sec=100.0,
                ),
            ]
            picked = unit._select_champion_hybrid(specs, kept, results)
            self.assertIsNotNone(picked)
            assert picked is not None
            self.assertEqual(picked[0].name, "h2")


if __name__ == "__main__":
    unittest.main()
