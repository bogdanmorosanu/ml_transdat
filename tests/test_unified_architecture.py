"""Unit tests for the unified architecture: GridManager, ModelEvaluator, PlotDispatcher, and models."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from pytransdat import TransRO
from pytransdat.spg_reader import SpgReader
from src.data import (
    DEFAULT_SUBGRID_OFFSETS_4X4,
    GridDataset,
    GridId,
    get_default_grid_manager,
    iter_densified_grid_chunks,
)
from src.evaluation import (
    MetricType,
    ModelEvaluator,
    compute_shift_errors,
    summarize_errors,
)
from src.models import (
    AdaptiveQuadtreeRegressor,
    DecisionTreeSpatialRegressor,
    Helmert2DParams,
    KNNSpatialRegressor,
    PyprojEpsg3844Transformer,
    ResidualCoordinateTransformer,
    TwoStepHelmert2DTransformer,
)
from src.viz import (
    PlotDispatcher,
    PlotGroup,
    PlotType,
    plot_model_comparison,
    resolve_plot_types,
)


class TestUnifiedArchitecture(unittest.TestCase):
    """Verify end-to-end contracts across data, models, evaluation, and viz."""

    def test_grid_manager_and_streaming(self) -> None:
        gm = get_default_grid_manager()
        ds = gm.get(GridId.SPG_NODES_VALID)
        self.assertIsInstance(ds, GridDataset)
        self.assertGreater(len(ds), 2000)
        self.assertEqual(ds.coords.shape, (len(ds), 2))
        self.assertEqual(ds.true_shifts.shape, (len(ds), 2))

        chunks = list(ds.iter_chunks(batch_size=500))
        self.assertGreater(len(chunks), 4)
        total_pts = sum(c[0].shape[0] for c in chunks)
        self.assertEqual(total_pts, len(ds))

    def test_lazy_chunk_streaming(self) -> None:
        gen = iter_densified_grid_chunks(DEFAULT_SUBGRID_OFFSETS_4X4, batch_size=256)
        c1, s1 = next(gen)
        self.assertEqual(c1.shape, (256, 2))
        self.assertEqual(s1.shape, (256, 2))

    def test_fit_stream_and_evaluator(self) -> None:
        gm = get_default_grid_manager()
        ds_nodes = gm.get(GridId.SPG_NODES_VALID)

        model = KNNSpatialRegressor(n_neighbors=4, weights="distance")
        model.fit_stream(ds_nodes.iter_chunks(batch_size=1000))
        self.assertTrue(model.is_fitted)
        self.assertGreater(model.memory_footprint_bytes(), 0)

        evaluator = ModelEvaluator()
        res = evaluator.evaluate_shift_model(model, ds_nodes, model_name="knn_k4_test")
        self.assertEqual(res.summary["count"], len(ds_nodes))
        self.assertLess(res.summary["mean_2d_m"], 1e-9)

    def test_residual_coordinate_transformer_matches_transro(self) -> None:
        gm = get_default_grid_manager()
        ds_nodes = gm.get(GridId.SPG_NODES_VALID)
        ds_4x4 = gm.get(GridId.RO_GRID_4X4)
        df_sub = ds_4x4.df.iloc[:100]

        reader = SpgReader()
        emb_params = Helmert2DParams(
            tn_m=float(reader.helmert_tn),
            te_m=float(reader.helmert_te),
            ppm=float(reader.helmert_ppm),
            rot_rad=float(reader.helmert_rot_rad),
            rot_arcsec=float(reader.helmert_rot_rad * 180.0 * 3600.0 / math.pi),
        )
        base = TwoStepHelmert2DTransformer(emb_params)
        knn = KNNSpatialRegressor(n_neighbors=4, weights="distance").fit(
            ds_nodes.coords, ds_nodes.true_shifts
        )
        comp = ResidualCoordinateTransformer(base_transformer=base, shift_model=knn)

        # Forward and inverse against TransRO ground-truth columns (must match to ~KNN accuracy, < 5 cm)
        ev = ModelEvaluator()
        res_inv = ev.evaluate_transformer(
            comp, GridDataset(GridId.CUSTOM, df_sub), "etrs89_to_st70", "knn_residual"
        )
        res_fwd = ev.evaluate_transformer(
            comp, GridDataset(GridId.CUSTOM, df_sub), "st70_to_etrs89", "knn_residual"
        )
        self.assertLess(res_inv.summary["mean_2d_m"], 0.05)
        self.assertLess(res_fwd.summary["mean_2d_m"], 0.05)

        # Direct comparison vs TransRO on a known point
        tr = TransRO()
        lat_gt, lon_gt = tr.st70_to_etrs89(
            float(df_sub["st70_n"].iloc[0]), float(df_sub["st70_e"].iloc[0])
        )
        lat_pred, lon_pred = comp.st70_to_etrs89(
            [float(df_sub["st70_n"].iloc[0])], [float(df_sub["st70_e"].iloc[0])]
        )
        self.assertLess(abs(float(lat_pred[0]) - lat_gt), 1e-6)
        self.assertLess(abs(float(lon_pred[0]) - lon_gt), 1e-6)

    def test_plot_registry_and_dispatcher(self) -> None:
        resolved = resolve_plot_types(
            [PlotGroup.MINIMAL, PlotType.TRANSECT_66KM, PlotType.SPATIAL_ERROR]
        )
        self.assertEqual(
            resolved,
            [PlotType.SPATIAL_ERROR, PlotType.HIST_ECDF, PlotType.TRANSECT_66KM],
        )

        gm = get_default_grid_manager()
        ds_nodes = gm.get(GridId.SPG_NODES_VALID)
        dt = DecisionTreeSpatialRegressor(max_depth=6).fit(
            ds_nodes.coords, ds_nodes.true_shifts
        )
        res = ModelEvaluator().evaluate_shift_model(dt, ds_nodes, model_name="dt_smoke")

        with tempfile.TemporaryDirectory() as tmpdir:
            dispatcher = PlotDispatcher()
            paths = dispatcher.render(
                res,
                out_dir=Path(tmpdir),
                plots=[PlotGroup.MINIMAL, PlotType.TRANSECT_66KM],
                dpi=80,
                transect_models={"DT depth=6": dt},
            )
            self.assertIn(PlotType.SPATIAL_ERROR, paths)
            self.assertIn(PlotType.HIST_ECDF, paths)
            self.assertIn(PlotType.TRANSECT_66KM, paths)
            for p in paths.values():
                self.assertTrue(p.is_file())

    def test_plot_dispatcher_rejects_unhandled_plot_types(self) -> None:
        gm = get_default_grid_manager()
        ds_nodes = gm.get(GridId.SPG_NODES_VALID)
        dt = DecisionTreeSpatialRegressor(max_depth=4).fit(
            ds_nodes.coords, ds_nodes.true_shifts
        )
        res = ModelEvaluator().evaluate_shift_model(dt, ds_nodes, model_name="dt_err")
        with tempfile.TemporaryDirectory() as tmpdir:
            dispatcher = PlotDispatcher()
            with self.assertRaises(ValueError):
                dispatcher.render(
                    res, out_dir=Path(tmpdir), plots=PlotType.MODEL_COMPARISON
                )
            with self.assertRaises(ValueError):
                dispatcher.render(
                    res, out_dir=Path(tmpdir), plots=PlotType.TRANSECT_66KM
                )

    def test_quadtree_vectorized_matches_exact_and_handles_nan(self) -> None:
        gm = get_default_grid_manager()
        dn_df, de_df = gm.get_adaptive_pair("2cm")
        corridor_df = gm.get(GridId.CORRIDOR_SNW).df.iloc[:100]

        qt = AdaptiveQuadtreeRegressor(dn_grid=dn_df.df, de_grid=de_df.df)
        self.assertGreater(qt.fit_time_sec, 0.0)
        self.assertGreater(qt.memory_breakdown().grid_lookup_bytes, 0)

        # Include exact cell corner and upper-right edges alongside corridor samples
        r0 = dn_df.df.iloc[0]
        edge_pts = np.array(
            [
                [r0["n_min"], r0["e_min"]],
                [r0["n_max"], r0["e_max"]],
                [r0["n_min"], r0["e_max"]],
            ],
            dtype=np.float64,
        )
        q_pts = np.vstack(
            [corridor_df[["st70_n", "st70_e"]].to_numpy(dtype=np.float64), edge_pts]
        )
        preds = qt.predict(q_pts)

        # Reference brute-force exact check on first 20 + edge points
        def _bf_query(grid_df: pd.DataFrame, pts: np.ndarray) -> np.ndarray:
            n_min = grid_df["n_min"].to_numpy()
            n_max = grid_df["n_max"].to_numpy()
            e_min = grid_df["e_min"].to_numpy()
            e_max = grid_df["e_max"].to_numpy()
            v00 = grid_df["v00"].to_numpy()
            v01 = grid_df["v01"].to_numpy()
            v10 = grid_df["v10"].to_numpy()
            v11 = grid_df["v11"].to_numpy()
            out = np.full(len(pts), np.nan, dtype=np.float64)
            for i, (n, e) in enumerate(pts):
                m = (n_min <= n) & (n <= n_max) & (e_min <= e) & (e <= e_max)
                if m.any():
                    idx = int(np.argmax(m))
                    u = (n - n_min[idx]) / (n_max[idx] - n_min[idx])
                    v = (e - e_min[idx]) / (e_max[idx] - e_min[idx])
                    out[i] = (
                        (1 - u) * (1 - v) * v00[idx]
                        + (1 - u) * v * v01[idx]
                        + u * (1 - v) * v10[idx]
                        + u * v * v11[idx]
                    )
            return out

        check_pts = np.vstack([q_pts[:20], edge_pts])
        expected_dn = _bf_query(dn_df.df, check_pts)
        expected_de = _bf_query(de_df.df, check_pts)
        actual = np.vstack([preds[:20], preds[-3:]])
        np.testing.assert_allclose(actual[:, 0], expected_dn, atol=1e-12)
        np.testing.assert_allclose(actual[:, 1], expected_de, atol=1e-12)

        # Out-of-bounds point returns NaN and summarize_errors remains finite
        oob_pts = np.vstack([q_pts[:5], np.array([[-999_999.0, -999_999.0]])])
        oob_preds = qt.predict(oob_pts)
        self.assertTrue(np.isnan(oob_preds[-1, 0]))
        df_tmp = pd.DataFrame({"st70_n": oob_pts[:, 0], "st70_e": oob_pts[:, 1]})
        err_df = compute_shift_errors(df_tmp, oob_preds, np.zeros_like(oob_preds))
        summary = summarize_errors(err_df)
        self.assertEqual(summary["count"], 5)
        self.assertEqual(summary["total_count"], 6)
        self.assertEqual(summary["unmatched_count"], 1)
        self.assertTrue(np.isfinite(summary["mean_2d_m"]))

    def test_knn_kneighbors_and_memory_breakdown(self) -> None:
        gm = get_default_grid_manager()
        ds_nodes = gm.get(GridId.SPG_NODES_VALID)
        knn = KNNSpatialRegressor(n_neighbors=4, weights="distance").fit(
            ds_nodes.coords, ds_nodes.true_shifts
        )
        dists, idxs = knn.kneighbors(ds_nodes.coords[:3])
        self.assertEqual(dists.shape, (3, 4))
        self.assertEqual(idxs.shape, (3, 4))

        mb = knn.memory_breakdown()
        self.assertGreater(mb.code_bytes, 0)
        self.assertGreater(mb.dependencies_bytes, 0)
        self.assertGreater(mb.grid_lookup_bytes, 0)

    def test_assess_custom_arrays_uses_custom_id(self) -> None:
        gm = get_default_grid_manager()
        ds_nodes = gm.get(GridId.SPG_NODES_VALID)
        knn = KNNSpatialRegressor(n_neighbors=4, weights="distance").fit(
            ds_nodes.coords, ds_nodes.true_shifts
        )
        eval_res = knn.assess(ds_nodes.coords[:20], ds_nodes.true_shifts[:20])
        self.assertEqual(eval_res.dataset_id, GridId.CUSTOM.value)
        self.assertEqual(eval_res.summary["count"], 20)

        report_txt = knn.report(eval_res)
        self.assertIn("Evaluation Scorecard", report_txt)

    def test_evaluate_models_and_tabulate_results_mm(self) -> None:
        gm = get_default_grid_manager()
        ds_nodes = gm.get(GridId.SPG_NODES_VALID)
        knn = KNNSpatialRegressor(n_neighbors=4, weights="distance").fit(
            ds_nodes.coords, ds_nodes.true_shifts
        )
        dt = DecisionTreeSpatialRegressor(max_depth=6).fit(
            ds_nodes.coords, ds_nodes.true_shifts
        )
        metrics = [
            MetricType.MEAN_2D_M,
            MetricType.RMSE_2D_M,
            MetricType.P95_2D_M,
            MetricType.EVAL_TIME_SEC,
        ]
        ev = ModelEvaluator()
        results = ev.evaluate_models([("KNN K=4", knn), ("DT d=6", dt)], ds_nodes)
        df_mm = ev.tabulate_results(results, metrics=metrics, precision_unit="mm")
        self.assertEqual([r.model_name for r in results], ["KNN K=4", "DT d=6"])
        self.assertIn("Mean 2D Error (mm)", df_mm.columns)
        self.assertAlmostEqual(
            float(df_mm.loc[1, "Mean 2D Error (mm)"]),
            results[1].summary["mean_2d_m"] * 1000.0,
            places=6,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_png = Path(tmpdir) / "cmp.png"
            plot_model_comparison(
                results, out_png, metrics=metrics, precision_unit="mm", dpi=80
            )
            self.assertTrue(out_png.is_file())

    def test_pyproj_transformer_and_memory_breakdown(self) -> None:
        gm = get_default_grid_manager()
        ds_4x4 = gm.get(GridId.RO_GRID_4X4)
        ds_sub = GridDataset(GridId.CUSTOM, ds_4x4.df.iloc[:25].copy())
        proj_tf = PyprojEpsg3844Transformer()
        res = proj_tf.assess(ds_sub, direction="etrs89_to_st70")
        self.assertEqual(res.summary["count"], 25)
        self.assertIsNotNone(res.memory_breakdown)
        self.assertGreater(res.memory_breakdown.dependencies_bytes, 0)  # type: ignore[union-attr]

    def test_analysis_unit_and_v2_brief_schema(self) -> None:
        from src.core import AnalysisExperimentUnit, ExperimentType

        class _DummyAnalysisUnit(AnalysisExperimentUnit):
            @property
            def slug(self) -> str:
                return "dummy_analysis"

            def run_analysis(self) -> dict[str, object]:
                return {
                    "key_metrics": {"count": 42},
                    "artifacts": {"plots": ["demo.png"]},
                    "extra": {"note": "ok"},
                }

        with tempfile.TemporaryDirectory() as tmpdir:
            unit = _DummyAnalysisUnit(Path(tmpdir))
            brief = unit.run()
            self.assertEqual(brief["schema_version"], "2.0")
            self.assertEqual(brief["slug"], "dummy_analysis")
            self.assertEqual(brief["experiment_type"], ExperimentType.ANALYSIS.value)
            self.assertEqual(brief["key_metrics"], {"count": 42})
            self.assertEqual(brief["benchmark_candidates"], [])
            self.assertTrue((Path(tmpdir) / "unit_brief.json").is_file())

    def test_specs_from_unit_and_geometry_spg_reader(self) -> None:
        from experiments.knn_spatial_regressor.run_unit import KnnSpatialRegressorUnit
        from src.core import specs_from_unit
        from src.geometry.plane import extract_grid_cell_planes

        specs = specs_from_unit(
            KnnSpatialRegressorUnit,
            suite="grid_shift_models",
            unit_dir=Path("experiments/knn_spatial_regressor"),
        )
        self.assertEqual(len(specs), 2)
        self.assertEqual(specs[0].source_experiment, "knn_spatial_regressor")

        cells = extract_grid_cell_planes()
        self.assertGreater(len(cells), 2000)


if __name__ == "__main__":
    unittest.main()
