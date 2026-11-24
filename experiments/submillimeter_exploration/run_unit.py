"""Multi-track exploratory experiment screening 7 sub-millimeter improvement tracks.

Evaluates feature engineering (multi-harmonic cyclic cell waves, multi-band RFF),
training optimization (100% grid fit, Huber loss, L-BFGS polishing), two-stage
coarse-to-fine residual decompositions, curvature-adaptive and dense datasets,
learnable 2D multi-resolution feature grids, untested model families (SIREN,
Matérn RBF, HistGBR), and bidirectional ETRS89 <-> Stereo70 verification.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.core.base_unit import ExperimentModelSpec, GridShiftExperimentUnit
from src.data import GridId
from src.evaluation.evaluator import EvaluationResult
from src.evaluation.metrics import dataframe_to_markdown
from src.models import (
    BaseGridShiftModel,
    CellBilinearShiftModel,
    CompositeResidualShiftModel,
    HistGBRGridShiftModel,
    NeuralFeatureGridShiftModel,
    RBFSurfaceRegressor,
    ResidualCoordinateTransformer,
    TorchMLPGridShiftModel,
    TwoStepHelmert2DTransformer,
    load_embedded_spg_helmert_params,
)
from src.viz import PlotGroup

logger = logging.getLogger(__name__)

EVAL_GRIDS: list[GridId] = [
    GridId.RO_TEST_UNIFORM_4X4,
    GridId.RO_TEST_ADAPTIVE_3MM_MID,
    GridId.CORRIDOR_SNW,
]


class SubmillimeterExplorationUnit(GridShiftExperimentUnit):
    """Multi-track exploratory screening unit for sub-millimeter coordinate shifts."""

    @property
    def slug(self) -> str:
        return "submillimeter_exploration"

    def _build_prefitted_tps_11km(self) -> RBFSurfaceRegressor:
        """Load or fit the compact 11 km Thin Plate Spline on valid SPG nodes for Track 3."""
        tps = RBFSurfaceRegressor(
            kernel="thin_plate_spline",
            degree=1,
            smoothing=2e-4,
        )
        if self._probe_only:
            return tps
        tps.load_or_fit(
            lambda: (
                self.load_grid(GridId.SPG_NODES_VALID).coords,
                self.load_grid(GridId.SPG_NODES_VALID).true_shifts,
            ),
            cache_path=self.models_dir / "_stage1_tps_11km.pkl",
            train_grid=GridId.SPG_NODES_VALID.value,
            label="stage1_tps_11km",
        )
        return tps

    def get_model_specs(self) -> list[ExperimentModelSpec]:
        tps_11km = self._build_prefitted_tps_11km()
        return [
            # ------------------------------------------------------------------
            # B0: Anchor Champion Baseline (torch_res_relu_cellpos_4x4)
            # ------------------------------------------------------------------
            ExperimentModelSpec(
                name="B0_torch_res_relu_cellpos_4x4",
                model=TorchMLPGridShiftModel(
                    hidden_dim=256,
                    num_res_blocks=2,
                    activation="relu",
                    use_fourier_features=True,
                    fourier_scale=3.0,
                    num_fourier_frequencies=64,
                    use_cyclic_cell_pos=True,
                    batch_size=256,
                    lr=1e-3,
                    max_epochs=200,
                    patience=25,
                ),
                train_grid=GridId.RO_GRID_4X4,
                eval_grids=EVAL_GRIDS,
                plots=PlotGroup.STANDARD_DIAGNOSTIC,
                keep_in_memory=False,
            ),
            # ------------------------------------------------------------------
            # Track 1 (Rank 3): Harmonic Cell Waves & Multi-Band Fourier
            # ------------------------------------------------------------------
            ExperimentModelSpec(
                name="T1_harmonics4_4x4",
                model=TorchMLPGridShiftModel(
                    hidden_dim=256,
                    num_res_blocks=2,
                    activation="relu",
                    use_fourier_features=True,
                    fourier_scale=3.0,
                    num_fourier_frequencies=64,
                    use_cyclic_cell_pos=True,
                    cyclic_cell_harmonics=4,
                    batch_size=256,
                    lr=1e-3,
                    max_epochs=200,
                    patience=25,
                ),
                train_grid=GridId.RO_GRID_4X4,
                eval_grids=EVAL_GRIDS,
                plots=None,
                keep_in_memory=True,
            ),
            ExperimentModelSpec(
                name="T1_multiscale_rff_4x4",
                model=TorchMLPGridShiftModel(
                    hidden_dim=256,
                    num_res_blocks=2,
                    activation="relu",
                    use_fourier_features=True,
                    fourier_scales=(1.0, 3.0, 8.0, 16.0),
                    num_fourier_frequencies=64,
                    use_cyclic_cell_pos=True,
                    batch_size=256,
                    lr=1e-3,
                    max_epochs=200,
                    patience=25,
                ),
                train_grid=GridId.RO_GRID_4X4,
                eval_grids=EVAL_GRIDS,
                plots=None,
                keep_in_memory=False,
            ),
            ExperimentModelSpec(
                name="T1_harmonics4_multiscale_4x4",
                model=TorchMLPGridShiftModel(
                    hidden_dim=256,
                    num_res_blocks=2,
                    activation="relu",
                    use_fourier_features=True,
                    fourier_scales=(1.0, 3.0, 8.0, 16.0),
                    num_fourier_frequencies=64,
                    use_cyclic_cell_pos=True,
                    cyclic_cell_harmonics=4,
                    batch_size=256,
                    lr=1e-3,
                    max_epochs=200,
                    patience=25,
                ),
                train_grid=GridId.RO_GRID_4X4,
                eval_grids=EVAL_GRIDS,
                plots=None,
                keep_in_memory=True,
            ),
            # ------------------------------------------------------------------
            # Track 2 (Rank 4): 100% Grid Fit, Huber Loss & L-BFGS Polishing
            # ------------------------------------------------------------------
            ExperimentModelSpec(
                name="T2_fullgrid_4x4",
                model=TorchMLPGridShiftModel(
                    hidden_dim=256,
                    num_res_blocks=2,
                    activation="relu",
                    use_fourier_features=True,
                    fourier_scale=3.0,
                    num_fourier_frequencies=64,
                    use_cyclic_cell_pos=True,
                    val_fraction=0.0,
                    batch_size=256,
                    lr=1e-3,
                    max_epochs=200,
                    patience=25,
                ),
                train_grid=GridId.RO_GRID_4X4,
                eval_grids=EVAL_GRIDS,
                plots=None,
                keep_in_memory=False,
            ),
            ExperimentModelSpec(
                name="T2_huber_fullgrid_4x4",
                model=TorchMLPGridShiftModel(
                    hidden_dim=256,
                    num_res_blocks=2,
                    activation="relu",
                    use_fourier_features=True,
                    fourier_scale=3.0,
                    num_fourier_frequencies=64,
                    use_cyclic_cell_pos=True,
                    val_fraction=0.0,
                    loss_type="huber",
                    batch_size=256,
                    lr=1e-3,
                    max_epochs=200,
                    patience=25,
                ),
                train_grid=GridId.RO_GRID_4X4,
                eval_grids=EVAL_GRIDS,
                plots=None,
                keep_in_memory=False,
            ),
            ExperimentModelSpec(
                name="T2_lbfgs_polish_4x4",
                model=TorchMLPGridShiftModel(
                    hidden_dim=256,
                    num_res_blocks=2,
                    activation="relu",
                    use_fourier_features=True,
                    fourier_scale=3.0,
                    num_fourier_frequencies=64,
                    use_cyclic_cell_pos=True,
                    val_fraction=0.0,
                    lbfgs_steps=40,
                    batch_size=256,
                    lr=1e-3,
                    max_epochs=200,
                    patience=25,
                ),
                train_grid=GridId.RO_GRID_4X4,
                eval_grids=EVAL_GRIDS,
                plots=None,
                keep_in_memory=True,
            ),
            # ------------------------------------------------------------------
            # Track 3 (Rank 1): Coarse-to-Fine Two-Stage Residual Decomposition
            # ------------------------------------------------------------------
            ExperimentModelSpec(
                name="T3_tps11km_plus_resnet_4x4",
                model=CompositeResidualShiftModel(
                    stage1_model=tps_11km,
                    stage2_model=TorchMLPGridShiftModel(
                        hidden_dim=256,
                        num_res_blocks=2,
                        activation="relu",
                        use_fourier_features=True,
                        fourier_scale=3.0,
                        num_fourier_frequencies=64,
                        use_cyclic_cell_pos=True,
                        cyclic_cell_harmonics=4,
                        val_fraction=0.0,
                        batch_size=256,
                        lr=1e-3,
                        max_epochs=200,
                        patience=25,
                    ),
                ),
                train_grid=GridId.RO_GRID_4X4,
                eval_grids=EVAL_GRIDS,
                plots=None,
                keep_in_memory=True,
            ),
            ExperimentModelSpec(
                name="T3_bilinear_plus_resnet_4x4",
                model=CompositeResidualShiftModel(
                    stage1_model=CellBilinearShiftModel(),
                    stage2_model=TorchMLPGridShiftModel(
                        hidden_dim=256,
                        num_res_blocks=2,
                        activation="relu",
                        use_fourier_features=True,
                        fourier_scale=3.0,
                        num_fourier_frequencies=64,
                        use_cyclic_cell_pos=True,
                        cyclic_cell_harmonics=4,
                        val_fraction=0.0,
                        batch_size=256,
                        lr=1e-3,
                        max_epochs=200,
                        patience=25,
                    ),
                ),
                train_grid=GridId.RO_GRID_4X4,
                eval_grids=EVAL_GRIDS,
                plots=None,
                keep_in_memory=True,
            ),
            ExperimentModelSpec(
                name="T3_coarse_to_fine_nn_4x4",
                model=CompositeResidualShiftModel(
                    stage1_model=TorchMLPGridShiftModel(
                        hidden_dim=128,
                        num_res_blocks=2,
                        activation="relu",
                        use_fourier_features=True,
                        fourier_scale=1.5,
                        num_fourier_frequencies=32,
                        use_cyclic_cell_pos=False,
                        val_fraction=0.0,
                        batch_size=256,
                        lr=1e-3,
                        max_epochs=120,
                        patience=20,
                    ),
                    stage2_model=TorchMLPGridShiftModel(
                        hidden_dim=256,
                        num_res_blocks=2,
                        activation="relu",
                        use_fourier_features=True,
                        fourier_scales=(3.0, 8.0, 16.0),
                        num_fourier_frequencies=64,
                        use_cyclic_cell_pos=True,
                        cyclic_cell_harmonics=4,
                        val_fraction=0.0,
                        batch_size=256,
                        lr=1e-3,
                        max_epochs=150,
                        patience=25,
                    ),
                ),
                train_grid=GridId.RO_GRID_4X4,
                eval_grids=EVAL_GRIDS,
                plots=None,
                keep_in_memory=True,
            ),
            # ------------------------------------------------------------------
            # Track 4 (Rank 2): Smarter Training Datasets (with B0 Architecture)
            # ------------------------------------------------------------------
            ExperimentModelSpec(
                name="T4_data_8x8",
                model=TorchMLPGridShiftModel(
                    hidden_dim=256,
                    num_res_blocks=2,
                    activation="relu",
                    use_fourier_features=True,
                    fourier_scale=3.0,
                    num_fourier_frequencies=64,
                    use_cyclic_cell_pos=True,
                    batch_size=512,
                    lr=1e-3,
                    max_epochs=150,
                    patience=25,
                ),
                train_grid=GridId.RO_GRID_8X8,
                eval_grids=EVAL_GRIDS,
                plots=None,
                keep_in_memory=True,
            ),
            ExperimentModelSpec(
                name="T4_data_adaptive_hybrid",
                model=TorchMLPGridShiftModel(
                    hidden_dim=256,
                    num_res_blocks=2,
                    activation="relu",
                    use_fourier_features=True,
                    fourier_scale=3.0,
                    num_fourier_frequencies=64,
                    use_cyclic_cell_pos=True,
                    batch_size=512,
                    lr=1e-3,
                    max_epochs=200,
                    patience=25,
                ),
                train_grid=GridId.RO_TRAIN_ADAPTIVE_HYBRID,
                eval_grids=EVAL_GRIDS,
                plots=PlotGroup.STANDARD_DIAGNOSTIC,
                keep_in_memory=True,
                benchmark_candidate=True,
                benchmark_suite="grid_shift_models",
                benchmark_label="PyTorch ResNet + Cyclic CellPos (Adaptive Hybrid)",
            ),
            ExperimentModelSpec(
                name="T4_data_border_enriched",
                model=TorchMLPGridShiftModel(
                    hidden_dim=256,
                    num_res_blocks=2,
                    activation="relu",
                    use_fourier_features=True,
                    fourier_scale=3.0,
                    num_fourier_frequencies=64,
                    use_cyclic_cell_pos=True,
                    batch_size=512,
                    lr=1e-3,
                    max_epochs=200,
                    patience=25,
                ),
                train_grid=GridId.RO_TRAIN_BORDER_ENRICHED_4X4,
                eval_grids=EVAL_GRIDS,
                plots=None,
                keep_in_memory=False,
            ),
            # ------------------------------------------------------------------
            # Track 5 (Rank 5): Learnable Multi-Resolution 2D Feature Grid + MLP
            # ------------------------------------------------------------------
            ExperimentModelSpec(
                name="T5_neural_feature_grid_4x4",
                model=NeuralFeatureGridShiftModel(
                    grid_resolutions=((53, 72), (105, 143)),
                    feature_dim=8,
                    hidden_dim=64,
                    cyclic_cell_harmonics=2,
                    batch_size=256,
                    lr=3e-3,
                    max_epochs=200,
                    val_fraction=0.0,
                ),
                train_grid=GridId.RO_GRID_4X4,
                eval_grids=EVAL_GRIDS,
                plots=None,
                keep_in_memory=True,
            ),
            ExperimentModelSpec(
                name="T5_neural_feature_grid_8x8",
                model=NeuralFeatureGridShiftModel(
                    grid_resolutions=((53, 72), (105, 143)),
                    feature_dim=8,
                    hidden_dim=64,
                    cyclic_cell_harmonics=2,
                    batch_size=512,
                    lr=3e-3,
                    max_epochs=150,
                    val_fraction=0.0,
                ),
                train_grid=GridId.RO_GRID_8X8,
                eval_grids=EVAL_GRIDS,
                plots=PlotGroup.STANDARD_DIAGNOSTIC,
                keep_in_memory=True,
            ),
            # ------------------------------------------------------------------
            # Track 6 (Rank 6): Untested Pure-ML Model Families
            # ------------------------------------------------------------------
            ExperimentModelSpec(
                name="T6_siren_4x4",
                model=TorchMLPGridShiftModel(
                    hidden_dim=256,
                    num_res_blocks=2,
                    activation="siren",
                    siren_omega_0=15.0,
                    use_fourier_features=False,
                    use_cyclic_cell_pos=True,
                    cyclic_cell_harmonics=2,
                    val_fraction=0.0,
                    batch_size=256,
                    lr=5e-4,
                    max_epochs=200,
                    patience=25,
                ),
                train_grid=GridId.RO_GRID_4X4,
                eval_grids=EVAL_GRIDS,
                plots=None,
                keep_in_memory=False,
            ),
            ExperimentModelSpec(
                name="T6_matern32_rbf_4x4",
                model=RBFSurfaceRegressor(
                    kernel="matern32",
                    smoothing=1e-4,
                    epsilon=15.0,
                    neighbors=60,
                ),
                train_grid=GridId.RO_GRID_4X4,
                eval_grids=EVAL_GRIDS,
                plots=PlotGroup.STANDARD_DIAGNOSTIC,
                keep_in_memory=False,
            ),
            ExperimentModelSpec(
                name="T6_hist_gbr_4x4",
                model=HistGBRGridShiftModel(
                    max_iter=500,
                    max_leaf_nodes=63,
                    learning_rate=0.05,
                    use_cyclic_cell_pos=True,
                ),
                train_grid=GridId.RO_GRID_4X4,
                eval_grids=EVAL_GRIDS,
                plots=None,
                keep_in_memory=False,
            ),
            # ------------------------------------------------------------------
            # Synthesis Candidate: Bilinear Stage 1 + Feature Grid Stage 2 (8x8)
            # ------------------------------------------------------------------
            ExperimentModelSpec(
                name="T7_bilinear_plus_featgrid_8x8",
                model=CompositeResidualShiftModel(
                    stage1_model=CellBilinearShiftModel(),
                    stage2_model=NeuralFeatureGridShiftModel(
                        grid_resolutions=((53, 72), (105, 143)),
                        feature_dim=8,
                        hidden_dim=64,
                        cyclic_cell_harmonics=2,
                        batch_size=512,
                        lr=3e-3,
                        max_epochs=150,
                        val_fraction=0.0,
                    ),
                ),
                train_grid=GridId.RO_GRID_8X8,
                eval_grids=EVAL_GRIDS,
                plots=PlotGroup.STANDARD_DIAGNOSTIC,
                keep_in_memory=True,
                benchmark_candidate=True,
                benchmark_suite="grid_shift_models",
                benchmark_label="Bilinear + Neural Feature Grid (8x8)",
            ),
        ]

    def get_extreme_scenario_models(
        self,
        fitted_models: dict[str, BaseGridShiftModel],
    ) -> dict[str, BaseGridShiftModel]:
        label_map = {
            "T3_tps11km_plus_resnet_4x4": "T3 TPS11km + ResNet (4x4)",
            "T3_bilinear_plus_resnet_4x4": "T3 Bilinear + ResNet (4x4)",
            "T4_data_adaptive_hybrid": "T4 Adaptive Hybrid ResNet",
            "T7_bilinear_plus_featgrid_8x8": "T7 Bilinear + FeatGrid (8x8)",
        }
        return {
            lbl: fitted_models[k] for k, lbl in label_map.items() if k in fitted_models
        }

    def after_run(
        self,
        fitted_models: dict[str, BaseGridShiftModel],
        results: list[EvaluationResult],
    ) -> dict[str, Any]:
        """Generate track attribution summary and run Track 7 bidirectional check."""
        # 1. Build Delta-vs-Baseline Attribution Summary via ModelEvaluator
        df_attr = self.evaluator.tabulate_delta_vs_baseline(
            results=results,
            baseline_model="B0_torch_res_relu_cellpos_4x4",
            precision_unit="mm",
        )
        df_attr.to_csv(self.results_dir / "track_attribution_summary.csv", index=False)
        md_lines = [
            "# Track Attribution Summary (Delta vs. B0 Baseline)",
            "",
            dataframe_to_markdown(df_attr),
            "",
        ]
        (self.results_dir / "track_attribution_summary.md").write_text(
            "\n".join(md_lines), encoding="utf-8"
        )

        # 2. Pick overall best kept model by lowest Uniform 4x4 RMSE
        by_model: dict[str, dict[str, EvaluationResult]] = {}
        for r in results:
            by_model.setdefault(r.model_name, {})[r.dataset_id] = r
        best_name = min(
            (k for k in fitted_models if k in by_model),
            key=lambda k: by_model[k][GridId.RO_TEST_UNIFORM_4X4.value].summary["rmse_2d_m"],
        )
        best_model = fitted_models[best_name]
        print(f"[after_run] Overall best kept shift model: {best_name}")

        # 3. Track 7: Full Bidirectional ETRS89 <-> Stereo70 Check
        helmert_params = load_embedded_spg_helmert_params()
        transformer = ResidualCoordinateTransformer(
            base_transformer=TwoStepHelmert2DTransformer(helmert_params),
            shift_model=best_model,
        )
        t7_results = self.evaluate_transformer(
            transformer=transformer,
            model_name=f"Track7_{best_name}",
            dataset_id=GridId.CORRIDOR_SNW,
            directions=("etrs89_to_st70", "st70_to_etrs89"),
            plots=None,
        )
        t7_payload = [
            {
                "model": r.model_name,
                "direction": r.direction,
                "dataset": r.dataset_id,
                "mean_2d_mm": round(float(r.summary["mean_2d_m"]) * 1000.0, 4),
                "rmse_2d_mm": round(float(r.summary["rmse_2d_m"]) * 1000.0, 4),
                "p95_2d_mm": round(float(r.summary["p95_2d_m"]) * 1000.0, 4),
                "max_2d_mm": round(float(r.summary["max_2d_m"]) * 1000.0, 4),
                "pass_lt_1mm_pct": round(
                    float(r.summary["pass_rates_pct"].get("lt_1mm", 0.0)), 2
                ),
                "pass_lt_1cm_pct": round(
                    float(r.summary["pass_rates_pct"].get("lt_1cm", 0.0)), 2
                ),
            }
            for r in t7_results
        ]
        (self.results_dir / "bidirectional_track7.json").write_text(
            json.dumps(t7_payload, indent=2), encoding="utf-8"
        )

        return {
            "best_kept_model": best_name,
            "best_uniform_rmse_mm": round(
                float(
                    by_model[best_name][GridId.RO_TEST_UNIFORM_4X4.value].summary[
                        "rmse_2d_m"
                    ]
                )
                * 1000.0,
                4,
            ),
            "track7_bidirectional": t7_payload,
        }


if __name__ == "__main__":
    unit = SubmillimeterExplorationUnit(Path("experiments/submillimeter_exploration"))
    unit.run()
