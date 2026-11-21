"""Analysis Unit: 3x3 Grid Cell Neighbor Feature Extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src.core.base_unit import AnalysisExperimentUnit
from src.geometry.neighbor_features import (
    NEIGHBOR_FEATURE_COLS,
    CellNeighborFeatureExtractor,
    extract_cell_neighbor_features,
)


class CellNeighborFeaturesUnit(AnalysisExperimentUnit):
    """Extracts 40 continuous 3x3 neighbor features for 11x11 km SPG cells."""

    @property
    def slug(self) -> str:
        return "cell_neighbor_features"

    def run_analysis(self) -> dict[str, Any]:
        self.data_features_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        df = extract_cell_neighbor_features()

        parquet_path = self.data_features_dir / "cell_neighbor_features.parquet"
        csv_path = self.results_dir / "cell_neighbor_features.csv"
        df.to_parquet(parquet_path, index=False)
        df.to_csv(csv_path, index=False)

        self._plot_neighbor_maps(df)

        # Verify O(1) spatial lookup
        extractor = CellNeighborFeatureExtractor(parquet_path)
        sample_feats = extractor.lookup_points(
            np.array([335000.0, 500000.0]),
            np.array([585000.0, 500000.0]),
        )

        slope_cols = [c for c in NEIGHBOR_FEATURE_COLS if "slope" in c]
        twist_cols = [c for c in NEIGHBOR_FEATURE_COLS if "twist" in c]

        max_slope_mm = float(df[slope_cols].abs().max().max() * 1000.0)
        mean_abs_slope_mm = float(df[slope_cols].abs().mean().mean() * 1000.0)
        max_twist_mm = float(df[twist_cols].abs().max().max() * 1000.0)
        mean_abs_twist_mm = float(df[twist_cols].abs().mean().mean() * 1000.0)

        return {
            "key_metrics": {
                "valid_interior_cells": len(df),
                "num_engineered_features": len(NEIGHBOR_FEATURE_COLS),
                "nan_count": int(df[NEIGHBOR_FEATURE_COLS].isna().sum().sum()),
                "max_abs_slope_mm": round(max_slope_mm, 3),
                "mean_abs_slope_mm": round(mean_abs_slope_mm, 3),
                "max_abs_twist_mm": round(max_twist_mm, 3),
                "mean_abs_twist_mm": round(mean_abs_twist_mm, 3),
                "o1_lookup_valid_rows": int(np.isfinite(sample_feats).all(axis=1).sum()),
            },
            "artifacts": {
                "shared_features": [parquet_path.as_posix()],
                "results": ["results/cell_neighbor_features.csv"],
                "plots": [
                    "national_neighbor_slope_max_map.png",
                    "national_neighbor_twist_max_map.png",
                ],
            },
        }

    def _plot_neighbor_maps(self, df: Any) -> None:
        slope_cols = [c for c in NEIGHBOR_FEATURE_COLS if "slope" in c]
        twist_cols = [c for c in NEIGHBOR_FEATURE_COLS if "twist" in c]

        max_slope_mm = df[slope_cols].abs().max(axis=1) * 1000.0
        max_twist_mm = df[twist_cols].abs().max(axis=1) * 1000.0

        for values, title, fname, cmap in (
            (
                max_slope_mm,
                "Max Corner One-Sided Slope Magnitude (mm / 11km)",
                "national_neighbor_slope_max_map.png",
                "viridis",
            ),
            (
                max_twist_mm,
                "Max Corner Diagonal Twist Magnitude (mm)",
                "national_neighbor_twist_max_map.png",
                "magma",
            ),
        ):
            fig, ax = plt.subplots(figsize=(8, 6))
            sc = ax.scatter(
                df["center_e"] / 1000.0,
                df["center_n"] / 1000.0,
                c=values,
                cmap=cmap,
                s=14,
                marker="s",
            )
            plt.colorbar(sc, ax=ax, label="mm")
            ax.set_title(title)
            ax.set_xlabel("Easting (km)")
            ax.set_ylabel("Northing (km)")
            ax.set_aspect("equal")
            fig.tight_layout()
            fig.savefig(self.plots_dir / fname, dpi=150)
            plt.close(fig)


if __name__ == "__main__":
    CellNeighborFeaturesUnit(Path(__file__).resolve().parent).run()
