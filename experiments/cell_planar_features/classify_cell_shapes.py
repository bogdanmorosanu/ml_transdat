"""Comprehensive Shape Classification Analysis of Romania's Distortion Grid Cells.

Classifies all valid 11x11 km cells in rom_grid3d_25.09.spg into curvature archetypes:
1. LINEAR (Green)
2. DOME_BOWL (Blue)
3. S_SHAPE (Orange)

Evaluates axis concordance between Northing (dN) and Easting (dE) and produces
national spatial distribution maps across multiple tolerances.
"""

import sys
from pathlib import Path

# Ensure workspace root is in sys.path
workspace_root = Path(__file__).resolve().parents[2]
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap

from pytransdat.trans_ro import TransRO
from src.geometry.plane import extract_grid_cell_planes
from src.geometry.shape_classifier import (
    CellShapeFeatureExtractor,
    ShapeCategory,
    classify_cells_dataset,
)


def run_shape_classification() -> pd.DataFrame:
    """Executes the full-grid shape classification sweep and exports metrics and maps."""
    out_dir = Path(__file__).resolve().parent
    results_dir = out_dir / "results"
    plots_dir = out_dir / "plots"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("Loading official SPG grid and cells...")
    trans = TransRO()
    cells = extract_grid_cell_planes()
    print(f"Loaded {len(cells)} valid land cells.")

    print(
        "Evaluating 25 interior test points per cell across polynomial orders 1, 2, and 3..."
    )
    metrics_list = classify_cells_dataset(cells, trans._t_gr2d.interp, grid_res=5)
    print(f"Successfully evaluated {len(metrics_list)} interior cells.")

    # Convert to DataFrame
    rows = [m.to_dict(default_tol_mm=5.0) for m in metrics_list]
    df = pd.DataFrame(rows)

    # Add classifications across multiple tolerances (3-class taxonomy)
    tolerances = [2.0, 3.0, 5.0, 10.0, 15.0]
    for tol in tolerances:
        tol_tag = f"{int(tol)}mm" if tol == int(tol) else f"{tol:.1f}mm"
        df[f"cat_dn_{tol_tag}"] = [int(m.get_category_dn(tol)) for m in metrics_list]
        df[f"cat_de_{tol_tag}"] = [int(m.get_category_de(tol)) for m in metrics_list]
        df[f"cat_2d_{tol_tag}"] = [int(m.get_category_2d(tol)) for m in metrics_list]

    # Save full datasets
    parquet_path = results_dir / "cell_shape_metrics.parquet"
    csv_path = results_dir / "cell_shape_metrics.csv"
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    print(f"Saved classification dataset to {parquet_path} and {csv_path}")

    # --- Print Multi-Tolerance Summary Table ---
    print("\n" + "=" * 75)
    print("CELL SHAPE CLASSIFICATION DISTRIBUTION ACROSS TOLERANCES (N = 2,297 Cells)")
    print("=" * 75)
    header = f"{'Tolerance':<10} | {'Field':<6} | {'1: Linear':<14} | {'2: Dome/Bowl':<16} | {'3: S-Shape':<14}"
    print(header)
    print("-" * 75)

    for tol in tolerances:
        tol_tag = f"{int(tol)}mm" if tol == int(tol) else f"{tol:.1f}mm"
        for field, name in [("dn", "dN"), ("de", "dE"), ("2d", "2D")]:
            col = f"cat_{field}_{tol_tag}"
            counts = df[col].value_counts(normalize=True).to_dict()
            p1 = counts.get(int(ShapeCategory.LINEAR), 0.0) * 100.0
            p2 = counts.get(int(ShapeCategory.DOME_BOWL), 0.0) * 100.0
            p3 = counts.get(int(ShapeCategory.S_SHAPE), 0.0) * 100.0
            print(
                f"{tol_tag:<10} | {name:<6} | {p1:5.1f}%         | {p2:5.1f}%           | {p3:5.1f}%"
            )
        print("-" * 75)

    # --- Print Axis Concordance Matrix (5.0 mm and 10.0 mm) ---
    print("\n" + "=" * 75)
    print("AXIS CONCORDANCE (dN vs. dE) AT 5.0 mm TOLERANCE")
    print("=" * 75)
    crosstab_5mm = pd.crosstab(
        df["cat_dn_5mm"].map(lambda x: ShapeCategory(x).label),
        df["cat_de_5mm"].map(lambda x: ShapeCategory(x).label),
        margins=True,
    )
    print(crosstab_5mm)

    exact_match_5mm = (df["cat_dn_5mm"] == df["cat_de_5mm"]).mean() * 100.0
    print(f"\nExact Shape Agreement between dN and dE at 5 mm: {exact_match_5mm:.1f}%")

    print("\n" + "=" * 75)
    print("AXIS CONCORDANCE (dN vs. dE) AT 10.0 mm TOLERANCE")
    print("=" * 75)
    crosstab_10mm = pd.crosstab(
        df["cat_dn_10mm"].map(lambda x: ShapeCategory(x).label),
        df["cat_de_10mm"].map(lambda x: ShapeCategory(x).label),
        margins=True,
    )
    print(crosstab_10mm)
    exact_match_10mm = (df["cat_dn_10mm"] == df["cat_de_10mm"]).mean() * 100.0
    print(
        f"\nExact Shape Agreement between dN and dE at 10 mm: {exact_match_10mm:.1f}%"
    )

    # --- PLOTTING: National Spatial Distribution Maps (3 Classes) ---
    # Colormap: 1=Green (Linear), 2=Blue (Dome), 3=Orange (S-Shape)
    cmap = ListedColormap(["#2ca02c", "#1f77b4", "#ff7f0e"])
    categories = [ShapeCategory.LINEAR, ShapeCategory.DOME_BOWL, ShapeCategory.S_SHAPE]

    for tol in [5.0, 10.0]:
        tol_tag = f"{int(tol)}mm"
        _fig, axes = plt.subplots(1, 3, figsize=(22, 6.5), sharex=True, sharey=True)

        for idx, (field, title) in enumerate(
            [
                ("dn", f"Northing Shift (dN) Shape [{tol_tag}]"),
                ("de", f"Easting Shift (dE) Shape [{tol_tag}]"),
                ("2d", f"Combined 2D Cell Shape [{tol_tag}]"),
            ]
        ):
            ax = axes[idx]
            col = f"cat_{field}_{tol_tag}"
            sc = ax.scatter(
                df["center_e"] / 1000.0,
                df["center_n"] / 1000.0,
                c=df[col],
                cmap=cmap,
                vmin=0.5,
                vmax=3.5,
                s=30,
                marker="s",
                alpha=0.9,
            )
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlabel("Stereo 70 Easting (km)")
            if idx == 0:
                ax.set_ylabel("Stereo 70 Northing (km)")
            ax.grid(True, linestyle="--", alpha=0.4)

        # Colorbar with 3 category labels
        cbar = _fig.colorbar(
            sc,
            ax=axes,
            orientation="horizontal",
            fraction=0.04,
            pad=0.12,
            ticks=[1, 2, 3],
        )
        cbar.ax.set_xticklabels(
            [c.label for c in categories], fontsize=11, fontweight="bold"
        )
        cbar.set_label("Curvature Shape Category", fontsize=11)

        plot_path = plots_dir / f"national_shape_map_{tol_tag}.png"
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved plot: {plot_path}")

    # Verify that CellShapeFeatureExtractor loads the generated parquet properly
    print("\nVerifying CellShapeFeatureExtractor on test coordinates...")
    extractor = CellShapeFeatureExtractor(parquet_path)
    # Test Bucharest coordinates (N=335000, E=585000)
    b_test = extractor.lookup_point(335000.0, 585000.0, tol_mm=5.0)
    print(
        f"Bucharest test point: valid={b_test['is_valid']}, cat_dn={ShapeCategory(b_test['cat_dn']).label}, cat_de={ShapeCategory(b_test['cat_de']).label}, cat_2d={ShapeCategory(b_test['cat_2d']).label}"
    )

    return df


if __name__ == "__main__":
    run_shape_classification()
