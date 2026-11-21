"""Exploratory Analysis of Grid Cell Planes across Romania.

Extracts planar and bilinear features for all 2,556 valid 11x11 km cells,
computes empirical summary statistics, saves tabular datasets, and
generates diagnostic national spatial distribution plots.
"""

import sys
from pathlib import Path

# Ensure workspace root is in sys.path
workspace_root = Path(__file__).resolve().parents[2]
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.geometry.plane import extract_grid_cell_planes


def run_planar_analysis() -> pd.DataFrame:
    """Executes the grid cell planar extraction and saves results and plots."""
    out_dir = Path(__file__).resolve().parent
    results_dir = out_dir / "results"
    plots_dir = out_dir / "plots"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("Extracting 2D cell planes from official SPG grid...")
    cells = extract_grid_cell_planes()
    print(f"Extracted {len(cells)} valid land cells.")

    rows = [c.to_dict() for c in cells]
    df = pd.DataFrame(rows)

    # Convert twists to millimeters for intuitive reporting
    df["dn_twist_mm"] = df["dn_twist"] * 1000.0
    df["de_twist_mm"] = df["de_twist"] * 1000.0
    df["total_twist_mm"] = df["total_twist_mag"] * 1000.0

    # Save datasets
    parquet_path = results_dir / "cell_planes.parquet"
    csv_path = results_dir / "cell_planes.csv"
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    print(f"Saved dataset to {parquet_path} and {csv_path}")

    # Generate Statistical Overview
    print("\n" + "=" * 60)
    print("CELL PLANAR PROPERTIES SUMMARY (2,556 Valid Cells)")
    print("=" * 60)
    print("1. Northing Shifts (dN):")
    print(f"   - Base Offset (min): [{df['dn_min'].min():.3f} m to {df['dn_min'].max():.3f} m]")
    print(f"   - Within-Cell Span: Mean = {df['dn_span'].mean():.4f} m, Max = {df['dn_span'].max():.4f} m")
    print(f"   - Slope Mag per 11km: Mean = {df['dn_slope_mag'].mean():.4f} m, Max = {df['dn_slope_mag'].max():.4f} m")
    print(f"   - Twist (Non-planarity): Mean = {df['dn_twist_mm'].abs().mean():.2f} mm, Max = {df['dn_twist_mm'].abs().max():.2f} mm")

    print("\n2. Easting Shifts (dE):")
    print(f"   - Base Offset (min): [{df['de_min'].min():.3f} m to {df['de_min'].max():.3f} m]")
    print(f"   - Within-Cell Span: Mean = {df['de_span'].mean():.4f} m, Max = {df['de_span'].max():.4f} m")
    print(f"   - Slope Mag per 11km: Mean = {df['de_slope_mag'].mean():.4f} m, Max = {df['de_slope_mag'].max():.4f} m")
    print(f"   - Twist (Non-planarity): Mean = {df['de_twist_mm'].abs().mean():.2f} mm, Max = {df['de_twist_mm'].abs().max():.2f} mm")

    # Percentage of cells that are highly planar
    sub_5mm = (df["total_twist_mm"].abs() < 5.0).mean() * 100.0
    sub_10mm = (df["total_twist_mm"].abs() < 10.0).mean() * 100.0
    sub_25mm = (df["total_twist_mm"].abs() < 25.0).mean() * 100.0
    sub_50mm = (df["total_twist_mm"].abs() < 50.0).mean() * 100.0

    print("\n3. Planarity Breakdown (Combined Total Twist):")
    print(f"   - Twist < 5 mm (Near Perfect Plane): {sub_5mm:.1f}%")
    print(f"   - Twist < 10 mm (1 cm Plane):         {sub_10mm:.1f}%")
    print(f"   - Twist < 25 mm:                      {sub_25mm:.1f}%")
    print(f"   - Twist < 50 mm (5 cm):               {sub_50mm:.1f}%")
    print(f"   - Twist >= 50 mm (Strong Saddles):     {100.0 - sub_50mm:.1f}%")
    print("=" * 60)

    # --- PLOT 1: National Planar Slope Maps ---
    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
    sc1 = ax1.scatter(
        df["center_e"] / 1000.0,
        df["center_n"] / 1000.0,
        c=df["dn_slope_mag"],
        cmap="viridis",
        s=35,
        marker="s",
        alpha=0.9,
    )
    ax1.set_title("Northing Shift (dN) Planar Slope Magnitude (m / 11 km)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Stereo 70 Easting (km)")
    ax1.set_ylabel("Stereo 70 Northing (km)")
    ax1.grid(True, linestyle="--", alpha=0.5)
    plt.colorbar(sc1, ax=ax1, label="Slope Magnitude (m per 11 km cell)")

    sc2 = ax2.scatter(
        df["center_e"] / 1000.0,
        df["center_n"] / 1000.0,
        c=df["de_slope_mag"],
        cmap="plasma",
        s=35,
        marker="s",
        alpha=0.9,
    )
    ax2.set_title("Easting Shift (dE) Planar Slope Magnitude (m / 11 km)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Stereo 70 Easting (km)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    plt.colorbar(sc2, ax=ax2, label="Slope Magnitude (m per 11 km cell)")

    plt.tight_layout()
    plot1_path = plots_dir / "national_cell_slopes_map.png"
    plt.savefig(plot1_path, dpi=200)
    plt.close()
    print(f"Saved plot: {plot1_path}")

    # --- PLOT 2: National Twist (Non-Planarity) Map ---
    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
    sc1 = ax1.scatter(
        df["center_e"] / 1000.0,
        df["center_n"] / 1000.0,
        c=df["dn_twist_mm"].abs(),
        cmap="YlOrRd",
        s=35,
        marker="s",
        alpha=0.9,
        vmax=150.0,  # Clip colorbar at 150 mm to reveal details in typical cells
    )
    ax1.set_title("Northing Shift (dN) Bilinear Twist |τ| (mm)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Stereo 70 Easting (km)")
    ax1.set_ylabel("Stereo 70 Northing (km)")
    ax1.grid(True, linestyle="--", alpha=0.5)
    plt.colorbar(sc1, ax=ax1, label="|Twist| (mm departure from flat plane)")

    sc2 = ax2.scatter(
        df["center_e"] / 1000.0,
        df["center_n"] / 1000.0,
        c=df["de_twist_mm"].abs(),
        cmap="YlOrRd",
        s=35,
        marker="s",
        alpha=0.9,
        vmax=150.0,
    )
    ax2.set_title("Easting Shift (dE) Bilinear Twist |τ| (mm)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Stereo 70 Easting (km)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    plt.colorbar(sc2, ax=ax2, label="|Twist| (mm departure from flat plane)")

    plt.tight_layout()
    plot2_path = plots_dir / "national_cell_twist_map.png"
    plt.savefig(plot2_path, dpi=200)
    plt.close()
    print(f"Saved plot: {plot2_path}")

    # --- PLOT 3: Aspect & Directional Tilt Distribution ---
    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={"projection": "polar"})
    # Convert aspect degrees to radians for polar plot
    theta_dn = np.radians(df["dn_aspect_deg"])
    theta_de = np.radians(df["de_aspect_deg"])

    ax1.hist(theta_dn, bins=36, color="teal", alpha=0.75, edgecolor="black")
    ax1.set_title("dN Tilt Aspect Distribution (Direction of Upward Slope)", fontsize=11, fontweight="bold", pad=15)
    ax1.set_theta_zero_location("E")  # Standard East=0
    ax1.set_theta_direction(1)

    ax2.hist(theta_de, bins=36, color="coral", alpha=0.75, edgecolor="black")
    ax2.set_title("dE Tilt Aspect Distribution (Direction of Upward Slope)", fontsize=11, fontweight="bold", pad=15)
    ax2.set_theta_zero_location("E")
    ax2.set_theta_direction(1)

    plt.tight_layout()
    plot3_path = plots_dir / "national_cell_aspect_distribution.png"
    plt.savefig(plot3_path, dpi=200)
    plt.close()
    print(f"Saved plot: {plot3_path}")

    return df


if __name__ == "__main__":
    run_planar_analysis()
