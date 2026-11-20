"""Optional CLI helper to materialize dynamic grids into a target directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from pytransdat import TransRO
from src.data.samplers import (
    DEFAULT_SUBGRID_OFFSETS_4X4,
    DEFAULT_SUBGRID_OFFSETS_8X8,
    DEFAULT_SUBGRID_OFFSETS_16X16,
    generate_densified_grid_df,
    generate_spg_nodes_df,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize dynamic grids to Parquet (optional; GridManager generates them on the fly)."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(".cache/grids"),
        help="Target output directory (default: .cache/grids).",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    engine = TransRO()
    spg_df = generate_spg_nodes_df()
    spg_df.to_parquet(args.out_dir / "spg_nodes_valid.parquet", index=False)
    print(f"Saved spg_nodes_valid: {len(spg_df):,} rows")

    for name, offsets in (
        ("ro_grid_4x4", DEFAULT_SUBGRID_OFFSETS_4X4),
        ("ro_grid_8x8", DEFAULT_SUBGRID_OFFSETS_8X8),
        ("ro_grid_16x16", DEFAULT_SUBGRID_OFFSETS_16X16),
    ):
        df = generate_densified_grid_df(offsets=offsets, engine=engine)
        df.to_parquet(args.out_dir / f"{name}.parquet", index=False)
        print(f"Saved {name}: {len(df):,} rows")


if __name__ == "__main__":
    main()
