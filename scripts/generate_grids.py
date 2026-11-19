"""Generator script for canonical grids:
1. data/grids/spg_nodes_valid.parquet (clean extracted SPG nodes)
2. data/grids/ro_grid_4x4.parquet (former ro_grid_3x3, 4x4 subgrid, 2.75 km step)
3. data/grids/ro_grid_8x8.parquet (densified half-step 8x8 subgrid, 1.375 km step)
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pytransdat import TransRO
from src.core.data_samplers import (
    DEFAULT_SUBGRID_OFFSETS_4X4,
    DEFAULT_SUBGRID_OFFSETS_8X8,
    build_grid_candidates,
)

OUTPUT_DIR = Path("data/grids")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_spg_nodes() -> None:
    print("Generating SPG nodes...")
    engine = TransRO()
    spg_data = np.load("pytransdat/grids/rom_grid3d_25.09.spg", allow_pickle=True)
    grid_arr = spg_data["grids"]["geodetic_shifts"]["grid"]
    dn_arr = grid_arr[0]
    de_arr = grid_arr[1]
    grid2d = engine._t_gr2d

    records = []
    node_id = 0
    for r in range(grid2d.r_count):
        for c in range(grid2d.c_count):
            dn_val = float(dn_arr[r, c])
            de_val = float(de_arr[r, c])
            if not np.isnan(dn_val) and not np.isnan(de_val):
                n_coord = float(grid2d.n_min + r * grid2d.n_step)
                e_coord = float(grid2d.e_min + c * grid2d.e_step)
                mag_2d = math.hypot(dn_val, de_val)

                # Check if pytransdat can transform it
                etrs_lat_deg = np.nan
                etrs_lon_deg = np.nan
                etrs_lat_rad = np.nan
                etrs_lon_rad = np.nan
                etrs_h = np.nan
                is_transdat_valid = False
                try:
                    res = engine.st70_to_etrs89(n_coord, e_coord, 0.0)
                    etrs_lat_rad = float(res[0])
                    etrs_lon_rad = float(res[1])
                    etrs_lat_deg = math.degrees(etrs_lat_rad)
                    etrs_lon_deg = math.degrees(etrs_lon_rad)
                    etrs_h = float(res[2])
                    is_transdat_valid = True
                except Exception:
                    pass

                records.append(
                    {
                        "node_id": node_id,
                        "row_idx": r,
                        "col_idx": c,
                        "st70_n": n_coord,
                        "st70_e": e_coord,
                        "st70_h": 0.0,
                        "shift_dn_m": dn_val,
                        "shift_de_m": de_val,
                        "shift_mag_2d_m": mag_2d,
                        "is_transdat_valid": is_transdat_valid,
                        "etrs89_lat_deg": etrs_lat_deg,
                        "etrs89_lon_deg": etrs_lon_deg,
                        "etrs89_lat_rad": etrs_lat_rad,
                        "etrs89_lon_rad": etrs_lon_rad,
                        "etrs89_h": etrs_h,
                    }
                )
                node_id += 1

    df_nodes = pd.DataFrame(records)
    out_pq = OUTPUT_DIR / "spg_nodes_valid.parquet"
    out_meta = OUTPUT_DIR / "spg_nodes_valid.meta.json"
    df_nodes.to_parquet(out_pq, index=False)

    meta = {
        "name": "spg_nodes_valid",
        "description": "Clean, pre-filtered valid 2D nodes directly from rom_grid3d_25.09.spg geodetic_shifts layer.",
        "spg_grid_file": "rom_grid3d_25.09.spg",
        "total_nodes_in_grid": grid2d.r_count * grid2d.c_count,
        "valid_land_nodes_count": len(df_nodes),
        "transdat_valid_nodes_count": int(df_nodes["is_transdat_valid"].sum()),
        "grid_step_m": [float(grid2d.n_step), float(grid2d.e_step)],
        "shift_dn_stats": {
            "min": float(df_nodes["shift_dn_m"].min()),
            "max": float(df_nodes["shift_dn_m"].max()),
            "mean": float(df_nodes["shift_dn_m"].mean()),
            "std": float(df_nodes["shift_dn_m"].std()),
        },
        "shift_de_stats": {
            "min": float(df_nodes["shift_de_m"].min()),
            "max": float(df_nodes["shift_de_m"].max()),
            "mean": float(df_nodes["shift_de_m"].mean()),
            "std": float(df_nodes["shift_de_m"].std()),
        },
        "parquet_file": "data/grids/spg_nodes_valid.parquet",
    }
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {len(df_nodes)} SPG nodes to {out_pq}")


def transform_and_filter_candidates(
    candidates: list[dict],
    engine: TransRO,
    name: str,
    subgrid_offsets: list[float],
    subgrid_spacing_m: float,
) -> pd.DataFrame:
    print(f"Transforming {len(candidates)} candidates for {name}...")
    t0 = time.perf_counter()

    valid_records = []
    pid = 0
    for c in candidates:
        n_in = c["st70_n"]
        e_in = c["st70_e"]
        h_in = c["st70_h"]
        try:
            # st70 -> etrs89
            res_etrs = engine.st70_to_etrs89(n_in, e_in, h_in)
            lat_rad = float(res_etrs[0])
            lon_rad = float(res_etrs[1])
            h_etrs = float(res_etrs[2])

            lat_deg = math.degrees(lat_rad)
            lon_deg = math.degrees(lon_rad)

            # etrs89 -> st70 roundtrip (pytransdat accepts radians for etrs89_to_st70)
            res_st70 = engine.etrs89_to_st70(lat_rad, lon_rad, h_etrs)
            n_rt = float(res_st70[0])
            e_rt = float(res_st70[1])
            h_rt = float(res_st70[2])

            err_n = abs(n_rt - n_in)
            err_e = abs(e_rt - e_in)
            err_h = abs(h_rt - h_in)

            # Check if this point also belongs to 4x4 grid (offsets in 4x4 set)
            is_4x4 = (c["u"] in DEFAULT_SUBGRID_OFFSETS_4X4) and (c["v"] in DEFAULT_SUBGRID_OFFSETS_4X4)

            valid_records.append(
                {
                    "point_id": pid,
                    "row_idx": c["row_idx"],
                    "col_idx": c["col_idx"],
                    "u": c["u"],
                    "v": c["v"],
                    "is_node": c["is_node"],
                    "is_4x4_point": is_4x4,
                    "st70_n": n_in,
                    "st70_e": e_in,
                    "st70_h": h_in,
                    "etrs89_lat_deg": lat_deg,
                    "etrs89_lon_deg": lon_deg,
                    "etrs89_lat_rad": lat_rad,
                    "etrs89_lon_rad": lon_rad,
                    "etrs89_h": h_etrs,
                    "roundtrip_err_n_m": err_n,
                    "roundtrip_err_e_m": err_e,
                    "roundtrip_err_h_m": err_h,
                }
            )
            pid += 1
        except Exception:
            continue

    df = pd.DataFrame(valid_records)
    elapsed = time.perf_counter() - t0
    print(f"{name}: {len(df)} bidirectionally valid points retained in {elapsed:.2f}s")

    out_pq = OUTPUT_DIR / f"{name}.parquet"
    out_meta = OUTPUT_DIR / f"{name}.meta.json"
    df.to_parquet(out_pq, index=False)

    meta = {
        "name": name,
        "description": f"Canonical bidirectional {name} subgrid ({subgrid_spacing_m} m spacing) across all valid cells of the official Romanian SPG distortion grid.",
        "spg_grid_file": "rom_grid3d_25.09.spg",
        "subgrid_offsets": subgrid_offsets,
        "grid_step_m": [11000.0, 11000.0],
        "subgrid_spacing_m": [subgrid_spacing_m, subgrid_spacing_m],
        "total_candidates": len(candidates),
        "bidirectional_valid_count": len(df),
        "node_count": int(df["is_node"].sum()),
        "interior_count": int((~df["is_node"]).sum()),
        "bounds": {
            "st70_n_min": float(df["st70_n"].min()),
            "st70_n_max": float(df["st70_n"].max()),
            "st70_e_min": float(df["st70_e"].min()),
            "st70_e_max": float(df["st70_e"].max()),
            "etrs89_lat_deg_min": float(df["etrs89_lat_deg"].min()),
            "etrs89_lat_deg_max": float(df["etrs89_lat_deg"].max()),
            "etrs89_lon_deg_min": float(df["etrs89_lon_deg"].min()),
            "etrs89_lon_deg_max": float(df["etrs89_lon_deg"].max()),
        },
        "roundtrip_max_err_m": {
            "northing": float(df["roundtrip_err_n_m"].max()),
            "easting": float(df["roundtrip_err_e_m"].max()),
            "height": float(df["roundtrip_err_h_m"].max()),
        },
        "generation_time_s": round(elapsed, 3),
        "parquet_file": f"data/grids/{name}.parquet",
    }
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return df


def main():
    engine = TransRO()
    generate_spg_nodes()

    # Generate ro_grid_4x4
    cand_4x4 = build_grid_candidates(engine, offsets=DEFAULT_SUBGRID_OFFSETS_4X4)
    transform_and_filter_candidates(
        candidates=cand_4x4,
        engine=engine,
        name="ro_grid_4x4",
        subgrid_offsets=list(DEFAULT_SUBGRID_OFFSETS_4X4),
        subgrid_spacing_m=2750.0,
    )

    # Generate ro_grid_8x8
    cand_8x8 = build_grid_candidates(engine, offsets=DEFAULT_SUBGRID_OFFSETS_8X8)
    transform_and_filter_candidates(
        candidates=cand_8x8,
        engine=engine,
        name="ro_grid_8x8",
        subgrid_offsets=list(DEFAULT_SUBGRID_OFFSETS_8X8),
        subgrid_spacing_m=1375.0,
    )


if __name__ == "__main__":
    main()
