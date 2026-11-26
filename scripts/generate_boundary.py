"""Generate static Stereo 70 (EPSG:3844) buffered boundary GeoJSON assets.

Reprojects an input boundary polygon (e.g. WGS84/CRS84 GeoJSON) into
EPSG:3844 (Dealul Piscului 1970 / Stereo 70), applies an outward planar
buffer in meters, and writes the resulting GeoJSON and metadata JSON under
``data/boundaries/``.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd

DEFAULT_INPUT = Path("scratch/ro_admin_country_simplified_polygon.geojson")
DEFAULT_OUTPUT = Path("data/boundaries/romania_border_st70.geojson")
TARGET_CRS = "EPSG:3844"
DEFAULT_BUFFER_M = 300.0


def generate_st70_boundary(
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    buffer_m: float = DEFAULT_BUFFER_M,
    target_crs: str = TARGET_CRS,
) -> Path:
    """Reproject and buffer a boundary polygon into Stereo 70 meters.

    :param input_path: Path to source boundary vector file (e.g. CRS84 GeoJSON).
    :type input_path: Path
    :param output_path: Destination GeoJSON path in projected Stereo 70 meters.
    :type output_path: Path
    :param buffer_m: Outward planar buffer distance in meters.
    :type buffer_m: float
    :param target_crs: Target projected CRS string (default ``EPSG:3844``).
    :type target_crs: str
    :return: Path to the written GeoJSON file.
    :rtype: Path
    :raises ValueError: If ``buffer_m < 0`` or input has no geometries.
    """
    if buffer_m < 0.0:
        raise ValueError(f"buffer_m must be >= 0, got {buffer_m}")
    if not input_path.is_file():
        raise FileNotFoundError(f"Input boundary file not found: {input_path}")

    gdf = gpd.read_file(input_path)
    if gdf.empty:
        raise ValueError(f"Input boundary file contains no features: {input_path}")

    gdf_proj = gdf.to_crs(target_crs)
    union_geom = gdf_proj.geometry.union_all()
    if buffer_m > 0.0:
        union_geom = union_geom.buffer(buffer_m)

    out_gdf = gpd.GeoDataFrame(
        [
            {
                "name": "Romania",
                "buffer_m": float(buffer_m),
                "crs": target_crs,
            }
        ],
        geometry=[union_geom],
        crs=target_crs,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_gdf.to_file(output_path, driver="GeoJSON")

    meta_path = output_path.with_suffix(".meta.json")
    meta = {
        "source_file": input_path.as_posix(),
        "output_file": output_path.as_posix(),
        "target_crs": target_crs,
        "buffer_m": float(buffer_m),
        "area_km2": round(float(union_geom.area) / 1_000_000.0, 3),
        "bounds_xy_m": [round(float(v), 3) for v in union_geom.bounds],
        "generated_utc": datetime.now(UTC).isoformat(),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate static Stereo 70 buffered boundary GeoJSON."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--buffer-m", type=float, default=DEFAULT_BUFFER_M)
    args = parser.parse_args()

    out = generate_st70_boundary(
        input_path=args.input,
        output_path=args.output,
        buffer_m=args.buffer_m,
    )
    print(f"Generated Stereo 70 boundary asset: {out.as_posix()}")


if __name__ == "__main__":
    main()
