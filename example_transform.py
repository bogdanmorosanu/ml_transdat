"""Sample script to verify coordinate transformation with pytransdat.

ETRS89 coordinates (B, L, h):
- Latitude B: in radians or degrees (converted to radians)
- Longitude L: in radians or degrees (converted to radians)
- Height h: in meters

Stereo70 coordinates (X, Y, Z):
- X (Northing) in meters
- Y (Easting) in meters
- Z (Height) in meters
"""

import math
from pytransdat.trans_ro import TransRO

def main():
    # 1. Initialize transformation engine (loads rom_grid3d_25.09.spg)
    trans = TransRO()

    # 2. Sample point in Romania (approx Brasov region)
    # B: 45.65 degrees North, L: 25.60 degrees East, h: 600.0 m
    lat_deg = 45.65
    lon_deg = 25.60
    h_m = 600.0

    lat_rad = math.radians(lat_deg)
    lon_rad = math.radians(lon_deg)

    print("--- 1. DIRECT TRANSFORMATION (ETRS89 -> Stereo70) ---")
    print(f"Input ETRS89: Lat = {lat_deg} deg, Lon = {lon_deg} deg, Height = {h_m:.3f} m")
    
    # Direct transformation takes (lat_rad, lon_rad, height_m)
    x_st70, y_st70, z_st70 = trans.etrs89_to_st70(lat_rad, lon_rad, h_m)
    print(f"Result Stereo70: X (North) = {x_st70:.4f} m, Y (East) = {y_st70:.4f} m, Z = {z_st70:.4f} m\n")

    print("--- 2. INVERSE TRANSFORMATION (Stereo70 -> ETRS89) ---")
    print(f"Input Stereo70: X = {x_st70:.4f} m, Y = {y_st70:.4f} m, Z = {z_st70:.4f} m")

    # Inverse transformation takes (x_m, y_m, z_m)
    res_lat_rad, res_lon_rad, res_h_m = trans.st70_to_etrs89(x_st70, y_st70, z_st70)
    res_lat_deg = math.degrees(res_lat_rad)
    res_lon_deg = math.degrees(res_lon_rad)

    print(f"Result ETRS89: Lat = {res_lat_deg:.8f} deg, Lon = {res_lon_deg:.8f} deg, Height = {res_h_m:.3f} m\n")

    # Verification: check delta between original and round-trip
    delta_lat = abs(res_lat_deg - lat_deg)
    delta_lon = abs(res_lon_deg - lon_deg)
    delta_h = abs(res_h_m - h_m)
    print("--- 3. ROUND-TRIP ACCURACY CHECK ---")
    print(f"Delta Lat: {delta_lat:.2e} deg")
    print(f"Delta Lon: {delta_lon:.2e} deg")
    print(f"Delta Height: {delta_h:.2e} m")
    assert delta_lat < 1e-7, "Latitude mismatch"
    assert delta_lon < 1e-7, "Longitude mismatch"
    assert delta_h < 1e-3, "Height mismatch"
    print("\nTransformation engine verified successfully!")

if __name__ == "__main__":
    main()
