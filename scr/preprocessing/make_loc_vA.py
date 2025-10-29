#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math, pathlib, argparse

def km_to_feet(km): return int(round(km * 3280.839895))

def loc_line(lat, lon, h_km, date="2002/01/00", hour="H0", D="D2", P="P0", C="C4", S="S0"):
    NS = "N" if lat >= 0 else "S"; EW = "E" if lon >= 0 else "W"
    lat_abs = abs(float(lat)); lon_abs = abs(float(lon)); alt_ft = km_to_feet(h_km)
    return f"{NS}, {lat_abs:8.5f}, {EW}, {lon_abs:9.5f}, F, {alt_ft:7d}, {date}, {hour}, {D}, {P}, {C}, {S}\n"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--date", default="2002/01/00") 
    ap.add_argument("--hour", default="H0")
    ap.add_argument("--D", default="D2")   
    ap.add_argument("--P", default="P0")  
    ap.add_argument("--C", default="C4")   
    ap.add_argument("--S", default="S0")
    ap.add_argument("--chunk", type=int, default=3000)
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # grids
    heights = [h for h in range(8, 26, 1)]                         
    lats    = [l for l in range(-80, 81, 5)]                        
    lons    = [l for l in range(-180, 176, 10)]                     

    # materialize list
    coords = [(lat, lon, h) for h in heights for lat in lats for lon in lons]
    N = len(coords)
    print(f"[make_loc_vA] total points = {N}")

    # chunking
    parts = []
    for i in range(0, N, args.chunk):
        chunk = coords[i:i+args.chunk]
        loc_path = outdir / f"part_{i//args.chunk:04d}.loc"
        with loc_path.open("w") as f:
            f.write("START--------------------------------------------------------\n")
            for (lat, lon, h) in chunk:
                f.write(loc_line(lat, lon, h, args.date, args.hour, args.D, args.P, args.C, args.S))
            f.write("STOP---------------------------------------------------------\n")
        parts.append(loc_path.name)

    # write index
    with (outdir / "parts.txt").open("w") as f:
        for p in parts:
            f.write(str(p) + "\n")

    print(f"[make_loc_vA] wrote {len(parts)} LOC parts to {outdir}")

if __name__ == "__main__":
    main()
