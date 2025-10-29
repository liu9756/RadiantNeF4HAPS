#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate CARI LOC chunks for vB:
- heights: 10..25 (step 1) + {27,29} km
- lats:    -80..80 (step 4)
- lons:    -176..176 (step 8)
- dates:   comma-separated yyyy/mm/00 list (monthly averages)
Writes START/STOP-wrapped *.loc into <outdir>/<date_key>/ with parts.txt index.
"""
import argparse, pathlib

def km_to_feet(km): return int(round(km * 3280.839895))

def loc_line(lat, lon, h_km, date, hour="H0", D="D2", P="P0", C="C4", S="S0"):
    NS = "N" if lat >= 0 else "S"; EW = "E" if lon >= 0 else "W"
    lat_abs = abs(float(lat)); lon_abs = abs(float(lon)); alt_ft = km_to_feet(h_km)
    return f"{NS}, {lat_abs:.5f}, {EW}, {lon_abs:.5f}, F, {alt_ft:d}, {date}, {hour}, {D}, {P}, {C}, {S}\n"

def build_heights():
    H = list(range(10, 26, 1)) 
    H += [27, 29]
    return H

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--dates", default="2002/01/00,2009/01/00,2014/01/00",
                    help="comma-separated yyyy/mm/00 list (monthly averages)")
    ap.add_argument("--chunk", type=int, default=4000)
    ap.add_argument("--hour", default="H0")
    ap.add_argument("--D", default="D2")
    ap.add_argument("--P", default="P0")
    ap.add_argument("--C", default="C4")
    ap.add_argument("--S", default="S0")
    args = ap.parse_args()

    base = pathlib.Path(args.outdir)
    heights = build_heights()
    lats  = list(range(-80, 81, 4))     
    lons  = list(range(-176, 177, 8))  

    for date in [x.strip() for x in args.dates.split(",") if x.strip()]:
        date_key = date.replace("/", "_")  
        outdir = base / date_key
        outdir.mkdir(parents=True, exist_ok=True)

        coords = [(lat, lon, h) for h in heights for lat in lats for lon in lons]
        N = len(coords)
        parts = []
        for i in range(0, N, args.chunk):
            chunk = coords[i:i+args.chunk]
            loc_path = outdir / f"part_{i//args.chunk:04d}.loc"
            with loc_path.open("w") as f:
                f.write("START--------------------------------------------------------\n")
                for (lat, lon, h) in chunk:
                    f.write(loc_line(lat, lon, h, date, args.hour, args.D, args.P, args.C, args.S))
                f.write("STOP---------------------------------------------------------\n")
            parts.append(loc_path.name)

        with (outdir / "parts.txt").open("w") as f:
            for p in parts: f.write(p + "\n")

        print(f"[make_loc_vB] {date}: total points={N}, parts={len(parts)}, outdir={outdir}")

if __name__ == "__main__":
    main()
