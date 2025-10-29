import math, csv, pathlib, pandas as pd

def km_to_feet(km): return km * 3280.839895  # 1 km = 3280.839895 ft

def write_loc(loc_path, lat_grid, lon_grid, h_km_grid,
              date="2002/01/00", hour="H0", D="D2", P="P0", C="C4", S="S0"):
    loc_path = pathlib.Path(loc_path)
    with loc_path.open("w", newline="") as f:
        f.write("START--------------------------------------------------------\n")
        for lat in lat_grid:
            NS = "N" if lat >= 0 else "S"
            lat_abs = abs(float(lat))
            for lon in lon_grid:
                EW = "E" if lon >= 0 else "W"
                lon_abs = abs(float(lon))
                for h in h_km_grid:
                    alt_ft = int(round(km_to_feet(h)))
                    line = f"{NS}, {lat_abs:8.5f}, {EW}, {lon_abs:9.5f}, F, {alt_ft:7d}, {date}, {hour}, {D}, {P}, {C}, {S}\n"
                    f.write(line)
        f.write("STOP---------------------------------------------------------\n")
    return str(loc_path)

def read_ans(ans_path):
    df = pd.read_csv(ans_path, header=None, skip_blank_lines=True)
    header_idx = df.index[df.iloc[:,0].astype(str).str.contains("LAT", na=False)].tolist()
    if not header_idx: raise RuntimeError("ANS header not found")
    hdr = df.iloc[header_idx[0]].tolist()
    data = df.iloc[header_idx[0]+1:].reset_index(drop=True)
    data.columns = [h.strip() for h in hdr]
    data = data[~data.iloc[:,0].astype(str).str.startswith("C")]
    data["h_km"] = data["ALTITUDE"].astype(float) / 3280.839895
    data.rename(columns={
        "LAT":"lat_deg","LON":"lon_deg","VCR(GV)":"vcr_GV",
        "DOSE RATE":"dose_rate","SIGMA":"sigma","UNIT":"unit","QUANTITY":"quantity",
        "DATE":"date","HR":"hour"
    }, inplace=True)
    keep = ["lat_deg","lon_deg","h_km","date","hour","vcr_GV","dose_rate","sigma","unit","quantity","PARTICLE"]
    return data[keep]