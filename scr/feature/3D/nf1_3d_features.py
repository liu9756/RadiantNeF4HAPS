from __future__ import annotations
import re, math, json
from dataclasses import dataclass
from pathlib import Path 
from typing import Iterable, List, Tuple, Optional, Dict

import numpy as np 
import pandas as pd 
from utils43D import wrap180, _deg2rad, _fourier_feats

# -80.00000,-179.99999, 26247.0000,F,2002/01/00, 0, 0.00,TOTAL , 1.8427E+00, 1.1004E-01, microSv/hr, ICRP Pub. 103 EFFECTIVE DOSE
# dict：LAT, LON, ALTITUDE(ft), F, DATE, HR, VCR(GV), PARTICLE, DOSE RATE, SIGMA, UNIT, QUANTITY
def parse_cari_ans_one(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(("START", "STOP")):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 12:
                continue
            try:
                lat = float(parts[0])
                lon = float(parts[1])
                alt_ft = float(parts[2])
                date_str = parts[4]
                hour = int(parts[5])
                vcr = float(parts[6]) 
                particle = parts[7]   
                dose = float(parts[8])
                sigma = float(parts[9])
                unit_text = parts[10]
                quantity_text = parts[11]
            except Exception:
                continue

            rows.append({
                "lat_deg": lat,
                "lon_deg": lon,
                "h_km": alt_ft / 3280.84,
                "date": date_str,
                "hour": hour,
                "vcr_GV": vcr,
                "particle": particle,
                "dose_rate_uSvph": dose,
                "sigma": sigma,
                "unit_text": unit_text,
                "quantity_text": quantity_text
            })
    df = pd.DataFrame(rows)
    if df.empty:
        df["source_file"] = str(path.name)
        return df
    df["source_file"] = str(path.name)
    return df

def gather_ans_tree(ans_root: Path) -> pd.DataFrame:
    ans_paths = list(ans_root.rglob("*.ANS"))
    all_df = []
    for p in sorted(ans_paths):
        df = parse_cari_ans_one(p)
        if df.empty:
            continue
        alt_tag = p.parent.name
        m = re.match(r"h(\d+)p(\d+)km", alt_tag)
        if m:
            km = float(m.group(1)) + float(m.group(2))/10.0
            df["h_km_nominal"] = km
            df["alt_tag"] = alt_tag
        else:
            df["h_km_nominal"] = df["h_km"].round(1)
            df["alt_tag"] = "h%.1fkm" % df["h_km_nominal"].iloc[0]
        all_df.append(df)
    if not all_df:
        return pd.DataFrame(columns=[
            "lat_deg","lon_deg","h_km","h_km_nominal","dose_rate_uSvph","sigma",
            "unit_text","quantity_text","vcr_GV","date","hour","particle","alt_tag","source_file"
        ])
    out = pd.concat(all_df, ignore_index=True)
    out["lon_deg"] = out["lon_deg"].map(wrap180)
    out = out.drop_duplicates(subset=["lat_deg","lon_deg","h_km","dose_rate_uSvph"]).reset_index(drop=True)
    return out

@dataclass
class NF1_3D_Config:
    lon_block_deg: float = 30.0        
    h_ref_km: float = 16.0            
    h_scale_km: float = 6.0            
    use_fourier: bool = True
    lat_freqs: Tuple[float, ...] = (1/90, 1/45, 1/22.5)
    lon_freqs: Tuple[float, ...] = (1/180, 1/90, 1/45, 1/22.5)
    h_freqs:   Tuple[float, ...] = (1/24, 1/12, 1/6)

def build_nf1_3d_features(
    src: str,
    out_csv: Optional[str] = None,
    out_parquet: Optional[str] = None,
    cfg: NF1_3D_Config = NF1_3D_Config()
) -> pd.DataFrame:

    p = Path(src)
    if p.is_dir():
        base = gather_ans_tree(p)
    else:
        if p.suffix.lower() in (".csv",):
            base = pd.read_csv(p)
        elif p.suffix.lower() in (".parquet", ".pq"):
            base = pd.read_parquet(p)
        else:
            raise ValueError(f"Unrecognized src: {src}")


    need_cols = {"lat_deg","lon_deg","h_km","dose_rate_uSvph"}
    missing = need_cols - set(base.columns)
    if missing:
        raise ValueError(f"Input missing columns: {missing}")


    unit_set = {str(x).strip() for x in base.get("unit_text", pd.Series([])).unique() if pd.notna(x)}
    particle_set = {str(x).strip() for x in base.get("particle", pd.Series([])).unique() if pd.notna(x)}


    lat = base["lat_deg"].astype(float).to_numpy()
    lon = base["lon_deg"].astype(float).map(wrap180).to_numpy()
    lat_rad = _deg2rad(lat)
    lon_rad = _deg2rad(lon)
    feats = pd.DataFrame({
        "x_sin_lat": np.sin(lat_rad),
        "x_cos_lat": np.cos(lat_rad),
        "x_sin_lon": np.sin(lon_rad),
        "x_cos_lon": np.cos(lon_rad),
    })


    h = base["h_km"].astype(float).to_numpy()
    h_norm = (h - cfg.h_ref_km) / (cfg.h_scale_km + 1e-9)
    feats["h_km"]   = h
    feats["h_norm"] = h_norm

    if cfg.use_fourier:
        f_lat = _fourier_feats(lat, cfg.lat_freqs, "latF")
        f_lon = _fourier_feats(lon, cfg.lon_freqs, "lonF")
        f_h   = _fourier_feats(h,   cfg.h_freqs,   "hF")
        feats = pd.concat([feats, f_lat, f_lon, f_h], axis=1)


    feats["y_log"] = np.log(base["dose_rate_uSvph"].astype(float) + 1e-9)
    feats["dose_rate_uSvph"] = base["dose_rate_uSvph"].astype(float)
    feats["lat_deg"] = lat
    feats["lon_deg"] = lon
    feats["alt_tag"] = base.get("alt_tag", pd.Series([f"h{round(h[0],1)}km"]*len(h)))
    feats["source_file"] = base.get("source_file", pd.Series([""]*len(h)))


    block = np.floor((lon + 180.0) / max(cfg.lon_block_deg, 1e-6)).astype(int)
    feats["group_id"] = block

    if out_parquet:
        Path(out_parquet).parent.mkdir(parents=True, exist_ok=True)
        feats.to_parquet(out_parquet, index=False)
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        feats.to_csv(out_csv, index=False)

    by_alt = pd.DataFrame({
        "h_km": pd.Series(h),
        "dose": feats["dose_rate_uSvph"]
    })
    alt_summary = by_alt.groupby(by_alt["h_km"].round(1)).dose.agg(["count","mean","std","min","max"])
    print("== NF1-3D summary ==")
    print("units:", unit_set, "| particles:", particle_set)
    print(alt_summary.head(20))

    return feats

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="ans_vA 根目录（含 hXXpXkm 子目录）或已聚合的 CSV/Parquet")
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--out-parquet", default=None)
    ap.add_argument("--no-fourier", action="store_true", help="关闭位置编码")
    ap.add_argument("--lon-block-deg", type=float, default=30.0)
    ap.add_argument("--h-ref", type=float, default=16.0)
    ap.add_argument("--h-scale", type=float, default=6.0)
    args = ap.parse_args()

    cfg = NF1_3D_Config(
        lon_block_deg=args.lon_block_deg,
        h_ref_km=args.h_ref,
        h_scale_km=args.h_scale,
        use_fourier=(not args.no_fourier)
    )
    build_nf1_3d_features(
        src=args.src,
        out_csv=args.out_csv,
        out_parquet=args.out_parquet,
        cfg=cfg
    )