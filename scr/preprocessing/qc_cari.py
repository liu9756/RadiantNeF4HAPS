#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CARI-7A ANS Quality Control
- Parses one or more .ANS files (robust to ALTITUDE unit column F/G/K).
- Checks schema/units/date, deduplicates, filters altitude window.
- Vertical profile checks (soft Pfotzer window), latitude trend checks.
- Exports CSV and JSON summaries for training readiness.

Refs:
- ANS header & units: LAT, LON, ALTITUDE, DATE, HR, VCR(GV), PARTICLE, DOSE RATE, SIGMA, UNIT, QUANTITY; rows show ALTITUDE followed by F/G/K; units microSv/hr. (FAA CARI-7A User's Guide)  # see citation in chat
- Monthly average: dd=00.  # see citation in chat
- Menu-less DEFAULT.INP + MENUS=NO!: line 5 .LOC -> .ANS.  # see citation in chat
- Pfotzer maximum may be weak/disappear in dose metrics (soft prior).  # see citation in chat
"""
import argparse, csv, json, math, re
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd

# ---------------------------
# Parsing
# ---------------------------

ANS_HEADER = ["LAT", "LON", "ALTITUDE", "DATE", "HR", "VCR(GV)",
              "PARTICLE", "DOSE RATE", "SIGMA", "UNIT", "QUANTITY"]

ALT_UNITS = {"F": "feet", "G": "meters", "K": "kilometers"}

def feet_to_km(ft: float) -> float:
    return float(ft) / 3280.839895

def _row_is_header(tokens: List[str]) -> bool:
    normalized = [t.strip().upper() for t in tokens]
    return ("LAT" in normalized and "LON" in normalized and "ALTITUDE" in normalized)

def parse_ans(path: Path) -> pd.DataFrame:
    """
    Robust parser for CARI .ANS (CSV-ish) files.
    - Finds the header row dynamically.
    - Handles ALTITUDE followed by a separate unit token (F/G/K).
    - Drops comment lines starting with "C".
    """
    records = []
    header_found = False
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if not header_found and _row_is_header(row):
                header_found = True
                continue
            if not header_found:
                continue
            first = str(row[0]).strip()
            if first.upper().startswith("C"):
                continue

            t = [c.strip() for c in row if c is not None]
            t = [x for x in t if x != ""]

            if len(t) < 11:
                continue

            try:
                lat = float(t[0]); lon = float(t[1])
                alt = float(t[2])
                alt_unit = None
                date_idx = 3
                if t[3] in ALT_UNITS:
                    alt_unit = t[3]
                    date_idx = 4

                date = t[date_idx]
                hour = t[date_idx+1]
                vcr = float(t[date_idx+2])
                particle = t[date_idx+3]
                dose_rate = float(t[date_idx+4])
                sigma = float(t[date_idx+5])
                unit = t[date_idx+6]
                quantity = ",".join(t[date_idx+7:]).strip()

                # normalize altitude to km
                if alt_unit == "F" or (alt_unit is None and alt > 1e3):
                    h_km = feet_to_km(alt)
                elif alt_unit == "K":
                    h_km = float(alt)
                else:
                    h_km = float(alt) / 1000.0

                records.append({
                    "source_file": path.name,
                    "lat_deg": lat,
                    "lon_deg": lon,
                    "h_km": h_km,
                    "date": date,
                    "hour": hour,
                    "vcr_GV": vcr,
                    "particle": particle.strip(),
                    "dose_rate_uSvph": dose_rate,  
                    "sigma": sigma,
                    "unit_text": unit,
                    "quantity_text": quantity
                })
            except Exception:
                continue

    df = pd.DataFrame.from_records(records)
    return df

# ---------------------------
# QC checks
# ---------------------------

def qc_schema_units(df: pd.DataFrame) -> Dict:
    issues = []
    if not df["unit_text"].str.contains("micro", case=False, na=False).any():
        issues.append("No 'microSv/hr' unit string found in ANS rows.")
    if df["quantity_text"].isna().any():
        issues.append("Some rows have empty QUANTITY text.")
    total_ratio = (df["particle"].str.upper().str.contains("TOTAL", na=False).sum()) / max(len(df),1)
    return {"unit_ok": len(issues)==0, "issues": issues, "particle_total_ratio": total_ratio}

def qc_date_monthavg(df: pd.DataFrame) -> Dict:
    monthavg_mask = df["date"].astype(str).str.match(r"^\d{4}/\d{2}/00$")
    ratio = monthavg_mask.mean()
    return {"monthly_avg_ratio": float(ratio)}

def qc_deduplicate(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    keys = ["lat_deg","lon_deg","h_km","date","hour"]
    before = len(df)
    deduped = df.drop_duplicates(subset=keys)
    removed = before - len(deduped)
    return deduped, {"duplicates_removed": int(removed)}

def qc_filter_altitude(df: pd.DataFrame, hmin: float, hmax: float) -> Tuple[pd.DataFrame, Dict]:
    mask = (df["h_km"] >= hmin) & (df["h_km"] <= hmax)
    kept = df[mask].copy()
    return kept, {"altitude_kept": int(mask.sum()), "altitude_dropped": int((~mask).sum())}

def moving_avg(x: np.ndarray, k: int = 3) -> np.ndarray:
    if k <= 1 or len(x) < k: return x
    pad = k//2
    xp = np.pad(x, (pad,pad), mode="edge")
    kernel = np.ones(k)/k
    return np.convolve(xp, kernel, mode="valid")

def profile_metrics(g: pd.DataFrame, smooth_k:int=5) -> Dict:
    gg = g.sort_values("h_km")
    h = gg["h_km"].to_numpy()
    y = gg["dose_rate_uSvph"].to_numpy()
    ylog = np.log1p(np.clip(y, a_min=0, a_max=None))
    ysm = moving_avg(ylog, k=smooth_k)
    dh = np.diff(h)
    dy = np.diff(ysm)
    signs = np.sign(dy)
    crossings = np.where((signs[:-1] > 0) & (signs[1:] <= 0))[0]
    n_peaks = int(len(crossings))
    imax = int(np.argmax(ysm))
    h_star = float(h[min(imax, len(h)-1)])
    pf_ok = (12.0 <= h_star <= 22.0)
    osc = float(np.mean(np.abs(np.diff(signs))))
    return {
        "n_points": int(len(h)),
        "n_peaks": n_peaks,
        "h_star_km": h_star,
        "pf_soft_ok": bool(pf_ok),
        "osc_metric": osc,
        "dose_max_uSvph": float(np.max(y)) if len(y)>0 else float("nan")
    }

def qc_vertical_profiles(df: pd.DataFrame, smooth_k:int=5) -> pd.DataFrame:
    rows = []
    for (lat, lon, date), g in df.groupby(["lat_deg","lon_deg","date"]):
        m = profile_metrics(g, smooth_k=smooth_k)
        m.update({"lat_deg":float(lat), "lon_deg":float(lon), "date":str(date)})
        rows.append(m)
    return pd.DataFrame(rows)

def qc_latitude_trend(df: pd.DataFrame, heights_km = (10,12,14,16,18,20,22)) -> pd.DataFrame:
    bins = [0,20,40,60,80,90]
    out_rows = []
    for hk in heights_km:
        slice_df = df.loc[(df["h_km"]>=hk-0.3)&(df["h_km"]<=hk+0.3)].copy()
        if len(slice_df)==0:
            continue
        slice_df["abs_lat"] = slice_df["lat_deg"].abs()
        slice_df["bin"] = pd.cut(slice_df["abs_lat"], bins=bins, right=False)
        med = slice_df.groupby("bin")["dose_rate_uSvph"].median().reset_index()
        vals = med["dose_rate_uSvph"].to_numpy()
        monotone = bool(np.all(np.diff(vals[~np.isnan(vals)]) >= -1e-9)) if len(vals)>1 else True
        out_rows.append({"height_km": hk, "monotone_nondec": monotone,
                         **{f"bin_{i}_med": float(vals[i]) if i<len(vals) and not math.isnan(vals[i]) else float("nan")
                            for i in range(len(vals))}})
    return pd.DataFrame(out_rows)

# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ans-glob", required=True, help="Glob pattern for .ANS files (quoted).")
    ap.add_argument("--recursive", action="store_true", help="Enable recursive globbing (**).")
    ap.add_argument("--outdir", required=True, help="Output directory.")
    ap.add_argument("--hmin", type=float, default=8.0)
    ap.add_argument("--hmax", type=float, default=25.0)
    ap.add_argument("--smooth", type=int, default=5, help="Moving average window for profiles.")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Expand glob(s)
    pattern = args.ans_glob
    paths = sorted(Path("/").glob(pattern.lstrip("/"))) if args.recursive else sorted(Path().glob(pattern))
    if len(paths)==0:
        print(f"[QC] No ANS files matched: {pattern}")
        return

    # Parse and concat
    dfs = []
    for p in paths:
        try:
            dfp = parse_ans(p)
            if len(dfp)==0:
                print(f"[QC] WARNING: empty after parse: {p}")
            dfs.append(dfp)
        except Exception as e:
            print(f"[QC] ERROR parsing {p}: {e}")

    if len(dfs)==0:
        print("[QC] No data parsed.")
        return

    df = pd.concat(dfs, ignore_index=True)

    sch = qc_schema_units(df)
    monthavg = qc_date_monthavg(df)
    df, ddup = qc_deduplicate(df)
    df_f, altstats = qc_filter_altitude(df, args.hmin, args.hmax)
    neg_rows = int((df_f["dose_rate_uSvph"] < 0).sum())
    prof_df = qc_vertical_profiles(df_f, smooth_k=args.smooth)
    lat_df = qc_latitude_trend(df_f)

    summary = {
        "n_files": len(paths),
        "n_rows_raw": int(sum(len(x) for x in dfs)),
        "n_rows_after_dedup": int(len(df)),
        "n_rows_in_alt_range": int(len(df_f)),
        "neg_dose_count": neg_rows,
        "schema_units": sch,
        "month_avg": monthavg,
        "altitude_filter": altstats,
        "profile_summary": {
            "n_profiles": int(len(prof_df)),
            "pf_soft_ok_ratio": float(prof_df["pf_soft_ok"].mean()) if len(prof_df)>0 else None,
            "median_h_star_km": float(prof_df["h_star_km"].median()) if len(prof_df)>0 else None,
            "median_n_peaks": float(prof_df["n_peaks"].median()) if len(prof_df)>0 else None
        },
        "latitude_trend": {
            "heights_checked": lat_df["height_km"].dropna().tolist(),
            "monotone_pass_ratio": float(lat_df["monotone_nondec"].mean()) if len(lat_df)>0 else None
        }
    }

    df_f.to_csv(outdir / "ans_concat_filtered.csv", index=False)
    prof_df.to_csv(outdir / "qc_profiles.csv", index=False)
    lat_df.to_csv(outdir / "qc_latitude_trend.csv", index=False)
    with (outdir / "qc_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print("[QC] Done. Outputs written to:", outdir)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
