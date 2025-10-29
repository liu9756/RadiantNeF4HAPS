# nf1_features.py
# -*- coding: utf-8 -*-
"""
NF-1: 特征工程（单高度二维神经场）
- 输入：CARI-7A 解析后的 CSV（推荐）或原始 .ANS
- 输出：DataFrame（可选择保存成 CSV），在原始列基础上追加以下列：
    x_sin_lat, x_cos_lat, x_sin_lon, x_cos_lon, alt_norm, y_log, group_id

设计说明
1) 列名自适配：如果是你QC产物（lat_deg 等），直接用；如果是原始 .ANS，
   会先尝试从你现有的 gen_cari_loc_and_parse_ans.read_ans() 解析，
   若不可用则用内置的轻量解析器（仅读表头后的数值行）。
2) 不做筛选/去重/单位改动：严格按你当前阶段的约束。
3) 数学依据：
   - 经纬 sin/cos 周期编码可缓解坐标-MLP 的频谱偏置，提升高频表达能力；
     参见 Tancik et al., NeurIPS 2020（Fourier features）与 Rahaman et al., ICML 2019（spectral bias）。
   - y_log = log1p(μSv/h) 常见于长尾/跨数量级回归的稳定化处理。

参考资料（用于本模块的设计依据）：
- CARI-7/-7A 用户指南：月平均 dd=00；LOC→ANS 字段与批处理用法。
- Fourier features / Spectral bias：Tancik 2020；Rahaman 2019.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Literal, Dict, Any

import math
import numpy as np
import pandas as pd

_ANS_TO_CANON = {
    "LAT": "lat_deg",
    "LON": "lon_deg",
    "ALTITUDE": "h_km",            
    "DATE": "date",
    "HR": "hour",
    "VCR(GV)": "vcr_GV",
    "PARTICLE": "particle",
    "DOSE RATE": "dose_rate_uSvph",
    "SIGMA": "sigma",
    "UNIT": "unit_text",
    "QUANTITY": "quantity_text",
}

_FT2KM = 1.0 / 3280.839895

def _try_import_external_parser():
    try:
        from gen_cari_loc_and_parse_ans import read_ans  # type: ignore
        return read_ans
    except Exception:
        return None

def _parse_ans_light(ans_path: Path) -> pd.DataFrame:
    recs = []
    header = None
    with ans_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            row = [c.strip() for c in line.strip().split(",")]
            if not row or len(row) < 3:
                continue
            uppers = [c.upper() for c in row]
            if header is None and "LAT" in uppers and "LON" in uppers and "ALTITUDE" in uppers:
                header = uppers
                continue
            if header is None:
                continue
            if row[0].startswith("C") or row[0].startswith("!"):
                continue
            try:
                lat = float(row[0]); lon = float(row[1]); alt = float(row[2])
                idx = 3
                alt_unit = None
                if len(row) > 3 and row[3] in ("F","G","K"):
                    alt_unit = row[3]; idx = 4

                date   = row[idx]   if len(row) > idx   else ""
                hour   = row[idx+1] if len(row) > idx+1 else ""
                vcr    = float(row[idx+2]) if len(row) > idx+2 and row[idx+2] not in ("",) else np.nan
                part   = row[idx+3] if len(row) > idx+3 else ""
                dose   = float(row[idx+4]) if len(row) > idx+4 else np.nan
                sigma  = float(row[idx+5]) if len(row) > idx+5 else np.nan
                unit   = row[idx+6] if len(row) > idx+6 else ""
                quant  = ",".join(row[idx+7:]).strip() if len(row) > idx+7 else ""
                # ALT → km
                if (alt_unit == "F") or (alt_unit is None and alt > 1e3):
                    h_km = alt * _FT2KM
                elif alt_unit == "K":
                    h_km = alt
                else:
                    h_km = alt / 1000.0

                recs.append({
                    "lat_deg": lat, "lon_deg": lon, "h_km": h_km,
                    "date": date, "hour": hour, "vcr_GV": vcr,
                    "particle": part, "dose_rate_uSvph": dose,
                    "sigma": sigma, "unit_text": unit, "quantity_text": quant
                })
            except Exception:
                continue

    if not recs:
        raise ValueError(f"Empty .ANS after parsing: {ans_path}")
    return pd.DataFrame.from_records(recs)

def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c: c for c in df.columns}
    ups = {c.upper(): c for c in df.columns}
    need_map = any(k in ups for k in _ANS_TO_CANON.keys())

    if need_map:
        rename = {}
        for src_upper, dst in _ANS_TO_CANON.items():
            if src_upper in ups:
                rename[ups[src_upper]] = dst
        df = df.rename(columns=rename)

    required = ["lat_deg","lon_deg","h_km","dose_rate_uSvph"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns after canonicalization: {missing}")
    return df

def build_nf1_features(
    input_path: str | Path,
    save_path: Optional[str | Path] = None,
    format_hint: Literal["auto","csv","ans"] = "auto"
) -> pd.DataFrame:

    p = Path(str(input_path))
    if format_hint == "auto":
        ext = p.suffix.lower()
        if ext == ".csv":
            format_hint = "csv"
        elif ext == ".ans":
            format_hint = "ans"
        else:
            format_hint = "csv"

    if format_hint == "csv":
        try:
            df = pd.read_csv(p)
        except Exception as e:
            df = _parse_ans_light(p)
    elif format_hint == "ans":
        reader = _try_import_external_parser()
        if reader is not None:
            df = reader(str(p))
        else:
            df = _parse_ans_light(p)
    else:
        raise ValueError(f"Unknown format_hint: {format_hint}")

    df = _canonicalize_columns(df).copy()

    # --- NF-1 feature ---
    phi = np.deg2rad(df["lat_deg"].astype(float).to_numpy())
    lam = np.deg2rad(df["lon_deg"].astype(float).to_numpy())
    hkm = df["h_km"].astype(float).to_numpy()
    dose = df["dose_rate_uSvph"].astype(float).to_numpy()

    df["x_sin_lat"] = np.sin(phi)
    df["x_cos_lat"] = np.cos(phi)
    df["x_sin_lon"] = np.sin(lam)
    df["x_cos_lon"] = np.cos(lam)
    df["alt_norm"]  = (hkm - 16.5) / 8.0
    df["y_log"]     = np.log1p(dose)
    df["group_id"]  = df["lat_deg"].map("{:.5f}".format) + "_" + df["lon_deg"].map("{:.5f}".format)

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
    return df



if __name__ == "__main__":
    pass
