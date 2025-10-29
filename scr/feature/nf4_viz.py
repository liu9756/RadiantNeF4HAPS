# nf4_viz.py
# -*- coding: utf-8 -*-
"""
NF-4: 单高度二维场的可视化与等值线导出
输入：含列 (lat_deg, lon_deg, dose_pred_uSvph) 的 DataFrame（推荐用 NF-3 的 pred_df）
输出：
  - 热力图 PNG（pcolormesh/contourf）
  - 等值线 CSV（level, path_id, seg_id, pt_idx, lon, lat）
  - 可选：网格 CSV（lon,lat,dose）用于其他工具

仅依赖 numpy/pandas/matplotlib。
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional, Tuple, List
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt

# ---------- 将散点表格拼网格 ----------
def to_grid(df: pd.DataFrame, field: str = "dose_pred_uSvph"
           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lats = np.sort(df["lat_deg"].to_numpy(dtype=float).round(10))
    lons = np.sort(df["lon_deg"].to_numpy(dtype=float).round(10))
    lats = np.unique(lats)
    lons = np.unique(lons)
    L, M = len(lats), len(lons)

    lat_to_i = {v:i for i,v in enumerate(lats)}
    lon_to_j = {v:j for j,v in enumerate(lons)}
    Z = np.full((L, M), np.nan, dtype=float)
    for r in df.itertuples(index=False):
        i = lat_to_i[round(float(r.lat_deg),10)]
        j = lon_to_j[round(float(r.lon_deg),10)]
        Z[i, j] = float(getattr(r, field))

    LAT, LON = np.meshgrid(lats, lons, indexing="ij")
    return LAT, LON, Z

# ---------- 热力图 ----------
def plot_heatmap(df: pd.DataFrame,
                 field: str = "dose_pred_uSvph",
                 title: str = "Dose rate at ~10 km (μSv/h)",
                 save_path: Optional[str] = None,
                 levels: Optional[Iterable[float]] = None,
                 dpi: int = 200) -> plt.Figure:
    LAT, LON, Z = to_grid(df, field=field)
    fig, ax = plt.subplots(figsize=(9,4.2), constrained_layout=True)

    hm = ax.pcolormesh(LON, LAT, Z, shading="auto")
    cb = fig.colorbar(hm, ax=ax, shrink=0.9, label="μSv/h")

    if levels is not None:
        cs = ax.contour(LON, LAT, Z, levels=sorted(levels), linewidths=0.8)
        ax.clabel(cs, cs.levels, inline=True, fontsize=8, fmt="%.2f")
    ax.set_title(title)
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_xlim([np.min(LON), np.max(LON)])
    ax.set_ylim([np.min(LAT), np.max(LAT)])
    ax.grid(True, lw=0.2, alpha=0.6)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi)
    return fig

# ---------- 等值线导出 ----------
def export_contours_csv(df: pd.DataFrame,
                        levels,
                        out_csv: str,
                        field: str = "dose_pred_uSvph") -> pd.DataFrame:

    LAT, LON, Z = to_grid(df, field=field)
    Zm = np.ma.masked_invalid(Z)

    fig, ax = plt.subplots(figsize=(6, 3))
    cs = ax.contour(LON, LAT, Zm, levels=sorted(levels))
    plt.close(fig)

    rows = []
    for li, level in enumerate(cs.levels):
        segs = cs.allsegs[li]
        if not segs: 
            continue
        for path_id, seg in enumerate(segs, start=1):
            for pt_idx, (x, y) in enumerate(seg):
                rows.append((float(level), int(path_id), 0, int(pt_idx),
                             float(x), float(y)))

    out = pd.DataFrame(rows, columns=["level","path_id","seg_id","pt_idx","lon_deg","lat_deg"])
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out

# ---------- 网格 CSV（可选） ----------
def export_grid_csv(df: pd.DataFrame, out_csv: str, field: str = "dose_pred_uSvph") -> None:
    LAT, LON, Z = to_grid(df, field=field)
    flat = pd.DataFrame({
        "lat_deg": LAT.ravel(),
        "lon_deg": LON.ravel(),
        field: Z.ravel()
    })
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    flat.to_csv(out_csv, index=False)
