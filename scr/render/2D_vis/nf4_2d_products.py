from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_volume(npz_path: str):
    P = np.load(npz_path)
    dose = P.get("dose_muSvph", None)
    if dose is None: dose = P["dose"] 
    lats = P["lat"] if "lat" in P else P["lat_deg"]
    lons = P["lon"] if "lon" in P else P["lon_deg"]
    hs   = P["heights"] if "heights" in P else P["h_km"]
    return np.asarray(dose), np.asarray(lats), np.asarray(lons), np.asarray(hs, float)

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def plot_slice(Z: np.ndarray, lats, lons, title, out_png,
               vmin=0.0, vmax=22.0, cmap="turbo"):
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=200)
    im = ax.pcolormesh(lons, lats, Z, shading="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02); cb.set_label("μSv/h")
    ax.set_xlim([float(lons.min()), float(lons.max())])
    ax.set_ylim([float(lats.min()), float(lats.max())])
    for lat in range(-60, 61, 30): ax.axhline(lat, lw=0.4, ls="--", c="k", alpha=0.3)
    for lon in range(-180, 181, 60): ax.axvline(lon, lw=0.4, ls="--", c="k", alpha=0.3)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png); plt.close(fig)

def gen_global_slices(dose_klm, lats, lons, hs, outdir: Path, vmin=0.0, vmax=22.0):
    od = outdir / "slices"; ensure_dir(od)
    for k, h in enumerate(hs):
        Z = dose_klm[k]  # [L,M]
        plot_slice(Z, lats, lons,
                   title=f"Dose rate (μSv/h)  @ h ≈ {h:.1f} km   [fixed 0–22]",
                   out_png=od/f"slice_{h:.1f}km.png", vmin=vmin, vmax=vmax)

def hovmoller_lat_height(dose_klm, lats, lons, hs, outdir: Path, vmin=0.0, vmax=22.0, cmap="turbo"):
    """纬度×高度（对经度求均值） -> [K,L]"""
    Z = dose_klm.mean(axis=2) 
    fig, ax = plt.subplots(figsize=(7.0, 5.0), dpi=200)
    im = ax.pcolormesh(lats, hs, Z[:, :].T if Z.shape[0]==len(lats) else Z, shading="auto",
                       vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title("Hovmöller: Latitude × Height (zonal mean)")
    ax.set_xlabel("Latitude (°)"); ax.set_ylabel("Height (km)")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02); cb.set_label("μSv/h")
    fig.tight_layout()
    ensure_dir(outdir/"hovmoller")
    fig.savefig(outdir/"hovmoller/lat_height.png"); plt.close(fig)

def nearest_index(arr, v):
    return int(np.clip(np.argmin(np.abs(arr - v)), 0, len(arr)-1))

def hovmoller_lon_height(dose_klm, lats, lons, hs, outdir: Path,
                         lat_lines=(-60,-30,0,30,60), vmin=0.0, vmax=22.0, cmap="turbo"):
    """经度×高度（固定某纬线，沿纬线最近格点取值） -> [K,M]"""
    ensure_dir(outdir/"hovmoller")
    for lat in lat_lines:
        i = nearest_index(lats, lat)
        Z = dose_klm[:, i, :]  # [K,M]
        fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=200)
        im = ax.pcolormesh(lons, hs, Z, shading="auto", vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_title(f"Hovmöller: Longitude × Height @ lat≈{lats[i]:.1f}°")
        ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Height (km)")
        cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02); cb.set_label("μSv/h")
        ax.set_xlim([float(lons.min()), float(lons.max())])
        for lon0 in range(-180, 181, 60): ax.axvline(lon0, lw=0.4, ls="--", c="k", alpha=0.25)
        fig.tight_layout()
        fig.savefig(outdir/f"hovmoller/lon_height_lat{lats[i]:+.0f}.png"); plt.close(fig)

def pfotzer_maps(dose_klm, lats, lons, hs, outdir: Path,
                 vmax_val=22.0, cmap_val="turbo", cmap_h="viridis"):
    idx = np.argmax(dose_klm, axis=0)    
    peak_val = np.take_along_axis(dose_klm, idx[None, ...], axis=0)[0]  
    peak_h   = hs[idx]  
    ensure_dir(outdir/"pfotzer")

    plot_slice(peak_val, lats, lons, title="Peak dose (μSv/h) within 8–22 km",
               out_png=outdir/"pfotzer/peak_value.png", vmin=0.0, vmax=vmax_val, cmap=cmap_val)

    vmin_h, vmax_h = float(hs.min()), float(hs.max())
    fig, ax = plt.subplots(figsize=(9,4.2), dpi=200)
    im = ax.pcolormesh(lons, lats, peak_h, shading="auto", vmin=vmin_h, vmax=vmax_h, cmap=cmap_h)
    ax.set_title("Peak height (km) within 8–22 km")
    ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02); cb.set_label("km")
    ax.set_xlim([float(lons.min()), float(lons.max())])
    ax.set_ylim([float(lats.min()), float(lats.max())])
    for lat in range(-60, 61, 30): ax.axhline(lat, lw=0.4, ls="--", c="k", alpha=0.3)
    for lon in range(-180, 181, 60): ax.axvline(lon, lw=0.4, ls="--", c="k", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir/"pfotzer/peak_height.png"); plt.close(fig)

def per_height_stats(dose_klm, hs, outdir: Path):
    ensure_dir(outdir/"stats")
    rows = []
    for k, h in enumerate(hs):
        vals = dose_klm[k].astype(np.float64)
        rows.append({
            "h_km": float(h),
            "mean_uSvph": float(np.mean(vals)),
            "std_uSvph":  float(np.std(vals)),
            "min_uSvph":  float(np.min(vals)),
            "max_uSvph":  float(np.max(vals)),
            "p50_uSvph":  float(np.percentile(vals, 50)),
            "p90_uSvph":  float(np.percentile(vals, 90)),
        })
    df = pd.DataFrame(rows).sort_values("h_km")
    df.to_csv(outdir/"stats/per_height.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.8, 3.6), dpi=200)
    ax.bar(df["h_km"], df["mean_uSvph"])
    ax.set_title("Global mean dose by height"); ax.set_xlabel("Height (km)"); ax.set_ylabel("μSv/h")
    fig.tight_layout(); fig.savefig(outdir/"stats/mean_by_h.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 3.6), dpi=200)
    ax.bar(df["h_km"], df["std_uSvph"])
    ax.set_title("Global std of dose by height"); ax.set_xlabel("Height (km)"); ax.set_ylabel("μSv/h")
    fig.tight_layout(); fig.savefig(outdir/"stats/std_by_h.png"); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", type=str, required=True, help="path to volume.npz")
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--fixed-min", type=float, default=0.0, help="fixed color min for slices/Hovmöller")
    ap.add_argument("--fixed-max", type=float, default=22.0, help="fixed color max for slices/Hovmöller")
    ap.add_argument("--lat-lines", type=str, default="-60,-30,0,30,60", help="lon×height Hovmöller纬线，逗号分隔")
    args = ap.parse_args()

    outdir = Path(args.outdir); ensure_dir(outdir)
    dose_klm, lats, lons, hs = load_volume(args.volume)


    gen_global_slices(dose_klm, lats, lons, hs, outdir, vmin=args.fixed_min, vmax=args.fixed_max)

    hovmoller_lat_height(dose_klm, lats, lons, hs, outdir, vmin=args.fixed_min, vmax=args.fixed_max)
    lat_lines = [float(x) for x in args.lat_lines.split(",") if len(x)]
    hovmoller_lon_height(dose_klm, lats, lons, hs, outdir, lat_lines=lat_lines,
                         vmin=args.fixed_min, vmax=args.fixed_max)


    pfotzer_maps(dose_klm, lats, lons, hs, outdir, vmax_val=args.fixed_max)

    per_height_stats(dose_klm, hs, outdir)

    print("[NF4-2D] Outputs written to:", outdir)

if __name__ == "__main__":
    main()
