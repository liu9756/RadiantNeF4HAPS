from __future__ import annotations
import argparse, json, math, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import torch
import torch.nn as nn

from utils43D import build_features, wrap180
class MLP(nn.Module):
    def __init__(self, in_dim=5, hidden=64, depth=3, act="relu"):
        super().__init__()
        acts = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}
        A = acts.get(act, nn.ReLU)
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), A()]
            d = hidden
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): 
        return self.net(x).squeeze(-1)

def load_mlp(ckpt_path: str, device="cpu", override_arch: dict | None = None):
    import torch
    from pathlib import Path

    payload = torch.load(ckpt_path, map_location=device)
    state = payload.get("state_dict", payload.get("model_state", payload))
    if "in_dim" in payload:
        in_dim = int(payload["in_dim"])
    else:
        if "0.weight" in state:
            in_dim = state["0.weight"].shape[1]
        else:
            first_w = next(v for k, v in state.items() if k.endswith(".weight"))
            in_dim = first_w.shape[1]
    arch = payload.get("arch", {"hidden": 64, "depth": 3, "act": "relu"})
    if override_arch:
        arch.update(override_arch)

    print(f"[load_mlp] in_dim={in_dim}, arch={arch}")
    model = MLP(in_dim, hidden=arch["hidden"], depth=arch["depth"], act=arch["act"]).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, payload.get("spec", None), payload.get("h_norm_cfg", None)


@torch.inference_mode()
def predict_grid(model, lats, lons, hs, h_min, h_max, device="cpu",
                 fourier_freqs=(1,2,4,8), use_vcr=False, vcr_fill=0.0,
                 batch_size=131072):
    import numpy as np
    import torch

    lats = np.asarray(lats, dtype=np.float32)
    lons = np.asarray(lons, dtype=np.float32)
    hs   = np.asarray(hs,   dtype=np.float32)
    L, M, K = len(lats), len(lons), len(hs)

    LAT, LON, H = np.meshgrid(lats, lons, hs, indexing="ij")   
    n_total = L * M * K

    lam = np.deg2rad(wrap180(LON.reshape(-1)))  
    phi = np.deg2rad(LAT.reshape(-1))   
    base = np.stack([np.sin(lam), np.cos(lam), np.sin(phi), np.cos(phi)], axis=-1) 
    h = H.reshape(-1).astype(np.float64)
    h_norm = ((h - h_min) / (h_max - h_min + 1e-12)).reshape(-1, 1)      
    cross = np.concatenate([h_norm, h_norm * base, h_norm**2], axis=-1)        
    def fourier_enc(x, freqs=(1,2,4,8)):
        outs = [x]
        for b in freqs:
            outs += [np.sin(2*np.pi*b*x), np.cos(2*np.pi*b*x)]
        return np.concatenate(outs, axis=-1)
    ff = fourier_enc(np.concatenate([base, h_norm], axis=-1), tuple(fourier_freqs)) 

    feats = [base, cross, ff]
    if use_vcr:
        vcr = np.full((n_total, 1), float(vcr_fill), dtype=np.float64)
        feats.append(vcr)

    X = np.concatenate(feats, axis=-1).astype(np.float32)  
    expected_in = int(model.net[0].weight.shape[1])
    if X.shape[1] != expected_in:
        if X.shape[1] == expected_in + 1:
            print(f"[predict_grid][WARN] X has {X.shape[1]} dims but model expects {expected_in}. "
                  f"Assuming stray VCR column — dropping the last column.")
            X = X[:, :expected_in].copy()
        else:
            raise RuntimeError(f"[predict_grid]: X={X.shape[1]}, model expects {expected_in}.")


    model.eval()
    ys = np.empty((n_total,), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, n_total, batch_size):
            e = min(s + batch_size, n_total)
            xb = torch.from_numpy(X[s:e]).to(device)
            y_log = model(xb)      
            ys[s:e] = torch.exp(y_log).cpu().numpy().reshape(-1)  

    dose_klm = ys.reshape(L, M, K).transpose(2, 0, 1).copy()
    return dose_klm


def quick_physics_checks(dose_klm, lats, lons, h_list, n=200):
    K,L,M = dose_klm.shape
    rng = np.random.default_rng(42)
    idx_lat = rng.integers(0, L, size=n)
    idx_lon = rng.integers(0, M, size=n)
    mono_ok = 0
    for i,j in zip(idx_lat, idx_lon):
        prof = dose_klm[:, i, j]  
        dif = np.diff(prof)
        n_down = int(np.sum(dif < -1e-6))
        if n_down <= 1:
            mono_ok += 1
    return {"monotone_pass_ratio": mono_ok / n, "n_points_checked": n}

def plot_slice(
    dose_klm, lats, lons, h_list, h_sel, out_png,
    global_vmin=0.0, global_vmax=22.0, cmap="viridis",
    show_clip_stats=True
):
    
    k = int(np.argmin(np.abs(np.asarray(h_list, float) - float(h_sel))))
    Z = np.asarray(dose_klm[k], float)
    norm = mcolors.Normalize(vmin=global_vmin, vmax=global_vmax, clip=True)
    cm   = plt.get_cmap(cmap).copy()
    cm.set_under(cm(0.0))  
    cm.set_over(cm(1.0))  

    if show_clip_stats:
        n = Z.size
        n_low  = int((Z < global_vmin).sum())
        n_high = int((Z > global_vmax).sum())
        if n_low or n_high:
            print(f"[slice @{h_list[k]:.2f} km] under={n_low/n:.2%}, over={n_high/n:.2%}")

    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=200)
    im = ax.pcolormesh(lons, lats, Z, shading="auto", cmap=cm, norm=norm)
    ax.set_title(f"Dose rate (μSv/h) @ h≈{h_list[k]:.2f} km")
    ax.set_xlabel("Longitude (deg)"); ax.set_ylabel("Latitude (deg)")

    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, extend="both")
    cb.set_label("μSv/h")
    cb.set_ticks([0, 5, 10, 15, 20, 22])

    ax.set_xlim([float(np.min(lons)), float(np.max(lons))])
    ax.set_ylim([float(np.min(lats)), float(np.max(lats))])
    for lat in range(-60, 61, 30): ax.axhline(lat, lw=0.4, ls="--", c="k", alpha=0.3)
    for lon in range(-180, 181, 60): ax.axvline(lon, lw=0.4, ls="--", c="k", alpha=0.3)

    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png); plt.close(fig)
    return out_png

def try_marching_cubes():
    try:
        from skimage.measure import marching_cubes
        return marching_cubes
    except Exception as e:
        return None

def export_isosurfaces_obj(dose_klm, lats, lons, h_list, levels, out_dir):
    mc = try_marching_cubes()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    outputs = []
    dh = float(np.median(np.diff(h_list))) if len(h_list)>1 else 1.0
    dlat = float(np.median(np.diff(lats))) if len(lats)>1 else 1.0
    dlon = float(np.median(np.diff(lons))) if len(lons)>1 else 1.0
    for lv in levels:
        verts, faces, _, _ = mc(dose_klm, level=lv, spacing=(dh, dlat, dlon))
        obj_path = Path(out_dir)/f"iso_{lv:.2f}_uSvph.obj"
        with open(obj_path, "w") as f:
            for v in verts:
                f.write(f"v {v[2]:.6f} {v[1]:.6f} {v[0]:.6f}\n") 
            for tri in faces:
                a,b,c = tri+1
                f.write(f"f {a} {b} {c}\n")
        outputs.append(str(obj_path))
    return outputs


class FieldAPI3D:
    def __init__(self, dose_klm, lats, lons, h_list):
        self.vol = dose_klm; self.lats = np.asarray(lats); self.lons=np.asarray(lons); self.hs=np.asarray(h_list)
    def _idx2(self, arr, v):
        j = int(np.searchsorted(arr, v) - 1)
        return max(0, min(j, len(arr)-2))
    def predict(self, lat_deg, lon_deg, h_km):
        lon_deg = wrap180(lon_deg)
        ii = self._idx2(self.lats, lat_deg)
        jj = self._idx2(self.lons, lon_deg)
        kk = self._idx2(self.hs,   h_km)
        y0,y1 = self.lats[ii], self.lats[ii+1]
        x0,x1 = self.lons[jj], self.lons[jj+1]
        z0,z1 = self.hs[kk],   self.hs[kk+1]
        ty = (lat_deg - y0)/max(1e-12,(y1-y0))
        tx = (lon_deg - x0)/max(1e-12,(x1-x0))
        tz = (h_km   - z0)/max(1e-12,(z1-z0))
        c000 = self.vol[kk  , ii  , jj  ]
        c100 = self.vol[kk  , ii  , jj+1]
        c010 = self.vol[kk  , ii+1, jj  ]
        c110 = self.vol[kk  , ii+1, jj+1]
        c001 = self.vol[kk+1, ii  , jj  ]
        c101 = self.vol[kk+1, ii  , jj+1]
        c011 = self.vol[kk+1, ii+1, jj  ]
        c111 = self.vol[kk+1, ii+1, jj+1]
        c00 = (1-tx)*c000 + tx*c100
        c01 = (1-tx)*c001 + tx*c101
        c10 = (1-tx)*c010 + tx*c110
        c11 = (1-tx)*c011 + tx*c111
        c0  = (1-ty)*c00  + ty*c10
        c1  = (1-ty)*c01  + ty*c11
        return float((1-tz)*c0 + tz*c1)

def main():
    ap = argparse.ArgumentParser(description="NF4-3D: out")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--report-json", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)

    ap.add_argument("--lat-min", type=float, default=-80)
    ap.add_argument("--lat-max", type=float, default= 80)
    ap.add_argument("--lat-step", type=float, default=2.0)
    ap.add_argument("--lon-step", type=float, default=2.0)
    ap.add_argument("--heights", type=str, default="8,10,12,14,16,18,20,22")

    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--depth",  type=int, default=3)
    ap.add_argument("--act",    type=str, default="relu")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--vmin", type=float, default=None)
    ap.add_argument("--vmax", type=float, default=None)
    ap.add_argument("--iso-levels", type=str, default="5,8,10")

    args = ap.parse_args()


    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    lats = np.arange(args.lat_min, args.lat_max + 1e-6, args.lat_step, dtype=np.float32)
    lons = np.arange(-180.0, 180.0 + 1e-6, args.lon_step, dtype=np.float32)
    hs   = [float(x) for x in args.heights.split(",")]


    rep = json.load(open(args.report_json, "r"))
    h_min = rep.get("h_norm_cfg", {}).get("h_min", 8.0)
    h_max = rep.get("h_norm_cfg", {}).get("h_max", 22.0)
    bp = rep.get("best_params", {})
    hidden = bp.get("hidden", args.hidden)
    depth  = bp.get("depth",  args.depth)
    act    = bp.get("act",    args.act)

    h_norm_cfg = None
    spec_from_ckpt = None
    model, spec_from_ckpt, h_norm_cfg = load_mlp(args.ckpt, device=args.device)

    if spec_from_ckpt is not None:
        fourier_freqs = tuple(spec_from_ckpt.get("fourier_freqs", (1,2,4,8)))
    else:
        fourier_freqs = (1,2,4,8)


    expected_in = int(model.net[0].weight.shape[1])

    def in_dim_given(freqs, use_vcr):
        return 4 + 6 + 5 * (1 + 2*len(freqs)) + (1 if use_vcr else 0)

    in_no_vcr  = in_dim_given(fourier_freqs, False) 
    in_with_vcr= in_dim_given(fourier_freqs, True)   

    if expected_in == in_no_vcr:
        use_vcr = False
    elif expected_in == in_with_vcr:
        use_vcr = True
    print(f"[nf4] expected_in={expected_in}, use_vcr={use_vcr}, freqs={fourier_freqs}")

    if h_norm_cfg is None and args.report_json:
        with open(args.report_json, "r") as f:
            rep2 = json.load(f)
        h_norm_cfg = rep2.get("h_norm_cfg", None)

    if h_norm_cfg is not None:
        h_min = h_norm_cfg.get("h_min", h_min)
        h_max = h_norm_cfg.get("h_max", h_max)

    if h_norm_cfg is None:
        raise RuntimeError("h_norm_cfg 缺失：请保证 ckpt 或 report JSON 中包含 {h_min,h_max}。")
    dose_klm = predict_grid(
        model, lats, lons, hs,
        h_min, h_max,
        device=args.device,
        fourier_freqs=fourier_freqs,  
        use_vcr=use_vcr,              
        vcr_fill=0.0
    )

    np.savez_compressed(outdir/"volume.npz",
                        dose_muSvph=dose_klm, lat_deg=lats, lon_deg=lons, h_km=np.array(hs))
    meta = {"h_min":h_min,"h_max":h_max,"lat_step":args.lat_step,"lon_step":args.lon_step,
            "hidden":hidden,"depth":depth,"act":act,"heights":hs,
            "fourier_freqs": list(fourier_freqs), "use_vcr": use_vcr}
    json.dump(meta, open(outdir/"meta.json","w"), indent=2)


    grid_dir = outdir/"grid_layers"; grid_dir.mkdir(exist_ok=True)
    for k, h in enumerate(hs):
        df = pd.DataFrame({
            "lat_deg": np.repeat(lats, len(lons)),
            "lon_deg": np.tile(lons, len(lats)),
            "h_km":    h,
            "dose_uSvph": dose_klm[k].reshape(-1)
        })
        df.to_csv(grid_dir/f"grid_{h:.1f}km.csv", index=False)


    qc = quick_physics_checks(dose_klm, lats, lons, hs, n=200)
    json.dump(qc, open(outdir/"quick_checks.json","w"), indent=2)

    for h_sel in [8,10,12,14,16,18,20,22]:
        if h_sel >= hs[0] - 1e-6 and h_sel <= hs[-1] + 1e-6:
            plot_slice(dose_klm, lats, lons, hs, h_sel,
                out_png=outdir/f"slice_{h_sel:.0f}km.png",
                global_vmin=0.0, global_vmax=22.0)


    iso_levels = [float(x) for x in args.iso_levels.split(",")]   
    iso_out = export_isosurfaces_obj(dose_klm, lats, lons, hs, iso_levels, out_dir=outdir/"isosurfaces")
    json.dump({"isosurf_objs": iso_out, "levels": iso_levels}, open(outdir/"isosurfaces_index.json","w"), indent=2)

    api = FieldAPI3D(dose_klm, lats, lons, hs)
    demo_pts = [ (35.0, 139.7, z) for z in np.linspace(hs[0], hs[-1], 10) ]
    demo = pd.DataFrame([{"lat":a,"lon":b,"h":c,"dose_pred_uSvph":api.predict(a,b,c)} for (a,b,c) in demo_pts])
    demo.to_csv(outdir/"demo_profile_tokyo.csv", index=False)

    print("[NF4-3D] 完成：", outdir)


if __name__ == "__main__":
    main()
