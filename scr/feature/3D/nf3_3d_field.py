from __future__ import annotations
import json, math, argparse, os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils43D import rmse_from_log, mape_from_log, wrap180, to_rad, mape

def wrap180(lon): return ((lon + 180.0) % 360.0) - 180.0

class FourierEncoder:
    def __init__(self, freqs=(1,2,4,8)):
        self.freqs = np.asarray(freqs, float)

    def __call__(self, x):
        """
        x: (..., D)
        return: (..., D * (1 + 2*len(freqs)))
        """
        x = np.asarray(x, float)
        outs = [x]
        for b in self.freqs:
            outs += [np.sin(2*np.pi*b*x), np.cos(2*np.pi*b*x)]
        return np.concatenate(outs, axis=-1)


@dataclass
class NF3InputSpec:
    fourier_freqs: tuple = (1,2,4,8)
    use_vcr: bool = True
    h_min: float = None   
    h_max: float = None

def build_inputs(df: pd.DataFrame, spec: NF3InputSpec):
    lam = to_rad(wrap180(df["lon_deg"].to_numpy()))
    phi = to_rad(df["lat_deg"].to_numpy())
    base = np.stack([np.sin(lam), np.cos(lam), np.sin(phi), np.cos(phi)], axis=-1)

    h = df["h_km"].to_numpy().astype(np.float64)
    h_min = float(np.min(h)) if spec.h_min is None else spec.h_min
    h_max = float(np.max(h)) if spec.h_max is None else spec.h_max
    h_norm = (h - h_min) / (h_max - h_min + 1e-12)
    h_norm = h_norm.reshape(-1,1)

    cross = np.concatenate([
        h_norm,
        h_norm * base[:,[0,1,2,3]],    
        h_norm**2
    ], axis=-1)

    enc = FourierEncoder(spec.fourier_freqs)
    ff = enc(np.concatenate([base, h_norm], axis=-1))

    feats = [base, cross, ff]

    if spec.use_vcr and "vcr_GV" in df.columns:
        vcr = df["vcr_GV"].to_numpy().astype(np.float64).reshape(-1,1)
        feats.append(vcr)

    X = np.concatenate(feats, axis=-1)
    return X, dict(h_min=h_min, h_max=h_max)


class FieldDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float().view(-1,1)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.y[i]


class MLP(nn.Module):
    def __init__(self, in_dim, hidden=64, depth=3, act="relu"):
        super().__init__()
        acts = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh
        }
        A = acts[act]
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), A()]
            d = hidden
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_one(model, train_loader, val_loader, epochs=5000, lr=1e-3, wd=1e-4,
              patience=500, device="cpu", mono_weight=0.0, latlonh=None,
              log_every=1, tag="cv"):

    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    mse = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    wait = 0

    n_train = len(train_loader.dataset)
    n_val   = len(val_loader.dataset)

    for ep in range(1, epochs + 1):
        model.train()
        train_sum = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            yp = model(xb)
            loss = mse(yp, yb) 
            loss.backward()
            opt.step()
            train_sum += float(loss.item()) * xb.size(0)
        train_mse = train_sum / max(1, n_train)
        train_rmse = math.sqrt(train_mse)

        model.eval()
        val_sum = 0.0
        y_true_log_list, y_pred_log_list = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                yp = model(xb)
                loss = mse(yp, yb)
                val_sum += float(loss.item()) * xb.size(0)
                y_true_log_list.append(yb.cpu().numpy().reshape(-1))
                y_pred_log_list.append(yp.cpu().numpy().reshape(-1))
        val_mse = val_sum / max(1, n_val)
        val_rmse = math.sqrt(val_mse)
        y_true_log = np.concatenate(y_true_log_list, axis=0)
        y_pred_log = np.concatenate(y_pred_log_list, axis=0)
        val_mape = mape_from_log(y_true_log, y_pred_log)

        improved = val_mse < best_val - 1e-9
        if improved:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if (ep % log_every == 0) or improved or (ep == 1):
            lr_cur = opt.param_groups[0]["lr"]
            print(
                f"[{tag}] Epoch {ep:04d}/{epochs} | "
                f"train_rmse_log={train_rmse:.6f}  "
                f"val_rmse_log={val_rmse:.6f}  val_mape%={val_mape:.3f}  "
                f"best_rmse_log={math.sqrt(best_val):.6f}  wait={wait}/{patience}  lr={lr_cur:g}",
                flush=True
            )

        if wait >= patience:
            if log_every > 0:
                print(f"[{tag}] Early stop at epoch {ep}, best_rmse_log={math.sqrt(best_val):.6f}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    return model, best_val


def block_id_from_lon(lon_deg, deg=30.0):
    lon_wrapped = wrap180(lon_deg)
    bid = np.floor((lon_wrapped + 180.0) / deg).astype(int)
    return bid

def rmse_log_metric(y_log_true, y_log_pred):
    return float(np.sqrt(np.mean((y_log_true - y_log_pred)**2)))


def run_nf3_3d(
    features_path: str,
    out_pred_csv: str,
    out_report_json: str,
    lon_block_deg: float = 30.0,
    hidden: int = 64, depth: int = 3, act: str = "relu",
    lr: float = 1e-3, weight_decay: float = 1e-4,
    epochs: int = 5000, batch_size: int = 256, patience: int = 500,
    mono_weight: float = 0.0,
    use_vcr: bool = True,
    export_grid_npz: str | None = None,
    grid_lat_step: float = 2.0,
    grid_lon_step: float = 2.0,
    grid_heights: list[float] | None = None,
    save_ckpt_path: str | None = None,
    device: str = "cpu"
):

    p = Path(features_path)
    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)


    if "y_log" in df.columns:
        y_log = df["y_log"].to_numpy().astype(np.float64)
    else:
        y_log = np.log(df["dose_rate_uSvph"].to_numpy().astype(np.float64) + 1e-12)


    spec = NF3InputSpec(use_vcr=use_vcr)
    X, h_cfg = build_inputs(df, spec)
    bids = block_id_from_lon(df["lon_deg"].to_numpy(), lon_block_deg)

    uniq = np.unique(bids)
    cv_metrics = []
    y_pred_cv = np.zeros_like(y_log)
    for b in uniq:
        val_m = (bids == b)
        tr_m  = ~val_m
        ds_tr = FieldDataset(X[tr_m], y_log[tr_m])
        ds_vl = FieldDataset(X[val_m], y_log[val_m])
        dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=False)
        dl_vl = DataLoader(ds_vl, batch_size=batch_size, shuffle=False, drop_last=False)

        model = MLP(in_dim=X.shape[1], hidden=hidden, depth=depth, act=act)
        model, best_v = train_one(
                            model, dl_tr, dl_vl,
                            epochs=epochs, lr=lr, wd=weight_decay,
                            patience=patience, device=device,
                            mono_weight=mono_weight, latlonh=None,
                            log_every=10,                       
                            tag=f"cv[b={int(b)}]"
                        )
        model.eval()
        with torch.no_grad():
            yhat = model(torch.from_numpy(X[val_m]).float().to(device)).cpu().numpy().reshape(-1)
        y_pred_cv[val_m] = yhat

        rmse_b = rmse_log_metric(y_log[val_m], yhat)
        mape_b = mape(np.exp(y_log[val_m]), np.exp(yhat))
        cv_metrics.append({"block": int(b), "val_rmse_log": rmse_b, "val_mape": mape_b})

    rmse_cv = rmse_log_metric(y_log, y_pred_cv)
    mape_cv = mape(np.exp(y_log), np.exp(y_pred_cv))

    ds_full = FieldDataset(X, y_log)
    dl_full = DataLoader(ds_full, batch_size=batch_size, shuffle=True, drop_last=False)
    dl_eval = DataLoader(ds_full, batch_size=batch_size, shuffle=False, drop_last=False)
    model_f = MLP(in_dim=X.shape[1], hidden=hidden, depth=depth, act=act)
    model_f, _ = train_one(
                        model_f, dl_full, dl_eval,
                        epochs=epochs, lr=lr, wd=weight_decay,
                        patience=patience, device=device,
                        mono_weight=mono_weight, latlonh=None,
                        log_every=1,                   
                        tag="full"
                    )
    model_f.eval()
    with torch.no_grad():
        yhat_full = model_f(torch.from_numpy(X).float().to(device)).cpu().numpy().reshape(-1)
    rmse_full = rmse_log_metric(y_log, yhat_full)
    mape_full = mape(np.exp(y_log), np.exp(yhat_full))

    per_h = []
    for hk in sorted(df["h_km"].unique()):
        m = (np.abs(df["h_km"].to_numpy()-hk) < 1e-6)
        if m.sum() == 0: continue
        per_h.append({
            "h_km": float(hk),
            "rmse_log": rmse_log_metric(y_log[m], y_pred_cv[m]),
            "mape_muSv": mape(np.exp(y_log[m]), np.exp(y_pred_cv[m])),
            "n": int(m.sum())
        })

    out_pred = Path(out_pred_csv)
    out_pred.parent.mkdir(parents=True, exist_ok=True)
    pred_df = df.copy()
    pred_df["y_log_pred_cv"]   = y_pred_cv
    pred_df["dose_pred_cv"]    = np.exp(y_pred_cv)
    pred_df["y_log_pred_full"] = yhat_full
    pred_df["dose_pred_full"]  = np.exp(yhat_full)
    pred_df.to_csv(out_pred, index=False)

    grid_info = None
    if export_grid_npz is not None:
        if grid_heights is None:
            grid_heights = sorted(df["h_km"].unique().tolist())
        lats = np.arange(-90.0, 90.0+1e-6, grid_lat_step)
        lons = np.arange(-180.0, 180.0+1e-6, grid_lon_step)
        LAT, LON, H = np.meshgrid(lats, lons, grid_heights, indexing="ij")
        model_f.eval()
        VOL = np.zeros_like(LAT, dtype=np.float32)
        for k, hk in enumerate(grid_heights):
            tmp = pd.DataFrame({
                "lat_deg": LAT[:,:,k].ravel(),
                "lon_deg": LON[:,:,k].ravel(),
                "h_km":    np.full(LAT[:,:,k].size, hk)
            })
            if "vcr_GV" in df.columns:
                tmp["vcr_GV"] = float(df["vcr_GV"].mean())
            Xg, _ = build_inputs(tmp, NF3InputSpec(
                use_vcr=("vcr_GV" in tmp.columns),
                h_min=h_cfg["h_min"], h_max=h_cfg["h_max"]
            ))
            with torch.no_grad():
                yk = model_f(torch.from_numpy(Xg).float().to(device)).cpu().numpy().reshape(LAT.shape[0], LAT.shape[1])
            VOL[:,:,k] = np.exp(yk).astype(np.float32)
        np.savez_compressed(export_grid_npz, lat=lats, lon=lons, heights=np.array(grid_heights, dtype=np.float32), dose=VOL)
        grid_info = {"npz": export_grid_npz, "lat_step": grid_lat_step, "lon_step": grid_lon_step, "heights": grid_heights}

    if save_ckpt_path is not None:
        Path(save_ckpt_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": model_f.state_dict(),
            "arch": {"in_dim": X.shape[1], "hidden": hidden, "depth": depth, "act": act},
            "optim": {"lr": lr, "weight_decay": weight_decay},
            "h_norm_cfg": h_cfg,
            "spec": {"fourier_freqs": list(NF3InputSpec().fourier_freqs), "use_vcr": use_vcr},
        }, save_ckpt_path)
        print(f"[full] saved checkpoint -> {save_ckpt_path}", flush=True)

    rep = {
        "cv": {
            "blocks": [int(b) for b in uniq.tolist()],
            "lon_block_deg": lon_block_deg,
            "per_block": cv_metrics,
            "rmse_log": rmse_cv,
            "mape_muSv": mape_cv
        },
        "full_fit": {
            "rmse_log_full": rmse_full,
            "mape_muSv_full": mape_full,
            "n_samples": int(df.shape[0])
        },
        "per_height": per_h,
        "h_norm_cfg": h_cfg,
        "grid_export": grid_info
    }
    out_rep = Path(out_report_json)
    out_rep.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rep, open(out_rep, "w"), indent=2)
    return rep, pred_df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out-pred", required=True)
    ap.add_argument("--out-report", required=True)
    ap.add_argument("--lon-block-deg", type=float, default=30.0)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--act", type=str, default="relu")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--patience", type=int, default=500)
    ap.add_argument("--mono-weight", type=float, default=0.0)
    ap.add_argument("--no-vcr", action="store_true")
    ap.add_argument("--export-grid-npz", type=str, default=None)
    ap.add_argument("--grid-lat-step", type=float, default=2.0)
    ap.add_argument("--grid-lon-step", type=float, default=2.0)
    ap.add_argument("--grid-heights", type=str, default=None,
                    help="comma separated heights, e.g. 10,12,14")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--save-ckpt", type=str, default=None)

    args = ap.parse_args()

    heights = None
    if args.grid_heights:
        heights = [float(x) for x in args.grid_heights.split(",")]

    run_nf3_3d(
        features_path=args.features,
        out_pred_csv=args.out_pred,
        out_report_json=args.out_report,
        lon_block_deg=args.lon_block_deg,
        hidden=args.hidden, depth=args.depth, act=args.act,
        lr=args.lr, weight_decay=args.weight_decay,
        epochs=args.epochs, batch_size=args.batch_size, patience=args.patience,
        mono_weight=args.mono_weight,
        use_vcr=(not args.no_vcr),
        export_grid_npz=args.export_grid_npz,
        grid_lat_step=args.grid_lat_step, grid_lon_step=args.grid_lon_step,
        grid_heights=heights,
        device=args.device,
        save_ckpt_path = args.save_ckpt
    )

if __name__ == "__main__":
    main()
