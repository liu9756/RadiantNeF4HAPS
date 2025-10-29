# nf3_mlp.py
# -*- coding: utf-8 -*-
"""
NF-3: 小型坐标-MLP（经度 Block-CV + 早停 + 可选单调先验）
依赖: numpy, pandas, torch

用法（与 NF-2 类似的纯代码接口）:
    from nf3_mlp import run_nf3_mlp
    pred_df, report, state = run_nf3_mlp(df_or_path, lon_block_deg=30.0)
返回:
    pred_df: 输入 DataFrame 追加 y_log_pred, dose_pred_uSvph, err_*
    report : {best_params, cv_metrics, full_fit_metrics, blocks, ...}
    state  : {"model_state_dict": ..., "config": ...}
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Dict, Any, Optional, Tuple
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

EPS = 1e-12
DEVICE = "cpu"  

# ---------- 数据与特征 ----------
FEAT_COLS = ["x_sin_lat","x_cos_lat","x_sin_lon","x_cos_lon","alt_norm"]

def _ensure_Xy(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    miss = [c for c in FEAT_COLS+["y_log"] if c not in df.columns]
    if miss:
        raise KeyError(f"NF-1 features missing: {miss}")
    X = df[FEAT_COLS].to_numpy(dtype=np.float32)
    y = df["y_log"].to_numpy(dtype=np.float32)
    return X, y

def make_lon_blocks(lon_deg: np.ndarray, width_deg: float = 30.0) -> np.ndarray:
    lon_shift = (lon_deg + 180.0) % 360.0
    bins = np.floor(lon_shift / float(width_deg)).astype(int)
    return bins  

# ---------- 模型 ----------
class MLP(nn.Module):
    def __init__(self, in_dim=5, hidden=64, depth=3, act="relu"):
        super().__init__()
        acts = {"relu": nn.ReLU, "silu": nn.SiLU}
        A = acts.get(act, nn.ReLU)
        layers = []
        d = in_dim
        for _ in range(max(0, depth-1)):
            layers += [nn.Linear(d, hidden), A(inplace=True)]
            d = hidden
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x).squeeze(-1)  

# ---------- 单调先验（可选） ----------
def monotone_lat_penalty(y_pred: torch.Tensor, lat_deg: np.ndarray, weight: float = 0.0) -> torch.Tensor:
    if weight <= 0 or len(y_pred) < 2:
        return y_pred.new_tensor(0.0)
    lat = np.abs(lat_deg.astype(np.float32))
    order = np.argsort(lat)
    yp = y_pred[order]
    diff = yp[:-1] - yp[1:]
    return weight * torch.clamp(diff, min=0).mean()

# ---------- 训练与评估 ----------
@dataclass
class TrainCfg:
    hidden: int = 64
    depth: int = 3
    act: str = "relu"
    lr: float = 1e-3
    weight_decay: float = 1e-4     
    epochs: int = 5000
    batch_size: int = 256
    patience: int = 300
    mono_weight: float = 0.0       

def _rmse(a, b): return float(np.sqrt(np.mean((a-b)**2)))
def _mape_from_log(y_true_log, y_pred_log):
    t = np.expm1(y_true_log); p = np.expm1(y_pred_log)
    return float(np.mean(np.abs(p - t) / np.maximum(t, EPS)) * 100.0)

def _train_one(X_tr, y_tr, X_va, y_va, lat_tr, cfg: TrainCfg) -> Tuple[np.ndarray, Dict[str, float], MLP]:
    torch.manual_seed(42); np.random.seed(42)
    model = MLP(in_dim=X_tr.shape[1], hidden=cfg.hidden, depth=cfg.depth, act=cfg.act).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_mse = nn.MSELoss()
    Xtr = torch.from_numpy(X_tr).to(DEVICE)
    ytr = torch.from_numpy(y_tr).to(DEVICE)
    Xva = torch.from_numpy(X_va).to(DEVICE)
    yva = torch.from_numpy(y_va).to(DEVICE)
    best = {"val": 1e9, "ep": -1}
    best_state = None
    patience = cfg.patience
    n_noimp = 0

    N = Xtr.shape[0]
    idx = np.arange(N)

    for ep in range(1, cfg.epochs+1):
        model.train()
        np.random.shuffle(idx)
        for i0 in range(0, N, cfg.batch_size):
            sel = idx[i0:i0+cfg.batch_size]
            xb = Xtr[sel]; yb = ytr[sel]
            opt.zero_grad()
            yhat = model(xb)
            loss = loss_mse(yhat, yb)
            if cfg.mono_weight > 0:
                mono = monotone_lat_penalty(yhat, lat_tr[sel], weight=cfg.mono_weight)
                loss = loss + mono
            loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            yv = model(Xva)
            val = loss_mse(yv, yva).item()
        if val + 1e-6 < best["val"]:
            best.update({"val": val, "ep": ep})
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            n_noimp = 0
        else:
            n_noimp += 1
            if n_noimp >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    with torch.no_grad():
        yv = model(Xva).cpu().numpy()
    rep = {"val_rmse_log": _rmse(y_va, yv), "val_mape": _mape_from_log(y_va, yv), "best_epoch": best["ep"]}
    return yv, rep, model

def _cv_blocks(df: pd.DataFrame, cfg: TrainCfg, lon_block_deg: float) -> Dict[str, Any]:
    X, y = _ensure_Xy(df)
    blocks = make_lon_blocks(df["lon_deg"].to_numpy(float), width_deg=lon_block_deg)
    block_ids = np.unique(blocks)
    lat = df["lat_deg"].to_numpy(float)

    y_pred_cv = np.zeros_like(y)
    fold_reports = []
    for b in block_ids:
        m_val = (blocks == b)
        m_tr = ~m_val
        _, rep, mdl = _train_one(X[m_tr], y[m_tr], X[m_val], y[m_val], lat[m_tr], cfg)

        mdl.eval()
        with torch.no_grad():
            y_pred_cv[m_val] = mdl(torch.from_numpy(X[m_val])).cpu().numpy()
        rep["block"] = int(b)
        fold_reports.append(rep)
    return {
        "blocks": block_ids.tolist(),
        "rmse_log": _rmse(y, y_pred_cv),
        "mape_muSv": _mape_from_log(y, y_pred_cv),
        "per_block": fold_reports
    }

def _grid(cfgs: Iterable[TrainCfg]) -> Iterable[TrainCfg]:
    for c in cfgs: yield c

def cv_search(df: pd.DataFrame,
              lon_block_deg: float = 30.0,
              search: Optional[Iterable[TrainCfg]] = None) -> Dict[str, Any]:
    if search is None:

        search = [
            TrainCfg(hidden=64, depth=3, lr=1e-3, weight_decay=1e-4, mono_weight=0.0),
            TrainCfg(hidden=64, depth=3, lr=1e-3, weight_decay=1e-4, mono_weight=0.05),
            TrainCfg(hidden=64, depth=3, lr=5e-4, weight_decay=5e-5, mono_weight=0.0),
        ]
    results = []
    for cfg in _grid(search):
        cv = _cv_blocks(df, cfg, lon_block_deg=lon_block_deg)
        results.append({"cfg": cfg.__dict__, "cv": cv})

    results = sorted(results, key=lambda r: r["cv"]["rmse_log"])
    best = results[0]
    return {"results": results, "best": best}

def run_nf3_mlp(features: pd.DataFrame | str,
                lon_block_deg: float = 30.0,
                search: Optional[Iterable[TrainCfg]] = None,
                save_pred_path: Optional[str] = None,
                save_report_path: Optional[str] = None,
                save_ckpt_path: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """主入口：CV 选参 → 全数据训练 → 预测与报告"""
    df = pd.read_csv(features) if isinstance(features, str) else features.copy()
    X, y = _ensure_Xy(df)
    lat = df["lat_deg"].to_numpy(float)

    sel = cv_search(df, lon_block_deg=lon_block_deg, search=search)
    best_cfg = TrainCfg(**sel["best"]["cfg"])

    N = len(df)
    idx = np.arange(N); np.random.seed(0); np.random.shuffle(idx)
    n_va = max(1, int(0.1*N))
    va = idx[:n_va]; tr = idx[n_va:]
    _, rep_hold, mdl = _train_one(X[tr], y[tr], X[va], y[va], lat[tr], best_cfg)

    mdl.eval()
    with torch.no_grad():
        y_pred = mdl(torch.from_numpy(X)).cpu().numpy()
    dose_pred = np.expm1(y_pred)
    out = df.copy()
    out["y_log_pred"] = y_pred
    out["dose_pred_uSvph"] = dose_pred
    out["err_log"] = y_pred - y
    out["err_uSvph"] = dose_pred - np.expm1(y)

    report = {
        "best_params": best_cfg.__dict__,
        "cv": sel["best"]["cv"],             
        "holdout": {
            "rmse_log": rep_hold["val_rmse_log"],
            "mape_muSv": rep_hold["val_mape"]
        },
        "full_fit": {
            "rmse_log": _rmse(y, y_pred),
            "mape_muSv": _mape_from_log(y, y_pred),
            "n_samples": int(N)
        }
    }

    if save_pred_path:
        out.to_csv(save_pred_path, index=False)
    if save_report_path:
        with open(save_report_path, "w") as f:
            json.dump(report, f, indent=2)
    state = {"model_state_dict": {k: v.cpu().numpy().tolist() for k,v in mdl.state_dict().items()},
             "config": best_cfg.__dict__}
    if save_ckpt_path:
        torch.save(mdl.state_dict(), save_ckpt_path)
    return out, report, state
