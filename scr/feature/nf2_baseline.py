# nf2_baseline.py
# -*- coding: utf-8 -*-
"""
NF-2: 可解释的基线回归（Ridge）+ 经度分块交叉验证
- 输入：NF-1 产物 DataFrame 或 CSV 路径（需含列：x_sin_lat, x_cos_lat, x_sin_lon, x_cos_lon, alt_norm, y_log,
        以及辅助列 lat_deg, lon_deg, h_km, dose_rate_uSvph）
- 输出：预测 DataFrame（追加 y_log_pred, dose_pred_uSvph），以及报告 dict（最佳 alpha、CV 指标等）
- 仅依赖 numpy/pandas；Ridge 采用闭式解，且**不对截距项**做惩罚。

设计依据：
- Ridge 回归与其正则化性质：Hastie, Tibshirani, Friedman《Elements of Statistical Learning》，
  线性回归与 Ridge 章节（第二版，Chapter 3, 18）。"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd

EPS = 1e-12

# -----------------------------
# 1) 设计矩阵 & 目标
# -----------------------------
def _ensure_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    needed = ["x_sin_lat","x_cos_lat","x_sin_lon","x_cos_lon","alt_norm","y_log"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"NF-1 features missing: {missing}")
    X = df[["x_sin_lat","x_cos_lat","x_sin_lon","x_cos_lon","alt_norm"]].to_numpy(dtype=float)
    y = df["y_log"].to_numpy(dtype=float)
    return X, y

# -----------------------------
# 2) 经度分块（Block-CV）
# -----------------------------
def make_lon_blocks(lon_deg: np.ndarray, width_deg: float = 30.0) -> np.ndarray:
    w = float(width_deg)
    lon_shift = (lon_deg + 180.0) % 360.0
    bins = np.floor(lon_shift / w).astype(int)
    return bins  

# -----------------------------
# 3) Ridge 闭式解（不惩罚截距）
# -----------------------------
@dataclass
class RidgeModel:
    w: np.ndarray  
    b: float       
    alpha: float

def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float) -> RidgeModel:
    X = np.asarray(X, float)
    y = np.asarray(y, float).reshape(-1)
    N, p = X.shape
    X1 = np.concatenate([np.ones((N,1)), X], axis=1)  
    I = np.eye(1+p, dtype=float)
    I[0,0] = 0.0
    A = X1.T @ X1 + alpha * I
    bvec = X1.T @ y
    theta = np.linalg.solve(A, bvec)  
    b0, w = theta[0], theta[1:]
    return RidgeModel(w=w, b=b0, alpha=alpha)

def ridge_predict(X: np.ndarray, model: RidgeModel) -> np.ndarray:
    return X @ model.w + model.b

# -----------------------------
# 4) 指标
# -----------------------------
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred)**2)))

def mape_muSv(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    t = np.expm1(y_true_log)
    p = np.expm1(y_pred_log)
    return float(np.mean(np.abs(p - t) / np.maximum(t, EPS)) * 100.0)

# -----------------------------
# 5) 经度 Block-CV + alpha 搜索
# -----------------------------
def cv_search_alpha(df: pd.DataFrame,
                    alphas: Iterable[float] = (1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0),
                    lon_block_deg: float = 30.0) -> Dict[str, Any]:
    X, y = _ensure_features(df)
    blocks = make_lon_blocks(df["lon_deg"].to_numpy(dtype=float), width_deg=lon_block_deg)
    block_ids = np.unique(blocks)
    results = []
    for alpha in alphas:
        y_pred_cv = np.zeros_like(y)
        for b in block_ids:
            m_val = (blocks == b)
            m_tr  = ~m_val
            mdl = ridge_fit(X[m_tr], y[m_tr], alpha=alpha)
            y_pred_cv[m_val] = ridge_predict(X[m_val], mdl)
        res = {
            "alpha": alpha,
            "rmse_log": rmse(y, y_pred_cv),
            "mape_muSv": mape_muSv(y, y_pred_cv)
        }
        results.append(res)
    results = sorted(results, key=lambda d: d["rmse_log"])
    best = results[0]
    return {"per_alpha": results, "best_alpha": best["alpha"], "best": best,
            "blocks": block_ids.tolist(), "lon_block_deg": lon_block_deg}

# -----------------------------
# 6) 主流程：训练+预测+报告
# -----------------------------
def run_nf2_baseline(features: pd.DataFrame | str,
                     lon_block_deg: float = 30.0,
                     alphas: Iterable[float] = (1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0),
                     save_pred_path: Optional[str] = None,
                     save_report_path: Optional[str] = None) -> Tuple[pd.DataFrame, RidgeModel, Dict[str, Any]]:
    df = pd.read_csv(features) if isinstance(features, str) else features.copy()
    rep = cv_search_alpha(df, alphas=alphas, lon_block_deg=lon_block_deg)
    alpha_star = rep["best_alpha"]
    X, y = _ensure_features(df)
    model = ridge_fit(X, y, alpha=alpha_star)
    y_pred_log = ridge_predict(X, model)
    dose_pred  = np.expm1(y_pred_log)
    out = df.copy()
    out["y_log_pred"] = y_pred_log
    out["dose_pred_uSvph"] = dose_pred
    out["err_log"] = y_pred_log - df["y_log"].to_numpy()
    out["err_uSvph"] = dose_pred - df["dose_rate_uSvph"].to_numpy(dtype=float)

    rep_final = rep.copy()
    rep_final["final_fit"] = {
        "alpha": model.alpha,
        "rmse_log_full": rmse(y, y_pred_log),
        "mape_muSv_full": mape_muSv(y, y_pred_log),
        "n_samples": len(df)
    }

    if save_pred_path:
        pd.DataFrame(out).to_csv(save_pred_path, index=False)
    if save_report_path:
        import json, os
        os.makedirs(str(__import__("pathlib").Path(save_report_path).parent), exist_ok=True)
        with open(save_report_path, "w") as f:
            import json; json.dump(rep_final, f, indent=2)
    return out, model, rep_final
