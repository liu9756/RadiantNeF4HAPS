# nf2_3d_baseline.py
import json, math, argparse
from pathlib import Path
import numpy as np, pandas as pd
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold

from utils43D import rmse_log, mape_muSv, build_design


@dataclass
class NF2Result:
    model: object
    preds: pd.DataFrame
    report: dict

def run_nf2_3d(features_path: str,
               alphas=(1e-6,1e-4,1e-3,1e-2,1e-1,1.0,10.0),
               lon_block_deg: float = 30.0) -> NF2Result:
    df = (pd.read_parquet(features_path) if features_path.endswith(".parquet")
          else pd.read_csv(features_path))
    if "particle" in df.columns:
        df = df[df["particle"]=="TOTAL"].copy()
    if "unit_text" in df.columns:
        df = df[df["unit_text"].str.contains("microSv/hr", case=False, na=False)].copy()

    X, y, block = build_design(df)

    gkf = GroupKFold(n_splits=int(round(360.0 / lon_block_deg)))
    per_alpha = []
    best = None
    best_model = None

    for alpha in alphas:
        fold_metrics = []
        for tr, va in gkf.split(X, y, groups=block):
            pipe = make_pipeline(StandardScaler(with_mean=True, with_std=True),
                                 Ridge(alpha=alpha, fit_intercept=True, random_state=42))
            pipe.fit(X[tr], y[tr])
            yv = pipe.predict(X[va])
            fold_metrics.append((rmse_log(y[va], yv), mape_muSv(y[va], yv)))
        rmse = float(np.mean([m[0] for m in fold_metrics]))
        mape = float(np.mean([m[1] for m in fold_metrics]))
        per_alpha.append({"alpha": alpha, "rmse_log": rmse, "mape_muSv": mape})
        if best is None or rmse < best["rmse_log"]:
            best = {"alpha": alpha, "rmse_log": rmse, "mape_muSv": mape}


    pipe = make_pipeline(StandardScaler(), Ridge(alpha=best["alpha"], fit_intercept=True, random_state=42))
    pipe.fit(X, y)
    yhat = pipe.predict(X)


    df_out = df[["lat_deg","lon_deg","h_km","dose_rate_uSvph"]].copy()
    df_out["y_true_log"] = y
    df_out["y_pred_log"] = yhat
    df_out["y_pred_uSvph"] = np.exp(yhat)
    df_out["resid"] = df_out["y_pred_uSvph"] - df_out["dose_rate_uSvph"]

    per_h = []
    for h_val, sub in df_out.groupby("h_km"):
        r = {
            "h_km": float(h_val),
            "rmse_log": rmse_log(sub["y_true_log"].to_numpy(), sub["y_pred_log"].to_numpy()),
            "mape_muSv": mape_muSv(sub["y_true_log"].to_numpy(), sub["y_pred_log"].to_numpy()),
            "n": int(len(sub))
        }
        per_h.append(r)

    phys_points = []
    rng = np.random.default_rng(0)
    keys = df_out.groupby(["lat_deg","lon_deg"]).size().index.tolist()
    picks = keys if len(keys) <= 100 else [keys[i] for i in rng.choice(len(keys), size=100, replace=False)]
    from scipy.stats import spearmanr
    ok, tot = 0, 0
    for lat, lon in picks:
        sub = df_out[(df_out.lat_deg==lat) & (df_out.lon_deg==lon)].sort_values("h_km")
        if len(sub) >= 4:
            rho, _ = spearmanr(sub["h_km"].to_numpy(), sub["y_pred_uSvph"].to_numpy())
            tot += 1
            ok  += int(rho > 0.0)
    monotone_pass_ratio = float(ok / max(tot,1))

    report = {
        "cv": {
            "blocks": list(range(int(round(360.0 / lon_block_deg)))),
            "lon_block_deg": lon_block_deg,
            "per_alpha": per_alpha,
            "best": best
        },
        "full_fit": {
            "alpha": best["alpha"],
            "rmse_log_full": rmse_log(y, yhat),
            "mape_muSv_full": mape_muSv(y, yhat),
            "n_samples": int(len(df))
        },
        "per_height": per_h,
        "phys_check": {"monotone_pass_ratio": monotone_pass_ratio, "n_points_checked": tot}
    }
    return NF2Result(model=pipe, preds=df_out, report=report)

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out_pred", required=True)
    ap.add_argument("--out_report", required=True)
    args = ap.parse_args()

    res = run_nf2_3d(args.features)
    Path(args.out_pred).parent.mkdir(parents=True, exist_ok=True)
    res.preds.to_csv(args.out_pred, index=False)
    with open(args.out_report, "w") as f:
        json.dump(res.report, f, indent=2)
    print("[NF2-3D] done.",
          "full RMSE_log=%.4f MAPE=%.2f%%" % (res.report["full_fit"]["rmse_log_full"],
                                              res.report["full_fit"]["mape_muSv_full"]))

if __name__ == "__main__":
    main()
