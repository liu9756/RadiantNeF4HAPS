from nf1_features import build_nf1_features
df = build_nf1_features("/users/PAS2177/liu9756/RadiantNeF4HAPS/data/processed/qc_vA_single/ans_concat_filtered.csv")
print(df.head())

from nf2_baseline import run_nf2_baseline

pred_path = "/users/PAS2177/liu9756/RadiantNeF4HAPS/data/processed/nf2/pred_baseline_ridge.csv"
rep_path  = "/users/PAS2177/liu9756/RadiantNeF4HAPS/data/processed/nf2/report_ridge.json"

pred_df, model, report = run_nf2_baseline(
    features=df,
    lon_block_deg=30.0,                            
    alphas=(1e-6,1e-4,1e-3,1e-2,1e-1,1.0,10.0),
    save_pred_path=pred_path,
    save_report_path=rep_path
)

print("best alpha:", report["best_alpha"])
print("CV (rmse_log, mape%):", report["best"]["rmse_log"], report["best"]["mape_muSv"])
print("FULL (rmse_log, mape%):", report["final_fit"]["rmse_log_full"], report["final_fit"]["mape_muSv_full"])
