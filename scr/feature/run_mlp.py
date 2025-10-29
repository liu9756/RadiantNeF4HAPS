from nf3_mlp import run_nf3_mlp
from nf1_features import build_nf1_features
df = build_nf1_features("/users/PAS2177/liu9756/RadiantNeF4HAPS/data/processed/qc_vA_single/ans_concat_filtered.csv")
print(df.head())
pred3, rep3, state3 = run_nf3_mlp(df, lon_block_deg=30.0,
                                  save_pred_path="/users/PAS2177/liu9756/RadiantNeF4HAPS/data/processed/nf3/pred_mlp.csv",
                                  save_report_path="/users/PAS2177/liu9756/RadiantNeF4HAPS/data/processed/nf3/report_mlp.json",
                                  save_ckpt_path="/users/PAS2177/liu9756/RadiantNeF4HAPS/data/processed/nf3/mlp.ckpt")
print(rep3["cv"]["rmse_log"], rep3["full_fit"]["rmse_log"])
