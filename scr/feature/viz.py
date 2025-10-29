from nf4_viz import plot_heatmap, export_contours_csv, export_grid_csv
import pandas as pd
path = "/users/PAS2177/liu9756/RadiantNeF4HAPS/data/processed/nf3/pred_mlp.csv"  
pred_df = pd.read_csv(path)
plot_heatmap(pred_df,
             levels=[0.5, 1.0, 2.0],
             save_path="/users/PAS2177/liu9756/RadiantNeF4HAPS/data/processed/nf4/map_10km.png")

export_contours_csv(pred_df,
                    levels=[0.5, 1.0, 2.0],
                    out_csv="/users/PAS2177/liu9756/RadiantNeF4HAPS/data/processed/nf4/contours_10km.csv")

export_grid_csv(pred_df,
                out_csv="/users/PAS2177/liu9756/RadiantNeF4HAPS/data/processed/nf4/grid_10km.csv")
