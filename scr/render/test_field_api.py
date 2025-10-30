from field_api import RadiationField2D

field = RadiationField2D.from_grid("/users/PAS2177/liu9756/RadiantNeF4HAPS/data/processed/nf4/grid_10km.csv")
print(field.predict(35.2, 139.7))  
lats = [30, 30, 30, 30]; lons = [-20, 0, 20, 40]
dose = field.sample_path(lats, lons)
