from __future__ import annotations
from dataclasses import dataclass
import numpy as np, pandas as pd 

def wrap180(lon: float) -> float:
    return ((lon + 180.0) % 360.0) - 180.0

def lonlat_to_xyz(lat_deg: float, lon_deg: float, h_km: float = 10.0, R_E_km: float = 6371.0):
    lat = np.deg2rad(lat_deg); lon = np.deg2rad(wrap180(lon_deg))
    r = R_E_km + h_km
    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)
    return float(x), float(y), float(z)

@dataclass
class RadiationField2D:
    lats: np.ndarray     # shape [L]
    lons: np.ndarray     # shape [M]
    Z:    np.ndarray     # shape [L, M], μSv/h

    @classmethod
    def from_grid(cls, grid_csv: str):
        df = pd.read_csv(grid_csv)
        for c in ("lat_deg","lon_deg","dose_pred_uSvph"):
            assert c in df.columns, f"Missing column: {c}"
        lats = np.sort(df["lat_deg"].astype(float).unique())
        lons = np.sort(df["lon_deg"].astype(float).unique())
        L, M = len(lats), len(lons)
        assert L >= 2 and M >= 2, "Grid must have at least 2x2 points"
        Z = np.full((L, M), np.nan, float)

        lat_i = {v:i for i,v in enumerate(lats)}
        lon_j = {v:j for j,v in enumerate(lons)}  

        for r in df.itertuples(index=False):
            i, j = lat_i[float(getattr(r, "lat_deg"))], lon_j[float(getattr(r, "lon_deg"))]
            Z[i, j] = float(getattr(r, "dose_pred_uSvph"))
        return cls(lats=lats, lons=lons, Z=Z)

    def predict(self, lat_deg: float, lon_deg: float) -> float:
        lon_deg = wrap180(lon_deg)
        i = np.searchsorted(self.lats, lat_deg) - 1
        j = np.searchsorted(self.lons, lon_deg) - 1
        i = int(np.clip(i, 0, len(self.lats)-2))
        j = int(np.clip(j, 0, len(self.lons)-2))
        lat0, lat1 = self.lats[i], self.lats[i+1]
        lon0, lon1 = self.lons[j], self.lons[j+1]
        Q11, Q21 = self.Z[i, j], self.Z[i, j+1]
        Q12, Q22 = self.Z[i+1, j], self.Z[i+1, j+1]
        tx = (lon_deg - lon0) / (lon1 - lon0 + 1e-12)
        ty = (lat_deg - lat0) / (lat1 - lat0 + 1e-12)
        return float((1-tx)*(1-ty)*Q11 + tx*(1-ty)*Q21 + (1-tx)*ty*Q12 + tx*ty*Q22)

    def sample_path(self, lat_list, lon_list):
        lat_arr = np.asarray(lat_list, float)
        lon_arr = np.asarray(lon_list, float)
        assert lat_arr.shape == lon_arr.shape, "lat_list/lon_list must have same shape"
        vals = [self.predict(float(la), float(lo)) for la,lo in zip(lat_arr, lon_arr)]
        return np.asarray(vals, float)


