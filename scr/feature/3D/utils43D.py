from __future__ import annotations
import re, math, json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Optional, Dict

import numpy as np
import pandas as pd

def wrap180(lon: float) -> float:
    return ((lon + 180.0) % 360.0) - 180.0

def _deg2rad(x: np.ndarray) -> np.ndarray:
    return np.deg2rad(x.astype(float))

def to_rad(deg):
    return np.deg2rad(deg)

def mape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return float(np.mean(np.abs((y_pred - y_true) / (y_true + eps))) * 100.0)

def _fourier_feats(x: np.ndarray, freqs: Iterable[float], prefix: str) -> pd.DataFrame:
    cols = {}
    for k, f in enumerate(freqs):
        cols[f"{prefix}_sin_{k}"] = np.sin(2.0 * math.pi * f * x)
        cols[f"{prefix}_cos_{k}"] = np.cos(2.0 * math.pi * f * x)
    return pd.DataFrame(cols)

def rmse_log(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred)**2)))

def mape_muSv(y_true_log, y_pred_log):
    y_true = np.exp(y_true_log)
    y_pred = np.exp(y_pred_log)
    return float(np.mean(np.abs((y_pred - y_true) / np.clip(y_true, 1e-9, None))) * 100.0)

def rmse_from_log(y_log_true, y_log_pred):
    y_log_true = np.asarray(y_log_true, float)
    y_log_pred = np.asarray(y_log_pred, float)
    return float(np.sqrt(np.mean((y_log_true - y_log_pred) ** 2)))

def mape_from_log(y_log_true, y_log_pred, eps=1e-8):
    yt = np.exp(np.asarray(y_log_true, float))
    yp = np.exp(np.asarray(y_log_pred, float))
    return float(np.mean(np.abs((yp - yt) / (yt + eps))) * 100.0)


def build_design(df: pd.DataFrame):
    lat = np.deg2rad(df["lat_deg"].to_numpy(float))
    lon = np.deg2rad(df["lon_deg"].to_numpy(float))
    h   = df["h_km"].to_numpy(float)

    s_lat, c_lat = np.sin(lat), np.cos(lat)
    s_lon, c_lon = np.sin(lon), np.cos(lon)

    h1 = h
    h2 = h * h
    h3 = h2 * h

    X = np.stack([s_lat, c_lat, s_lon, c_lon, h1, h2, h3], axis=1)
    y = np.log(np.clip(df["dose_rate_uSvph"].to_numpy(float), 1e-9, None))
    lon_deg = df["lon_deg"].to_numpy(float)
    block = np.floor((lon_deg + 180.0) / 30.0).astype(int) 
    return X, y, block

def build_features(lat_deg, lon_deg, h_km, h_min, h_max):
    lat = np.deg2rad(lat_deg); lon = np.deg2rad(wrap180(lon_deg))
    x = np.stack([
        np.sin(lat), np.cos(lat),
        np.sin(lon), np.cos(lon),
        (h_km - h_min) / max(1e-12, (h_max - h_min))
    ], axis=-1).astype(np.float32)
    return x