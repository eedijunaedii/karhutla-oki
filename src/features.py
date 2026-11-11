"""Rekayasa fitur (feature engineering) berbasis cuaca per sel grid.

Fitur yang dibuat per (cell_id, date):
  - hari tanpa hujan beruntun (dry_days)
  - hujan kumulatif 7/14/30 hari
  - lag harian suhu/kelembapan/hujan (t-1, t-2, t-3)
  - rata-rata bergerak suhu & kelembapan
Target:
  - fire_today (label di hari t)
  - fire_h1..fire_hk (kebakaran di t+k) untuk setiap horizon
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Nama kolom cuaca standar (hasil rename dari parameter POWER)
WEATHER_COLS = {
    "T2M_MAX": "t2m_max",
    "T2M": "t2m",
    "RH2M": "rh2m",
    "PRECTOTCORR": "prec",
    "WS2M": "ws2m",
}


def _dry_days(is_rain: pd.Series) -> pd.Series:
    """Jumlah hari beruntun tanpa hujan sampai hari t (termasuk t)."""
    # reset penghitung setiap kali hujan
    no_rain = (is_rain == 0).astype(int)  # is_rain berisi 0/1, jadi jangan pakai ~ (bitwise)
    grp = is_rain.cumsum()  # grup naik setiap hari hujan
    return no_rain.groupby(grp).cumsum()


def add_features(panel: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Tambahkan fitur lagged & target ke panel (cell_id, date, cuaca, fire_today).

    panel harus sudah terurut dan lengkap per cell_id x date.
    """
    feat = cfg["features"]
    horizons = cfg["time"]["horizons"]
    df = panel.sort_values(["cell_id", "date"]).copy()
    g = df.groupby("cell_id", group_keys=False)

    # Hari hujan & hari tanpa hujan beruntun
    df["is_rain"] = (df["prec"] >= feat["rain_threshold_mm"]).astype(int)
    df["dry_days"] = g["is_rain"].apply(_dry_days)

    # Hujan kumulatif (rolling sum, termasuk hari t)
    for w in feat["rain_cum_windows"]:
        df[f"rain_cum_{w}"] = (
            g["prec"].apply(lambda s, w=w: s.rolling(w, min_periods=1).sum())
        )

    # Lag harian
    lag_src = ["t2m_max", "t2m", "rh2m", "prec", "ws2m"]
    for lag in feat["lag_days"]:
        for c in lag_src:
            if c in df.columns:
                df[f"{c}_lag{lag}"] = g[c].shift(lag)

    # Rata-rata bergerak suhu & kelembapan
    for w in feat["roll_windows"]:
        for c in ("t2m_max", "rh2m"):
            if c in df.columns:
                df[f"{c}_roll{w}"] = (
                    g[c].apply(lambda s, w=w: s.rolling(w, min_periods=1).mean())
                )

    # Target horizon: kebakaran di t+k pada sel yang sama
    for k in horizons:
        df[f"fire_h{k}"] = g["fire_today"].shift(-k)

    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Kolom yang dipakai sebagai prediktor (buang id, tanggal, target, bantu)."""
    exclude = {"cell_id", "col", "row", "date", "fire_today", "is_rain",
               "lon_center", "lat_center", "lon_min", "lon_max", "lat_min", "lat_max"}
    exclude |= {c for c in df.columns if c.startswith("fire_h")}
    cols = [c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    return cols
