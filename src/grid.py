"""Pembuatan grid spasial ~5 km untuk wilayah studi (Kab. OKI).

Grid dibangun dalam koordinat lon/lat. Ukuran sel (km) dikonversi ke derajat
memakai 1 derajat ~ 111 km. Setiap sel punya cell_id = col * n_rows + row.
"""
from __future__ import annotations

import sys
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.utils import ensure_dir, get_logger, resolve_path

log = get_logger("grid")


def grid_spec(cfg: dict) -> dict:
    """Hitung parameter grid (langkah derajat, jumlah baris/kolom)."""
    reg, g = cfg["region"], cfg["grid"]
    step = g["cell_size_km"] / g["km_per_degree"]  # langkah dalam derajat
    n_cols = int(ceil((reg["lon_max"] - reg["lon_min"]) / step))
    n_rows = int(ceil((reg["lat_max"] - reg["lat_min"]) / step))
    return {
        "step": step,
        "lon_min": reg["lon_min"],
        "lat_min": reg["lat_min"],
        "n_cols": n_cols,
        "n_rows": n_rows,
    }


def assign_cell(lon, lat, spec: dict):
    """Petakan koordinat (lon, lat) -> cell_id. Mendukung array numpy/pandas.

    Mengembalikan cell_id (int) dan -1 bila di luar grid.
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    col = np.floor((lon - spec["lon_min"]) / spec["step"]).astype(int)
    row = np.floor((lat - spec["lat_min"]) / spec["step"]).astype(int)
    valid = (col >= 0) & (col < spec["n_cols"]) & (row >= 0) & (row < spec["n_rows"])
    cell = col * spec["n_rows"] + row
    cell = np.where(valid, cell, -1)
    return cell


def build_grid(cfg: dict) -> pd.DataFrame:
    """Bangun DataFrame seluruh sel grid dengan batas dan titik tengahnya."""
    spec = grid_spec(cfg)
    step = spec["step"]
    rows = []
    for col in range(spec["n_cols"]):
        for row in range(spec["n_rows"]):
            lon0 = spec["lon_min"] + col * step
            lat0 = spec["lat_min"] + row * step
            rows.append({
                "cell_id": col * spec["n_rows"] + row,
                "col": col,
                "row": row,
                "lon_min": lon0,
                "lon_max": lon0 + step,
                "lat_min": lat0,
                "lat_max": lat0 + step,
                "lon_center": lon0 + step / 2,
                "lat_center": lat0 + step / 2,
            })
    df = pd.DataFrame(rows)
    log.info("Grid dibuat: %d sel (%d kolom x %d baris), langkah %.4f derajat (~%.1f km)",
             len(df), spec["n_cols"], spec["n_rows"], step, cfg["grid"]["cell_size_km"])
    return df


def main() -> None:
    cfg = load_config()
    df = build_grid(cfg)
    out = ensure_dir(cfg["paths"]["processed"]) / "grid.csv"
    df.to_csv(out, index=False)
    log.info("Tersimpan: %s", resolve_path(out))


if __name__ == "__main__":
    main()
