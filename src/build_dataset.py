"""Bangun dataset latih dari grid + FIRMS + POWER.

Langkah:
  1. Muat grid, hotspot FIRMS gabungan, cuaca POWER gabungan.
  2. Buat panel lengkap: seluruh sel grid x seluruh tanggal.
  3. Petakan setiap sel ke titik POWER terdekat (snap) -> gabung cuaca harian.
  4. Beri label fire_today = 1 bila ada hotspot di sel pada hari itu.
  5. Tambahkan fitur lagged & target horizon (lihat features.py).

Output: data/processed/dataset.parquet (atau .csv bila pyarrow tidak ada).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.features import WEATHER_COLS, add_features
from src.grid import assign_cell, build_grid, grid_spec
from src.peat import load_peat_fraction
from src.spatial_features import add_spatial_features
from src.utils import ensure_dir, get_logger, resolve_path

log = get_logger("build")


def _load_inputs(cfg: dict):
    interim = resolve_path(cfg["paths"]["interim"])
    firms = pd.read_csv(interim / "firms_hotspots.csv", parse_dates=["acq_date"])
    power = pd.read_csv(interim / "power_daily.csv", parse_dates=["date"])
    grid = build_grid(cfg)
    return grid, firms, power


def _snap(values: np.ndarray, snap: float) -> np.ndarray:
    return np.round(np.round(values / snap) * snap, 4)


def build(cfg: dict) -> pd.DataFrame:
    grid, firms, power = _load_inputs(cfg)
    spec = grid_spec(cfg)
    snap = cfg["power"]["snap_degree"]

    # --- Rentang tanggal penuh ---
    dates = pd.date_range(cfg["time"]["start_date"], cfg["time"]["end_date"], freq="D")
    log.info("Panel: %d sel x %d hari = %d baris", len(grid), len(dates), len(grid) * len(dates))

    # --- Panel lengkap (cell x date) ---
    panel = (
        grid[["cell_id", "lon_center", "lat_center"]]
        .assign(key=1)
        .merge(pd.DataFrame({"date": dates, "key": 1}), on="key")
        .drop(columns="key")
    )

    # --- Gabung cuaca POWER (via titik snap terdekat) ---
    power = power.rename(columns={k: v for k, v in WEATHER_COLS.items() if k in power.columns})
    power["lon_s"] = _snap(power["lon"].to_numpy(), snap)
    power["lat_s"] = _snap(power["lat"].to_numpy(), snap)
    wcols = [v for v in WEATHER_COLS.values() if v in power.columns]
    power_g = power.groupby(["lon_s", "lat_s", "date"], as_index=False)[wcols].mean()

    panel["lon_s"] = _snap(panel["lon_center"].to_numpy(), snap)
    panel["lat_s"] = _snap(panel["lat_center"].to_numpy(), snap)
    panel = panel.merge(power_g, on=["lon_s", "lat_s", "date"], how="left")
    panel = panel.drop(columns=["lon_s", "lat_s", "lon_center", "lat_center"])

    missing = panel[wcols].isna().any(axis=1).mean() if wcols else 1.0
    if missing > 0.2:
        log.warning("%.0f%% baris panel tanpa data cuaca lengkap - periksa cakupan titik POWER.",
                    100 * missing)

    # --- Label hotspot ---
    firms = firms.copy()
    firms["cell_id"] = assign_cell(firms["longitude"], firms["latitude"], spec)
    firms = firms[firms["cell_id"] >= 0]
    firms["date"] = firms["acq_date"].dt.normalize()
    fire = (
        firms.groupby(["cell_id", "date"]).size().rename("n_hotspot").reset_index()
    )
    panel = panel.merge(fire, on=["cell_id", "date"], how="left")
    panel["n_hotspot"] = panel["n_hotspot"].fillna(0)
    panel["fire_today"] = (panel["n_hotspot"] > 0).astype(int)
    panel = panel.drop(columns="n_hotspot")

    log.info("Positif (hari-sel terbakar): %d (%.3f%%)",
             int(panel["fire_today"].sum()), 100 * panel["fire_today"].mean())

    # --- Fitur spasial-temporal (riwayat api, aman dari kebocoran) ---
    panel = add_spatial_features(panel, cfg)

    # --- Fitur gambut (opsional; aktif bila peta tersedia) ---
    peat = load_peat_fraction(cfg, grid)
    if peat is not None:
        panel["peat_frac"] = panel["cell_id"].map(peat).fillna(0.0).astype("float32")

    # --- Fitur cuaca (lagged) & target ---
    panel = add_features(panel, cfg)

    # Buang baris tanpa target/fitur lengkap
    target_cols = [f"fire_h{k}" for k in cfg["time"]["horizons"]]
    panel = panel.dropna(subset=target_cols)
    for c in target_cols:
        panel[c] = panel[c].astype(int)
    panel["year"] = panel["date"].dt.year

    return panel.reset_index(drop=True)


def save(panel: pd.DataFrame, cfg: dict) -> Path:
    out_dir = ensure_dir(cfg["paths"]["processed"])
    try:
        out = out_dir / "dataset.parquet"
        panel.to_parquet(out, index=False)
    except Exception:  # noqa: BLE001 (pyarrow/fastparquet tak ada)
        out = out_dir / "dataset.csv"
        panel.to_csv(out, index=False)
        log.warning("pyarrow tidak tersedia; disimpan sebagai CSV.")
    log.info("Dataset tersimpan: %s (%d baris, %d kolom)", out, len(panel), panel.shape[1])
    return out


def main() -> None:
    cfg = load_config()
    panel = build(cfg)
    save(panel, cfg)


if __name__ == "__main__":
    main()
