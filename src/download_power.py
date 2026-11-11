"""Unduh data cuaca harian NASA POWER untuk titik-titik grid wilayah studi.

NASA POWER TIDAK butuh API key. Resolusi POWER ~0.5 derajat, jadi kita hanya
perlu query pada sejumlah kecil titik (dibulatkan ke grid snap_degree) yang
menutupi bounding box. Satu request bisa mencakup seluruh rentang tahun.

Setiap titik disimpan sebagai CSV di data/raw/power/power_<lon>_<lat>.csv
dengan kolom: lon, lat, date, <parameter POWER...>.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.utils import ensure_dir, get_logger, resolve_path

log = get_logger("power")

API_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
FILL = -999.0


def power_points(cfg: dict) -> list[tuple[float, float]]:
    """Daftar titik (lon, lat) terbulatkan ke snap_degree yang menutupi bbox."""
    reg = cfg["region"]
    snap = cfg["power"]["snap_degree"]
    lons = np.arange(np.floor(reg["lon_min"] / snap) * snap,
                     np.ceil(reg["lon_max"] / snap) * snap + snap, snap)
    lats = np.arange(np.floor(reg["lat_min"] / snap) * snap,
                     np.ceil(reg["lat_max"] / snap) * snap + snap, snap)
    pts = [(round(float(lo), 4), round(float(la), 4)) for lo in lons for la in lats]
    return pts


def _fetch_point(cfg: dict, lon: float, lat: float) -> pd.DataFrame:
    p = cfg["power"]
    params = {
        "parameters": ",".join(p["parameters"]),
        "community": p["community"],
        "longitude": lon,
        "latitude": lat,
        "start": cfg["time"]["start_date"].replace("-", ""),
        "end": cfg["time"]["end_date"].replace("-", ""),
        "format": "JSON",
    }
    resp = requests.get(API_URL, params=params, timeout=180)
    resp.raise_for_status()
    data = resp.json()["properties"]["parameter"]
    # data: {PARAM: {YYYYMMDD: value}}
    df = pd.DataFrame(data)
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df = df.replace(FILL, np.nan)
    df.insert(0, "date", df.index)
    df.insert(0, "lat", lat)
    df.insert(0, "lon", lon)
    return df.reset_index(drop=True)


def download_power(cfg: dict) -> None:
    pts = power_points(cfg)
    out_dir = ensure_dir(Path(cfg["paths"]["raw"]) / "power")
    log.info("Akan mengunduh %d titik POWER (resolusi %.2f derajat)",
             len(pts), cfg["power"]["snap_degree"])
    for lon, lat in pts:
        fname = out_dir / f"power_{lon}_{lat}.csv"
        if fname.exists():
            continue
        try:
            df = _fetch_point(cfg, lon, lat)
            df.to_csv(fname, index=False)
            log.info("Titik (%.2f, %.2f): %d hari -> %s", lon, lat, len(df), fname.name)
        except (requests.RequestException, KeyError, ValueError) as e:
            log.error("Gagal unduh titik (%.2f, %.2f): %s", lon, lat, e)
        time.sleep(cfg["power"].get("request_delay_sec", 1.0))


def combine_power(cfg: dict) -> pd.DataFrame:
    """Gabungkan semua CSV POWER menjadi satu tabel panjang (long)."""
    raw_dir = resolve_path(Path(cfg["paths"]["raw"]) / "power")
    files = sorted(raw_dir.glob("power_*.csv")) if raw_dir.exists() else []
    if not files:
        raise FileNotFoundError(
            f"Tidak ada CSV POWER di {raw_dir}. Jalankan download_power dulu."
        )
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    out_dir = ensure_dir(cfg["paths"]["interim"])
    df.to_csv(out_dir / "power_daily.csv", index=False)
    log.info("POWER gabungan: %d baris dari %d titik", len(df), len(files))
    return df


def main() -> None:
    cfg = load_config()
    download_power(cfg)
    combine_power(cfg)


if __name__ == "__main__":
    main()
