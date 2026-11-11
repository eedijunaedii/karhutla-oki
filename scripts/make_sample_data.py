"""Generator DATA SINTETIS untuk menguji pipeline tanpa API/kredensial.

Membuat file mentah yang kompatibel dengan combine_firms() & combine_power():
  - data/raw/power/power_<lon>_<lat>.csv   (cuaca harian per titik)
  - data/raw/firms/sample_firms.csv        (titik panas)

Data dibuat dengan pola musiman khas Sumatra Selatan: musim kemarau (Jun-Okt)
lebih kering & lebih banyak titik panas, sehingga model menghasilkan sinyal
yang tidak trivial. INI BUKAN DATA ASLI — hanya untuk demo/uji coba.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.download_power import power_points
from src.utils import ensure_dir, get_logger

log = get_logger("sample")


def _seasonal(day_of_year: np.ndarray):
    """Faktor kemarau 0..1 (puncak ~hari ke-240, akhir Agustus/September).

    Musim kemarau Sumatra Selatan kira-kira Jun-Okt; faktor mendekati 1 di
    sekitar hari ke-240 dan mendekati 0 di musim hujan (Des-Mar).
    """
    return np.clip(np.cos((day_of_year - 240) / 365 * 2 * np.pi) * 0.5 + 0.5, 0, 1) ** 1.5


def make(cfg: dict, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(cfg["time"]["start_date"], cfg["time"]["end_date"], freq="D")
    doy = dates.dayofyear.to_numpy()
    dry = _seasonal(doy)  # 0 (basah) .. 1 (kering)

    # --- Cuaca POWER per titik ---
    pdir = ensure_dir(Path(cfg["paths"]["raw"]) / "power")
    for lon, lat in power_points(cfg):
        local = dry * (1 + 0.1 * rng.standard_normal(len(dates)))
        local = np.clip(local, 0, 1.3)
        prec = rng.gamma(shape=np.clip(1.5 * (1 - local) + 0.05, 0.05, None),
                         scale=6.0)  # mm/hari; kecil saat kering
        prec = np.where(rng.random(len(dates)) < (0.15 + 0.5 * local), prec, 0.0)
        df = pd.DataFrame({
            "lon": lon, "lat": lat, "date": dates,
            "T2M_MAX": 31 + 3 * local + rng.normal(0, 0.6, len(dates)),
            "T2M": 27 + 2 * local + rng.normal(0, 0.5, len(dates)),
            "RH2M": np.clip(92 - 22 * local + rng.normal(0, 3, len(dates)), 40, 100),
            "PRECTOTCORR": np.round(prec, 2),
            "WS2M": np.clip(1.5 + 0.8 * local + rng.normal(0, 0.3, len(dates)), 0, None),
        })
        df.to_csv(pdir / f"power_{lon}_{lat}.csv", index=False)
    log.info("Data cuaca sintetis dibuat untuk %d titik.", len(power_points(cfg)))

    # --- Titik panas FIRMS ---
    reg = cfg["region"]
    rows = []
    for d, dr in zip(dates, dry):
        # jumlah titik panas naik tajam saat kemarau
        lam = 25 * (dr ** 3)
        n = rng.poisson(lam)
        if n == 0:
            continue
        lons = rng.uniform(reg["lon_min"], reg["lon_max"], n)
        lats = rng.uniform(reg["lat_min"], reg["lat_max"], n)
        for lo, la in zip(lons, lats):
            rows.append({
                "latitude": round(float(la), 4), "longitude": round(float(lo), 4),
                "acq_date": d.date().isoformat(),
                "confidence": rng.choice(["nominal", "high"], p=[0.6, 0.4]),
                "bright_ti4": round(float(rng.uniform(300, 360)), 1),
                "frp": round(float(rng.uniform(1, 50)), 1),
            })
    fdir = ensure_dir(Path(cfg["paths"]["raw"]) / "firms")
    pd.DataFrame(rows).to_csv(fdir / "sample_firms.csv", index=False)
    log.info("Titik panas sintetis: %d baris.", len(rows))


def main() -> None:
    make(load_config())


if __name__ == "__main__":
    main()
