"""Unduh titik panas (hotspot) NASA FIRMS untuk bounding box wilayah studi.

Memakai FIRMS Area API (butuh MAP_KEY gratis). API membatasi maksimal 10 hari
per permintaan, jadi rentang panjang dipecah menjadi jendela 10 hari.
Setiap jendela disimpan sebagai CSV di data/raw/firms/ sehingga proses bisa
dilanjutkan (resume) bila terputus.

CATATAN: Ketersediaan arsip (produk *_SP) lewat API bergantung pada akun.
Alternatif: unduh manual arsip dari https://firms.modaps.eosdis.nasa.gov/download/
lalu letakkan file CSV-nya di data/raw/firms/ — fungsi combine_firms() akan
tetap membacanya.
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.utils import ensure_dir, get_logger, resolve_path

log = get_logger("firms")

API_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
# VIIRS memakai huruf tunggal (l/n/h); MODIS memakai label/angka.
_CONF_ORDER = {"low": 0, "l": 0, "nominal": 1, "n": 1, "high": 2, "h": 2}


def _date_windows(start: date, end: date, size: int = 10):
    cur = start
    while cur <= end:
        win_end = min(cur + timedelta(days=size - 1), end)
        yield cur, win_end
        cur = win_end + timedelta(days=1)


def _fetch_window(cfg, key, bbox, win_start, win_end, out_dir) -> str:
    """Unduh satu jendela tanggal. Mengembalikan status untuk logging."""
    f = cfg["firms"]
    fname = out_dir / f"firms_{win_start}_{win_end}.csv"
    if fname.exists():
        return "skip"
    days = (win_end - win_start).days + 1
    url = f"{API_URL}/{key}/{f['product']}/{bbox}/{days}/{win_start}"
    delay = f.get("request_delay_sec", 0.4)
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=180)
            resp.raise_for_status()
            text = resp.text.strip()
            if not text or text.lower().startswith("invalid") or "," not in text.split("\n")[0]:
                return f"invalid: {text[:80]}"
            fname.write_text(text, encoding="utf-8")
            n = max(text.count("\n"), 1) - 1
            return f"ok:{n}"
        except requests.RequestException as e:
            if attempt == 2:
                return f"error: {e}"
            time.sleep(1.0 + attempt)  # backoff sebelum coba lagi
    return "error"


def download_firms(cfg: dict) -> None:
    f = cfg["firms"]
    reg = cfg["region"]
    key = f["map_key"]
    if not key or key.startswith("ISI_"):
        raise SystemExit(
            "MAP_KEY FIRMS belum diisi. Dapatkan gratis di "
            "https://firms.modaps.eosdis.nasa.gov/api/map_key/ lalu isi "
            "firms.map_key di config.yaml (atau config.local.yaml)."
        )

    bbox = f"{reg['lon_min']},{reg['lat_min']},{reg['lon_max']},{reg['lat_max']}"
    start = date.fromisoformat(cfg["time"]["start_date"])
    end = date.fromisoformat(cfg["time"]["end_date"])
    out_dir = ensure_dir(Path(cfg["paths"]["raw"]) / "firms")
    win_days = int(f.get("window_days", 5))   # arsip *_SP maks 5 hari/permintaan
    workers = int(f.get("max_workers", 8))    # unduhan paralel (latensi server tinggi)

    windows = [w for w in _date_windows(start, end, win_days)
               if not (out_dir / f"firms_{w[0]}_{w[1]}.csv").exists()]
    total = len(windows)
    log.info("Mengunduh %d jendela (%d hari/jendela) dengan %d worker paralel...",
             total, win_days, workers)

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_fetch_window, cfg, key, bbox, ws, we, out_dir): (ws, we)
                   for ws, we in windows}
        for fut in as_completed(futures):
            ws, we = futures[fut]
            status = fut.result()
            done += 1
            if status.startswith(("error", "invalid")):
                log.warning("Jendela %s..%s: %s", ws, we, status)
            if done % 25 == 0 or done == total:
                log.info("Progres: %d/%d jendela selesai", done, total)


def _confidence_ok(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """Filter keyakinan. VIIRS: label low/nominal/high. MODIS: angka 0-100."""
    if "confidence" not in df.columns:
        return pd.Series(True, index=df.index)
    conf = df["confidence"]
    f = cfg["firms"]
    # Coba sebagai angka (MODIS)
    num = pd.to_numeric(conf, errors="coerce")
    if num.notna().mean() > 0.5:
        return num >= f["confidence_min_numeric"]
    # Sebagai label (VIIRS)
    min_rank = _CONF_ORDER.get(str(f["confidence_min_label"]).lower(), 0)
    rank = conf.astype(str).str.lower().map(_CONF_ORDER).fillna(0)
    return rank >= min_rank


def combine_firms(cfg: dict) -> pd.DataFrame:
    """Baca semua CSV FIRMS di data/raw/firms/, standarkan & gabungkan.

    Output kolom: latitude, longitude, acq_date (datetime), confidence.
    """
    raw_dir = resolve_path(Path(cfg["paths"]["raw"]) / "firms")
    files = sorted(raw_dir.glob("*.csv")) if raw_dir.exists() else []
    if not files:
        raise FileNotFoundError(
            f"Tidak ada CSV FIRMS di {raw_dir}. Jalankan download_firms dulu, "
            "atau letakkan file arsip manual di folder tsb."
        )
    frames = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
        except Exception as e:  # noqa: BLE001
            log.warning("Lewati %s: %s", fp.name, e)
            continue
        if df.empty:
            continue
        frames.append(df)

    raw = pd.concat(frames, ignore_index=True)
    raw.columns = [c.strip().lower() for c in raw.columns]

    # Nama kolom tanggal berbeda antar produk
    date_col = next((c for c in ("acq_date", "acquisition_date", "date") if c in raw.columns), None)
    if date_col is None:
        raise ValueError(f"Kolom tanggal tidak ditemukan. Kolom tersedia: {list(raw.columns)}")

    raw["acq_date"] = pd.to_datetime(raw[date_col], errors="coerce")
    raw = raw.dropna(subset=["acq_date", "latitude", "longitude"])
    raw = raw[_confidence_ok(raw, cfg)]

    keep = ["latitude", "longitude", "acq_date"]
    if "confidence" in raw.columns:
        keep.append("confidence")
    out = raw[keep].drop_duplicates().reset_index(drop=True)
    log.info("FIRMS gabungan: %d titik panas (setelah filter keyakinan)", len(out))

    out_dir = ensure_dir(cfg["paths"]["interim"])
    out.to_csv(out_dir / "firms_hotspots.csv", index=False)
    return out


def main() -> None:
    cfg = load_config()
    download_firms(cfg)
    combine_firms(cfg)


if __name__ == "__main__":
    main()
