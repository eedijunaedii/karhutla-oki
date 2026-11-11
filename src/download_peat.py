"""Unduh layer sebaran gambut Sumatera Selatan (mencakup OKI) via WFS.

Sumber: portal data gambut Sumsel (BRGM/GeoNode)
  https://datagambut.sumselprov.go.id/
Layer relevan:
  - geonode:lahan_gambut    -> sebaran/lahan gambut  (dipakai untuk peat_frac)
  - geonode:ketebalan_gambut-> ketebalan gambut
  - geonode:batas_khg       -> batas Kesatuan Hidrologis Gambut

Hasil disimpan ke data/external/peat/<layer>.geojson, lalu build_dataset.py
otomatis menghitung kolom peat_frac.

CATATAN: server portal ini kadang menolak GetFeature bila basis datanya sedang
bermasalah (ServiceException). Bila itu terjadi, coba lagi nanti, atau unduh
manual lewat antarmuka web portal / portal resmi KLHK-BRGM (lihat src/peat.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.utils import ensure_dir, get_logger

log = get_logger("peat-dl")

WFS = "https://datagambut.sumselprov.go.id/geoserver/ows"


def download_layer(cfg: dict, typename: str = "geonode:lahan_gambut",
                   clip_to_region: bool = True) -> Path | None:
    reg = cfg["region"]
    params = {
        "service": "WFS", "version": "1.0.0", "request": "GetFeature",
        "typeName": typename, "outputFormat": "application/json", "srsName": "EPSG:4326",
    }
    if clip_to_region:
        # WFS 1.0.0 EPSG:4326 -> urutan minx,miny,maxx,maxy (lon,lat)
        params["bbox"] = f"{reg['lon_min']},{reg['lat_min']},{reg['lon_max']},{reg['lat_max']},EPSG:4326"

    log.info("Mengunduh %s (clip ke OKI=%s)...", typename, clip_to_region)
    try:
        r = requests.get(WFS, params=params, timeout=180)
    except requests.RequestException as e:
        log.error("Gagal menghubungi server: %s", e)
        return None

    ct = r.headers.get("content-type", "")
    if "json" not in ct:
        snippet = r.text[:300].replace("\n", " ")
        log.error("Server tidak mengembalikan GeoJSON (content-type=%s). Cuplikan: %s", ct, snippet)
        log.error("Kemungkinan server/basis-data portal sedang bermasalah. Coba lagi nanti "
                  "atau unduh manual (lihat src/peat.py).")
        return None

    out_dir = ensure_dir("data/external/peat")
    safe = typename.replace(":", "_")
    out = out_dir / f"{safe}.geojson"
    out.write_bytes(r.content)
    try:
        n = len(r.json().get("features", []))
    except Exception:  # noqa: BLE001
        n = "?"
    log.info("Tersimpan: %s (%s fitur, %.1f KB)", out, n, len(r.content) / 1024)
    log.info("Jalankan ulang: python src/build_dataset.py  -> kolom peat_frac akan ikut.")
    return out


def main() -> None:
    cfg = load_config()
    typename = sys.argv[1] if len(sys.argv) > 1 else "geonode:lahan_gambut"
    download_layer(cfg, typename)


if __name__ == "__main__":
    main()
