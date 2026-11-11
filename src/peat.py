"""Ingest peta gambut (KLHK/BRGM) -> fitur fraksi gambut per sel grid.

Fitur `peat_frac` = proporsi luas sel yang berupa lahan gambut (0..1). Gambut
adalah faktor dominan karhutla Sumatra, jadi fitur ini bisa sangat menaikkan
PR-AUC bila peta tersedia.

CARA PAKAI:
  1. Unduh peta gambut (pilih salah satu sumber gratis):
       - KLHK "Peta Lahan Gambut" / BRGM "Kesatuan Hidrologis Gambut (KHG)":
         https://sipalaga.brgm.go.id/  atau  https://tanahair.indonesia.go.id/
       - Alternatif global: Global Peatland Map / PEATMAP.
  2. Letakkan file-nya di  data/external/peat/
       - Shapefile poligon gambut (*.shp + file pendampingnya), ATAU
       - GeoTIFF raster gambut (*.tif) bernilai 1=gambut / 0=bukan.
  3. (Sekali) pasang dependensi GIS:
       pip install geopandas shapely rasterio
  4. Jalankan ulang build_dataset.py -> kolom peat_frac otomatis ikut.

Bila file atau dependensi tidak ada, fungsi mengembalikan None dan pipeline
tetap jalan tanpa fitur gambut (dengan peringatan).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import get_logger, resolve_path

log = get_logger("peat")


def _peat_dir() -> Path:
    return resolve_path("data/external/peat")


def load_peat_fraction(cfg: dict, grid: pd.DataFrame) -> pd.Series | None:
    """Kembalikan Series peat_frac (indeks cell_id, 0..1) atau None bila tak ada."""
    pdir = _peat_dir()
    if not pdir.exists():
        log.info("Folder %s tidak ada; fitur gambut dilewati. Lihat src/peat.py.", pdir)
        return None

    vectors = (sorted(pdir.glob("*.shp")) + sorted(pdir.glob("*.geojson"))
               + sorted(pdir.glob("*.json")) + sorted(pdir.glob("*.gpkg")))
    tifs = sorted(pdir.glob("*.tif")) + sorted(pdir.glob("*.tiff"))
    if vectors:
        return _from_vector(vectors[0], grid)
    if tifs:
        return _from_raster(tifs[0], grid)
    log.info("Tidak ada file vektor (*.shp/.geojson/.gpkg) atau raster (*.tif) di %s; "
             "fitur gambut dilewati.", pdir)
    return None


def _from_vector(shp: Path, grid: pd.DataFrame) -> pd.Series | None:
    try:
        import geopandas as gpd
        from shapely.geometry import box
    except ImportError:
        log.warning("geopandas/shapely belum terpasang (pip install geopandas shapely); "
                    "fitur gambut dilewati.")
        return None

    log.info("Membaca peta gambut (poligon): %s", shp.name)
    peat = gpd.read_file(shp)
    if peat.crs is None:
        peat = peat.set_crs(4326)
    peat = peat.to_crs(4326)

    cells = gpd.GeoDataFrame(
        grid[["cell_id"]].copy(),
        geometry=[box(r.lon_min, r.lat_min, r.lon_max, r.lat_max)
                  for r in grid.itertuples()],
        crs=4326,
    )
    # Proyeksi setara-luas (meter) untuk menghitung fraksi luas
    cells_m = cells.to_crs(3857)
    peat_m = peat.to_crs(3857)
    peat_union = peat_m.union_all() if hasattr(peat_m, "union_all") else peat_m.unary_union

    cells_m["cell_area"] = cells_m.area
    cells_m["peat_area"] = cells_m.geometry.intersection(peat_union).area
    frac = (cells_m["peat_area"] / cells_m["cell_area"]).clip(0, 1)
    out = pd.Series(frac.to_numpy(), index=cells_m["cell_id"].to_numpy(), name="peat_frac")
    log.info("peat_frac dihitung untuk %d sel (rata-rata %.3f)", len(out), float(out.mean()))
    return out


def _from_raster(tif: Path, grid: pd.DataFrame) -> pd.Series | None:
    try:
        import rasterio
        from rasterio.windows import from_bounds
    except ImportError:
        log.warning("rasterio belum terpasang (pip install rasterio); fitur gambut dilewati.")
        return None

    log.info("Membaca peta gambut (raster): %s", tif.name)
    fracs = {}
    with rasterio.open(tif) as src:
        band = 1
        for r in grid.itertuples():
            try:
                win = from_bounds(r.lon_min, r.lat_min, r.lon_max, r.lat_max, src.transform)
                data = src.read(band, window=win, boundless=True, fill_value=0).astype(float)
                fracs[r.cell_id] = float(np.clip(np.nanmean(data > 0), 0, 1)) if data.size else 0.0
            except Exception:  # noqa: BLE001
                fracs[r.cell_id] = 0.0
    out = pd.Series(fracs, name="peat_frac")
    log.info("peat_frac dihitung untuk %d sel (rata-rata %.3f)", len(out), float(out.mean()))
    return out
