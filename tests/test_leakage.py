"""Uji kebocoran (leakage) fitur.

Prinsipnya: fitur pada tanggal t hanya boleh memakai data sampai t, sedangkan
label yang diprediksi adalah api pada t+1..t+3. Tiga uji di bawah memakai data
nyata dari data/processed/dataset.parquet (semua sel, Juni-Oktober 2019):

1. Masa depan diacak  -> fitur sampai tanggal T tidak boleh berubah.
2. Label tergeser     -> fire_h{k} di tanggal t harus sama dengan fire_today di t+k,
                         jadi tanggal label selalu > tanggal fitur.
3. Fitur riwayat api  -> hanya memakai data sampai kemarin (t-1).

Jalankan:  python tests/test_leakage.py   (atau pytest bila terpasang)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import load_config  # noqa: E402
from src.features import add_features, feature_columns  # noqa: E402
from src.spatial_features import add_spatial_features  # noqa: E402

START, END, CUTOFF = "2019-06-01", "2019-10-31", pd.Timestamp("2019-09-01")
RAW = ["cell_id", "date", "t2m_max", "t2m", "rh2m", "prec", "ws2m", "fire_today"]
SPATIAL = ["self_fire_7d", "self_fire_30d", "neigh_fire_1d", "neigh_fire_7d",
           "cell_fire_rate", "days_since_last_fire"]

_cache: dict = {}


def _panel() -> pd.DataFrame:
    if "panel" not in _cache:
        df = pd.read_parquet(ROOT / "data/processed/dataset.parquet", columns=RAW,
                             filters=[("date", ">=", pd.Timestamp(START)),
                                      ("date", "<=", pd.Timestamp(END))])
        _cache["panel"] = df.sort_values(["cell_id", "date"]).reset_index(drop=True)
    return _cache["panel"].copy()


def _build(panel: pd.DataFrame) -> pd.DataFrame:
    cfg = load_config()
    out = add_features(add_spatial_features(panel, cfg), cfg)
    return out.sort_values(["cell_id", "date"]).reset_index(drop=True)


def _scramble(panel: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """Ganti semua cuaca & label api pada baris `mask` dengan nilai acak."""
    rng = np.random.default_rng(0)
    p = panel.copy()
    n = int(mask.sum())
    for c in ["t2m_max", "t2m", "rh2m", "prec", "ws2m"]:
        p.loc[mask, c] = rng.uniform(0, 100, n)
    p.loc[mask, "fire_today"] = rng.integers(0, 2, n)
    return p


def test_fitur_tidak_berubah_saat_masa_depan_diacak():
    base = _panel()
    a = _build(base)
    b = _build(_scramble(base, base["date"] > CUTOFF))
    cols = feature_columns(a)
    past = a["date"] <= CUTOFF
    for c in cols:
        assert np.allclose(a.loc[past, c], b.loc[past, c], equal_nan=True), \
            f"{c} berubah padahal hanya data sesudah {CUTOFF.date()} yang diacak"
    return f"{len(cols)} fitur identik untuk {int(past.sum()):,} baris sampai {CUTOFF.date()}"


def test_tanggal_label_selalu_sesudah_tanggal_fitur():
    a = _build(_panel())
    cols = feature_columns(a)
    assert not any(c.startswith("fire_") for c in cols), "label ikut menjadi fitur"
    fire = a.set_index(["cell_id", "date"])["fire_today"]
    for k in load_config()["time"]["horizons"]:
        lab = a.dropna(subset=[f"fire_h{k}"])
        future = fire.reindex(pd.MultiIndex.from_arrays(
            [lab["cell_id"], lab["date"] + pd.Timedelta(days=k)])).to_numpy()
        assert np.array_equal(lab[f"fire_h{k}"].to_numpy(), future), \
            f"fire_h{k} bukan api pada t+{k}"
    return "fire_h1..h3 = api pada t+1..t+3; max(tanggal fitur) = t < tanggal label"


def test_fitur_riwayat_api_hanya_sampai_kemarin():
    base = _panel()
    a = _build(base)
    b = _build(_scramble(base, base["date"] >= CUTOFF))  # hari T ikut diacak
    day = a["date"] == CUTOFF
    for c in SPATIAL:
        assert np.allclose(a.loc[day, c], b.loc[day, c]), f"{c} memakai data hari ini"
    return f"{len(SPATIAL)} fitur riwayat api tidak terpengaruh data hari t"


if __name__ == "__main__":
    import time
    t0 = time.time()
    tests = [test_fitur_tidak_berubah_saat_masa_depan_diacak,
             test_tanggal_label_selalu_sesudah_tanggal_fitur,
             test_fitur_riwayat_api_hanya_sampai_kemarin]
    print(f"Data uji: {len(_panel()):,} baris (semua sel, {START} s.d. {END})")
    failed = 0
    for t in tests:
        try:
            print(f"PASSED  {t.__name__}: {t()}")
        except AssertionError as e:
            failed += 1
            print(f"FAILED  {t.__name__}: {e}")
    print(f"{len(tests) - failed} passed, {failed} failed in {time.time() - t0:.1f}s")
    sys.exit(1 if failed else 0)
