"""Fitur spasial-temporal dari riwayat titik panas (tanpa data eksternal).

Semua fitur dihitung HANYA dari masa lalu (shift >= 1 hari) sehingga tidak
terjadi kebocoran (leakage) ke target masa depan. Kebakaran sangat
terautokorelasi secara ruang & waktu, jadi fitur ini biasanya paling menaikkan
PR-AUC.

Fitur yang ditambahkan per (cell_id, date):
  - self_fire_7d, self_fire_30d : jumlah hari-api di sel ini pada 7/30 hari lalu
  - days_since_last_fire        : hari sejak api terakhir di sel ini (dibatasi)
  - neigh_fire_1d               : jumlah sel tetangga (8-arah) yang terbakar kemarin
  - neigh_fire_7d               : jumlah hari-api sel tetangga pada 7 hari lalu
  - cell_fire_rate              : laju kebakaran historis sel (climatology, expanding)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.grid import grid_spec
from src.utils import get_logger

log = get_logger("spatial")


def _past_window_sum(arr: np.ndarray, w: int) -> np.ndarray:
    """Jumlah pada [t-w, t-1] sepanjang sumbu hari (axis 0). Tidak termasuk t."""
    cs = np.cumsum(arr, axis=0)
    prev = np.zeros_like(cs)
    prev[1:] = cs[:-1]                     # cs[t-1]
    sub = np.zeros_like(cs)
    if w + 1 <= arr.shape[0]:
        sub[w + 1:] = cs[:-(w + 1)]        # cs[t-w-1]
    return prev - sub


def _neighbor_sum(A: np.ndarray) -> np.ndarray:
    """Jumlah 8-tetangga yang terbakar pada hari yang sama. A: (D, C, R)."""
    P = np.pad(A, ((0, 0), (1, 1), (1, 1)))
    return (
        P[:, :-2, :-2] + P[:, :-2, 1:-1] + P[:, :-2, 2:] +
        P[:, 1:-1, :-2] +                   P[:, 1:-1, 2:] +
        P[:, 2:, :-2] + P[:, 2:, 1:-1] + P[:, 2:, 2:]
    )


def add_spatial_features(panel: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Tambahkan fitur spasial-temporal ke panel LENGKAP (sel x tanggal).

    panel harus memuat cell_id, date, fire_today, dan belum dibuang barisnya.
    """
    spec = grid_spec(cfg)
    n_cols, n_rows = spec["n_cols"], spec["n_rows"]

    df = panel
    dates = np.sort(df["date"].unique())
    d_index = pd.Series(np.arange(len(dates)), index=pd.DatetimeIndex(dates))
    n_days = len(dates)

    cell = df["cell_id"].to_numpy()
    col = cell // n_rows
    row = cell % n_rows
    d = d_index.reindex(df["date"].values).to_numpy()
    fire = df["fire_today"].to_numpy().astype(np.int32)

    # Tensor api: A[hari, kolom, baris]
    A = np.zeros((n_days, n_cols, n_rows), dtype=np.int32)
    A[d, col, row] = fire

    # Riwayat sel sendiri
    SF7 = _past_window_sum(A, 7)
    SF30 = _past_window_sum(A, 30)

    # Tetangga
    NS = _neighbor_sum(A)                 # 8-tetangga terbakar per hari
    NF1 = _past_window_sum(NS, 1)         # tetangga terbakar kemarin
    NF7 = _past_window_sum(NS, 7)         # hari-api tetangga pada 7 hari lalu

    # Climatology: laju kebakaran historis sel (expanding, sampai kemarin)
    cs = np.cumsum(A, axis=0)
    cnt_prev = np.zeros_like(cs)
    cnt_prev[1:] = cs[:-1]
    day_arr = np.arange(n_days)[:, None, None]
    rate = cnt_prev / np.clip(day_arr, 1, None)

    # Hari sejak api terakhir (per kemarin, dibatasi 365)
    lf = np.maximum.accumulate(np.where(A > 0, day_arr, -1), axis=0)
    lf_prev = np.full_like(lf, -1)
    lf_prev[1:] = lf[:-1]
    dsl = np.where(lf_prev >= 0, day_arr - lf_prev, 365)
    dsl = np.clip(dsl, 0, 365).astype(np.int32)

    # Petakan kembali ke baris panel
    idx = (d, col, row)
    out = df.copy()
    out["self_fire_7d"] = SF7[idx]
    out["self_fire_30d"] = SF30[idx]
    out["neigh_fire_1d"] = NF1[idx]
    out["neigh_fire_7d"] = NF7[idx]
    out["cell_fire_rate"] = rate[idx].astype(np.float32)
    out["days_since_last_fire"] = dsl[idx]

    log.info("Fitur spasial ditambahkan: self_fire_7d/30d, neigh_fire_1d/7d, "
             "cell_fire_rate, days_since_last_fire")
    return out
