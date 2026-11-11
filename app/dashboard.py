"""Dashboard peta interaktif zona risiko karhutla (Streamlit + Folium).

Jalankan:  streamlit run app/dashboard.py

Menampilkan peta grid Kab. OKI dengan warna risiko harian:
  MERAH  (prob >= thresholds.red)    = risiko tinggi
  KUNING (prob >= thresholds.yellow) = waspada
  HIJAU  (lainnya)                   = aman
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.grid import build_grid
from src.utils import resolve_path

import streamlit as st

st.set_page_config(page_title="Peringatan Dini Karhutla - OKI", layout="wide")


@st.cache_data
def _load():
    cfg = load_config()
    proc = resolve_path(cfg["paths"]["processed"])
    pq, csv = proc / "dataset.parquet", proc / "dataset.csv"
    if pq.exists():
        df = pd.read_parquet(pq)
    elif csv.exists():
        df = pd.read_csv(csv, parse_dates=["date"])
    else:
        return cfg, None, None
    df["date"] = pd.to_datetime(df["date"])
    grid = build_grid(cfg).set_index("cell_id")
    return cfg, df, grid


@st.cache_resource
def _load_model(horizon: int):
    import joblib
    cfg = load_config()
    path = resolve_path(cfg["paths"]["models"]) / f"model_h{horizon}.joblib"
    return joblib.load(path) if path.exists() else None


def _color(prob: float, thr: dict) -> str:
    if prob >= thr["red"]:
        return "#e03131"
    if prob >= thr["yellow"]:
        return "#f59f00"
    return "#2f9e44"


def main() -> None:
    cfg, df, grid = _load()
    st.title("🔥 Sistem Peringatan Dini Karhutla — Kab. Ogan Komering Ilir")

    if df is None:
        st.warning("Dataset belum tersedia. Jalankan pipeline dulu "
                   "(lihat README: build_dataset.py & train.py).")
        return

    horizons = cfg["time"]["horizons"]
    thr = cfg["model"]["thresholds"]

    with st.sidebar:
        st.header("Parameter")
        k = st.selectbox("Horizon prediksi", horizons,
                         format_func=lambda x: f"H+{x} (prediksi {x} hari ke depan)")
        dates = sorted(df["date"].dt.date.unique())
        # Tanggal default bisa lewat URL, mis. ?date=2019-09-20 (tautan bisa dibagikan)
        default_date = dates[-1]
        qp = st.query_params.get("date")
        if qp:
            try:
                d = pd.to_datetime(qp).date()
                if d in set(dates):
                    default_date = d
            except (ValueError, TypeError):
                pass
        sel_date = st.select_slider("Tanggal acuan (hari-0)", options=dates, value=default_date)
        st.caption(f"Ambang risiko — Merah ≥ {thr['red']}, Kuning ≥ {thr['yellow']}")

    bundle = _load_model(k)
    if bundle is None:
        st.error(f"Model H+{k} belum dilatih. Jalankan src/train.py.")
        return

    model, feats = bundle["model"], bundle["features"]
    day = df[df["date"].dt.date == sel_date].copy()
    if day.empty:
        st.info("Tidak ada data pada tanggal ini.")
        return

    day["prob"] = model.predict_proba(day[feats].fillna(0))[:, 1]
    day["zona"] = day["prob"].apply(
        lambda p: "Merah" if p >= thr["red"] else ("Kuning" if p >= thr["yellow"] else "Hijau")
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total sel", len(day))
    c2.metric("🔴 Merah", int((day["zona"] == "Merah").sum()))
    c3.metric("🟡 Kuning", int((day["zona"] == "Kuning").sum()))
    c4.metric("🟢 Hijau", int((day["zona"] == "Hijau").sum()))
    st.caption(f"Prediksi kebakaran H+{k} dihitung dari kondisi pada {sel_date}.")

    # --- Peta Folium ---
    try:
        import folium
        from streamlit_folium import st_folium
    except ImportError:
        st.warning("folium / streamlit-folium belum terpasang. "
                   "pip install folium streamlit-folium. Menampilkan tabel sebagai gantinya.")
        st.dataframe(day[["cell_id", "prob", "zona"]].sort_values("prob", ascending=False))
        return

    reg = cfg["region"]
    fmap = folium.Map(
        location=[(reg["lat_min"] + reg["lat_max"]) / 2,
                  (reg["lon_min"] + reg["lon_max"]) / 2],
        zoom_start=9, tiles="OpenStreetMap",
    )
    for _, r in day.iterrows():
        g = grid.loc[r["cell_id"]]
        folium.Rectangle(
            bounds=[[g["lat_min"], g["lon_min"]], [g["lat_max"], g["lon_max"]]],
            color=None, fill=True, fill_color=_color(r["prob"], thr), fill_opacity=0.55,
            popup=f"Sel {int(r['cell_id'])} — {r['zona']} (p={r['prob']:.2f})",
        ).add_to(fmap)
    st_folium(fmap, width=None, height=560)

    with st.expander("Tabel sel berisiko tertinggi"):
        st.dataframe(day[["cell_id", "prob", "zona"]].sort_values("prob", ascending=False).head(50))


if __name__ == "__main__":
    main()
