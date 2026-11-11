"""Orkestrasi pipeline end-to-end.

Contoh:
  # Pakai data sintetis (tanpa internet/API) untuk uji coba cepat:
  python src/run_pipeline.py --sample

  # Pakai data asli (harus sudah mengunduh FIRMS & POWER lebih dulu):
  python src/download_firms.py        # butuh MAP_KEY
  python src/download_power.py
  python src/run_pipeline.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.build_dataset import build, save
from src.config import load_config
from src.download_firms import combine_firms
from src.download_power import combine_power
from src.train import load_dataset, train_horizon
from src.features import feature_columns
from src.utils import get_logger, resolve_path, save_json

log = get_logger("pipeline")


def main() -> None:
    ap = argparse.ArgumentParser(description="Pipeline karhutla EWS (OKI)")
    ap.add_argument("--sample", action="store_true",
                    help="buat data sintetis dulu (tanpa internet)")
    ap.add_argument("--skip-train", action="store_true", help="hanya bangun dataset")
    args = ap.parse_args()

    cfg = load_config()

    if args.sample:
        from scripts.make_sample_data import make
        log.info("== 0/4 Membuat data sintetis ==")
        make(cfg)

    log.info("== 1/4 Menggabungkan data mentah ==")
    combine_firms(cfg)
    combine_power(cfg)

    log.info("== 2/4 Membangun dataset + fitur ==")
    panel = build(cfg)
    save(panel, cfg)

    if args.skip_train:
        return

    log.info("== 3/4 Melatih model per horizon ==")
    df = load_dataset(cfg)
    feats = feature_columns(df)
    summ = {}
    for k in cfg["time"]["horizons"]:
        summ[f"h{k}"] = train_horizon(df, cfg, k, feats)
    save_json(summ, resolve_path(cfg["paths"]["reports"]) / "metrics_all.json")

    log.info("== 4/4 Selesai. Cek folder models/ & reports/. ==")
    log.info("Jalankan dashboard: streamlit run app/dashboard.py")


if __name__ == "__main__":
    main()
