"""Latih & evaluasi model prediksi karhutla per horizon (H+1..H+k).

Model:
  - Regresi Logistik (baseline, dengan penskalaan)
  - Random Forest
  - XGBoost (bila paket xgboost terpasang)

Penanganan imbalance: class_weight / scale_pos_weight, atau SMOTE (opsional).
Pembagian data: berdasarkan tahun (train_years vs test_years) -> split temporal.

Output per horizon:
  - models/model_h{k}.joblib  (model terbaik berdasarkan PR-AUC)
  - reports/metrics_h{k}.json
  - reports/pr_h{k}_{model}.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.evaluate import best_f1_threshold, compute_metrics, plot_pr_curve
from src.features import feature_columns
from src.utils import ensure_dir, get_logger, resolve_path, save_json

log = get_logger("train")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False


def load_dataset(cfg: dict) -> pd.DataFrame:
    proc = resolve_path(cfg["paths"]["processed"])
    pq, csv = proc / "dataset.parquet", proc / "dataset.csv"
    if pq.exists():
        return pd.read_parquet(pq)
    if csv.exists():
        return pd.read_csv(csv, parse_dates=["date"])
    raise FileNotFoundError("dataset belum dibangun. Jalankan src/build_dataset.py dulu.")


def _models(cfg: dict, n_pos: int, n_neg: int) -> dict:
    seed = cfg["project"]["random_seed"]
    cw = "balanced" if cfg["model"]["class_weight"] else None
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    models = {
        "logreg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight=cw, random_state=seed)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=5,
            class_weight=cw, n_jobs=-1, random_state=seed,
        ),
    }
    if HAS_XGB:
        spw = (n_neg / max(n_pos, 1)) if cfg["model"]["class_weight"] else 1.0
        models["xgboost"] = XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
            eval_metric="aucpr", n_jobs=-1, random_state=seed,
        )
    else:
        log.warning("xgboost tidak terpasang; dilewati (pip install xgboost).")
    return models


def _maybe_undersample(X, y, cfg):
    """Undersample kelas negatif pada data latih (opsional)."""
    ratio = cfg["model"].get("negative_undersample_ratio")
    if not ratio:
        return X, y
    rng = np.random.default_rng(cfg["project"]["random_seed"])
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_keep = min(len(neg_idx), int(ratio * len(pos_idx)))
    keep = np.concatenate([pos_idx, rng.choice(neg_idx, n_keep, replace=False)])
    rng.shuffle(keep)
    return X.iloc[keep], y.iloc[keep]


def train_horizon(df: pd.DataFrame, cfg: dict, k: int, feats: list[str]) -> dict:
    import joblib

    target = f"fire_h{k}"
    train = df[df["year"].isin(cfg["time"]["train_years"])]
    test = df[df["year"].isin(cfg["time"]["test_years"])]
    if train.empty or test.empty:
        raise ValueError(f"Split kosong untuk H+{k}. Periksa train_years/test_years vs data.")

    X_tr, y_tr = train[feats].fillna(0), train[target]
    X_te, y_te = test[feats].fillna(0), test[target]
    X_tr, y_tr = _maybe_undersample(X_tr, y_tr, cfg)

    # SMOTE opsional (hanya pada data latih)
    if cfg["model"].get("smote"):
        if HAS_SMOTE and y_tr.sum() > 5:
            X_tr, y_tr = SMOTE(random_state=cfg["project"]["random_seed"]).fit_resample(X_tr, y_tr)
            log.info("H+%d: SMOTE diterapkan -> %d baris latih", k, len(X_tr))
        else:
            log.warning("H+%d: SMOTE diminta tapi tak tersedia / positif terlalu sedikit.", k)

    n_pos, n_neg = int(y_tr.sum()), int((y_tr == 0).sum())
    reports = ensure_dir(cfg["paths"]["reports"])
    results, best = {}, None

    for name, model in _models(cfg, n_pos, n_neg).items():
        model.fit(X_tr, y_tr)
        y_prob = model.predict_proba(X_te)[:, 1]
        thr, _ = best_f1_threshold(y_te, y_prob)
        m_default = compute_metrics(y_te, y_prob, 0.5)
        m_best = compute_metrics(y_te, y_prob, thr)
        results[name] = {"threshold_default": m_default, "threshold_bestF1": m_best}
        plot_pr_curve(y_te, y_prob, f"PR H+{k} - {name}", reports / f"pr_h{k}_{name}.png")
        log.info("H+%d %-14s PR-AUC=%.3f  Recall@0.5=%.3f  Precision@0.5=%.3f",
                 k, name, m_default["pr_auc"] or 0, m_default["recall"], m_default["precision"])

        score = m_default["pr_auc"] or 0
        if best is None or score > best["score"]:
            best = {"name": name, "model": model, "score": score, "threshold": thr}

    # Simpan model terbaik + metadata
    models_dir = ensure_dir(cfg["paths"]["models"])
    joblib.dump(
        {"model": best["model"], "features": feats, "horizon": k,
         "threshold": best["threshold"], "model_name": best["name"],
         "thresholds_risk": cfg["model"]["thresholds"]},
        models_dir / f"model_h{k}.joblib",
    )
    log.info("H+%d: model terbaik = %s (PR-AUC=%.3f) tersimpan.", k, best["name"], best["score"])

    summary = {
        "horizon": k, "best_model": best["name"], "best_pr_auc": best["score"],
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        "features": feats, "results": results,
    }
    save_json(summary, reports / f"metrics_h{k}.json")
    return summary


def main() -> None:
    cfg = load_config()
    df = load_dataset(cfg)
    feats = feature_columns(df)
    log.info("Fitur (%d): %s", len(feats), ", ".join(feats))
    all_summ = {}
    for k in cfg["time"]["horizons"]:
        all_summ[f"h{k}"] = train_horizon(df, cfg, k, feats)
    save_json(all_summ, resolve_path(cfg["paths"]["reports"]) / "metrics_all.json")
    log.info("Selesai. Lihat folder reports/ dan models/.")


if __name__ == "__main__":
    main()
