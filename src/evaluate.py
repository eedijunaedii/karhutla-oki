"""Metrik & plot evaluasi untuk model klasifikasi tak seimbang (karhutla)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.utils import get_logger

log = get_logger("eval")


def best_f1_threshold(y_true, y_prob) -> tuple[float, float]:
    """Cari ambang yang memaksimalkan F1 pada kurva precision-recall."""
    from sklearn.metrics import precision_recall_curve

    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    # thr berukuran len(prec)-1
    idx = int(np.nanargmax(f1[:-1])) if len(thr) else 0
    return (float(thr[idx]) if len(thr) else 0.5), float(f1[idx])


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    from sklearn.metrics import (average_precision_score, confusion_matrix,
                                 precision_score, recall_score, roc_auc_score)

    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    pos = int(np.sum(y_true))
    out = {
        "n": int(len(y_true)),
        "n_positif": pos,
        "prevalensi": float(np.mean(y_true)),
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_prob)) if pos else None,
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if 0 < pos < len(y_true) else None,
    }
    p, r = out["precision"], out["recall"]
    out["f1"] = float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out["confusion_matrix"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    return out


def plot_pr_curve(y_true, y_prob, title: str, path: str | Path) -> None:
    """Simpan kurva precision-recall; lewati diam-diam bila matplotlib tak ada."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import PrecisionRecallDisplay
    except ImportError:
        log.warning("matplotlib tidak tersedia; lewati plot PR.")
        return

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    PrecisionRecallDisplay.from_predictions(y_true, y_prob, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
