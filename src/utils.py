"""Utilitas umum: path proyek, logging, I/O JSON."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


def project_root() -> Path:
    """Folder root proyek (satu tingkat di atas folder src/)."""
    return Path(__file__).resolve().parents[1]


def resolve_path(rel: str | Path) -> Path:
    """Ubah path relatif terhadap root proyek menjadi path absolut."""
    p = Path(rel)
    return p if p.is_absolute() else (project_root() / p)


def ensure_dir(path: str | Path) -> Path:
    """Pastikan folder ada; kembalikan Path-nya."""
    p = resolve_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_logger(name: str = "karhutla") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def save_json(obj: Any, path: str | Path) -> None:
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def load_json(path: str | Path) -> Any:
    with open(resolve_path(path), "r", encoding="utf-8") as f:
        return json.load(f)
