"""Pemuat konfigurasi. Membaca config.yaml (dan config.local.yaml bila ada)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.utils import project_root


def _deep_update(base: dict, override: dict) -> dict:
    """Gabungkan dict secara rekursif (override menimpa base)."""
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Muat konfigurasi proyek.

    Urutan: config.yaml -> (opsional) config.local.yaml menimpa nilai.
    """
    root = project_root()
    cfg_path = Path(path) if path else (root / "config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    local = root / "config.local.yaml"
    if local.exists():
        with open(local, "r", encoding="utf-8") as f:
            cfg = _deep_update(cfg, yaml.safe_load(f) or {})

    return cfg


if __name__ == "__main__":
    import json
    print(json.dumps(load_config(), indent=2, ensure_ascii=False))
