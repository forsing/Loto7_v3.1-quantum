"""Copula konfiguracija — zajednička raspodela 7 brojeva, seed 39."""

from __future__ import annotations

import json
from pathlib import Path

from config import RNG_SEED

ROOT = Path(__file__).resolve().parent
DEFAULT_COPULA_MODEL = ROOT / "loto739_copula.json"
DEFAULT_COPULA_CONFIG = ROOT / "loto739_copula_config.json"


def get_loto739_copula_config() -> dict:
    return {
        "random_seed": RNG_SEED,
        "lottery": {
            "name": "Loto Serbia 7/39",
            "pick": 7,
        },
        "copula": {
            "type": "gaussian",
            "use_empirical_combo": True,
            "use_position_copula": True,
        },
        "blend": {
            "objective_weight": 2.0,
        },
    }


def save_config(config: dict, path: Path | None = None) -> Path:
    out = path or DEFAULT_COPULA_CONFIG
    out.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def load_config(path: Path | None = None) -> dict:
    p = path or DEFAULT_COPULA_CONFIG
    if not p.is_file():
        return get_loto739_copula_config()
    return json.loads(p.read_text(encoding="utf-8"))
