"""Copula trening i učitavanje — zajednička raspodela 7 brojeva, ceo CSV."""

from __future__ import annotations

from pathlib import Path

from config import RNG_SEED
from models import Draw
from copula_config import DEFAULT_COPULA_CONFIG, DEFAULT_COPULA_MODEL, load_config, save_config
from copula_core import Loto739Copula


def save_model(model: Loto739Copula, path: Path | None = None) -> Path:
    out = path or DEFAULT_COPULA_MODEL
    return model.save(out)


def load_model(path: Path | None = None) -> Loto739Copula | None:
    p = path or DEFAULT_COPULA_MODEL
    return Loto739Copula.load(p)


def run_training_pipeline(
    draws: list[Draw],
    model_path: Path | None = None,
    config_path: Path | None = None,
    seed: int = RNG_SEED,
) -> dict:
    cfg = load_config(config_path)
    cfg["random_seed"] = seed

    print(f"\n[Copula] Trening na {len(draws)} izvlačenjima (zajednička sedmerka, seed={seed})")
    model = Loto739Copula()
    model.fit(draws)
    mp = save_model(model, model_path)
    cp = save_config(cfg, config_path)

    unique_combos = len(model.combo_counts)
    dup = sum(1 for c in model.combo_counts.values() if c > 1)
    print(f"[Copula] Završeno. Jedinstvenih sedmerki: {unique_combos}, duplikata: {dup}")

    return {
        "draws": len(draws),
        "unique_combos": unique_combos,
        "duplicate_combos": dup,
        "model_path": str(mp),
        "config_path": str(cp),
    }
