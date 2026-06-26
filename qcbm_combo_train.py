"""Trening QCBM combo — ceo CSV, pozicijska raspodela sedmerke."""

from __future__ import annotations

from pathlib import Path

from config import DEFAULT_CSV, RNG_SEED
from qcbm_config import DEFAULT_QCBM_COMBO_MODEL, DEFAULT_QCBM_CONFIG, load_config, save_config
from qcbm_combo import (
    load_positional_matrix,
    save_combo_model,
    train_all_positions,
)


def run_combo_training_pipeline(
    csv_path: str | Path | None = None,
    model_path: Path | None = None,
    config_path: Path | None = None,
    seed: int = RNG_SEED,
) -> dict:
    cfg = load_config(config_path)
    cfg["random_seed"] = seed
    combo_cfg = cfg.setdefault("combo", {})
    num_layers = int(combo_cfg.get("num_layers", 2))
    maxiter = int(combo_cfg.get("maxiter", 200))

    path = Path(csv_path or DEFAULT_CSV)
    matrix = load_positional_matrix(path)
    model = train_all_positions(
        matrix,
        seed=seed,
        num_layers=num_layers,
        maxiter=maxiter,
        verbose=True,
    )
    mp = save_combo_model(model, model_path or DEFAULT_QCBM_COMBO_MODEL)
    cfg["combo"]["model_type"] = "positional_qc25"
    cp = save_config(cfg, config_path or DEFAULT_QCBM_CONFIG)

    losses = [p["loss"] for p in model["positions"]]
    return {
        "model_type": "combo_qc25",
        "draws": len(matrix),
        "csv": str(path),
        "model_path": str(mp),
        "config_path": str(cp),
        "mean_kl_loss": float(sum(losses) / max(len(losses), 1)),
        "position_losses": losses,
        "seed": seed,
    }
