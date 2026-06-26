"""QCBM konfiguracija — Loto 7/39, seed 39."""

from __future__ import annotations

import json
from pathlib import Path

from config import RNG_SEED

ROOT = Path(__file__).resolve().parent
DEFAULT_QCBM_CONFIG = ROOT / "loto739_qcbm_config.json"
DEFAULT_QCBM_MODEL = ROOT / "loto739_qcbm.pt"
DEFAULT_QCBM_COMBO_MODEL = ROOT / "loto739_qcbm_combo.json"


def get_loto739_config() -> dict:
    """Podrazumevana konfiguracija za trenirani QCBM (7/39)."""
    return {
        "random_seed": RNG_SEED,
        "lottery": {
            "name": "Loto Serbia 7/39",
            "short": "loto739",
            "num_range": 39,
            "pick": 7,
        },
        "qcbm": {
            "ansatz_type": "hea",
            "n_layers": 4,
            "entanglement_mode": "circular",
            "rotation_type": "xyz",
            "initial_state": "uniform",
        },
        "training": {
            "epochs": 200,
            "learning_rate": 0.005,
            "optimizer": "adam",
            "scheduler": "cosine",
            "early_stop_patience": 20,
            "grad_clip": 1.0,
        },
        "loss_function": "mmd",
        "mmd": {
            "kernel": "rbf",
            "sigma": [1.0, 2.0, 5.0, 10.0],
        },
        "blend": {
            "qcbm_weight": 0.5,
            "classical_weight": 0.5,
        },
        "combo": {
            "model_type": "positional_qc25",
            "num_layers": 2,
            "maxiter": 200,
        },
    }


def save_config(config: dict, path: Path | None = None) -> Path:
    out = path or DEFAULT_QCBM_CONFIG
    out.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def load_config(path: Path | None = None) -> dict:
    p = path or DEFAULT_QCBM_CONFIG
    if not p.is_file():
        return get_loto739_config()
    return json.loads(p.read_text(encoding="utf-8"))
