"""QCBM trening i učitavanje — Loto 7/39, ceo CSV."""

from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam, RMSprop, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR

from config import RNG_SEED
from models import Draw, PoolSpec
from qcbm_config import DEFAULT_QCBM_CONFIG, DEFAULT_QCBM_MODEL, load_config, save_config
from qcbm_core import QCBM, QuantumLossFunctions

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def target_distribution_from_draws(draws: list[Draw], pool: PoolSpec) -> torch.Tensor:
    """Frekvencija svakog broja 1..39 preko celog CSV-a."""
    counts = Counter(num for draw in draws for num in draw.main if pool.minimum <= num <= pool.maximum)
    total_slots = len(draws) * pool.pick
    probs = np.array(
        [counts.get(value, 0) / max(total_slots, 1) for value in pool.values],
        dtype=np.float32,
    )
    if probs.sum() <= 0:
        probs = np.ones(len(pool.values), dtype=np.float32) / len(pool.values)
    else:
        probs = probs / probs.sum()
    return torch.tensor(probs, dtype=torch.float32)


def train_qcbm(
    model: QCBM,
    target_probs: torch.Tensor,
    config: dict,
    device: torch.device,
) -> tuple[QCBM, dict]:
    """Trenira QCBM (KL + CE + MMD) na ciljnoj raspodeli."""
    train_cfg = config["training"]
    epochs = int(train_cfg["epochs"])
    lr = float(train_cfg["learning_rate"])
    patience = int(train_cfg["early_stop_patience"])
    grad_clip = float(train_cfg["grad_clip"])

    if train_cfg["optimizer"] == "adam":
        optimizer = Adam(model.parameters(), lr=lr)
    elif train_cfg["optimizer"] == "sgd":
        optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        optimizer = RMSprop(model.parameters(), lr=lr)

    scheduler = None
    if train_cfg["scheduler"] == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    elif train_cfg["scheduler"] == "step":
        scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
    elif train_cfg["scheduler"] == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, patience=10)

    target = target_probs.to(device)
    if target.dim() > 1:
        target = target.squeeze()
    target = target / (target.sum() + 1e-10)

    history: dict = {"loss": []}
    best_loss = float("inf")
    best_state = None
    patience_counter = 0

    print(
        f"\n[QCBM] Trening | gubitak={config['loss_function']} | "
        f"optimizer={train_cfg['optimizer']} | epoha={epochs} | uređaj={device}"
    )

    epoch_iter = range(epochs)
    if tqdm is not None:
        epoch_iter = tqdm(epoch_iter, desc="  QCBM", ncols=80)

    model.train()
    for epoch in epoch_iter:
        optimizer.zero_grad()
        generated = model.forward(batch_size=1).squeeze(0)

        kl_loss = F.kl_div(torch.log(generated + 1e-10), target, reduction="sum")
        ce_loss = -(target * torch.log(generated + 1e-10)).sum()
        gen_2d = generated.unsqueeze(0)
        tgt_2d = target.unsqueeze(0)
        mmd_val = QuantumLossFunctions.mmd_loss(gen_2d, tgt_2d, config)
        loss = 0.4 * kl_loss + 0.3 * ce_loss + 0.3 * mmd_val

        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if scheduler and train_cfg["scheduler"] != "plateau":
            scheduler.step()

        loss_val = float(loss.item())
        history["loss"].append(loss_val)

        if loss_val < best_loss:
            best_loss = loss_val
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if scheduler and train_cfg["scheduler"] == "plateau":
            scheduler.step(loss_val)

        if tqdm is not None and hasattr(epoch_iter, "set_postfix"):
            epoch_iter.set_postfix({"loss": f"{loss_val:.6f}", "best": f"{best_loss:.6f}"})

        if patience_counter >= patience:
            print(f"\n[QCBM] Rana zaustavljanja na epohi {epoch + 1} (patience={patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"[QCBM] Završeno. Najbolji gubitak: {best_loss:.6f}")
    history["best_loss"] = best_loss
    return model, history


def save_model(model: QCBM, path: Path | None = None) -> Path:
    out = path or DEFAULT_QCBM_MODEL
    torch.save(model.state_dict(), out)
    return out


def load_model(
    config: dict | None = None,
    path: Path | None = None,
    device: torch.device | None = None,
) -> QCBM | None:
    p = path or DEFAULT_QCBM_MODEL
    if not p.is_file():
        return None
    cfg = config or load_config()
    num_range = int(cfg["lottery"]["num_range"])
    dev = device or pick_device()
    model = QCBM(num_range, cfg).to(dev)
    model.load_state_dict(torch.load(p, map_location=dev, weights_only=True))
    model.eval()
    return model


def qcbm_probability_vector(
    config: dict | None = None,
    model_path: Path | None = None,
    device: torch.device | None = None,
) -> list[float] | None:
    """39 težina — prvo combo (pozicijski QCBM), zatim stari marginalni PyTorch."""
    from qcbm_combo import combo_weight_vector, load_combo_model
    from qcbm_config import DEFAULT_QCBM_COMBO_MODEL

    combo = combo_weight_vector(load_combo_model(DEFAULT_QCBM_COMBO_MODEL))
    if combo is not None:
        return combo

    model = load_model(config=config, path=model_path, device=device)
    if model is None:
        return None
    dev = device or pick_device()
    with torch.no_grad():
        probs = model.forward(batch_size=1).squeeze(0).cpu().numpy()
    return [float(x) for x in probs]


def blend_weights(
    classical: np.ndarray,
    qcbm_probs: list[float],
    qcbm_weight: float = 0.5,
) -> list[float]:
    """Meša klasične skorove i QCBM raspodelu za qc25 seed_weights."""
    q = np.asarray(qcbm_probs, dtype=float)
    c = np.asarray(classical, dtype=float)
    if q.size != c.size:
        raise ValueError(f"QCBM ({q.size}) i klasični ({c.size}) vektor moraju biti iste dužine")
    q = q / max(q.sum(), 1e-12)
    c_min, c_max = float(c.min()), float(c.max())
    if c_max - c_min < 1e-12:
        c_norm = np.ones_like(c) / len(c)
    else:
        c_norm = (c - c_min) / (c_max - c_min)
        c_norm = c_norm / max(c_norm.sum(), 1e-12)
    alpha = float(np.clip(qcbm_weight, 0.0, 1.0))
    mixed = alpha * q + (1.0 - alpha) * c_norm
    mixed = mixed / max(mixed.sum(), 1e-12)
    return [float(x) for x in mixed]


def run_training_pipeline(
    draws: list[Draw],
    pool: PoolSpec,
    config: dict | None = None,
    model_path: Path | None = None,
    config_path: Path | None = None,
    seed: int = RNG_SEED,
    csv_path: str | Path | None = None,
) -> dict:
    """Puni tok: pozicijski QCBM combo (empirijska sedmerka po kolonama)."""
    from qcbm_combo_train import run_combo_training_pipeline

    _ = draws, pool  # zadržano za CLI potpis; combo koristi pozicijski CSV
    return run_combo_training_pipeline(
        csv_path=csv_path,
        model_path=model_path,
        config_path=config_path,
        seed=seed,
    )
