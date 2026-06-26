"""QCBM po poziciji — empirijska raspodela izvučenih kombinacija (obrazac QCBM_qc25_7 / q_1_QCBM)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from qiskit.circuit.library import n_local
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize as scipy_minimize

from config import RNG_SEED
from qc25 import MAX_VAL, MIN_VAL, NUM_LAYERS, NUM_POSITIONS, NUM_QUBITS

ROOT = Path(__file__).resolve().parent
DEFAULT_QCBM_COMBO_MODEL = ROOT / "loto739_qcbm_combo.json"


def load_positional_matrix(csv_path: str | Path) -> np.ndarray:
    """Num1..Num7 iz CSV-a — redosled kao u izvlačenju (bez sortiranja)."""
    import pandas as pd

    path = Path(csv_path)
    head = pd.read_csv(path, nrows=1, encoding="utf-8")
    cols = [f"Num{i}" for i in range(1, 8)]
    if not all(c in head.columns for c in cols):
        raise ValueError(f"CSV mora imati kolone {cols}")
    df = pd.read_csv(path, encoding="utf-8")
    return df[cols].to_numpy(dtype=np.int64)


def build_empirical_column(matrix: np.ndarray, pos: int) -> dict[int, float]:
    """Empirijska raspodela 5-bit stanja za jednu poziciju sedmerke."""
    dist: dict[int, int] = {}
    n_states = 1 << NUM_QUBITS
    mv, mv_max = MIN_VAL[pos], MAX_VAL[pos]
    span = mv_max - mv + 1
    for row in matrix:
        actual = int(row[pos])
        v = actual - mv
        if v < 0 or v >= span:
            v = v % span
        if v >= n_states:
            v = v % n_states
        dist[v] = dist.get(v, 0) + 1
    total = max(1, sum(dist.values()))
    return {k: c / total for k, c in dist.items()}


def make_ansatz(num_layers: int = NUM_LAYERS):
    return n_local(
        NUM_QUBITS,
        rotation_blocks="ry",
        entanglement_blocks="cz",
        entanglement="linear",
        reps=num_layers,
    )


def exact_probs(ansatz, theta: np.ndarray) -> dict[int, float]:
    circ = ansatz.assign_parameters(theta)
    sv = Statevector.from_instruction(circ)
    return {i: float(p) for i, p in enumerate(sv.probabilities()) if p > 1e-15}


def kl_loss(target: dict[int, float], generated: dict[int, float]) -> float:
    loss = 0.0
    for k, pt in target.items():
        ps = generated.get(k, 1e-10)
        if ps <= 0:
            ps = 1e-10
        loss += pt * np.log(pt / ps)
    return float(loss)


def train_position(
    target: dict[int, float],
    theta0: np.ndarray,
    num_layers: int,
    maxiter: int,
) -> tuple[np.ndarray, float, dict[int, float]]:
    ansatz = make_ansatz(num_layers)

    def cost(theta: np.ndarray) -> float:
        return kl_loss(target, exact_probs(ansatz, theta))

    res = scipy_minimize(
        cost,
        theta0,
        method="COBYLA",
        options={"maxiter": int(maxiter), "rhobeg": 0.5},
    )
    final = exact_probs(ansatz, res.x)
    return res.x.astype(float), float(res.fun), final


def train_all_positions(
    matrix: np.ndarray,
    seed: int = RNG_SEED,
    num_layers: int = NUM_LAYERS,
    maxiter: int = 200,
    verbose: bool = True,
) -> dict:
    rng = np.random.default_rng(seed)
    ansatz = make_ansatz(num_layers)
    n_params = ansatz.num_parameters
    positions: list[dict] = []

    if verbose:
        print(
            f"\n[QCBM combo] 7 pozicija × {NUM_QUBITS}q | COBYLA maxiter={maxiter} | "
            f"uzoraka={len(matrix)}"
        )

    for pos in range(7):
        target = build_empirical_column(matrix, pos)
        theta0 = rng.uniform(0, 2 * np.pi, n_params)
        params, loss, dist = train_position(target, theta0, num_layers, maxiter)
        if verbose:
            top = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:3]
            info = " | ".join(
                f"{(int(v) % (MAX_VAL[pos] - MIN_VAL[pos] + 1)) + MIN_VAL[pos]}:{p:.3f}"
                for v, p in top
            )
            print(f"  Poz {pos + 1} [{MIN_VAL[pos]}-{MAX_VAL[pos]}]: loss={loss:.4f}  top: {info}")
        positions.append(
            {
                "pos": pos,
                "min_val": MIN_VAL[pos],
                "max_val": MAX_VAL[pos],
                "params": params.tolist(),
                "loss": loss,
                "empirical_states": len(target),
                "distribution": {str(k): v for k, v in dist.items()},
            }
        )

    return {
        "version": "combo_qc25",
        "seed": int(seed),
        "num_qubits": NUM_QUBITS,
        "num_layers": int(num_layers),
        "maxiter": int(maxiter),
        "n_draws": int(len(matrix)),
        "positions": positions,
    }


def save_combo_model(model: dict, path: Path | None = None) -> Path:
    out = path or DEFAULT_QCBM_COMBO_MODEL
    out.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def load_combo_model(path: Path | None = None) -> dict | None:
    p = path or DEFAULT_QCBM_COMBO_MODEL
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _state_to_number(pos: int, state_idx: int) -> int:
    mv, mv_max = MIN_VAL[pos], MAX_VAL[pos]
    span = mv_max - mv + 1
    return int((int(state_idx) % span) + mv)


def combo_weight_vector(model: dict | None = None) -> list[float] | None:
    """39 težina iz 7 pozicijskih QCBM raspodela (za blend sa klasičnim skorovima)."""
    m = model or load_combo_model()
    if m is None:
        return None
    weights = np.zeros(39, dtype=float)
    for entry in m.get("positions", []):
        pos = int(entry["pos"])
        dist = entry.get("distribution") or {}
        for key, prob in dist.items():
            num = _state_to_number(pos, int(key))
            if 1 <= num <= 39:
                weights[num - 1] += float(prob)
    s = float(weights.sum())
    if s <= 0:
        return None
    return [float(x) for x in (weights / s)]


def combo_params_for_qc25(model: dict | None = None) -> list[np.ndarray] | None:
    """Trenirani parametri za prvih 5 pozicija qc25 kola (NUM_LAYERS × NUM_QUBITS)."""
    m = model or load_combo_model()
    if m is None:
        return None
    need = NUM_LAYERS * NUM_QUBITS
    out: list[np.ndarray] = []
    for entry in m.get("positions", [])[:NUM_POSITIONS]:
        raw = np.asarray(entry["params"], dtype=float)
        if raw.size == need:
            out.append(raw.copy())
        elif raw.size > need:
            out.append(raw[:need].copy())
        else:
            padded = np.zeros(need, dtype=float)
            padded[: raw.size] = raw
            out.append(padded)
    if len(out) < NUM_POSITIONS:
        return None
    return out
