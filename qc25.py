"""qc25: 5 blokova × 5 kubita = 25; 7/39 pozicijski opsezi (kao QCBM_qc25_7_v2)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit

from config import RNG_SEED

# Dozvoljeni opsezi po poziciji (Num1..Num7)
MIN_VAL = [1, 2, 3, 4, 5, 6, 7]
MAX_VAL = [33, 34, 35, 36, 37, 38, 39]

NUM_QUBITS = 5
NUM_LAYERS = 2
NUM_POSITIONS = 5
TOTAL_QUBITS = NUM_QUBITS * NUM_POSITIONS


def encode_position(value: int, n_q: int = NUM_QUBITS) -> QuantumCircuit:
    v = int(value)
    bin_full = format(v, "b")
    if len(bin_full) > n_q:
        bin_repr = bin_full[-n_q:]
    else:
        bin_repr = bin_full.zfill(n_q)
    qc = QuantumCircuit(n_q)
    for i, bit in enumerate(reversed(bin_repr)):
        if bit == "1":
            qc.x(i)
    return qc


def variational_layer(params: np.ndarray, n_q: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_q)
    for i in range(n_q):
        qc.ry(params[i], i)
    for i in range(n_q - 1):
        qc.cx(i, i + 1)
    return qc


def qcbm_ansatz(params: np.ndarray, n_q: int, n_lay: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_q)
    for layer in range(n_lay):
        start = layer * n_q
        end = (layer + 1) * n_q
        qc.compose(variational_layer(params[start:end], n_q), inplace=True)
    return qc


def full_qcbm(
    params_list: list[np.ndarray],
    values: list[int],
    n_q: int = NUM_QUBITS,
    n_lay: int = 2,
    n_pos: int = NUM_POSITIONS,
) -> QuantumCircuit:
    total_qubits = n_q * n_pos
    qc = QuantumCircuit(total_qubits)
    for pos in range(n_pos):
        start_q = pos * n_q
        end_q = start_q + n_q
        qc_enc = encode_position(values[pos], n_q)
        qc.compose(qc_enc, qubits=range(start_q, end_q), inplace=True)
        qc_var = qcbm_ansatz(params_list[pos], n_q, n_lay)
        qc.compose(qc_var, qubits=range(start_q, end_q), inplace=True)
    qc.measure_all()
    return qc


def bitstring_to_loto_with_7(
    bitstring_int: int,
    n_qubits: int = NUM_QUBITS,
    num_pos: int = NUM_POSITIONS,
) -> list[int]:
    num_bits = n_qubits * num_pos
    bitstring = format(int(bitstring_int), "b").zfill(num_bits)
    main_numbers: list[int] = []
    for pos in range(num_pos):
        start = pos * n_qubits
        chunk = bitstring[start : start + n_qubits]
        val = int(chunk, 2)
        mv = MIN_VAL[pos]
        mv_max = MAX_VAL[pos]
        rng = mv_max - mv + 1
        mapped = (val % rng) + mv
        main_numbers.append(int(mapped))

    def find_unique(start_val: int, used_set: set[int], idx: int) -> int:
        mv = MIN_VAL[idx]
        mv_max = MAX_VAL[idx]
        rng = mv_max - mv + 1
        v = ((start_val - mv) % rng) + mv
        tries = 0
        while v in used_set and tries < rng:
            v = mv + ((v - mv + 1) % rng)
            tries += 1
        if v in used_set:
            for cand in range(mv, mv_max + 1):
                if cand not in used_set:
                    v = cand
                    break
        return int(v)

    sum_main = sum(main_numbers)
    start6 = (sum_main) % (MAX_VAL[5] - MIN_VAL[5] + 1) + MIN_VAL[5]
    sixth = find_unique(start6, set(main_numbers), 5)
    used = set(main_numbers) | {sixth}
    start7 = (sum_main + sixth) % (MAX_VAL[6] - MIN_VAL[6] + 1) + MIN_VAL[6]
    seventh = find_unique(start7, used, 6)
    return main_numbers + [sixth, seventh]


def positional_tail_from_csv(csv_path: str | Path, n_tail: int = 500) -> np.ndarray:
    """Num1..Num7 bez sortiranja — za enkodiranje po poziciji."""
    path = Path(csv_path)
    head = pd.read_csv(path, nrows=1, encoding="utf-8")
    cols = [f"Num{i}" for i in range(1, 8)]
    if not all(c in head.columns for c in cols):
        raise ValueError(f"CSV mora imati kolone {cols} za qc25 enkodiranje")
    df = pd.read_csv(path, encoding="utf-8")
    n = min(max(1, n_tail), len(df))
    return df[cols].iloc[-n:].to_numpy(dtype=np.int64)


def encode_values_from_tail(tail: np.ndarray) -> list[int]:
    values: list[int] = []
    for pos in range(NUM_POSITIONS):
        col = tail[:, pos]
        mv, mv_max = MIN_VAL[pos], MAX_VAL[pos]
        mean_v = int(np.round(float(col.mean())))
        values.append(int(max(mv, min(mv_max, mean_v))))
    return values


def position_weight(seed_weights: list[float], pos: int) -> float:
    weights = list(seed_weights) or [1.0]
    mv, mv_max = MIN_VAL[pos], MAX_VAL[pos]
    if len(weights) < mv_max:
        weights = weights + [weights[-1]] * (mv_max - len(weights))
    segment = weights[mv - 1 : mv_max]
    return float(np.mean(segment)) if segment else 0.0


def build_params_list(
    encode_values: list[int],
    seed_weights: list[float],
    num_layers: int,
    batch: int = 0,
    seed: int = RNG_SEED,
    trained_params_list: list[np.ndarray] | None = None,
) -> list[np.ndarray]:
    import math

    golden = (math.sqrt(5.0) - 1.0) / 2.0
    rng = np.random.default_rng(int(seed) + batch * 39)
    params_list: list[np.ndarray] = []
    for pos in range(NUM_POSITIONS):
        w = position_weight(seed_weights, pos)
        mv_max = MAX_VAL[pos]
        if trained_params_list is not None and pos < len(trained_params_list):
            base = np.asarray(trained_params_list[pos], dtype=float)
            noise = 0.03 * (1.0 - min(w, 1.0)) + 0.015 * batch
            layer_params = (base + rng.uniform(-noise, noise, size=base.shape)) % (2 * math.pi)
        else:
            layer_params = np.zeros(num_layers * NUM_QUBITS, dtype=float)
            for layer in range(num_layers):
                for q in range(NUM_QUBITS):
                    idx = layer * NUM_QUBITS + q
                    base = (
                        w * 0.12
                        + golden * ((pos + 1) * (q + 1) * (layer + 1) * 0.013)
                        + 0.037 * batch
                        + encode_values[pos] / max(mv_max, 1) * 0.05
                    )
                    layer_params[idx] = float((2 * math.pi * base + rng.uniform(0, 0.08)) % (2 * math.pi))
        params_list.append(layer_params)
    return params_list


def counts_key_to_int(key: str | int) -> int:
    if isinstance(key, int):
        return int(key)
    s = str(key).replace(" ", "")
    if not s:
        return 0
    return int(s, 2)
