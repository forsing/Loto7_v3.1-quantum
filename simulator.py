"""Lokalni Qiskit Aer simulator — qc25 (25 kubita), bez IBM hardvera."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from config import DEFAULT_CSV, RNG_SEED
from qc25 import (
    NUM_LAYERS,
    TOTAL_QUBITS,
    bitstring_to_loto_with_7,
    build_params_list,
    counts_key_to_int,
    encode_values_from_tail,
    full_qcbm,
    positional_tail_from_csv,
)


def run_heavy_sampling(
    qubits: int,
    layers: int,
    batch_circuits: int,
    shots: int,
    seed_weights: list[float],
    output_counts: Path | None = None,
    csv_path: str | Path | None = None,
    seed: int = RNG_SEED,
) -> tuple[list[int], dict]:
    from qiskit_aer import AerSimulator

    np.random.seed(seed)
    qubits = min(int(qubits), TOTAL_QUBITS)
    if qubits != TOTAL_QUBITS:
        qubits = TOTAL_QUBITS

    path = csv_path or DEFAULT_CSV
    tail = positional_tail_from_csv(path)
    encode_values = encode_values_from_tail(tail)

    trained_params = None
    try:
        from qcbm_combo import combo_params_for_qc25, load_combo_model

        trained_params = combo_params_for_qc25(load_combo_model())
    except Exception:
        trained_params = None

    circuits = []
    circuit_layers = NUM_LAYERS if trained_params else max(1, int(layers))
    for batch in range(max(1, int(batch_circuits))):
        params_list = build_params_list(
            encode_values,
            seed_weights,
            num_layers=circuit_layers,
            batch=batch,
            seed=seed,
            trained_params_list=trained_params,
        )
        circuits.append(
            full_qcbm(
                params_list,
                encode_values,
                n_lay=circuit_layers,
            )
        )

    sim = AerSimulator()
    counts: dict[str, int] = {}
    for qc in circuits:
        result = sim.run(qc, shots=int(shots), seed_simulator=seed).result()
        for bitstring, count in result.get_counts(qc).items():
            key = str(bitstring)
            counts[key] = counts.get(key, 0) + int(count)

    bits: list[int] = []
    for bitstring, _count in sorted(counts.items(), key=lambda item: -item[1])[:128]:
        bits.extend(1 if ch == "1" else 0 for ch in bitstring[::-1])

    top_combos: list[dict] = []
    sorted_counts = sorted(counts.items(), key=lambda item: -item[1])
    total_shots = max(1, sum(counts.values()))
    for bitstring, count in sorted_counts[:10]:
        val = counts_key_to_int(bitstring)
        combo = bitstring_to_loto_with_7(val)
        top_combos.append(
            {
                "combo": combo,
                "count": int(count),
                "probability": float(count / total_shots),
            }
        )

    payload = {
        "backend": "aer_simulator",
        "simulator": "local",
        "qc25": True,
        "qubits": qubits,
        "layers": layers,
        "batch_circuits": batch_circuits,
        "shots_per_circuit": shots,
        "total_shots": total_shots,
        "encode_values": encode_values,
        "qcbm_combo_params": trained_params is not None,
        "counts": counts,
        "top_combos": top_combos,
        "seed": seed,
    }
    if output_counts:
        import json

        output_counts.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return bits, payload


def run_profiled_sampling(
    qubits: int,
    layers: int,
    batch_circuits: int,
    shots: int,
    seed_weights: list[float],
    output_counts: Path | None = None,
    repeat_jobs: int = 1,
    profile: str = "custom",
    csv_path: str | Path | None = None,
    seed: int = RNG_SEED,
) -> tuple[list[int], dict]:
    all_bits: list[int] = []
    jobs = []
    for repeat in range(max(1, int(repeat_jobs))):
        repeat_output = output_counts.with_suffix(f".counts.{repeat + 1:02d}.json") if output_counts else None
        bits, payload = run_heavy_sampling(
            qubits=qubits,
            layers=layers,
            batch_circuits=batch_circuits,
            shots=shots,
            seed_weights=seed_weights,
            output_counts=repeat_output,
            csv_path=csv_path,
            seed=seed + repeat,
        )
        all_bits.extend(bits)
        jobs.append(payload)
    first = jobs[0] if jobs else {}
    return all_bits, {
        "profile": profile,
        "backend": first.get("backend", "aer_simulator"),
        "simulator": "local",
        "qc25": True,
        "qubits": first.get("qubits", qubits),
        "layers": layers,
        "batch_circuits": batch_circuits,
        "shots_per_circuit": shots,
        "repeat_jobs": repeat_jobs,
        "total_requested_shots": sum(job.get("total_shots", 0) for job in jobs),
        "encode_values": first.get("encode_values"),
        "top_combos": first.get("top_combos"),
        "jobs": jobs,
    }
