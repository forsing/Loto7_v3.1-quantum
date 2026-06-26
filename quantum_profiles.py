from __future__ import annotations

from qc25 import TOTAL_QUBITS

LOCAL_QC25_QUBITS = TOTAL_QUBITS

# qc25: uvek 25 kubita lokalno; profili skaliraju slojeve / batch / shots
PROFILES = {
    "standard": {"qubits": 25, "layers": 2, "batch_circuits": 2, "shots": 4096, "repeat_jobs": 1},
    "long": {"qubits": 25, "layers": 4, "batch_circuits": 4, "shots": 8192, "repeat_jobs": 1},
    "deep": {"qubits": 25, "layers": 6, "batch_circuits": 6, "shots": 8192, "repeat_jobs": 2},
    "extreme": {"qubits": 25, "layers": 8, "batch_circuits": 8, "shots": 16384, "repeat_jobs": 2},
}


def resolve_quantum_profile(name: str, backend_qubits: int = LOCAL_QC25_QUBITS) -> dict:
    if name not in PROFILES:
        raise ValueError(f"unknown quantum profile {name}")
    profile = dict(PROFILES[name])
    profile["profile"] = name
    profile["requested_qubits"] = profile["qubits"]
    profile["qubits"] = min(profile["qubits"], int(backend_qubits), LOCAL_QC25_QUBITS)
    profile["total_requested_shots"] = profile["shots"] * profile["batch_circuits"] * profile["repeat_jobs"]
    return profile
