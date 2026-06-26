from __future__ import annotations

from collections import Counter
import itertools
import math

import numpy as np

from models import Draw, PoolSpec
from randomness import (
    audit_pool_randomness,
    binary_matrix,
    gap_fingerprint,
    lag_overlap,
    randomness_fingerprint,
    rolling_drift,
    runs_z,
    seasonality_fingerprint,
)


def simulate_uniform_draws(template: list[Draw], pool: PoolSpec, seed: int) -> list[Draw]:
    rng = np.random.default_rng(seed)
    values = pool.values
    out: list[Draw] = []
    for draw in template:
        nums = tuple(sorted(int(x) for x in rng.choice(values, size=pool.pick, replace=False)))
        out.append(Draw(draw.date, nums))
    return out


def pair_max_lift(draws: list[Draw], pool: PoolSpec, field: str) -> float:
    audit = audit_pool_randomness(draws, pool, field)
    return float(audit["pair_lift"]["max_lift"])


def triple_max_lift(draws: list[Draw], pool: PoolSpec, field: str) -> float:
    counts: Counter[tuple[int, int, int]] = Counter()
    for draw in draws:
        nums = draw.main
        valid = sorted(num for num in nums if pool.minimum <= num <= pool.maximum)
        counts.update(itertools.combinations(valid, 3))
    if not counts or pool.pick < 3:
        return 0.0
    probability = math.comb(len(pool.values) - 3, pool.pick - 3) / math.comb(len(pool.values), pool.pick)
    expected = len(draws) * probability
    return max(count / max(expected, 1e-12) for count in counts.values())


def frequency_chi_square(draws: list[Draw], pool: PoolSpec, field: str) -> float:
    return float(audit_pool_randomness(draws, pool, field)["frequency"]["chi_square"])


def lag_max_delta(draws: list[Draw], pool: PoolSpec, field: str) -> float:
    return float(lag_overlap(draws, pool, field)["max_abs_lift_delta"])


def drift_js(draws: list[Draw], pool: PoolSpec, field: str) -> float:
    return float(rolling_drift(draws, pool, field)["js_divergence"])


def runs_max_abs_z(draws: list[Draw], pool: PoolSpec, field: str) -> float:
    mat = binary_matrix(draws, pool, field)
    if mat.shape[1] == 0:
        return 0.0
    values = [abs(runs_z(mat[:, idx])) for idx in range(mat.shape[1])]
    return float(max(values, default=0.0))


def gap_max_abs_lift(draws: list[Draw], pool: PoolSpec, field: str) -> float:
    report = gap_fingerprint(draws, pool, field)
    return float(max((abs(row["mean_lift"] - 1.0) for row in report["top_gap_anomalies"]), default=0.0))


def calendar_max_js(draws: list[Draw], pool: PoolSpec, field: str) -> float:
    report = seasonality_fingerprint(draws, pool, field)
    return float(max(report["month"]["max_js"], report["weekday"]["max_js"]))


def empirical_p_value(observed: float, null_values: list[float]) -> float:
    exceed = sum(value >= observed for value in null_values)
    return (exceed + 1.0) / (len(null_values) + 1.0)


def calibrated_randomness_fingerprint(
    draws: list[Draw],
    pool: PoolSpec,
    field: str = "main",
    null_trials: int = 500,
    seed: int = 39,
) -> dict:
    base = randomness_fingerprint(draws, pool, field)
    observed = {
        "frequency_chi_square": frequency_chi_square(draws, pool, field),
        "pair_max_lift": pair_max_lift(draws, pool, field),
        "triple_max_lift": triple_max_lift(draws, pool, field),
        "lag_max_delta": lag_max_delta(draws, pool, field),
        "drift_js": drift_js(draws, pool, field),
        "runs_max_abs_z": runs_max_abs_z(draws, pool, field),
        "gap_max_abs_lift": gap_max_abs_lift(draws, pool, field),
        "calendar_max_js": calendar_max_js(draws, pool, field),
    }
    nulls = {name: [] for name in observed}
    for trial in range(max(1, int(null_trials))):
        null_draws = simulate_uniform_draws(draws, pool, seed + trial)
        nulls["frequency_chi_square"].append(frequency_chi_square(null_draws, pool, field))
        nulls["pair_max_lift"].append(pair_max_lift(null_draws, pool, field))
        nulls["triple_max_lift"].append(triple_max_lift(null_draws, pool, field))
        nulls["lag_max_delta"].append(lag_max_delta(null_draws, pool, field))
        nulls["drift_js"].append(drift_js(null_draws, pool, field))
        nulls["runs_max_abs_z"].append(runs_max_abs_z(null_draws, pool, field))
        nulls["gap_max_abs_lift"].append(gap_max_abs_lift(null_draws, pool, field))
        nulls["calendar_max_js"].append(calendar_max_js(null_draws, pool, field))

    calibration = {}
    for name, value in observed.items():
        arr = np.asarray(nulls[name], dtype=float)
        calibration[name] = {
            "observed": value,
            "empirical_p": empirical_p_value(value, nulls[name]),
            "null_mean": float(arr.mean()),
            "null_p95": float(np.percentile(arr, 95)),
            "null_p99": float(np.percentile(arr, 99)),
        }

    metric_to_type = {
        "frequency_chi_square": "frequency_bias",
        "pair_max_lift": "pair_clustering",
        "triple_max_lift": "triple_clustering",
        "lag_max_delta": "temporal_memory",
        "drift_js": "distribution_drift",
        "runs_max_abs_z": "runs_irregularity",
        "gap_max_abs_lift": "gap_anomaly",
        "calendar_max_js": "calendar_effect",
    }
    scores = dict(base["randomness_type"]["scores"])
    for metric, family in metric_to_type.items():
        scores[family] = 1.0 - calibration[metric]["empirical_p"]
    dominant = [
        metric_to_type[metric]
        for metric, row in sorted(calibration.items(), key=lambda item: item[1]["empirical_p"])
        if row["empirical_p"] <= 0.01
    ][:5]
    if not dominant:
        dominant = ["near_uniform"]
    base["calibration"] = calibration
    base["randomness_type"] = {
        "scores": scores,
        "dominant_types": dominant,
        "threshold": "dominant requires calibrated empirical p <= 0.01",
    }
    base["plain_language"] = {
        "summary": (
            "The calibrated null test did not find a strong reusable deviation from uniform randomness."
            if dominant == ["near_uniform"]
            else "Calibrated dominant randomness fingerprints: "
            + ", ".join(dominant)
            + ". These still require walk-forward validation."
        )
    }
    return base
