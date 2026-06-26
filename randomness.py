from __future__ import annotations

import itertools
import math
from collections import Counter
from statistics import NormalDist

import numpy as np

from math_model import number_scores, pair_centrality, robust_z
from models import Draw, LotterySpec, PoolSpec


def values_for(draw: Draw, field: str) -> tuple[int, ...]:
    if field != "main":
        raise ValueError(f"unsupported field {field}")
    return draw.main


def normal_two_sided_p(z: float) -> float:
    return max(0.0, min(1.0, 2.0 * (1.0 - NormalDist().cdf(abs(z)))))


def audit_pool_randomness(draws: list[Draw], pool: PoolSpec, field: str = "main") -> dict:
    if not draws:
        raise ValueError("draws cannot be empty")
    values = pool.values
    total_slots = len(draws) * pool.pick
    expected = total_slots / len(values)
    counts = Counter(num for draw in draws for num in values_for(draw, field) if pool.minimum <= num <= pool.maximum)
    observed = np.array([counts[value] for value in values], dtype=float)
    chi_square = float(np.sum((observed - expected) ** 2 / max(expected, 1e-12)))
    z_scores = (observed - expected) / math.sqrt(max(expected, 1e-12))
    top_positive = sorted(
        ({"number": value, "count": int(counts[value]), "z": float(z_scores[idx])} for idx, value in enumerate(values)),
        key=lambda row: row["z"],
        reverse=True,
    )[:10]
    top_negative = sorted(
        ({"number": value, "count": int(counts[value]), "z": float(z_scores[idx])} for idx, value in enumerate(values)),
        key=lambda row: row["z"],
    )[:10]

    gaps: dict[int, list[int]] = {value: [] for value in values}
    last_seen: dict[int, int] = {}
    for idx, draw in enumerate(draws):
        present = set(values_for(draw, field))
        for value in values:
            if value in present:
                if value in last_seen:
                    gaps[value].append(idx - last_seen[value])
                last_seen[value] = idx
    gap_rows = []
    for value in values:
        row = gaps[value]
        gap_rows.append(
            {
                "number": value,
                "mean_gap": float(np.mean(row)) if row else None,
                "max_gap": int(max(row)) if row else None,
                "current_gap": len(draws) - 1 - last_seen[value] if value in last_seen else len(draws),
            }
        )
    most_overdue = sorted(gap_rows, key=lambda row: row["current_gap"], reverse=True)[:10]

    pair_counts = Counter()
    for draw in draws:
        nums = sorted(num for num in values_for(draw, field) if pool.minimum <= num <= pool.maximum)
        for pair in itertools.combinations(nums, 2):
            pair_counts[pair] += 1
    expected_pair = len(draws) * (pool.pick / len(values)) * ((pool.pick - 1) / max(1, len(values) - 1))
    pair_rows = [
        {"pair": list(pair), "count": count, "lift": float(count / max(expected_pair, 1e-12))}
        for pair, count in pair_counts.items()
    ]
    pair_rows.sort(key=lambda row: row["lift"], reverse=True)

    months = Counter(draw.date.month for draw in draws)
    weekdays = Counter(draw.date.weekday() for draw in draws)
    month_skew = max(months.values()) / max(1, min(months.values())) if months else 1.0
    weekday_skew = max(weekdays.values()) / max(1, min(weekdays.values())) if weekdays else 1.0

    max_abs_z = float(np.max(np.abs(z_scores))) if len(z_scores) else 0.0
    max_pair_lift = float(pair_rows[0]["lift"]) if pair_rows else 0.0
    signal_score = 0
    if max_abs_z >= 2.0:
        signal_score += 1
    if max_abs_z >= 3.0:
        signal_score += 1
    if max_pair_lift >= 1.5:
        signal_score += 1
    if month_skew >= 1.8 or weekday_skew >= 1.8:
        signal_score += 1
    strength = ("none", "weak", "moderate", "strong", "strong")[min(signal_score, 4)]
    plain = (
        "No obvious deviation from a simple random baseline was detected."
        if strength == "none"
        else "The history shows measurable deviations from a simple random baseline. This is a signal to backtest, not proof of predictability."
    )

    return {
        "draws": len(draws),
        "pool": {"min": pool.minimum, "max": pool.maximum, "pick": pool.pick},
        "frequency": {
            "expected_per_number": expected,
            "chi_square": chi_square,
            "max_abs_z": max_abs_z,
            "top_positive": top_positive,
            "top_negative": top_negative,
            "approx_strongest_p": normal_two_sided_p(max_abs_z),
        },
        "gap": {"most_overdue": most_overdue},
        "pair_lift": {"expected_pair_count": expected_pair, "max_lift": max_pair_lift, "top_pairs": pair_rows[:10]},
        "seasonality": {"month_skew": float(month_skew), "weekday_skew": float(weekday_skew)},
        "verdict": {"signal_strength": strength, "plain": plain},
    }


def shannon_entropy(probs: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=float)
    probs = probs[probs > 0]
    if len(probs) == 0:
        return 0.0
    return float(-np.sum(probs * np.log2(probs)))


def jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / max(float(p.sum()), 1e-12)
    q = q / max(float(q.sum()), 1e-12)
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / np.maximum(b[mask], 1e-12))))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) == 0 or np.allclose(values.sum(), 0):
        return 0.0
    sorted_values = np.sort(values)
    n = len(sorted_values)
    return float((2 * np.arange(1, n + 1).dot(sorted_values) / (n * sorted_values.sum())) - (n + 1) / n)


def binary_matrix(draws: list[Draw], pool: PoolSpec, field: str) -> np.ndarray:
    index = {value: idx for idx, value in enumerate(pool.values)}
    mat = np.zeros((len(draws), len(pool.values)), dtype=np.int8)
    for row_idx, draw in enumerate(draws):
        for num in values_for(draw, field):
            if num in index:
                mat[row_idx, index[num]] = 1
    return mat


def runs_z(row: np.ndarray) -> float:
    row = np.asarray(row, dtype=int)
    n1 = int(row.sum())
    n0 = int(len(row) - n1)
    if n0 == 0 or n1 == 0:
        return 0.0
    runs = 1 + int(np.sum(row[1:] != row[:-1]))
    expected = 1 + (2 * n0 * n1) / (n0 + n1)
    var = (2 * n0 * n1 * (2 * n0 * n1 - n0 - n1)) / (((n0 + n1) ** 2) * (n0 + n1 - 1))
    if var <= 0:
        return 0.0
    return float((runs - expected) / math.sqrt(var))


def triple_lift(draws: list[Draw], pool: PoolSpec, field: str) -> dict:
    counts = Counter()
    for draw in draws:
        nums = sorted(num for num in values_for(draw, field) if pool.minimum <= num <= pool.maximum)
        for triple in itertools.combinations(nums, 3):
            counts[triple] += 1
    if not counts or pool.pick < 3:
        return {"expected_triple_count": 0.0, "max_lift": 0.0, "top_triples": []}
    p = 1.0
    for offset in range(3):
        p *= (pool.pick - offset) / max(1, len(pool.values) - offset)
    expected = len(draws) * p
    rows = [
        {"triple": list(triple), "count": count, "lift": float(count / max(expected, 1e-12))}
        for triple, count in counts.items()
    ]
    rows.sort(key=lambda row: row["lift"], reverse=True)
    return {"expected_triple_count": expected, "max_lift": float(rows[0]["lift"]), "top_triples": rows[:10]}


def lag_overlap(draws: list[Draw], pool: PoolSpec, field: str, max_lag: int = 10) -> dict:
    sets = [set(values_for(draw, field)) for draw in draws]
    expected = pool.pick * pool.pick / len(pool.values)
    rows = []
    for lag in range(1, min(max_lag, len(sets) - 1) + 1):
        overlaps = [len(sets[idx] & sets[idx - lag]) for idx in range(lag, len(sets))]
        mean = float(np.mean(overlaps)) if overlaps else 0.0
        rows.append(
            {"lag": lag, "mean_overlap": mean, "expected_overlap": expected, "lift": mean / max(expected, 1e-12)}
        )
    max_lift = max((abs(row["lift"] - 1.0) for row in rows), default=0.0)
    return {"expected_overlap": expected, "max_abs_lift_delta": float(max_lift), "lags": rows}


def rolling_drift(draws: list[Draw], pool: PoolSpec, field: str) -> dict:
    mat = binary_matrix(draws, pool, field)
    if len(draws) < 20:
        return {"js_divergence": 0.0, "top_movers": []}
    split = len(draws) // 2
    early = mat[:split].sum(axis=0).astype(float) + 0.5
    late = mat[split:].sum(axis=0).astype(float) + 0.5
    early_probs = early / early.sum()
    late_probs = late / late.sum()
    diff = late_probs - early_probs
    movers = sorted(
        (
            {
                "number": value,
                "delta_probability": float(diff[idx]),
                "early_p": float(early_probs[idx]),
                "late_p": float(late_probs[idx]),
            }
            for idx, value in enumerate(pool.values)
        ),
        key=lambda row: abs(row["delta_probability"]),
        reverse=True,
    )[:10]
    return {"js_divergence": jensen_shannon(early_probs, late_probs), "top_movers": movers}


def seasonality_fingerprint(draws: list[Draw], pool: PoolSpec, field: str) -> dict:
    month_counts = {month: np.zeros(len(pool.values), dtype=float) for month in range(1, 13)}
    weekday_counts = {day: np.zeros(len(pool.values), dtype=float) for day in range(7)}
    index = {value: idx for idx, value in enumerate(pool.values)}
    for draw in draws:
        for num in values_for(draw, field):
            if num in index:
                month_counts[draw.date.month][index[num]] += 1
                weekday_counts[draw.date.weekday()][index[num]] += 1
    global_counts = sum(month_counts.values()) + 0.5
    global_probs = global_counts / global_counts.sum()

    def max_js(groups: dict[int, np.ndarray]) -> dict:
        rows = []
        for key, counts in groups.items():
            if counts.sum() == 0:
                continue
            probs = (counts + 0.5) / (counts.sum() + 0.5 * len(counts))
            rows.append({"bucket": key, "js_divergence": jensen_shannon(probs, global_probs)})
        rows.sort(key=lambda row: row["js_divergence"], reverse=True)
        return {"max_js": float(rows[0]["js_divergence"]) if rows else 0.0, "top_buckets": rows[:5]}

    return {"month": max_js(month_counts), "weekday": max_js(weekday_counts)}


def gap_fingerprint(draws: list[Draw], pool: PoolSpec, field: str) -> dict:
    mat = binary_matrix(draws, pool, field)
    p = pool.pick / len(pool.values)
    expected_mean = 1.0 / max(p, 1e-12)
    rows = []
    for idx, value in enumerate(pool.values):
        hit_idx = np.where(mat[:, idx] > 0)[0]
        if len(hit_idx) < 2:
            continue
        gaps = np.diff(hit_idx)
        mean_gap = float(np.mean(gaps))
        var_gap = float(np.var(gaps))
        rows.append(
            {
                "number": value,
                "mean_gap": mean_gap,
                "expected_geometric_mean": expected_mean,
                "variance_ratio": var_gap / max(expected_mean * (1.0 - p) / max(p, 1e-12), 1e-12),
                "mean_lift": mean_gap / max(expected_mean, 1e-12),
            }
        )
    rows.sort(key=lambda row: abs(row["mean_lift"] - 1.0) + abs(row["variance_ratio"] - 1.0), reverse=True)
    return {"expected_geometric_mean": expected_mean, "top_gap_anomalies": rows[:10]}


def randomness_fingerprint(draws: list[Draw], pool: PoolSpec, field: str = "main") -> dict:
    audit = audit_pool_randomness(draws, pool, field)
    mat = binary_matrix(draws, pool, field)
    counts = mat.sum(axis=0).astype(float)
    probs = (counts + 0.5) / (counts.sum() + 0.5 * len(counts))
    entropy = shannon_entropy(probs)
    max_entropy = math.log2(len(pool.values))
    normalized_entropy = entropy / max(max_entropy, 1e-12)
    run_z = np.array([runs_z(mat[:, idx]) for idx in range(mat.shape[1])], dtype=float)
    triple = triple_lift(draws, pool, field)
    lag = lag_overlap(draws, pool, field)
    drift = rolling_drift(draws, pool, field)
    season = seasonality_fingerprint(draws, pool, field)
    gap = gap_fingerprint(draws, pool, field)

    pair_counts = np.array([row["count"] for row in audit["pair_lift"]["top_pairs"]], dtype=float)
    scores = {
        "frequency_bias": min(1.0, audit["frequency"]["max_abs_z"] / 4.0),
        "entropy_compression": min(1.0, max(0.0, 1.0 - normalized_entropy) * 8.0),
        "pair_clustering": min(1.0, max(0.0, audit["pair_lift"]["max_lift"] - 1.0) / 3.0),
        "triple_clustering": min(1.0, max(0.0, triple["max_lift"] - 1.0) / 3.5),
        "temporal_memory": min(1.0, lag["max_abs_lift_delta"]),
        "runs_irregularity": min(1.0, float(np.max(np.abs(run_z))) / 4.0 if len(run_z) else 0.0),
        "distribution_drift": min(1.0, drift["js_divergence"] * 8.0),
        "calendar_effect": min(1.0, max(season["month"]["max_js"], season["weekday"]["max_js"]) * 10.0),
        "gap_anomaly": min(
            1.0,
            max((abs(row["mean_lift"] - 1.0) for row in gap["top_gap_anomalies"]), default=0.0),
        ),
        "graph_concentration": min(1.0, gini(pair_counts) if len(pair_counts) else 0.0),
    }
    dominant = [
        name for name, value in sorted(scores.items(), key=lambda item: item[1], reverse=True) if value >= 0.35
    ][:5]
    if not dominant:
        dominant = ["near_uniform"]
    summary = (
        "The draw history looks closest to uniform randomness under this battery."
        if dominant == ["near_uniform"]
        else "Dominant randomness fingerprints: "
        + ", ".join(dominant)
        + ". These are hypotheses and must survive walk-forward tests."
    )
    return {
        "draws": len(draws),
        "entropy": {
            "shannon_bits": entropy,
            "max_entropy_bits": max_entropy,
            "normalized_entropy": normalized_entropy,
            "entropy_deficit": 1.0 - normalized_entropy,
        },
        "frequency": audit["frequency"],
        "gap": gap,
        "pair_lift": audit["pair_lift"],
        "triple_lift": triple,
        "serial_dependence": lag,
        "runs": {
            "max_abs_runs_z": float(np.max(np.abs(run_z))) if len(run_z) else 0.0,
            "top_runs_anomalies": sorted(
                ({"number": pool.values[idx], "runs_z": float(run_z[idx])} for idx in range(len(run_z))),
                key=lambda row: abs(row["runs_z"]),
                reverse=True,
            )[:10],
        },
        "drift": drift,
        "seasonality": season,
        "graph_structure": {
            "top_pair_lift": audit["pair_lift"]["max_lift"],
            "top_triple_lift": triple["max_lift"],
            "pair_count_gini_top10": gini(pair_counts) if len(pair_counts) else 0.0,
        },
        "randomness_type": {"scores": scores, "dominant_types": dominant},
        "plain_language": {"summary": summary},
    }


def score_vector(draws: list[Draw], spec: LotterySpec, field: str, model: str) -> np.ndarray:
    pool = spec.main
    if pool is None:
        return np.array([])
    values = pool.values
    if model == "uniform":
        return np.zeros(len(values), dtype=float)
    mat = binary_matrix(draws, pool, field)
    mat_counts = Counter(num for draw in draws for num in values_for(draw, field))
    frequency = robust_z(np.array([mat_counts[value] for value in values], dtype=float))
    recent = draws[-min(52, len(draws)) :]
    recent_counts = Counter(num for draw in recent for num in values_for(draw, field))
    recent_frequency = robust_z(np.array([recent_counts[value] for value in values], dtype=float))
    old = draws[: max(1, len(draws) - len(recent))]
    old_counts = Counter(num for draw in old for num in values_for(draw, field))
    old_frequency = robust_z(np.array([old_counts[value] for value in values], dtype=float))
    last_seen = {}
    for idx, draw in enumerate(draws):
        for num in values_for(draw, field):
            last_seen[num] = idx
    gap = robust_z(np.array([len(draws) - 1 - last_seen.get(value, -1) for value in values], dtype=float))
    pair = pair_centrality(draws, pool)
    weights = np.exp(np.linspace(-3.0, 0.0, max(1, len(draws))))
    ewma = robust_z((mat * weights[:, None]).sum(axis=0) / max(float(weights.sum()), 1e-12))
    alpha = 0.7
    bayes = robust_z(
        np.log(
            (np.array([mat_counts[value] for value in values], dtype=float) + alpha)
            / (len(draws) * pool.pick + alpha * len(values))
        )
    )
    drift = robust_z(recent_frequency - old_frequency)
    block_count = min(6, max(2, len(draws) // 20))
    blocks = np.array_split(mat, block_count)
    block_rates = np.vstack([block.mean(axis=0) if len(block) else np.zeros(len(values)) for block in blocks])
    stability = robust_z(-block_rates.std(axis=0))
    if model == "frequency_all":
        return frequency
    if model == "recent_frequency":
        return recent_frequency
    if model == "ewma_recency":
        return ewma
    if model == "bayesian_dirichlet":
        return bayes
    if model == "gap_overdue":
        return gap
    if model == "pair_centrality":
        return pair
    if model == "anti_frequency":
        return -frequency
    if model == "anti_recent":
        return -recent_frequency
    if model == "drift_recent_vs_old":
        return drift
    if model == "stability":
        return stability
    if model == "hybrid_gap_pair":
        return robust_z(0.55 * gap + 0.45 * pair)
    if model == "hybrid_recency_pair":
        return robust_z(0.55 * recent_frequency + 0.45 * pair)
    if model == "ensemble":
        return robust_z(
            0.20 * frequency + 0.20 * recent_frequency + 0.15 * ewma + 0.15 * gap + 0.18 * pair + 0.12 * drift
        )
    if model == "legacy_weighted":
        return number_scores(draws, pool)
    raise ValueError(f"unknown model {model}")


def pick_top(scores: np.ndarray, pool: PoolSpec) -> set[int]:
    ranked = np.argsort(scores)[::-1]
    return set(pool.values[idx] for idx in ranked[: pool.pick])


def summarize_hits(hits: list[int], pick: int, uniform_hits: list[int] | None = None) -> dict:
    arr = np.array(hits, dtype=float)
    if len(arr) == 0:
        return {}
    distribution = {str(k): int(np.sum(arr == k)) for k in range(pick + 1)}
    mean_hits = float(arr.mean())
    uniform_mean = float(np.mean(uniform_hits)) if uniform_hits else mean_hits
    return {
        "mean_hits": mean_hits,
        "any_1_plus": float(np.mean(arr >= 1)),
        "any_2_plus": float(np.mean(arr >= 2)),
        "any_3_plus": float(np.mean(arr >= 3)),
        "hit_distribution": distribution,
        "lift_vs_uniform": float(mean_hits - uniform_mean),
    }


def walk_forward_models(
    draws: list[Draw],
    spec: LotterySpec,
    field: str = "main",
    train_min: int = 80,
    top_k: int | None = None,
) -> dict:
    pool = spec.main
    if pool is None:
        return {}
    if top_k is None:
        top_k = pool.pick
    if len(draws) <= train_min:
        train_min = max(5, min(len(draws) - 1, train_min))
    models = [
        "uniform",
        "frequency_all",
        "recent_frequency",
        "ewma_recency",
        "bayesian_dirichlet",
        "gap_overdue",
        "pair_centrality",
        "anti_frequency",
        "anti_recent",
        "drift_recent_vs_old",
        "stability",
        "hybrid_gap_pair",
        "hybrid_recency_pair",
        "ensemble",
        "legacy_weighted",
    ]
    hits_by_model: dict[str, list[int]] = {model: [] for model in models}

    for idx in range(train_min, len(draws)):
        train = draws[:idx]
        actual = set(values_for(draws[idx], field))
        for model in models:
            scores = score_vector(train, spec, field, model)
            chosen = pick_top(scores, pool)
            hits_by_model[model].append(len(chosen & actual))

    uniform_hits = hits_by_model["uniform"]
    summaries = {model: summarize_hits(hits, top_k, uniform_hits) for model, hits in hits_by_model.items()}
    best_model = max(summaries, key=lambda model: (summaries[model]["mean_hits"], summaries[model]["any_3_plus"]))
    best = summaries[best_model]
    uniform = summaries["uniform"]
    useful = best["mean_hits"] > uniform["mean_hits"] and best["any_2_plus"] >= uniform["any_2_plus"]
    return {
        "field": field,
        "train_min": train_min,
        "test_draws": max(0, len(draws) - train_min),
        "best_model": best_model,
        "models": summaries,
        "verdict": {
            "useful_signal": bool(useful),
            "plain": (
                f"Best model '{best_model}' beat the uniform baseline in walk-forward mean hits."
                if useful
                else "No model beat the uniform baseline strongly enough to claim a reusable edge."
            ),
        },
    }
