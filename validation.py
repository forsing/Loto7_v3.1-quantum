from __future__ import annotations

from collections import Counter

import numpy as np

from math_model import optimize_tickets
from models import Draw, LotterySpec
from randomness import score_vector, walk_forward_models


def nested_ticket_backtest(
    spec: LotterySpec,
    draws: list[Draw],
    columns: int,
    train_min: int = 160,
    seed: int = 39,
    candidate_pool: int = 6000,
    max_test_draws: int | None = None,
) -> dict:
    if len(draws) <= train_min:
        return {
            "test_draws": 0,
            "leakage_guard": "not enough draws for nested validation",
            "selected_models": {},
            "best_main_distribution": {},
        }

    best_hits: list[int] = []
    selected_models: Counter[str] = Counter()
    test_indices = list(range(train_min, len(draws)))
    if max_test_draws is not None and len(test_indices) > max_test_draws:
        test_indices = test_indices[-max_test_draws:]

    for idx in test_indices:
        train = draws[:idx]
        actual = set(draws[idx].main)
        walk = walk_forward_models(train, spec, field="main", train_min=max(30, train_min // 2))
        model = walk["best_model"]
        selected_models[model] += 1
        scores = score_vector(train, spec, "main", model)
        tickets = optimize_tickets(
            spec,
            train,
            columns=columns,
            seed=seed + idx,
            candidate_pool=candidate_pool,
            score_override=scores,
        )
        best_hits.append(max(len(set(ticket.main) & actual) for ticket in tickets))

    arr = np.asarray(best_hits, dtype=int)
    return {
        "test_draws": int(len(arr)),
        "leakage_guard": "tickets generated only from draws before the tested draw",
        "selected_models": dict(selected_models),
        "best_main_mean": float(arr.mean()) if len(arr) else 0.0,
        "any_1_plus": float(np.mean(arr >= 1)) if len(arr) else 0.0,
        "any_2_plus": float(np.mean(arr >= 2)) if len(arr) else 0.0,
        "any_3_plus": float(np.mean(arr >= 3)) if len(arr) else 0.0,
        "best_main_distribution": {str(k): int(np.sum(arr == k)) for k in range(spec.main.pick + 1)},
    }
