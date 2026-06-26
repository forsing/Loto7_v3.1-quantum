from __future__ import annotations

import heapq
import itertools
import math

import numpy as np


def combo_score(combo: tuple[int, ...], index: dict[int, int], scores: np.ndarray) -> float:
    return float(sum(scores[index[num]] for num in combo))


def stream_top_combinations(
    values: list[int],
    pick: int,
    scores: np.ndarray,
    top_k: int = 5000,
) -> dict:
    index = {value: idx for idx, value in enumerate(values)}
    total = math.comb(len(values), pick)
    heap: list[tuple[float, tuple[int, ...]]] = []
    evaluated = 0
    for combo in itertools.combinations(values, pick):
        score = combo_score(combo, index, scores)
        evaluated += 1
        item = (score, combo)
        if len(heap) < top_k:
            heapq.heappush(heap, item)
        elif score > heap[0][0]:
            heapq.heapreplace(heap, item)
    top = sorted(heap, key=lambda item: item[0], reverse=True)
    return {
        "total_combinations": total,
        "evaluated_combinations": evaluated,
        "top": [{"score": score, "combo": list(combo)} for score, combo in top],
    }
