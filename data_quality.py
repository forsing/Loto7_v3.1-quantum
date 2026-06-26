from __future__ import annotations

from collections import Counter

from models import Draw, LotterySpec


def validate_draw_history(draws: list[Draw], spec: LotterySpec) -> dict:
    date_counts = Counter(str(draw.date) for draw in draws)
    duplicate_dates = sorted(date for date, count in date_counts.items() if count > 1)
    range_errors = []
    size_errors = []
    duplicate_number_errors = []
    for draw in draws:
        if len(draw.main) != spec.main.pick:
            size_errors.append(str(draw.date))
        if len(set(draw.main)) != len(draw.main):
            duplicate_number_errors.append(str(draw.date))
        for num in draw.main:
            if num < spec.main.minimum or num > spec.main.maximum:
                range_errors.append({"date": str(draw.date), "number": num})
    usable = (
        not duplicate_dates
        and not range_errors
        and not size_errors
        and not duplicate_number_errors
        and len(draws) >= 30
    )
    return {
        "draws": len(draws),
        "first_draw": str(draws[0].date) if draws else None,
        "last_draw": str(draws[-1].date) if draws else None,
        "duplicate_dates": duplicate_dates,
        "range_errors": range_errors,
        "size_errors": size_errors,
        "duplicate_number_errors": duplicate_number_errors,
        "usable": usable,
        "minimum_required_draws": 30,
        "minimum_recommended_draws": 80,
    }
