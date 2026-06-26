from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from config import DEFAULT_CSV
from models import Draw, LotterySpec


def _num_columns(df: pd.DataFrame, pick: int) -> list[str] | None:
    lower = {str(c).strip().lower(): str(c) for c in df.columns}
    cols: list[str] = []
    for i in range(1, pick + 1):
        key = f"num{i}"
        if key not in lower:
            return None
        cols.append(lower[key])
    return cols


def parse_loto739_csv(path_or_url: str | Path, spec: LotterySpec) -> list[Draw]:
    """GHQ: Num1..Num7, bez datuma; red 0 = najstarije, poslednji = najnovije."""
    df = pd.read_csv(path_or_url, encoding="utf-8")
    num_cols = _num_columns(df, spec.main.pick)
    if not num_cols:
        raise ValueError(f"CSV mora imati kolone Num1..Num{spec.main.pick}.")
    base = date(1990, 1, 1)
    draws: list[Draw] = []
    for i, (_, row) in enumerate(df.iterrows()):
        try:
            main = tuple(sorted(int(row[c]) for c in num_cols))
        except (ValueError, TypeError):
            continue
        if len(main) != spec.main.pick:
            continue
        draws.append(Draw(base + timedelta(days=i), main))
    return sorted(draws, key=lambda item: item.date)


def load_draws(spec: LotterySpec, csv_path: str | Path | None = None) -> list[Draw]:
    path = csv_path or DEFAULT_CSV
    return parse_loto739_csv(path, spec)
