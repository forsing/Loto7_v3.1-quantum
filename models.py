from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class PoolSpec:
    name: str
    minimum: int
    maximum: int
    pick: int

    @property
    def values(self) -> list[int]:
        return list(range(self.minimum, self.maximum + 1))


@dataclass(frozen=True)
class LotterySpec:
    name: str
    region: str
    main: PoolSpec
    source_note: str = ""


@dataclass(frozen=True)
class Draw:
    date: date
    main: tuple[int, ...]


@dataclass
class Ticket:
    main: tuple[int, ...]
    source: str = "model"

    def as_dict(self) -> dict:
        return {"main": list(self.main), "source": self.source}
