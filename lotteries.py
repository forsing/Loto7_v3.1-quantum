from __future__ import annotations

from models import LotterySpec, PoolSpec

LOTTERY = LotterySpec(
    name="Loto Serbia 7/39",
    region="Serbia",
    main=PoolSpec("numbers", 1, 39, 7),
    source_note="GHQ CSV Num1..Num7 (loto7hh_4638_k50.csv).",
)
