"""Gaussian copula — zajednička raspodela 7 brojeva (sortirana sedmerka)."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import multivariate_normal, norm

from models import Draw


class Loto739Copula:
    """P(combo) = marginala po poziciji × copula(z1..z7); skor na nivou cele sedmerke."""

    def __init__(self) -> None:
        self.n_draws = 0
        self.pick = 7
        self.correlation: np.ndarray | None = None
        self.position_samples: np.ndarray | None = None
        self.combo_counts: dict[str, int] = {}

    def fit(self, draws: list[Draw]) -> None:
        rows = np.array([tuple(sorted(d.main)) for d in draws], dtype=float)
        if rows.size == 0:
            raise ValueError("Nema izvlačenja za copula trening.")
        n, pick = rows.shape
        self.n_draws = int(n)
        self.pick = int(pick)

        uniform = np.zeros_like(rows)
        for col in range(pick):
            values = rows[:, col]
            ranks = np.argsort(np.argsort(values)) + 1
            uniform[:, col] = (ranks - 0.5) / n
        uniform = np.clip(uniform, 1e-6, 1.0 - 1e-6)
        z = np.clip(norm.ppf(uniform), -6.0, 6.0)

        corr = np.corrcoef(z.T)
        self.correlation = corr + 1e-5 * np.eye(pick)
        self.position_samples = rows

        counts = Counter(tuple(sorted(d.main)) for d in draws)
        self.combo_counts = {",".join(str(x) for x in key): int(v) for key, v in counts.items()}

    def joint_log_score(self, combo: tuple[int, ...]) -> float:
        if self.correlation is None or self.position_samples is None or self.n_draws == 0:
            return 0.0

        combo = tuple(sorted(int(x) for x in combo))
        if len(combo) != self.pick:
            return float("-inf")

        key = ",".join(str(x) for x in combo)
        emp_count = self.combo_counts.get(key, 0)
        emp_log = math.log((emp_count + 0.5) / (self.n_draws + 0.5 * max(len(self.combo_counts), 1)))

        z = np.zeros(self.pick, dtype=float)
        for col, num in enumerate(combo):
            col_vals = self.position_samples[:, col]
            u = (float(np.sum(col_vals <= num)) - 0.5) / self.n_draws
            u = float(np.clip(u, 1e-6, 1.0 - 1e-6))
            z[col] = float(norm.ppf(u))
        z = np.clip(z, -6.0, 6.0)

        mvn_log = float(
            multivariate_normal.logpdf(
                z,
                mean=np.zeros(self.pick),
                cov=self.correlation,
                allow_singular=True,
            )
        )
        marg_log = float(np.sum(norm.logpdf(z)))
        copula_log = mvn_log - marg_log
        return emp_log + copula_log

    def to_dict(self) -> dict:
        return {
            "n_draws": self.n_draws,
            "pick": self.pick,
            "correlation": self.correlation.tolist() if self.correlation is not None else None,
            "position_samples": self.position_samples.tolist() if self.position_samples is not None else None,
            "combo_counts": self.combo_counts,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> Loto739Copula:
        model = cls()
        model.n_draws = int(payload.get("n_draws", 0))
        model.pick = int(payload.get("pick", 7))
        corr = payload.get("correlation")
        samples = payload.get("position_samples")
        model.correlation = np.asarray(corr, dtype=float) if corr is not None else None
        model.position_samples = np.asarray(samples, dtype=float) if samples is not None else None
        model.combo_counts = dict(payload.get("combo_counts") or {})
        return model

    def save(self, path: Path) -> Path:
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> Loto739Copula | None:
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(payload)
