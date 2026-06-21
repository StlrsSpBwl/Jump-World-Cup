from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


DEFAULT_COMPETITION_STRENGTH: dict[str, float] = {
    "uefa champions league": 1.08,
    "uefa europa league": 1.02,
    "england premier league": 1.06,
    "premier league": 1.06,
    "spain laliga": 1.05,
    "la liga": 1.05,
    "germany bundesliga": 1.04,
    "italy serie a": 1.04,
    "france ligue 1": 1.01,
    "netherlands eredivisie": 0.96,
    "portugal primeira liga": 0.95,
    "belgium pro league": 0.91,
    "mls": 0.88,
    "saudi pro league": 0.86,
    "efl championship": 0.86,
    "south africa premier division": 0.80,
    "czech first league": 0.84,
    "international uefa": 1.00,
    "international conmebol": 0.99,
    "international concacaf": 0.92,
    "international caf": 0.90,
    "international afc": 0.89,
    "international ofc": 0.78,
}


@dataclass(frozen=True)
class StrengthAdjustment:
    competition: str
    multiplier: float
    source: str


class CompetitionStrengthModel:
    """Canonical competition/opposition-strength model shared by data layers."""

    def __init__(self, strengths: dict[str, float], source: str) -> None:
        self._strengths = {_normalize(key): float(value) for key, value in strengths.items()}
        self.source = source

    @classmethod
    def load(cls, path: str | Path | None = None) -> "CompetitionStrengthModel":
        strengths = dict(DEFAULT_COMPETITION_STRENGTH)
        source = "fixed_defaults"
        if path is not None:
            candidate = Path(path)
            if candidate.exists():
                with candidate.open(newline="", encoding="utf-8-sig") as handle:
                    for row in csv.DictReader(handle):
                        name = row.get("competition") or row.get("name")
                        value = row.get("strength_multiplier") or row.get("multiplier")
                        if name and value not in {None, ""}:
                            strengths[name] = float(value)
                source = str(candidate)
        return cls(strengths, source)

    def adjustment(self, competition: str | None) -> StrengthAdjustment:
        key = _normalize(competition or "")
        if key in self._strengths:
            return StrengthAdjustment(competition or "", self._strengths[key], self.source)
        return StrengthAdjustment(competition or "", 1.0, "neutral_fallback")

    def adjusted_rate(self, value: float | None, competition: str | None) -> float | None:
        if value is None:
            return None
        adjustment = self.adjustment(competition)
        return float(value) * adjustment.multiplier


def _normalize(value: str) -> str:
    return " ".join(value.strip().casefold().replace("-", " ").split())
