from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class Stat(str, Enum):
    FOULS = "fouls"
    CORNERS = "corners"
    OFFSIDES = "offsides"
    SHOTS_ON_TARGET = "shots_on_target"
    CARDS = "cards"


class QuestionType(str, Enum):
    MORE_THAN = "more_than"
    HALFTIME_MORE_THAN = "halftime_more_than"
    SECOND_HALF_MORE_THAN = "second_half_more_than"
    THRESHOLD = "threshold"


class TieHandling(str, Enum):
    STRICT = "strict"
    HALF = "half"
    HOME = "home"
    AWAY = "away"


@dataclass(frozen=True)
class Question:
    home: str
    away: str
    stat: Stat
    question_type: QuestionType
    k: int | None = None
    referee: str | None = None
    competition_type: str = "world_cup"
    home_elo: float | None = None
    away_elo: float | None = None
    neutral: bool = True

    def __post_init__(self) -> None:
        if self.question_type == QuestionType.THRESHOLD and self.k is None:
            raise ValueError("threshold questions require k")
        if self.k is not None and self.k < 0:
            raise ValueError("k must be non-negative")

    @property
    def key(self) -> str:
        parts = [
            self.home,
            self.away,
            self.stat.value,
            self.question_type.value,
            "" if self.k is None else str(self.k),
        ]
        return "|".join(part.strip().casefold() for part in parts)


@dataclass
class Forecast:
    probability: float
    model_probability: float
    model_only_probability: float
    market_probability: float | None
    p_home_more: float | None
    p_tie: float | None
    p_away_more: float | None
    interval_80: tuple[float, float]
    effective_sample_size_home: float
    effective_sample_size_away: float
    raw_model_probability: float
    metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "probability": self.probability,
            "model_probability": self.model_probability,
            "model_only_probability": self.model_only_probability,
            "market_probability": self.market_probability,
            "p_home_more": self.p_home_more,
            "p_tie": self.p_tie,
            "p_away_more": self.p_away_more,
            "interval_80": list(self.interval_80),
            "effective_sample_size_home": self.effective_sample_size_home,
            "effective_sample_size_away": self.effective_sample_size_away,
            "raw_model_probability": self.raw_model_probability,
            "metadata": self.metadata,
        }
