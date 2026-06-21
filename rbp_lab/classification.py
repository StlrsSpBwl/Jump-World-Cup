from __future__ import annotations

import re

from .models import Category


# Ordered from specific to broad so "shots on target" is not classified as "shots".
CATEGORY_PATTERNS: list[tuple[Category, tuple[str, ...]]] = [
    (
        Category.COMPOUND,
        (
            r"\band\b.*\b(goal|score|btts|corner|card|win)\b",
            r"\b(btts|both teams to score)\b.*\b(over|[3-9]\+)\b",
        ),
    ),
    (
        Category.PERIOD_SPLIT,
        (r"\b(first|second|1st|2nd)\s*half\b", r"\bhalf[- ]?time\b", r"\bby halftime\b"),
    ),
    (Category.PENALTY, (r"\bpenalt(y|ies)\b", r"\bspot kick\b")),
    (
        Category.PLAYER_SCORER,
        (r"\b(anytime|first|last)\s+(goal)?scorer\b", r"\bto score\b", r"\bplayer goal\b"),
    ),
    (Category.SHOTS_ON_TARGET, (r"\bshots? on target\b", r"\bsot\b")),
    (Category.SHOTS, (r"\bshots?\b", r"\battempts?\b")),
    (Category.CORNERS, (r"\bcorners?\b",)),
    (Category.CARDS_BOOKINGS, (r"\bcards?\b", r"\bbookings?\b", r"\byellow\b", r"\bred card\b")),
    (Category.FOULS, (r"\bfouls?\b",)),
    (Category.BTTS, (r"\bbtts\b", r"\bboth teams (to )?score\b")),
    (
        Category.SUPREMACY_HANDICAP,
        (r"\bhandicap\b", r"\bsupremacy\b", r"\bwin by\b", r"\bmargin\b", r"[+-]\d+(\.\d+)?"),
    ),
    (
        Category.GOALS_TOTALS,
        (r"\b(total|over|under)\b.*\bgoals?\b", r"\bgoals?\b.*\b(over|under|total)\b", r"\d+\+ goals?"),
    ),
    (
        Category.MATCH_RESULT,
        (r"\bmatch result\b", r"\b(to )?win\b", r"\bdraw\b", r"\bdouble chance\b"),
    ),
]


def classify_question(text: str) -> str:
    normalized = " ".join(str(text).casefold().split())
    for category, patterns in CATEGORY_PATTERNS:
        if any(re.search(pattern, normalized) for pattern in patterns):
            return category.value
    return Category.OTHER.value

