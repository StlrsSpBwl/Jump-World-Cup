"""Player-prop layer — the lineup/sub-discount edge, as code.

The contest's RBP edge has come almost entirely from one thing: confirmed
benched stars that the crowd (and previews) still price as starters. This module
turns that read into a reproducible calculation instead of hand reasoning.

For a confirmed substitute, a prop is decomposed:

    P(prop) = P(player appears) * P(prop | expected sub minutes)

where the conditional uses the player's per-90 rate scaled to the cameo minutes.
A confirmed starter uses the full-match rate; a confirmed-out player is 0.

Per-90 rates come from `player_profiles` when available, else a role prior.
"""

from __future__ import annotations

import re

import numpy as np

# Event -> which per-90 rate field drives it.
EVENT_RATE_FIELD = {
    "goals": "goals_per90",
    "score": "goals_per90",
    "goal_or_assist": "goal_assist_per90",
    "score_or_assist": "goal_assist_per90",
    "shots_on_target": "sot_per90",
    "second_half_shots_on_target": "sot_per90",
}

# Role priors for per-90 rates when a player has no profile data.
ROLE_RATES = {
    "striker":      {"goals_per90": 0.48, "goal_assist_per90": 0.62, "sot_per90": 1.45},
    "forward":      {"goals_per90": 0.40, "goal_assist_per90": 0.58, "sot_per90": 1.35},
    "winger":       {"goals_per90": 0.30, "goal_assist_per90": 0.52, "sot_per90": 1.20},
    "attacking_mid":{"goals_per90": 0.28, "goal_assist_per90": 0.50, "sot_per90": 1.10},
    "midfielder":   {"goals_per90": 0.12, "goal_assist_per90": 0.26, "sot_per90": 0.60},
    "deep_mid":     {"goals_per90": 0.05, "goal_assist_per90": 0.14, "sot_per90": 0.35},
    "defender":     {"goals_per90": 0.05, "goal_assist_per90": 0.12, "sot_per90": 0.25},
}

# P(a confirmed sub actually gets onto the pitch), by role.
SUB_APPEARANCE = {
    "striker": 0.62, "forward": 0.60, "winger": 0.58, "attacking_mid": 0.55,
    "midfielder": 0.45, "deep_mid": 0.40, "defender": 0.30,
}

DEFAULT_SUB_MINUTES = 25.0
DEFAULT_START_MINUTES = 82.0


def parse_player_event(question: str) -> str | None:
    """Map a free-text player question to an event key."""
    q = question.lower()
    if "score or assist" in q or ("assist" in q and "score" in q):
        return "goal_or_assist"
    if "shot on target" in q or "shots on target" in q:
        return "second_half_shots_on_target" if "second half" in q else "shots_on_target"
    if "score" in q or "goal" in q:
        return "goals"
    return None


def _rate(event: str, role: str, per90_override: float | None) -> float:
    if per90_override is not None:
        return float(per90_override)
    field = EVENT_RATE_FIELD.get(event, "sot_per90")
    return ROLE_RATES.get(role, ROLE_RATES["midfielder"])[field]


def _event_given_minutes(event: str, rate_per90: float, minutes: float) -> float:
    """P(at least one event in `minutes`), constant-hazard on the per-90 rate."""
    span = minutes
    if event == "second_half_shots_on_target":
        # only the second-half window counts; cap the span at a half
        span = min(minutes, 45.0)
    lam = rate_per90 * span / 90.0
    return float(1.0 - np.exp(-lam))


def player_prop_probability(
    event: str,
    role: str,
    status: str,
    *,
    per90: float | None = None,
    sub_minutes: float = DEFAULT_SUB_MINUTES,
    start_minutes: float = DEFAULT_START_MINUTES,
    appearance_prob: float | None = None,
) -> dict:
    """Probability of a player prop given confirmed lineup status.

    status: 'starter' | 'sub'/'bench' | 'out'
    Returns {'probability', 'basis', and components}.
    """
    status = status.strip().lower()
    rate = _rate(event, role, per90)

    if status in {"out", "unavailable"}:
        return {"probability": 0.0, "basis": "confirmed_out", "rate_per90": rate}

    if status in {"sub", "bench", "substitute"}:
        appear = appearance_prob if appearance_prob is not None else SUB_APPEARANCE.get(role, 0.45)
        cond = _event_given_minutes(event, rate, sub_minutes)
        p = float(appear * cond)
        return {
            "probability": p,
            "basis": "sub_discount",
            "rate_per90": rate,
            "appearance_prob": appear,
            "sub_minutes": sub_minutes,
            "p_event_given_minutes": cond,
        }

    # starter
    p = _event_given_minutes(event, rate, start_minutes)
    return {"probability": float(p), "basis": "starter", "rate_per90": rate,
            "start_minutes": start_minutes}
