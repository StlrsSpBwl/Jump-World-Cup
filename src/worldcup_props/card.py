"""Phase 3 — orchestration: a full card from one call.

`forecast_card` ties the closed-form goal engine (Phase 1), the count-prop
evaluators with validated corrections, and the player-prop sub-discount layer
(Phase 2) together. It routes each free-text question to the right evaluator and
returns a probability plus a `basis` label so every row is auditable: which
engine produced it, or whether it fell back to a parameterized base rate for the
genuinely unmodeled types (penalty/red, "a sub scores", hydration-timing cards).

Inputs the caller supplies: the market lambdas (from odds), the DB (for count
rates), confirmed lineup statuses, and the question list.
"""

from __future__ import annotations

import re
from pathlib import Path

from .closed_form import (
    advance_probability,
    apply_favorite_dominance,
    count_more_than,
    count_threshold,
    count_total_threshold,
    goal_before_minute,
    goal_props,
    team_count_rate,
)
from .player_props import parse_player_event, player_prop_probability

STATS = ("fouls", "corners", "offsides", "cards", "shots_on_target")
STAT_WORDS = {
    "foul": "fouls", "corner": "corners", "offside": "offsides",
    "card": "cards", "shot on target": "shots_on_target", "shots on target": "shots_on_target",
}
# Parameterized base rates for genuinely unmodeled question types (no closed form).
BASE_RATES = {
    "penalty_or_red": 0.40,
    "penalty_only": 0.30,
    "sub_scores": 0.21,
    "any_brace": 0.18,
    "offside_before_break": 0.52,
    "card_after_break": 0.63,
    "goal_before_break": None,  # computed from lambda
}


def _num_threshold(q: str) -> int | None:
    m = re.search(r"(\d+)\s*(?:or more|\+|or more total)", q)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)\s*or fewer", q)
    if m:
        return int(m.group(1))
    return None


def _stat(q: str) -> str | None:
    for word, stat in STAT_WORDS.items():
        if word in q:
            return stat
    return None


def _period(q: str) -> str:
    ql = q.lower()
    if "second half" in ql or "2nd half" in ql:
        return "second_half"
    if "first half" in ql or "halftime" in ql or "1st half" in ql:
        return "first_half"
    return "full"


def _subject_team(q: str, home: str, away: str) -> str | None:
    """Which named team is the grammatical subject (appears first / before 'more')."""
    ih, ia = q.find(home), q.find(away)
    if ih == -1 and ia == -1:
        return None
    if ih == -1:
        return away
    if ia == -1:
        return home
    return home if ih < ia else away


def _route(q, home, away, g, lam_h, lam_a, db, lineup_status, roles):
    ql = q.lower()
    fav_win = max(g["home_win"], g["away_win"])

    # --- player props (highest-value: the sub-discount edge) ---
    event = parse_player_event(q)
    if event and lineup_status is not None:
        for name, status in lineup_status.items():
            if name.lower() in ql:
                role = (roles or {}).get(name, "forward")
                r = player_prop_probability(event, role, status)
                return r["probability"], f"player:{r['basis']}:{event}"

    # --- knockout / outcome ---
    if "advance" in ql:
        side = "home" if _subject_team(q, home, away) == home else "away"
        return advance_probability(g["home_win"], g["draw"], g["away_win"], side), "advance"
    if "win the match" in ql:
        return (g["home_win"] if _subject_team(q, home, away) == home else g["away_win"]), "win"
    if "end in a tie" in ql or ("halftime" in ql and "tied" in ql) or ("at halftime" in ql and "tie" in ql):
        return (g["ht_draw"] if "halftime" in ql else g["draw"]), "draw"
    if "ahead at halftime" in ql or ("winning" in ql and "halftime" in ql):
        sub = _subject_team(q, home, away)
        return (g["ht_home_lead"] if sub == home else g["ht_away_lead"]), "ht_lead"

    # --- totals / BTTS / goals timing ---
    if "both teams score" in ql:
        return (g["btts_and_3plus"] if "3 or more" in ql or "3+" in ql else g["btts"]), "btts"
    if "2 or fewer total goals" in ql:
        return g["under_2_5"], "under_2_5"
    if "3 or more total goals" in ql or "3+ total" in ql:
        return g["three_or_more"], "three_plus"
    if "second half" in ql and "more" in ql and "first half" in ql:
        return g["second_half_more_goals"], "2h>1h"
    if "second half" in ql and ("2 or more" in ql and "goal" in ql):
        return g["second_half_2plus"], "2h_2plus"
    if "score in the second half" in ql or "score in the 2nd half" in ql:
        sub = _subject_team(q, home, away)
        return (g["home_scores_2h"] if sub == home else g["away_scores_2h"]), "team_scores_2h"
    if "goal" in ql and "before the first hydration" in ql:
        return goal_before_minute(lam_h, lam_a, 23.0), "goal_before_break"

    # --- count props ---
    stat = _stat(q)
    period = _period(q)
    if stat and db:
        k = _num_threshold(q)
        # "X+ total <stat>" — both teams combined, no subject needed.
        if "total" in ql and k is not None:
            ra = team_count_rate(db, home, stat, period)
            rb = team_count_rate(db, away, stat, period)
            if ra and rb:
                return count_total_threshold(ra, rb, k), f"count_total:{stat}:{period}"
        sub = _subject_team(q, home, away)
        if sub:
            opp = away if sub == home else home
            ra = team_count_rate(db, sub, stat, period)
            rb = team_count_rate(db, opp, stat, period)
            if "more" in ql and "than" in ql and ra and rb:  # comparison
                p = count_more_than(ra, rb)
                # validated favorite corrections: SOT/corners, subject = favorite only
                if stat in ("shots_on_target", "corners"):
                    kind = "sot" if stat == "shots_on_target" else "corners"
                    p = apply_favorite_dominance(p, sub == _favorite_side(g, home, away), fav_win, kind)
                return p, f"count_more_than:{stat}:{period}"
            if k is not None and ra:  # single-team threshold
                return count_threshold(ra, k), f"count_threshold:{stat}:{period}"

    # --- parameterized base rates (unmodeled types) ---
    if "penalty" in ql and "red card" in ql:
        return BASE_RATES["penalty_or_red"], "base:penalty_or_red"
    if "penalty" in ql:
        return BASE_RATES["penalty_only"], "base:penalty"
    if "substitute score" in ql or "a sub" in ql:
        return BASE_RATES["sub_scores"], "base:sub_scores"
    if "more than 1 goal" in ql or "2 or more goal" in ql:
        return BASE_RATES["any_brace"], "base:brace"
    if "offside" in ql and "hydration" in ql:
        return BASE_RATES["offside_before_break"], "base:offside_before_break"
    if "card" in ql and "hydration" in ql:
        return BASE_RATES["card_after_break"], "base:card_after_break"
    return None, "unrouted"


def _favorite_side(g, home, away):
    return home if g["home_win"] >= g["away_win"] else away


def forecast_card(
    home: str,
    away: str,
    lambda_home: float,
    lambda_away: float,
    questions: list[str],
    *,
    db: str | Path | None = None,
    lineup_status: dict[str, str] | None = None,
    roles: dict[str, str] | None = None,
    rho: float = -0.08,
) -> list[dict]:
    """Route every question to an engine and return [{question, probability, basis}]."""
    g = goal_props(lambda_home, lambda_away, rho)
    out = []
    for q in questions:
        prob, basis = _route(q, home, away, g, lambda_home, lambda_away, db, lineup_status, roles)
        out.append({"question": q, "probability": prob, "basis": basis})
    return out
