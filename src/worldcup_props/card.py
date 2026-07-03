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

import math
import re
from pathlib import Path

from scipy.stats import poisson

from .closed_form import (
    advance_probability,
    apply_favorite_dominance,
    count_more_than,
    count_threshold,
    count_total_threshold,
    goal_after_minute,
    goal_before_minute,
    goal_props,
    half_split_lambdas,
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
    "any_player_2plus_sot": 0.85,  # ~10 SOT across few shooters -> very likely someone gets 2+
    "red_card": 0.22,              # a red card shown (knockout base rate)
    "fh_stoppage_goal": 0.11,      # a goal in first-half stoppage time (~3' window, late-half tilt)
    "goal_before_break": None,  # computed from lambda
}


def _team_total_goals(lam: float, k: int) -> float:
    """P(a team scores >= k goals), Poisson on its goal expectancy."""
    return float(1.0 - poisson.cdf(k - 1, lam))


def _team_both_halves(lam_h, lam_a, subject_is_home):
    """P(team scores in BOTH halves) = P(1H goal) * P(2H goal), per-half lambdas."""
    l1h, l1a, l2h, l2a = half_split_lambdas(lam_h, lam_a)
    l1, l2 = (l1h, l2h) if subject_is_home else (l1a, l2a)
    return float((1.0 - math.exp(-l1)) * (1.0 - math.exp(-l2)))


def _team_scores_first(lam_h, lam_a, subject_is_home):
    """P(team scores the first goal) = goal-rate share * P(>=1 goal in the match)."""
    total = lam_h + lam_a
    if total <= 0:
        return 0.0
    share = (lam_h if subject_is_home else lam_a) / total
    return float(share * (1.0 - math.exp(-total)))


def _half_total_goals(lam_h, lam_a, first_half: bool, k: int):
    """P(a given half produces >= k goals), Poisson on that half's combined lambda."""
    l1h, l1a, l2h, l2a = half_split_lambdas(lam_h, lam_a)
    lam = (l1h + l1a) if first_half else (l2h + l2a)
    return float(1.0 - poisson.cdf(k - 1, lam))


# Stats the current-tournament (Footiqo) source covers; offsides/fouls fall back to DB.
_FOOTIQO_STATS = {"shots_on_target", "corners", "cards", "shots"}


def _team_cr(footiqo, db, team, stat, period):
    """Team count rate: shrunk current-tournament (Footiqo) when available, else stale DB."""
    if footiqo is not None and stat in _FOOTIQO_STATS and period == "full":
        cr = footiqo.team_rate(team, stat)
        if cr is not None:
            return cr
    return team_count_rate(db, team, stat, period) if db else None


def _total_prob(footiqo, db, stat, k, home, away, period):
    """P(total stat >= k): Footiqo empirical distribution (banked, n≈85) if available, else DB."""
    if footiqo is not None and stat in _FOOTIQO_STATS and period == "full":
        p = footiqo.total_rate(stat, k)
        if p is not None:
            return p, f"footiqo_total:{stat}"
    ra = team_count_rate(db, home, stat, period) if db else None
    rb = team_count_rate(db, away, stat, period) if db else None
    if ra and rb:
        return count_total_threshold(ra, rb, k), f"count_total:{stat}:{period}"
    return None, None


def _both_teams_card(footiqo, db, home, away, period="full"):
    """P(both teams receive >= 1 card) = P(home>=1) * P(away>=1) from card rates."""
    rh = _team_cr(footiqo, db, home, "cards", period)
    ra = _team_cr(footiqo, db, away, "cards", period)
    if not rh or not ra:
        return None
    return float(count_threshold(rh, 1) * count_threshold(ra, 1))


def _num_threshold(q: str) -> int | None:
    m = re.search(r"(\d+)\s*(?:or more|\+|or more total)", q)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)\s*or fewer", q)
    if m:
        return int(m.group(1))
    return None


def _stat(q: str) -> str | None:
    ql = q.lower()
    if "shot on target" in ql or "shots on target" in ql:
        return "shots_on_target"
    if "shot" in ql:  # "total shots (on and off target)" — after the SOT check
        return "shots"
    for word, stat in STAT_WORDS.items():
        if word in ql:
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


def _route(q, home, away, g, lam_h, lam_a, db, lineup_status, roles, footiqo=None):
    ql = q.lower()
    fav_win = max(g["home_win"], g["away_win"])

    # --- player props (highest-value: the sub-discount edge) ---
    event = parse_player_event(q)
    if event and lineup_status is not None:
        for name, status in lineup_status.items():
            if name.lower() in ql:
                role = (roles or {}).get(name, "forward")
                k = _num_threshold(q) or 1
                r = player_prop_probability(event, role, status, k=k)
                return r["probability"], f"player:{r['basis']}:{event}"

    # --- knockout / outcome ---
    if "advance" in ql:
        side = "home" if _subject_team(q, home, away) == home else "away"
        return advance_probability(g["home_win"], g["draw"], g["away_win"], side), "advance"
    if "ahead at halftime" in ql or ("winning" in ql and "halftime" in ql):
        sub = _subject_team(q, home, away)
        return (g["ht_home_lead"] if sub == home else g["ht_away_lead"]), "ht_lead"
    if re.search(r"\bwin\b", ql) and "halftime" not in ql and _subject_team(q, home, away):
        return (g["home_win"] if _subject_team(q, home, away) == home else g["away_win"]), "win"
    if "end in a tie" in ql or ("halftime" in ql and "tied" in ql) or ("at halftime" in ql and "tie" in ql):
        return (g["ht_draw"] if "halftime" in ql else g["draw"]), "draw"

    # --- totals / BTTS / goals timing ---
    if "both teams" in ql and "card" in ql:  # both teams >=1 card (guard before btts)
        p = _both_teams_card(footiqo, db, home, away)
        if p is not None:
            return p, "both_teams_card"
    if "both teams score" in ql:
        return (g["btts_and_3plus"] if "3 or more" in ql or "3+" in ql else g["btts"]), "btts"
    if "2 or fewer total goals" in ql:
        return g["under_2_5"], "under_2_5"
    if "3 or more total goals" in ql or "3+ total" in ql:
        return g["three_or_more"], "three_plus"
    if "second half" in ql and "more" in ql and "first half" in ql:
        return g["second_half_more_goals"], "2h>1h"
    # half-total goals: "[first/second] half produce N+ goals" (no team) — before the brace rule
    m_hh = re.search(r"(\d+)\s*or more goals", ql)
    if m_hh and ("first half" in ql or "second half" in ql) and not _subject_team(q, home, away):
        return _half_total_goals(lam_h, lam_a, "first half" in ql, int(m_hh.group(1))), "half_total_goals"
    if "second half" in ql and ("2 or more" in ql and "goal" in ql):
        return g["second_half_2plus"], "2h_2plus"
    if "score in the second half" in ql or "score in the 2nd half" in ql:
        sub = _subject_team(q, home, away)
        return (g["home_scores_2h"] if sub == home else g["away_scores_2h"]), "team_scores_2h"
    if "goal" in ql and "before the first hydration" in ql:
        return goal_before_minute(lam_h, lam_a, 23.0), "goal_before_break"
    if "goal" in ql and "after the second hydration" in ql:
        return goal_after_minute(lam_h, lam_a, 67.0), "goal_after_break"
    if "goal" in ql and ("first-half stoppage" in ql or "first half stoppage" in ql):
        return BASE_RATES["fh_stoppage_goal"], "base:fh_stoppage_goal"

    # --- team-level goal props (MUST precede the any-player brace fallback) ---
    if "both halves" in ql and "score" in ql:
        sub = _subject_team(q, home, away)
        if sub:
            return _team_both_halves(lam_h, lam_a, sub == home), "team_both_halves"
    m_tg = re.search(r"(\d+)\s*or more goals", ql)
    if m_tg and "total" not in ql:
        sub = _subject_team(q, home, away)
        if sub:
            lam = lam_h if sub == home else lam_a
            return _team_total_goals(lam, int(m_tg.group(1))), "team_total_goals"
    if "more than 1 goal" in ql and _subject_team(q, home, away):  # team brace, named team
        sub = _subject_team(q, home, away)
        return _team_total_goals(lam_h if sub == home else lam_a, 2), "team_total_goals"
    if "first goal" in ql or "open the scoring" in ql or "score first" in ql:
        sub = _subject_team(q, home, away)
        if sub:
            return _team_scores_first(lam_h, lam_a, sub == home), "team_scores_first"

    # --- count props ---
    stat = _stat(q)
    period = _period(q)
    if stat and (db or footiqo):
        k = _num_threshold(q)
        # match-total stat: "X+ total <stat>" OR "will there be X+ <stat>" (no team, no player)
        is_match_total = "total" in ql or (
            ("there be" in ql or "there are" in ql) and "player" not in ql
        )
        if is_match_total and k is not None:
            p, basis = _total_prob(footiqo, db, stat, k, home, away, period)
            if p is not None:
                return p, basis
        sub = _subject_team(q, home, away)
        if sub:
            opp = away if sub == home else home
            ra = _team_cr(footiqo, db, sub, stat, period)
            rb = _team_cr(footiqo, db, opp, stat, period)
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
    if "red card" in ql:
        return BASE_RATES["red_card"], "base:red_card"
    if "substitute score" in ql or "a sub" in ql:
        return BASE_RATES["sub_scores"], "base:sub_scores"
    if ("shot on target" in ql or "shots on target" in ql) and _num_threshold(q) and _num_threshold(q) >= 2:
        return BASE_RATES["any_player_2plus_sot"], "base:any_player_2plus_sot"
    if "more than 1 goal" in ql or "2 or more goal" in ql:
        return BASE_RATES["any_brace"], "base:brace"
    if "offside" in ql and "hydration" in ql:
        return BASE_RATES["offside_before_break"], "base:offside_before_break"
    if "card" in ql and "hydration" in ql:
        return BASE_RATES["card_after_break"], "base:card_after_break"
    return None, "unrouted"


def _favorite_side(g, home, away):
    return home if g["home_win"] >= g["away_win"] else away


def _match_team(name, home, away):
    """Resolve an LLM-provided team name to home/away (case/substring tolerant)."""
    if not name:
        return None
    n = name.lower()
    if n in home.lower() or home.lower() in n:
        return home
    if n in away.lower() or away.lower() in n:
        return away
    return None


# fixed base rates by key (goal_before_break is computed, so excluded here)
_FIXED_BASE = {k: v for k, v in BASE_RATES.items() if isinstance(v, (int, float))}


def route_spec(spec, home, away, g, lam_h, lam_a, db, lineup_status, roles, footiqo=None):
    """Map a structured QuestionSpec (from the LLM parser) to (probability, basis)."""
    kind = spec.kind
    fav_win = max(g["home_win"], g["away_win"])
    sub = _match_team(spec.subject_team, home, away)

    if kind == "player_prop" and spec.event and spec.player:
        status = "starter"
        if lineup_status:
            for name, st in lineup_status.items():
                if name.lower() in spec.player.lower() or spec.player.lower() in name.lower():
                    status = st
                    break
        role = (roles or {}).get(spec.player, "forward")
        r = player_prop_probability(spec.event, role, status, k=spec.threshold or 1)
        return r["probability"], f"player:{r['basis']}:{spec.event}"
    if kind == "advance":
        side = "home" if sub == home else "away"
        return advance_probability(g["home_win"], g["draw"], g["away_win"], side), "advance"
    if kind == "win":
        return (g["home_win"] if sub == home else g["away_win"]), "win"
    if kind == "draw":
        return g["draw"], "draw"
    if kind == "ht_lead":
        return (g["ht_home_lead"] if sub == home else g["ht_away_lead"]), "ht_lead"
    if kind == "btts":
        return g["btts"], "btts"
    if kind == "btts_and_3plus":
        return g["btts_and_3plus"], "btts"
    if kind == "total_under":
        return g["under_2_5"], "under_2_5"
    if kind == "total_over":
        return g["three_or_more"], "three_plus"
    if kind == "second_half_more_goals":
        return g["second_half_more_goals"], "2h>1h"
    if kind == "second_half_2plus":
        return g["second_half_2plus"], "2h_2plus"
    if kind == "team_scores_2h":
        return (g["home_scores_2h"] if sub == home else g["away_scores_2h"]), "team_scores_2h"
    if kind == "goal_before_break":
        return goal_before_minute(lam_h, lam_a, 23.0), "goal_before_break"
    if kind == "team_total_goals" and spec.threshold:
        lam = lam_h if sub == home else lam_a
        return _team_total_goals(lam, spec.threshold), "team_total_goals"
    if kind == "team_both_halves":
        return _team_both_halves(lam_h, lam_a, sub == home), "team_both_halves"
    if kind == "first_goal":
        return _team_scores_first(lam_h, lam_a, sub == home), "team_scores_first"
    if kind == "half_total_goals" and spec.threshold:
        return _half_total_goals(lam_h, lam_a, spec.period == "first_half", spec.threshold), "half_total_goals"
    if kind == "goal_after_break":
        return goal_after_minute(lam_h, lam_a, 67.0), "goal_after_break"
    if kind == "both_teams_card":
        p = _both_teams_card(footiqo, db, home, away)
        if p is not None:
            return p, "both_teams_card"
    if kind in ("count_threshold", "count_total", "count_compare") and (db or footiqo) and spec.stat:
        stat, period, k = spec.stat, spec.period, spec.threshold
        if kind == "count_total" and k is not None:
            p, basis = _total_prob(footiqo, db, stat, k, home, away, period)
            if p is not None:
                return p, basis
        subj = sub or home
        opp = away if subj == home else home
        ra = _team_cr(footiqo, db, subj, stat, period)
        rb = _team_cr(footiqo, db, opp, stat, period)
        if kind == "count_compare" and ra and rb:
            p = count_more_than(ra, rb)
            if stat in ("shots_on_target", "corners"):
                corr = "sot" if stat == "shots_on_target" else "corners"
                p = apply_favorite_dominance(p, subj == _favorite_side(g, home, away), fav_win, corr)
            return p, f"count_more_than:{stat}:{period}"
        if kind == "count_threshold" and k is not None and ra:
            return count_threshold(ra, k), f"count_threshold:{stat}:{period}"
    if kind == "base_rate" and spec.base_rate_key in _FIXED_BASE:
        return _FIXED_BASE[spec.base_rate_key], f"base:{spec.base_rate_key}"
    return None, "unrouted"


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
    specs: list | None = None,
    footiqo=None,
    rho: float = -0.08,
) -> list[dict]:
    """Route every question to an engine and return [{question, probability, basis}].

    If `specs` (structured QuestionSpec objects from the LLM parser) are supplied,
    route via those; otherwise fall back to the built-in rule-based parser.
    `footiqo` (a Footiqo instance) sources structural count props from the current
    tournament instead of the stale DB; None keeps the DB path.
    """
    g = goal_props(lambda_home, lambda_away, rho)
    out = []
    if specs is not None:
        for spec in specs:
            prob, basis = route_spec(
                spec, home, away, g, lambda_home, lambda_away, db, lineup_status, roles, footiqo
            )
            out.append({"question": spec.question, "probability": prob, "basis": basis})
        return out
    for q in questions:
        prob, basis = _route(q, home, away, g, lambda_home, lambda_away, db, lineup_status, roles, footiqo)
        out.append({"question": q, "probability": prob, "basis": basis})
    return out
