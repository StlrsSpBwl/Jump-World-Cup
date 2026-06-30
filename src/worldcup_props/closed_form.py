"""Closed-form prop engine.

A thin, reproducible alternative to the Monte Carlo simulator for the contest's
prop questions. Goal/outcome/BTTS/totals come straight from the Dixon-Coles
score matrix; half-split props (HT result, 2H>1H, team-scores-in-2H) from
per-half lambdas; count props (corners/SOT/cards) from each team's historical
rate fed through a negative-binomial, with the held-out-validated favorite
corrections (2H-SOT dominance, corner inflation) applied where they earned it.

Everything here is a closed-form / numerical evaluation -- no simulation. The
market is the input (via lambdas) and the historical DB supplies count rates.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import nbinom, poisson

from .goals import (
    both_teams_score_probability,
    dixon_coles_matrix,
    scoreline_probabilities,
    total_over_probability,
)

# Second halves carry slightly more goals than first halves (fatigue, chasing).
DEFAULT_FIRST_HALF_SHARE = 0.46
DEFAULT_MATCH_MINUTES = 95.0  # ~90 + average stoppage


# ----------------------------------------------------------------------------- goals
def _p_poisson_greater(mu_a: float, mu_b: float, max_n: int = 30) -> float:
    """P(A > B) for independent A~Poisson(mu_a), B~Poisson(mu_b)."""
    a = np.arange(max_n + 1)
    pa = poisson.pmf(a, mu_a)
    cdf_b_below = poisson.cdf(a - 1, mu_b)  # P(B <= a-1) = P(B < a)
    return float(np.sum(pa * cdf_b_below))


def half_split_lambdas(lh: float, la: float, first_half_share: float = DEFAULT_FIRST_HALF_SHARE):
    s1 = first_half_share
    s2 = 1.0 - s1
    return lh * s1, la * s1, lh * s2, la * s2


def goal_props(
    lambda_home: float,
    lambda_away: float,
    rho: float = -0.08,
    first_half_share: float = DEFAULT_FIRST_HALF_SHARE,
    max_goals: int = 12,
) -> dict[str, float]:
    """All goal/outcome/half-split props from market lambdas (closed form)."""
    m = dixon_coles_matrix(lambda_home, lambda_away, rho, max_goals)
    sl = scoreline_probabilities(m)
    btts = both_teams_score_probability(m)
    over_2_5 = total_over_probability(m, 2.5)

    l1h, l1a, l2h, l2a = half_split_lambdas(lambda_home, lambda_away, first_half_share)
    m1 = dixon_coles_matrix(l1h, l1a, rho, max_goals)
    ht = scoreline_probabilities(m1)

    tot_2h = l2h + l2a
    return {
        "home_win": sl["home"],
        "draw": sl["draw"],
        "away_win": sl["away"],
        "btts": btts,
        "over_2_5": over_2_5,
        "under_2_5": 1.0 - over_2_5,          # "2 or fewer total goals"
        "three_or_more": over_2_5,            # 3+ goals == over 2.5
        "btts_and_3plus": max(0.0, btts - float(m[1, 1])),  # BTTS minus exact 1-1
        "home_scores": 1.0 - float(np.exp(-lambda_home)),
        "away_scores": 1.0 - float(np.exp(-lambda_away)),
        # halves
        "ht_home_lead": ht["home"],
        "ht_draw": ht["draw"],
        "ht_away_lead": ht["away"],
        "second_half_more_goals": _p_poisson_greater(tot_2h, l1h + l1a),
        "second_half_2plus": float(1.0 - poisson.cdf(1, tot_2h)),
        "home_scores_2h": 1.0 - float(np.exp(-l2h)),
        "away_scores_2h": 1.0 - float(np.exp(-l2a)),
        "home_scores_1h": 1.0 - float(np.exp(-l1h)),
        "away_scores_1h": 1.0 - float(np.exp(-l1a)),
    }


def goal_before_minute(
    lambda_home: float, lambda_away: float, minute: float, match_minutes: float = DEFAULT_MATCH_MINUTES
) -> float:
    """P(at least one goal in the first `minute` minutes), constant-hazard approx."""
    total = lambda_home + lambda_away
    return float(1.0 - np.exp(-total * minute / match_minutes))


def goal_after_minute(
    lambda_home: float, lambda_away: float, minute: float, match_minutes: float = DEFAULT_MATCH_MINUTES
) -> float:
    """P(at least one goal AFTER `minute`), constant-hazard approx (mirror of before)."""
    total = lambda_home + lambda_away
    return float(1.0 - np.exp(-total * (match_minutes - minute) / match_minutes))


def advance_probability(home_win: float, draw: float, away_win: float, side: str,
                        et_pen_edge: float | None = None) -> float:
    """Knockout 'team advances' = win in 90 + draw * P(win the ET/shootout).

    et_pen_edge is the team's share of the tied-game tiebreak. Default: the
    team's relative win strength, so the favorite carries ET/penalties too.
    """
    win = home_win if side == "home" else away_win
    other = away_win if side == "home" else home_win
    if et_pen_edge is None:
        et_pen_edge = win / (win + other) if (win + other) > 0 else 0.5
    return float(win + draw * et_pen_edge)


# ----------------------------------------------------------------------------- counts
@dataclass
class CountRate:
    mean: float
    var: float


def _count_pmf(rate: CountRate, support: np.ndarray) -> np.ndarray:
    """PMF over `support` from mean/var: NB if overdispersed, else Poisson."""
    mean = max(rate.mean, 1e-6)
    var = rate.var
    if var <= mean * 1.05:  # not meaningfully overdispersed
        return poisson.pmf(support, mean)
    size = mean * mean / (var - mean)
    p = size / (size + mean)
    return nbinom.pmf(support, size, p)


def count_threshold(rate: CountRate, k: int, max_n: int = 40) -> float:
    """P(count >= k)."""
    support = np.arange(max_n + 1)
    pmf = _count_pmf(rate, support)
    return float(pmf[support >= k].sum())


def count_more_than(rate_a: CountRate, rate_b: CountRate, max_n: int = 40) -> float:
    """P(A > B) for two independent count variables."""
    support = np.arange(max_n + 1)
    pa = _count_pmf(rate_a, support)
    cb = np.cumsum(_count_pmf(rate_b, support))  # cdf
    cb_below = np.concatenate([[0.0], cb[:-1]])  # P(B < a)
    return float(np.sum(pa * cb_below))


def count_total_threshold(rate_a: CountRate, rate_b: CountRate, k: int, max_n: int = 60) -> float:
    """P(A + B >= k) via pmf convolution."""
    support = np.arange(max_n + 1)
    conv = np.convolve(_count_pmf(rate_a, support), _count_pmf(rate_b, support))
    return float(conv[k:].sum())


def team_count_rate(
    db: str | Path, team: str, stat: str, period: str = "full"
) -> CountRate | None:
    """Mean/var of a team's `stat` over its matches. period: full|first_half|second_half."""
    con = sqlite3.connect(str(db))
    col = stat if period != "first_half" else f"first_half_{stat}"
    rows = con.execute(
        f"SELECT {stat} AS total, first_half_{stat} AS fh FROM team_match_stats WHERE team=?",
        (team,),
    ).fetchall()
    vals = []
    for total, fh in rows:
        if total is None:
            continue
        if period == "full":
            vals.append(float(total))
        elif period == "first_half" and fh is not None:
            vals.append(float(fh))
        elif period == "second_half" and fh is not None:
            vals.append(float(total) - float(fh))
    if len(vals) < 3:
        return None
    arr = np.array(vals)
    return CountRate(mean=float(arr.mean()), var=float(arr.var(ddof=1)))


# ------------------------------------------------------ validated favorite corrections
def _favorite_target(win_probability: float, kind: str) -> float | None:
    """Held-out-validated P(favorite has more {stat}) by supremacy bucket.

    SOT (2H) and corners measured over 2,408 matches; see tools/holdout_validation.
    """
    if win_probability < 0.50:
        return None
    if kind == "sot":
        return 0.57 if win_probability < 0.56 else 0.63 if win_probability < 0.68 else 0.76
    if kind == "corners":
        return 0.57 if win_probability < 0.56 else 0.65 if win_probability < 0.68 else 0.80
    return None


def apply_favorite_dominance(
    probability: float, favorite_is_subject: bool, win_probability: float, kind: str,
    weight: float = 0.55,
) -> float:
    """Blend a 'subject more {stat} than opponent' prob toward the empirical target."""
    target_fav = _favorite_target(win_probability, kind)
    if target_fav is None:
        return probability
    target = target_fav if favorite_is_subject else 1.0 - target_fav
    return float(min(1.0, max(0.0, (1.0 - weight) * probability + weight * target)))
