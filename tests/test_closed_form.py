import numpy as np
import pytest

from worldcup_props.closed_form import (
    CountRate,
    advance_probability,
    apply_favorite_dominance,
    count_more_than,
    count_threshold,
    count_total_threshold,
    goal_before_minute,
    goal_props,
)


def test_goal_props_are_coherent():
    p = goal_props(1.8, 1.0)
    # 1X2 sums to 1
    assert abs(p["home_win"] + p["draw"] + p["away_win"] - 1.0) < 1e-6
    # favorite (home, higher lambda) wins more than away
    assert p["home_win"] > p["away_win"]
    # under + over partition
    assert abs(p["under_2_5"] + p["over_2_5"] - 1.0) < 1e-9
    # BTTS-and-3+ is a subset of BTTS
    assert p["btts_and_3plus"] <= p["btts"]
    # HT 1X2 sums to 1
    assert abs(p["ht_home_lead"] + p["ht_draw"] + p["ht_away_lead"] - 1.0) < 1e-6
    # all probabilities in [0,1]
    assert all(0.0 <= v <= 1.0 for v in p.values())


def test_second_half_skews_above_first():
    # default share gives 2nd half more goals, so home scores 2H >= 1H
    p = goal_props(1.6, 1.2)
    assert p["home_scores_2h"] >= p["home_scores_1h"]
    # 2H>1H should be a plausible probability, not degenerate
    assert 0.3 < p["second_half_more_goals"] < 0.6


def test_goal_before_minute_monotonic():
    a = goal_before_minute(1.5, 1.2, 23)
    b = goal_before_minute(1.5, 1.2, 45)
    assert a < b
    assert 0.0 < a < 1.0


def test_advance_probability():
    # heavy favorite: win 0.70, draw 0.20, lose 0.10 -> advance ~0.80
    adv = advance_probability(0.70, 0.20, 0.10, "home", et_pen_edge=0.5)
    assert abs(adv - 0.80) < 1e-9
    # underdog advance is lower
    und = advance_probability(0.70, 0.20, 0.10, "away", et_pen_edge=0.5)
    assert und < adv


def test_count_props():
    # mean 5 corners, mild overdispersion
    r = CountRate(mean=5.0, var=6.5)
    p5 = count_threshold(r, 5)
    assert 0.4 < p5 < 0.7
    # higher threshold -> lower prob
    assert count_threshold(r, 7) < p5
    # more_than: a stronger side has more
    strong = CountRate(mean=6.0, var=7.0)
    weak = CountRate(mean=3.0, var=4.0)
    assert count_more_than(strong, weak) > 0.6
    # total threshold sane
    pt = count_total_threshold(strong, weak, 9)
    assert 0.3 < pt < 0.8


def test_favorite_dominance_blend():
    # strong favorite (win 0.80) as subject -> pushed up toward 0.76 target
    base = 0.55
    after = apply_favorite_dominance(base, favorite_is_subject=True, win_probability=0.80, kind="sot")
    assert after > base
    # when subject is the underdog, pushed down
    under = apply_favorite_dominance(base, favorite_is_subject=False, win_probability=0.80, kind="sot")
    assert under < base
    # near-even match: no change
    even = apply_favorite_dominance(base, favorite_is_subject=True, win_probability=0.45, kind="sot")
    assert even == base
