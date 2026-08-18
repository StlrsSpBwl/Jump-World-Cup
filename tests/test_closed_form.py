import sqlite3

import numpy as np
import pytest

from worldcup_props import db as db_module
from worldcup_props.closed_form import (
    CountRate,
    advance_probability,
    apply_favorite_dominance,
    count_more_than,
    count_threshold,
    count_total_threshold,
    goal_before_minute,
    goal_props,
    opponent_adjusted_team_count_rate,
    race_probability,
    team_count_rate,
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


def test_equal_half_goals_is_plausible_and_partitions_with_more_less():
    p = goal_props(1.6, 1.2)
    # equal/more/less across the two halves must partition to 1
    less_or_equal_or_more = p["equal_half_goals"] + p["second_half_more_goals"]
    # second_half_more_goals is P(2H > 1H); the complement of (equal + 2H>1H) is P(1H > 2H)
    assert 0.0 < less_or_equal_or_more < 1.0
    assert 0.15 < p["equal_half_goals"] < 0.45


def test_race_probability_favors_higher_rate_and_handles_zero():
    assert race_probability(3.0, 1.0) > 0.5
    assert race_probability(1.0, 3.0) < 0.5
    assert race_probability(2.0, 2.0) == pytest.approx(0.5)
    assert race_probability(0.0, 0.0) == 0.5


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


def _seed_team_match_stats(db_path, rows):
    """rows: [(match_id, team, opponent, shots_on_target, fouls), ...]."""
    db_module.initialize(db_path)
    con = sqlite3.connect(db_path)
    for mid, team, opp, sot, fouls in rows:
        con.execute(
            "INSERT OR IGNORE INTO matches "
            "(id, source, source_match_id, match_date, competition, competition_type, "
            "home_team, away_team) VALUES (?, 'test', ?, '2026-01-01', 'wc', 'group', ?, ?)",
            (mid, str(mid), team, opp),
        )
        con.execute(
            "INSERT INTO team_match_stats (match_id, team, opponent, is_home, "
            "shots_on_target, fouls) VALUES (?, ?, ?, 1, ?, ?)",
            (mid, team, opp, sot, fouls),
        )
    con.commit()
    con.close()


def test_opponent_adjusted_team_count_rate(tmp_path):
    db_path = tmp_path / "props.sqlite"
    rows = []
    # Team A's own attack: SOT [4, 5, 6, 5] against four throwaway opponents.
    for i, sot in enumerate([4, 5, 6, 5]):
        rows.append((i, "A", f"X{i}", sot, 10))
    # Team B is leaky: opponents post high SOT against them.
    for i, sot in enumerate([9, 10, 8, 9]):
        rows.append((100 + i, f"Q{i}", "B", sot, 10))
    # Team C is stingy: opponents post low SOT against them.
    for i, sot in enumerate([1, 2, 1, 2]):
        rows.append((200 + i, f"R{i}", "C", sot, 10))
    _seed_team_match_stats(db_path, rows)

    base = team_count_rate(db_path, "A", "shots_on_target")
    assert base is not None and base.mean == pytest.approx(5.0)

    vs_leaky = opponent_adjusted_team_count_rate(db_path, "A", "B", "shots_on_target")
    assert vs_leaky.mean > base.mean  # leaky defense -> pushed up

    vs_stingy = opponent_adjusted_team_count_rate(db_path, "A", "C", "shots_on_target")
    assert vs_stingy.mean < base.mean  # stingy defense -> pushed down

    # unvalidated stat (cards/fouls) falls back to the flat rate untouched
    base_fouls = team_count_rate(db_path, "A", "fouls")
    adj_fouls = opponent_adjusted_team_count_rate(db_path, "A", "B", "fouls")
    assert adj_fouls.mean == pytest.approx(base_fouls.mean)

    # half-split periods aren't validated -> fall back to the flat rate
    assert opponent_adjusted_team_count_rate(db_path, "A", "B", "shots_on_target", "second_half") is None
    assert team_count_rate(db_path, "A", "shots_on_target", "second_half") is None


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
