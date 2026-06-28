import pytest

from worldcup_props.player_props import parse_player_event, player_prop_probability


def test_parse_player_event():
    assert parse_player_event("Will Cyle Larin score a goal (excluding own goals)?") == "goals"
    assert parse_player_event("Will Rayners have at least 1 shot on target?") == "shots_on_target"
    assert parse_player_event("Will X have a shot on target in the second half?") == "second_half_shots_on_target"
    assert parse_player_event("Will Y score or assist a goal?") == "goal_or_assist"


def test_confirmed_out_is_zero():
    r = player_prop_probability("goals", "striker", "out")
    assert r["probability"] == 0.0


def test_sub_discount_reproduces_benched_striker_reads():
    # Rayners: benched forward, 1+ SOT -> ~0.18 (the +21 RBP row)
    rayners = player_prop_probability("shots_on_target", "forward", "sub")
    assert 0.14 < rayners["probability"] < 0.22
    assert rayners["basis"] == "sub_discount"
    # Larin: benched striker, score a goal -> ~0.10 (the +13 RBP row)
    larin = player_prop_probability("goals", "striker", "sub")
    assert 0.07 < larin["probability"] < 0.14


def test_starter_beats_sub_for_same_player():
    start = player_prop_probability("shots_on_target", "striker", "starter")["probability"]
    sub = player_prop_probability("shots_on_target", "striker", "sub")["probability"]
    assert start > sub
    # a starting focal striker getting a shot on target is a coin-flip-plus
    assert 0.55 < start < 0.85


def test_second_half_window_is_capped():
    # 2H SOT can't exceed full-match SOT for the same player/status
    full = player_prop_probability("shots_on_target", "winger", "starter")["probability"]
    half = player_prop_probability("second_half_shots_on_target", "winger", "starter")["probability"]
    assert half < full


def test_real_per90_overrides_role_prior():
    low = player_prop_probability("goals", "striker", "starter", per90=0.1)["probability"]
    high = player_prop_probability("goals", "striker", "starter", per90=0.8)["probability"]
    assert high > low
