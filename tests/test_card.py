from pathlib import Path

from worldcup_props.card import forecast_card

DB = Path(__file__).resolve().parents[1] / "data" / "worldcup_props.sqlite"


def _by_q(res, fragment):
    return next(r for r in res if fragment.lower() in r["question"].lower())


def test_full_card_routes_every_question():
    qs = [
        "Will both teams score in regulation?",
        "Will Canada be ahead at halftime?",
        "Will regulation end in a tie?",
        "Will South Africa advance to the Round of 16?",
        "Will the second half produce more goals than the first half?",
        "Will a goal be scored before the first hydration break?",
        "Will there be 9 or more total corner kicks in regulation?",
        "Will there be 4 or more total cards shown in regulation?",
        "Will a penalty kick be awarded?",
        "Will a substitute score a goal?",
        "Will any player score more than 1 goal?",
        "Will either team be ruled offside before the first hydration break?",
        "Will Cyle Larin score a goal in regulation?",
        "Will Iqraam Rayners have at least 1 shot on target in regulation?",
    ]
    res = forecast_card(
        "Canada", "South Africa", 1.4, 1.0, qs,
        db=str(DB), lineup_status={"Cyle Larin": "sub", "Iqraam Rayners": "sub"},
    )
    # everything routed to a probability, nothing unrouted
    assert all(r["probability"] is not None for r in res)
    assert all(0.0 <= r["probability"] <= 1.0 for r in res)


def test_sub_discount_runs_through_the_card():
    res = forecast_card(
        "Canada", "South Africa", 1.4, 1.0,
        ["Will Iqraam Rayners have at least 1 shot on target in regulation?"],
        db=str(DB), lineup_status={"Iqraam Rayners": "sub"},
    )
    r = res[0]
    assert r["basis"].startswith("player:sub_discount")
    assert 0.14 < r["probability"] < 0.24  # the +21 RBP row, from code


def test_favorite_advances_more_than_underdog():
    # Canada (lambda 1.6) clear favorite over SA (lambda 0.8)
    res = forecast_card("Canada", "South Africa", 1.6, 0.8,
                        ["Will Canada advance to the Round of 16?",
                         "Will South Africa advance to the Round of 16?"],
                        db=str(DB))
    can = _by_q(res, "canada advance")["probability"]
    sa = _by_q(res, "south africa advance")["probability"]
    assert can > sa
    assert abs((can + sa) - 1.0) < 0.05  # one of them advances


def test_team_total_goals_not_misrouted_to_brace():
    # regression: "[team] score 2 or more goals" must be a team-goal calc, NOT base:brace
    res = forecast_card("Brazil", "Japan", 1.65, 0.875,
                        ["Will Brazil score 2 or more goals in regulation?",
                         "Will any player score more than 1 goal in regulation?"],
                        db=str(DB))
    team = _by_q(res, "brazil score 2")
    brace = _by_q(res, "any player score more than 1")
    assert team["basis"] == "team_total_goals"
    assert 0.45 < team["probability"] < 0.55   # 1 - P(0) - P(1) on lambda 1.65
    assert brace["basis"] == "base:brace"       # any-player phrasing stays a brace


def test_team_both_halves_and_any_player_sot_route():
    res = forecast_card("Brazil", "Japan", 1.65, 0.875,
                        ["Will Brazil score in both halves of regulation?",
                         "Will any player record 2 or more shots on target in regulation?"],
                        db=str(DB))
    bh = _by_q(res, "both halves")
    sot = _by_q(res, "any player record 2")
    assert bh["basis"] == "team_both_halves" and 0.2 < bh["probability"] < 0.45
    assert sot["basis"] == "base:any_player_2plus_sot"
    assert all(r["probability"] is not None for r in res)


def test_player_prop_threshold_lowers_2plus_sot():
    # "2+ SOT" must be far lower than "1+ SOT" for the same benched player
    res = forecast_card("Germany", "Paraguay", 2.2, 0.7,
                        ["Will Jamal Musiala have 2 or more shots on target in regulation?",
                         "Will Jamal Musiala have 1 or more shots on target in regulation?"],
                        db=str(DB), lineup_status={"Jamal Musiala": "sub"},
                        roles={"Jamal Musiala": "attacking_mid"})
    two = _by_q(res, "2 or more shots")["probability"]
    one = _by_q(res, "1 or more shots")["probability"]
    assert two < one * 0.5          # 2+ is much rarer than 1+
    assert two < 0.10               # benched + 2+ threshold -> low


def test_win_first_goal_and_there_be_phrasings_route():
    res = forecast_card("Germany", "Paraguay", 2.2, 0.7,
                        ["Will Germany win in regulation?",
                         "Will Germany score the first goal of the match?",
                         "Will there be 3 or more offside calls in regulation?"],
                        db=str(DB))
    assert _by_q(res, "germany win")["basis"] == "win"
    assert _by_q(res, "first goal")["basis"] == "team_scores_first"
    assert _by_q(res, "offside calls")["basis"] == "count_total:offsides:full"
    assert all(r["probability"] is not None for r in res)


def test_team_alias_resolves_count_props():
    # "Ivory Coast" (contest name) must resolve to the DB's "Côte d'Ivoire"
    res = forecast_card("Norway", "Ivory Coast", 1.575, 1.175,
                        ["Will Ivory Coast have more corner kicks than Norway in regulation?",
                         "Will there be 4 or more total cards shown in regulation?"],
                        db=str(DB))
    assert all(r["probability"] is not None for r in res)
    assert _by_q(res, "corner kicks")["basis"] == "count_more_than:corners:full"


def test_winning_at_halftime_not_misrouted_to_win():
    res = forecast_card("Germany", "Paraguay", 2.2, 0.7,
                        ["Will Germany be ahead at halftime?"], db=str(DB))
    assert res[0]["basis"] == "ht_lead"


def test_half_total_goals_not_misrouted_to_brace():
    # "first half produce 2 or more goals" must be a half-total calc, NOT base:brace
    res = forecast_card("Netherlands", "Morocco", 1.275, 1.05,
                        ["Will the first half produce 2 or more goals?"], db=str(DB))
    r = res[0]
    assert r["basis"] == "half_total_goals"
    assert 0.22 < r["probability"] < 0.36   # Poisson on 1H combined lambda ~1.07


def test_redcard_bothcard_and_timing_route():
    res = forecast_card("Netherlands", "Morocco", 1.275, 1.05,
                        ["Will a red card be shown in the match?",
                         "Will both teams receive at least one card in regulation?",
                         "Will a goal be scored after the second hydration break?",
                         "Will a goal be scored in first-half stoppage time?"],
                        db=str(DB))
    assert _by_q(res, "red card")["basis"] == "base:red_card"
    assert _by_q(res, "both teams receive")["basis"] == "both_teams_card"
    assert _by_q(res, "after the second hydration")["basis"] == "goal_after_break"
    assert _by_q(res, "first-half stoppage")["basis"] == "base:fh_stoppage_goal"
    assert all(r["probability"] is not None for r in res)


def test_outcome_and_total_routing():
    res = forecast_card("Canada", "South Africa", 1.4, 1.0,
                        ["Will both teams score AND the match have 3 or more total goals?",
                         "Will the match have 2 or fewer total goals?"],
                        db=str(DB))
    assert _by_q(res, "both teams score")["basis"] == "btts"
    assert _by_q(res, "2 or fewer")["basis"] == "under_2_5"


def test_player_prop_uses_real_profile_per90_not_role_prior():
    # Haaland's real goals_per90 (~1.40) is far above the generic "forward" role
    # prior (0.40) -- the card must prefer the DB profile when one exists.
    res = forecast_card("Brazil", "Norway", 1.67, 1.04,
                        ["Will Erling Haaland (Norway) score a goal in regulation?"],
                        db=str(DB), lineup_status={"Erling Haaland": "starter"},
                        roles={"Erling Haaland": "striker"})
    r = res[0]
    assert r["basis"] == "player:starter:goals:profile"
    # role-prior striker rate (0.48/90) over 82 min -> ~0.35; real per90 (~1.40) -> much higher
    assert r["probability"] > 0.55


def test_more_cards_than_goals_route():
    res = forecast_card("England", "Mexico", 1.25, 1.0,
                        ["Will there be more total cards than total goals in regulation?"],
                        db=str(DB))
    r = res[0]
    assert r["basis"] == "more_stat_than_goals:cards"
    assert 0.0 < r["probability"] < 1.0


def test_exact_total_goals_and_penalty_shootout_route():
    res = forecast_card("Brazil", "Norway", 1.67, 1.04,
                        ["Will the match finish with exactly 2 total goals in regulation?",
                         "Will the match be decided by a penalty shootout?"],
                        db=str(DB))
    exact = _by_q(res, "exactly 2")
    assert exact["basis"] == "exact_total_goals"
    assert 0.0 < exact["probability"] < 0.4
    shootout = _by_q(res, "penalty shootout")
    assert shootout["basis"] == "draw*base:penalty_shootout_given_draw"
    # draw probability * 0.33 conditional, must be well under the draw prob itself
    assert 0.0 < shootout["probability"] < 0.15


def test_extra_time_equal_halves_and_card_before_goal_race_route():
    res = forecast_card("Portugal", "Spain", 1.04, 1.58,
                        ["Will the match go to extra time?",
                         "Will both halves have the same number of goals in regulation?",
                         "Will the first card of the match be shown before the first goal is scored?"],
                        db=str(DB))
    et = _by_q(res, "extra time")
    assert et["basis"] == "draw"
    eq = _by_q(res, "same number of goals")
    assert eq["basis"] == "equal_half_goals"
    assert 0.0 < eq["probability"] < 1.0
    race = _by_q(res, "before the first goal")
    assert race["basis"] == "race:cards_before_goals"
    assert 0.0 < race["probability"] < 1.0
