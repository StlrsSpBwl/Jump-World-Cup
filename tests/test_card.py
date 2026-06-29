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


def test_outcome_and_total_routing():
    res = forecast_card("Canada", "South Africa", 1.4, 1.0,
                        ["Will both teams score AND the match have 3 or more total goals?",
                         "Will the match have 2 or fewer total goals?"],
                        db=str(DB))
    assert _by_q(res, "both teams score")["basis"] == "btts"
    assert _by_q(res, "2 or fewer")["basis"] == "under_2_5"
