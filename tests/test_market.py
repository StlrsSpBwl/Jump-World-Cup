import pytest

from worldcup_props.data import CachedHttpClient
from worldcup_props.goals import calibrate_match_goals
from worldcup_props.market import (
    aggregate_match_markets,
    devig_proportional,
    devig_shin,
    ingest_market_csv,
    lookup_direct_market_probability,
)


@pytest.mark.parametrize("method", [devig_proportional, devig_shin])
def test_devig_probabilities_sum_to_one(method):
    probabilities = method([1.80, 2.20])
    assert sum(probabilities) == pytest.approx(1.0)
    assert all(0.0 < probability < 1.0 for probability in probabilities)


def test_favorite_remains_favorite_after_devig():
    probabilities = devig_shin([1.50, 3.00, 7.00])
    assert probabilities[0] > probabilities[1] > probabilities[2]


def test_manual_market_ingestion_and_goal_calibration(tmp_path):
    market_csv = tmp_path / "markets.csv"
    market_csv.write_text(
        "match,market,selection,decimal_odds,book,timestamp,line\n"
        "A vs B,1X2,A,2.0,Pinnacle,2026-01-01T00:00:00Z,\n"
        "A vs B,1X2,Draw,3.4,Pinnacle,2026-01-01T00:00:00Z,\n"
        "A vs B,1X2,B,4.0,Pinnacle,2026-01-01T00:00:00Z,\n"
        "A vs B,totals,Over,1.9,Pinnacle,2026-01-01T00:00:00Z,2.5\n"
        "A vs B,totals,Under,2.0,Pinnacle,2026-01-01T00:00:00Z,2.5\n",
        encoding="utf-8",
    )
    database = tmp_path / "props.sqlite"
    assert ingest_market_csv(database, market_csv) == 5
    quotes = aggregate_match_markets(database, "A", "B")
    assert {quote["selection"] for quote in quotes if quote["market_type"] == "h2h"} == {
        "home",
        "draw",
        "away",
    }
    calibration = calibrate_match_goals(database, "A", "B", 1.3, 1.2)
    assert calibration.source == "market_calibrated"
    assert calibration.targets_used == 4
    assert calibration.residuals


def test_fair_probability_market_ingestion_and_goal_calibration(tmp_path):
    market_csv = tmp_path / "fair_markets.csv"
    market_csv.write_text(
        "match,market,selection,probability,book,timestamp,line\n"
        "A vs B,1X2,A,0.47,Consensus,2026-01-01T00:00:00Z,\n"
        "A vs B,1X2,Draw,0.29,Consensus,2026-01-01T00:00:00Z,\n"
        "A vs B,1X2,B,0.24,Consensus,2026-01-01T00:00:00Z,\n"
        "A vs B,totals,Over,0.43,Consensus,2026-01-01T00:00:00Z,2.5\n"
        "A vs B,totals,Under,0.57,Consensus,2026-01-01T00:00:00Z,2.5\n",
        encoding="utf-8",
    )
    database = tmp_path / "props.sqlite"
    assert ingest_market_csv(database, market_csv) == 5
    quotes = aggregate_match_markets(database, "A", "B")
    home_quote = next(
        quote
        for quote in quotes
        if quote["market_type"] == "h2h" and quote["selection"] == "home"
    )
    assert home_quote["probability"] == pytest.approx(0.47)
    calibration = calibrate_match_goals(database, "A", "B", 1.0, 1.0)
    assert calibration.source == "market_calibrated"
    assert calibration.targets_used == 4
    assert calibration.lambda_home > calibration.lambda_away


def test_direct_player_goal_or_assist_probability_lookup(tmp_path):
    market_csv = tmp_path / "player_markets.csv"
    market_csv.write_text(
        "match,market,selection,probability,book,timestamp,line\n"
        "A vs B,player_goal_or_assist,Star Player,0.32,Consensus,2026-01-01T00:00:00Z,\n",
        encoding="utf-8",
    )
    definitions = tmp_path / "market_definitions.json"
    definitions.write_text(
        '{"player_goal_or_assist": {"market_type": "player_goal_or_assist", '
        '"definition_match": true}}',
        encoding="utf-8",
    )
    database = tmp_path / "props.sqlite"
    assert ingest_market_csv(database, market_csv) == 1
    probability, details = lookup_direct_market_probability(
        database,
        definitions,
        home="A",
        away="B",
        question_type="player_goal_or_assist",
        selection="Star Player",
    )
    assert probability == pytest.approx(0.32)
    assert details["definition_match"] is True


def test_force_refresh_bypasses_cached_market_response(tmp_path, monkeypatch):
    payloads = iter(['{"version": 1}', '{"version": 2}'])
    calls = []

    class Response:
        def __init__(self, text):
            self.text = text

        def raise_for_status(self):
            return None

    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append(url)
        return Response(next(payloads))

    monkeypatch.setattr("worldcup_props.data.requests.get", fake_get)
    client = CachedHttpClient(tmp_path, min_delay_seconds=0.0)
    assert client.get_json("https://example.test/odds")["version"] == 1
    assert client.get_json("https://example.test/odds")["version"] == 1
    assert (
        client.get_json("https://example.test/odds", force_refresh=True)["version"]
        == 2
    )
    assert len(calls) == 2
