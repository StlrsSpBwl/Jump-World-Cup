from pathlib import Path

from worldcup_props.draftkings import parse_har_file, write_market_csv
from worldcup_props.market import ingest_market_csv, lookup_direct_market_probability

FIXTURE = Path(__file__).parent / "fixtures" / "dk_jordan_algeria.har"
DEFINITIONS = Path(__file__).resolve().parents[1] / "data" / "market_definitions.json"


def test_parse_har_extracts_event_and_rows():
    result = parse_har_file(FIXTURE)
    assert result.event.home == "Jordan"
    assert result.event.away == "Algeria"
    # Every mapped market/selection parsed; nothing guessed and dropped.
    assert result.skipped == []
    markets = {row["market"] for row in result.rows}
    assert markets == {"h2h", "totals", "player_anytime_goalscorer"}


def test_draw_outcome_is_mapped_from_tie():
    # DraftKings labels the draw outcomeType "Tie", not "Draw".
    result = parse_har_file(FIXTURE)
    h2h = {row["selection"] for row in result.rows if row["market"] == "h2h"}
    assert h2h == {"home", "draw", "away"}


def test_player_props_stored_as_raw_implied_probability():
    result = parse_har_file(FIXTURE)
    players = [r for r in result.rows if r["market"] == "player_anytime_goalscorer"]
    assert players
    for row in players:
        assert row["decimal_odds"] == ""
        assert 0.0 < float(row["probability"]) < 1.0


def test_har_to_lookup_end_to_end(tmp_path):
    result = parse_har_file(FIXTURE)
    csv_path = tmp_path / "dk.csv"
    write_market_csv(result.rows, csv_path)
    db = tmp_path / "markets.sqlite"
    assert ingest_market_csv(db, csv_path) > 0

    # 3-way h2h is de-vigged to a coherent probability the forecast can anchor on.
    home, _ = lookup_direct_market_probability(
        db, DEFINITIONS, home="Jordan", away="Algeria",
        question_type="match_winner", selection="home",
    )
    assert home is not None and 0.0 < home < 1.0

    under, _ = lookup_direct_market_probability(
        db, DEFINITIONS, home="Jordan", away="Algeria",
        question_type="total_goals_2_or_fewer",
    )
    assert under is not None and 0.0 < under < 1.0


def test_missing_markets_raises(tmp_path):
    empty = tmp_path / "empty.har"
    empty.write_text('{"log": {"entries": []}}', encoding="utf-8")
    try:
        parse_har_file(empty)
    except ValueError as exc:
        assert "No DraftKings market responses" in str(exc)
    else:
        raise AssertionError("expected ValueError for HAR with no market responses")
