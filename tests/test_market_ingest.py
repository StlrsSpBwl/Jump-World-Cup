from __future__ import annotations

from datetime import datetime, timezone

import pytest

from worldcup_props.market_ingest import (
    SearchResult,
    ingest_market_profile,
    summarize_market_profile,
    to_implied,
)


class FakeSearchClient:
    def __init__(self, results):
        self.results = results
        self.queries: list[str] = []

    def search(self, query: str):
        self.queries.append(query)
        return self.results


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("+150", 100 / 250),
        ("-200", 200 / 300),
        ("5/4", 1 / 2.25),
        (2.25, 1 / 2.25),
        ("44%", 0.44),
    ],
)
def test_to_implied_normalizes_odds_formats(value, expected):
    assert to_implied(value) == pytest.approx(expected)


def test_to_implied_rejects_ambiguous_or_invalid_values():
    with pytest.raises(ValueError):
        to_implied("not odds")
    with pytest.raises(ValueError):
        to_implied("1.0")


def test_market_profile_query_ladder_and_golden_fixture():
    result = SearchResult(
        url="https://www.tips.gg/match/norway-vs-senegal-2026-06-22/odds/",
        title="Norway vs Senegal odds",
        snippet=(
            "Norway +130 draw +230 Senegal +210. "
            "Total goals 2.5 over -108 under +104. "
            "Norway team total 1.5 over +125 under -161. "
            "Senegal team total 1.5 over +175 under -200. "
            "Only player shots on target props listed: Haaland over 0.5 -400."
        ),
    )
    client = FakeSearchClient([result])

    profile = ingest_market_profile(
        "Norway",
        "Senegal",
        client,
        captured_at=datetime(2026, 6, 22, tzinfo=timezone.utc),
    )

    assert len(client.queries) == 6
    assert any("team total goals" in query for query in client.queries)
    assert profile.one_x_two is not None
    assert profile.total_goals is not None
    assert profile.team_total_home is not None
    assert profile.team_total_away is not None
    assert profile.one_x_two.devig_prob is not None
    assert sum(profile.one_x_two.devig_prob.values()) == pytest.approx(1.0, abs=1e-3)
    assert sum(profile.total_goals.devig_prob.values()) == pytest.approx(1.0, abs=1e-3)
    assert profile.team_total_home.devig_prob["over"] == pytest.approx(0.42, abs=0.02)
    assert profile.team_total_away.devig_prob["over"] == pytest.approx(0.35, abs=0.02)
    # These lambdas are the mathematically consistent Poisson inversion of
    # the posted O1.5 team-total prices after de-vigging. They deliberately do
    # not force a higher external narrative number when the posted line does
    # not support it.
    assert profile.derived["lambda_home"] == pytest.approx(1.42, abs=0.03)
    assert profile.derived["lambda_away"] == pytest.approx(1.23, abs=0.04)
    assert profile.derived["total"] == pytest.approx(2.65, abs=0.06)
    assert profile.derived["supremacy"] == pytest.approx(0.19, abs=0.06)
    assert "team_sot" in profile.unpriced
    assert "team_sot" not in profile.prop_lines
    assert profile.reconciliation_warnings == []


def test_market_profile_summary_exposes_unpriced_block():
    client = FakeSearchClient([])
    profile = ingest_market_profile("A", "B", client)

    summary = summarize_market_profile(profile)

    assert "one_x_two: UNPRICED" in summary
    assert "UNPRICED:" in summary
    assert "team_sot" in summary


def test_unpriced_market_cannot_be_required_as_anchor():
    client = FakeSearchClient([])
    profile = ingest_market_profile("A", "B", client)

    with pytest.raises(ValueError, match="unpriced"):
        profile.require_priced("team_sot")
