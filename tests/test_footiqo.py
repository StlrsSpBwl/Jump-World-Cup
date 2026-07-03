from pathlib import Path

import pytest

from worldcup_props.card import forecast_card
from worldcup_props.footiqo import Footiqo

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "Footiqo data"
DB = ROOT / "data" / "worldcup_props.sqlite"

pytestmark = pytest.mark.skipif(not FOLDER.exists(), reason="Footiqo data folder not present")


@pytest.fixture(scope="module")
def fq():
    return Footiqo(FOLDER)


def test_loads_all_tournament_matches(fq):
    assert len(fq.matches) > 60  # ~85 games


def test_base_rates_are_empirical_and_sane(fq):
    # cards run ~2.4/game this tournament -> 5+ is rare, ~0.10
    assert 0.05 < fq.total_rate("cards", 5) < 0.20
    # shots run ~24/game -> 20+ is common
    assert 0.60 < fq.total_rate("shots", 20) < 0.80


def test_team_rate_shrinks_extremes_toward_center(fq):
    # a team with an extreme small-sample mean gets pulled toward the tournament mean
    raw = fq._team_vals("Spain", "shots_on_target")
    shrunk = fq.team_rate("Spain", "shots_on_target").mean
    tourn = fq._single_team_mean("shots_on_target")
    raw_mean = sum(raw) / len(raw)
    assert min(raw_mean, tourn) - 0.01 <= shrunk <= max(raw_mean, tourn) + 0.01
    assert abs(shrunk - tourn) < abs(raw_mean - tourn)  # strictly closer to the prior


def test_match_lambdas_from_local_odds(fq):
    lam = fq.match_lambdas("Portugal", "Croatia")
    assert lam is not None
    lb, la = lam
    assert lb > la and 1.0 < lb < 2.5 and 0.5 < la < 1.6  # Portugal favored, sane totals


def test_footiqo_overrides_stale_db_cards(fq):
    q = ["Will there be 5 or more total cards shown in regulation?"]
    db_row = forecast_card("Portugal", "Croatia", 1.7, 0.9, q, db=str(DB))[0]
    fq_row = forecast_card("Portugal", "Croatia", 1.7, 0.9, q, db=str(DB), footiqo=fq)[0]
    assert fq_row["basis"] == "footiqo_total:cards"
    assert fq_row["probability"] < db_row["probability"] - 0.10  # DB was ~2x too high


def test_total_shots_routes_via_footiqo(fq):
    q = ["Will there be 20 or more total shots in regulation?"]
    row = forecast_card("Portugal", "Croatia", 1.7, 0.9, q, db=str(DB), footiqo=fq)[0]
    assert row["basis"] == "footiqo_total:shots"
    assert 0.55 < row["probability"] < 0.80
