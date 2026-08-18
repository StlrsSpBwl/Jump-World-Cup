"""BTTS pins the home/away goal split that h2h+totals leave under-determined."""

from worldcup_props.goals import (
    _targets_from_quotes,
    both_teams_score_probability,
    dixon_coles_matrix,
)


def _quote(market_type, selection, probability, line=None):
    return {
        "market_type": market_type,
        "selection": selection,
        "line": line,
        "probability": probability,
    }


def test_btts_evaluator_matches_matrix_mass():
    matrix = dixon_coles_matrix(2.55, 0.55)
    # P(both score) = mass with home>=1 and away>=1
    assert abs(both_teams_score_probability(matrix) - matrix[1:, 1:].sum()) < 1e-12
    # a heavy favorite vs a weak side lands BTTS well under a coin flip
    assert 0.35 < both_teams_score_probability(matrix) < 0.45


def test_btts_quote_becomes_a_calibration_target():
    quotes = [
        _quote("h2h", "home", 0.80),
        _quote("h2h", "away", 0.06),
        _quote("totals", "over", 0.60, line=2.5),
        _quote("btts", "yes", 0.39),
    ]
    targets = _targets_from_quotes(quotes)
    by_name = {t.name: t for t in targets}
    assert "btts:yes" in by_name
    # weighted like an h2h leg so it actually moves the split
    assert by_name["btts:yes"].weight == 2.0
    # evaluator wired to the BTTS mass
    matrix = dixon_coles_matrix(2.4, 0.7)
    assert abs(by_name["btts:yes"].evaluator(matrix) - both_teams_score_probability(matrix)) < 1e-12


def test_btts_target_absent_when_no_btts_quote():
    targets = _targets_from_quotes([_quote("h2h", "home", 0.8), _quote("h2h", "away", 0.06)])
    assert all(t.name != "btts:yes" for t in targets)
