import pytest

from worldcup_props.calibration import (
    apply_calibration_map,
    apply_shrinkage,
    brier_decomposition,
    brier_score,
    fit_platt_brier,
    tune_shrinkage_lambda,
)


def test_brier_and_decomposition_are_consistent():
    probabilities = [0.1, 0.3, 0.7, 0.9]
    outcomes = [0, 0, 1, 1]
    score = brier_score(probabilities, outcomes)
    decomposition = brier_decomposition(probabilities, outcomes, bins=4)
    assert score == pytest.approx(0.05)
    assert decomposition.brier == pytest.approx(score)
    assert (
        decomposition.uncertainty
        - decomposition.resolution
        + decomposition.reliability
    ) == pytest.approx(score)


def test_shrinkage_never_extrapolates():
    coefficient, base_rate, score = tune_shrinkage_lambda(
        [0.01, 0.99, 0.01, 0.99], [1, 0, 1, 0]
    )
    assert coefficient == 0.0
    assert base_rate == 0.5
    assert score == pytest.approx(0.25)
    assert apply_shrinkage(0.9, coefficient, base_rate) == 0.5


def test_platt_map_is_valid_probability():
    mapping, score = fit_platt_brier(
        [0.1, 0.25, 0.7, 0.9], [0, 0, 1, 1]
    )
    assert mapping["method"] == "platt"
    assert 0.0 <= apply_calibration_map(0.6, mapping) <= 1.0
    assert score >= 0.0
