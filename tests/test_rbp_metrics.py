import pandas as pd
import pytest

from rbp_lab.config import DashboardSettings
from rbp_lab.metrics import (
    aggregate_metric,
    apply_sign_convention,
    brier,
    compute_question_metrics,
)


def sample_frame(**overrides):
    row = {
        "question_text": "Home team wins",
        "category": "match_result",
        "p_model": 0.8,
        "p_claude": 0.6,
        "p_crowd": 0.5,
        "outcome": 1,
        "weight": 2.0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_perfect_prediction_has_zero_brier():
    assert brier(1.0, 1) == 0
    assert brier(0.0, 0) == 0


def test_matching_crowd_has_zero_rbp():
    scored = compute_question_metrics(sample_frame(p_model=0.5))
    assert scored.loc[0, "rbp_model"] == pytest.approx(0)


def test_positive_sign_means_model_beat_crowd_and_claude():
    scored = compute_question_metrics(sample_frame())
    assert scored.loc[0, "rbp_model"] > 0
    assert scored.loc[0, "model_vs_llm"] > 0


def test_sign_convention_can_be_flipped_without_recomputing_brier():
    scored = compute_question_metrics(sample_frame())
    flipped = apply_sign_convention(
        scored, DashboardSettings(sign_convention="negative_beats_crowd")
    )
    assert flipped.loc[0, "rbp_model"] == pytest.approx(-scored.loc[0, "rbp_model"])
    assert flipped.loc[0, "brier_model"] == scored.loc[0, "brier_model"]


def test_void_is_retained_and_excluded_from_metrics():
    scored = compute_question_metrics(sample_frame(outcome=None))
    assert len(scored) == 1
    assert pd.isna(scored.loc[0, "rbp_model"])


def test_weighted_mean_divides_by_settled_weight():
    scored = compute_question_metrics(sample_frame())
    assert aggregate_metric(scored, "rbp_model", "weighted_mean") == pytest.approx(
        scored.loc[0, "rbp_model"] / 2
    )

