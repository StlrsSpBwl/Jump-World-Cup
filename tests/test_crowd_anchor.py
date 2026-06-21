import pytest

from worldcup_props.config import Settings
from worldcup_props.crowd import apply_crowd_anchor
from worldcup_props.evaluation import log_result


def test_fixated_offside_anchor_clamps_extreme_low_model():
    settings = Settings()

    probability, details = apply_crowd_anchor(
        0.20,
        "offsides:threshold:2",
        settings,
    )

    assert probability == pytest.approx(0.43)
    assert details["applied"] is True
    assert details["band"] == [0.43, 0.54]
    assert details["target"] == pytest.approx(0.43)


def test_loose_match_winner_bucket_passes_through():
    settings = Settings()

    probability, details = apply_crowd_anchor(
        0.72,
        "match_winner",
        settings,
    )

    assert probability == pytest.approx(0.72)
    assert details["applied"] is False
    assert details["reason"] == "loose_crowd_bucket"


def test_recent_crowd_drift_moves_fixated_anchor_with_cap(tmp_path):
    database = tmp_path / "props.sqlite"
    for index, crowd in enumerate([0.42, 0.43, 0.44, 0.50, 0.52, 0.54]):
        log_result(
            database,
            match_key=f"M{index}",
            question_key=f"Q{index}",
            question_type="offsides:threshold:2",
            submitted_probability=0.45,
            crowd_probability=crowd,
            outcome=0,
            observed_at=f"2026-06-20T00:0{index}:00+00:00",
        )
    settings = Settings(crowd_anchor_recent_window=3, crowd_anchor_min_drift_rows=6)

    probability, details = apply_crowd_anchor(
        0.43,
        "offsides:threshold:2",
        settings,
        database_path=database,
    )

    assert details["trend"]["usable"] is True
    assert details["trend"]["recent_mean"] > details["trend"]["earlier_mean"]
    assert details["drift_adjustment"] > 0.0
    assert probability > 0.43
