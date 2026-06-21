import pytest

from worldcup_props.evaluation import log_result, results_report


def test_results_report_tracks_brier_and_crowd_bias(tmp_path):
    database = tmp_path / "props.sqlite"
    log_result(
        database,
        match_key="A vs B",
        question_key="A wins",
        question_type="match_winner",
        submitted_probability=0.7,
        crowd_probability=0.6,
        market_blended_probability=0.72,
        outcome=1,
    )
    report = results_report(database)
    group = report["question_types"]["match_winner"]
    assert group["submitted"]["brier"] == pytest.approx(0.09)
    assert group["crowd_bias"] == pytest.approx(-0.4)
    assert group["market_blended"]["events"] == 1
