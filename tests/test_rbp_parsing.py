import io

import pytest

from rbp_lab.parsing import (
    normalize_probability,
    parse_outcome,
    parse_settlement_csv,
    reconcile_sources,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("62%", 0.62), ("0.62", 0.62), (62, 0.62), ("101%", None)],
)
def test_probability_normalization(raw, expected):
    assert normalize_probability(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("yes", 1), ("NO", 0), ("void", None), ("", None)],
)
def test_outcome_normalization(raw, expected):
    assert parse_outcome(raw) == expected


def test_csv_aliases_and_fuzzy_reconciliation():
    settlement = parse_settlement_csv(
        io.BytesIO(
            b"Question Text,Crowd Probability,Outcome,Weight\n"
            b"Over 2.5 total goals,55%,yes,2\n"
        )
    ).data
    model = settlement[["question_text"]].copy()
    model["question_text"] = "Total goals over 2.5"
    model["prob"] = 0.64
    combined, notices = reconcile_sources(model, None, settlement, threshold=60)
    assert len(combined) == 1
    assert combined.loc[0, "p_model"] == pytest.approx(0.64)
    assert combined.loc[0, "p_crowd"] == pytest.approx(0.55)
    assert notices

