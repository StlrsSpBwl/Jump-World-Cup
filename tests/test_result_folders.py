from pathlib import Path

import pytest

from rbp_lab.result_folders import (
    ResultReport,
    combine_result_reports,
    display_label_from_key,
    discover_result_pairs,
    fixture_key_from_filename,
    parse_ocr_lines,
)
import pandas as pd


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("SPA_CV.pdf", "ESP_CPV"),
        ("SPN_CVD.pdf", "ESP_CPV"),
        ("CAN_BHN.pdf", "CAN_BHR"),
        ("CAN_BHI_Claude.pdf", "CAN_BHR"),
        ("Mex_RS.pdf", "MEX_RSA"),
        ("MEX_RSA_Claude.pdf", "MEX_RSA"),
        ("SWD_TUN.pdf", "SWE_TUN"),
    ],
)
def test_fixture_filename_aliases(filename, expected):
    assert fixture_key_from_filename(filename) == expected


def test_folder_discovery_pairs_aliases(tmp_path):
    model = tmp_path / "Model"
    claude = tmp_path / "Claude"
    model.mkdir()
    claude.mkdir()
    (model / "SPA_CV.pdf").touch()
    (claude / "SPN_CVD.pdf").touch()
    pairs, warnings = discover_result_pairs(tmp_path)
    assert len(pairs) == 1
    assert pairs[0].key == "ESP_CPV"
    assert warnings == []


def test_display_label_expands_country_codes():
    assert display_label_from_key("ESP_CPV") == "Spain vs Cape Verde"


def test_parse_ocr_question_card():
    page = [
        {"text": "Jun 12, 2026", "x": 0.63, "y": 0.86, "width": 0.09, "height": 0.01},
        {"text": "United States", "x": 0.33, "y": 0.84, "width": 0.11, "height": 0.01},
        {"text": "Paraguay", "x": 0.59, "y": 0.83, "width": 0.08, "height": 0.01},
        {"text": "Q1", "x": 0.28, "y": 0.61, "width": 0.02, "height": 0.01},
        {"text": "RESULT: YES", "x": 0.62, "y": 0.61, "width": 0.09, "height": 0.01},
        {
            "text": "Will United States win the match?",
            "x": 0.30,
            "y": 0.59,
            "width": 0.30,
            "height": 0.01,
        },
        {"text": "YOU", "x": 0.28, "y": 0.55, "width": 0.03, "height": 0.01},
        {"text": "59%", "x": 0.68, "y": 0.55, "width": 0.03, "height": 0.01},
        {"text": "CROWD", "x": 0.28, "y": 0.52, "width": 0.05, "height": 0.01},
        {"text": "56%", "x": 0.68, "y": 0.52, "width": 0.03, "height": 0.01},
        {"text": "OUTCOME", "x": 0.28, "y": 0.49, "width": 0.06, "height": 0.01},
        {"text": "100%", "x": 0.67, "y": 0.49, "width": 0.04, "height": 0.01},
    ]
    metadata, questions, warnings = parse_ocr_lines([page])
    assert metadata["match_label"] == "United States vs Paraguay"
    assert metadata["match_date"].isoformat() == "2026-06-12"
    assert warnings == []
    assert questions.loc[0, "prob"] == pytest.approx(0.59)
    assert questions.loc[0, "p_crowd"] == pytest.approx(0.56)
    assert questions.loc[0, "outcome"] == 1


def test_reports_join_by_question_text_not_question_number():
    model = ResultReport(
        "A vs B",
        None,
        12.5,
        pd.DataFrame(
            [
                {
                    "question_number": "Q1",
                    "question_text": "Will A win the match?",
                    "prob": 0.6,
                    "p_crowd": 0.5,
                    "outcome": 1,
                    "weight": 1.0,
                }
            ]
        ),
        [],
    )
    claude = ResultReport(
        "A vs B",
        None,
        8.0,
        pd.DataFrame(
            [
                {
                    "question_number": "Q7",
                    "question_text": "Will A win the match?",
                    "prob": 0.55,
                    "p_crowd": 0.5,
                    "outcome": 1,
                    "weight": 1.0,
                }
            ]
        ),
        [],
    )
    combined, warnings = combine_result_reports(model, claude)
    assert warnings == []
    assert len(combined) == 1
    assert combined.loc[0, "p_model"] == pytest.approx(0.6)
    assert combined.loc[0, "p_claude"] == pytest.approx(0.55)


def test_parse_page_one_official_rbp():
    page = [
        {"text": "Jun 12, 2026", "x": 0.63, "y": 0.86, "width": 0.09, "height": 0.01},
        {"text": "RBP + 28.58", "x": 0.46, "y": 0.79, "width": 0.08, "height": 0.01},
    ]
    metadata, _, _ = parse_ocr_lines([page])
    assert metadata["match_rbp"] == pytest.approx(28.58)
