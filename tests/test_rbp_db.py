from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd

from rbp_lab.config import DashboardSettings
from rbp_lab.db import (
    initialize_database,
    list_fixtures,
    load_match,
    mark_passed_fixtures_missed,
    save_fixture,
    save_match,
    update_fixture_status,
)
from rbp_lab.models import FixtureRecord, MatchRecord, SubmissionStatus


def test_match_round_trip_keeps_void_rows(tmp_path):
    database = tmp_path / "rbp.db"
    initialize_database(database, seed=False)
    records = pd.DataFrame(
        [
            {
                "question_text": "Penalty awarded",
                "category": "penalty",
                "p_model": 0.2,
                "p_claude": 0.25,
                "p_crowd": 0.3,
                "outcome": None,
                "weight": 1.0,
            }
        ]
    )
    match_id = save_match(
        MatchRecord(
            match_label="A vs B",
            competition_stage="Group",
            match_date=date(2026, 6, 15),
            official_rbp_model=28.58,
            official_rbp_claude=12.25,
        ),
        records,
        DashboardSettings(sign_convention="negative_beats_crowd"),
        database,
    )
    match, loaded = load_match(match_id, database)
    assert match["match_label"] == "A vs B"
    assert match["official_rbp_model"] == 28.58
    assert match["official_rbp_claude"] == 12.25
    assert len(loaded) == 1
    assert pd.isna(loaded.loc[0, "outcome"])
    assert pd.isna(loaded.loc[0, "rbp_model"])


def test_fixture_migration_round_trip_and_status_updates(tmp_path):
    database = tmp_path / "rbp.db"
    initialize_database(database, seed=False)
    kickoff = datetime(2026, 6, 15, 20, 0, tzinfo=ZoneInfo("UTC"))
    fixture_id = save_fixture(
        FixtureRecord(
            match_label="Belgium vs Egypt",
            kickoff_utc=kickoff,
        ),
        database,
    )
    loaded = list_fixtures(database)
    assert loaded[0].id == fixture_id
    assert loaded[0].kickoff_utc == kickoff

    update_fixture_status(fixture_id, SubmissionStatus.SUBMITTED, database)
    submitted = list_fixtures(database)[0]
    assert submitted.submission_status == SubmissionStatus.SUBMITTED
    assert submitted.submitted_at is not None


def test_passed_pending_fixture_is_marked_missed(tmp_path):
    database = tmp_path / "rbp.db"
    initialize_database(database, seed=False)
    kickoff = datetime(2026, 6, 15, 12, 0, tzinfo=ZoneInfo("UTC"))
    save_fixture(
        FixtureRecord(match_label="A vs B", kickoff_utc=kickoff),
        database,
    )
    changed = mark_passed_fixtures_missed(
        datetime(2026, 6, 15, 13, 0, tzinfo=ZoneInfo("UTC")),
        database,
    )
    assert changed == 1
    assert list_fixtures(database)[0].submission_status == SubmissionStatus.MISSED
