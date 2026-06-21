from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from rbp_lab.models import FixtureRecord, SubmissionStatus
from rbp_lab.fixtures import parse_fixtures_csv
from rbp_lab.reminders import (
    due_fixtures,
    overdue_fixtures,
    to_ics,
    upcoming_fixtures,
)


UTC = ZoneInfo("UTC")
NOW = datetime(2026, 6, 15, 16, 0, tzinfo=UTC)


def fixture(
    *,
    fixture_id=1,
    kickoff=None,
    status=SubmissionStatus.PENDING,
    reminded_at=None,
):
    return FixtureRecord(
        id=fixture_id,
        match_label="Belgium vs Egypt",
        competition_stage="Group",
        kickoff_utc=kickoff or NOW + timedelta(minutes=30),
        submission_status=status,
        reminded_at=reminded_at,
    )


def test_fixture_exactly_at_lead_boundary_is_due():
    assert due_fixtures([fixture()], NOW, 30)


def test_submitted_fixture_is_never_due():
    submitted = fixture(status=SubmissionStatus.SUBMITTED)
    assert due_fixtures([submitted], NOW, 30) == []


def test_past_pending_fixture_is_overdue():
    past = fixture(kickoff=NOW - timedelta(minutes=1))
    assert overdue_fixtures([past], NOW) == [past]


def test_upcoming_respects_window_and_pending_status():
    inside = fixture(kickoff=NOW + timedelta(hours=2))
    outside = fixture(fixture_id=2, kickoff=NOW + timedelta(hours=25))
    assert upcoming_fixtures([inside, outside], NOW, 24) == [inside]


def test_ics_contains_alarm_and_one_event_per_fixture():
    fixtures = [fixture(), fixture(fixture_id=2, kickoff=NOW + timedelta(hours=2))]
    calendar = to_ics(fixtures, 30, "America/New_York")
    assert "TRIGGER:-PT30M" in calendar
    assert calendar.count("BEGIN:VEVENT") == 2
    assert "SUMMARY:SUBMIT: Belgium vs Egypt" in calendar


def test_ics_uids_are_stable_between_exports():
    first = to_ics([fixture()], 30, "America/New_York")
    second = to_ics([fixture()], 30, "America/New_York")
    first_uid = next(line for line in first.splitlines() if line.startswith("UID:"))
    second_uid = next(line for line in second.splitlines() if line.startswith("UID:"))
    assert first_uid == second_uid


def test_reminded_inside_current_window_is_not_due_again():
    reminded = fixture(reminded_at=NOW)
    assert due_fixtures([reminded], NOW, 30) == []


def test_fixture_csv_assumes_default_timezone_when_missing():
    frame, warnings = parse_fixtures_csv(
        b"match_label,kickoff\nBelgium vs Egypt,2026-06-15 12:00\n",
        "America/New_York",
    )
    assert warnings == []
    assert frame.loc[0, "kickoff_utc"].hour == 16
