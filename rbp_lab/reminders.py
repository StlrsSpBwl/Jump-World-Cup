from __future__ import annotations

import hashlib
from datetime import datetime, timedelta
from typing import Iterable
from zoneinfo import ZoneInfo

from .config import get_timezone
from .models import FixtureRecord, SubmissionStatus


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")
    return value.astimezone(ZoneInfo("UTC"))


def due_fixtures(
    fixtures: Iterable[FixtureRecord],
    now_utc: datetime,
    lead_minutes: int,
) -> list[FixtureRecord]:
    now = _utc(now_utc)
    lead = timedelta(minutes=lead_minutes)
    due: list[FixtureRecord] = []
    for fixture in fixtures:
        kickoff = _utc(fixture.kickoff_utc)
        window_start = kickoff - lead
        reminded_before_window = (
            fixture.reminded_at is None or _utc(fixture.reminded_at) < window_start
        )
        if (
            fixture.submission_status == SubmissionStatus.PENDING
            and window_start <= now < kickoff
            and reminded_before_window
        ):
            due.append(fixture)
    return due


def overdue_fixtures(
    fixtures: Iterable[FixtureRecord], now_utc: datetime
) -> list[FixtureRecord]:
    now = _utc(now_utc)
    return [
        fixture
        for fixture in fixtures
        if fixture.submission_status == SubmissionStatus.PENDING
        and _utc(fixture.kickoff_utc) <= now
    ]


def upcoming_fixtures(
    fixtures: Iterable[FixtureRecord],
    now_utc: datetime,
    window_hours: int,
) -> list[FixtureRecord]:
    now = _utc(now_utc)
    horizon = now + timedelta(hours=window_hours)
    return [
        fixture
        for fixture in fixtures
        if fixture.submission_status == SubmissionStatus.PENDING
        and now < _utc(fixture.kickoff_utc) <= horizon
    ]


def _escape_ics(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace(",", "\\,")
        .replace(";", "\\;")
    )


def _ics_timestamp(value: datetime) -> str:
    return _utc(value).strftime("%Y%m%dT%H%M%SZ")


def _fixture_uid(fixture: FixtureRecord) -> str:
    stable_value = (
        str(fixture.id)
        if fixture.id is not None
        else f"{fixture.match_label.casefold()}|{_ics_timestamp(fixture.kickoff_utc)}"
    )
    digest = hashlib.sha256(stable_value.encode("utf-8")).hexdigest()[:24]
    return f"fixture-{digest}@rbp-lab.local"


def to_ics(
    fixtures: Iterable[FixtureRecord],
    lead_minutes: int,
    local_tz: str,
) -> str:
    # DTSTART is deliberately UTC. Calendar clients localize it on import.
    _ = get_timezone(local_tz)
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//RBP Lab//Fixture Submission Reminders//EN",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
    ]
    for fixture in fixtures:
        kickoff = _utc(fixture.kickoff_utc)
        lines.extend(
            [
                "BEGIN:VEVENT",
                f"UID:{_fixture_uid(fixture)}",
                f"DTSTAMP:{_ics_timestamp(datetime.now(tz=ZoneInfo('UTC')))}",
                f"DTSTART:{_ics_timestamp(kickoff)}",
                f"DTEND:{_ics_timestamp(kickoff + timedelta(minutes=120))}",
                f"SUMMARY:{_escape_ics(f'SUBMIT: {fixture.match_label}')}",
                f"DESCRIPTION:{_escape_ics('Prediction submission closes at kickoff.')}",
                "BEGIN:VALARM",
                f"TRIGGER:-PT{int(lead_minutes)}M",
                "ACTION:DISPLAY",
                f"DESCRIPTION:{_escape_ics(f'Submit predictions for {fixture.match_label}')}",
                "END:VALARM",
                "END:VEVENT",
            ]
        )
    lines.append("END:VCALENDAR")
    return "\r\n".join(lines) + "\r\n"


def format_reminder_text(
    fixture: FixtureRecord,
    now_utc: datetime,
    local_tz: str,
) -> str:
    now = _utc(now_utc)
    kickoff = _utc(fixture.kickoff_utc)
    minutes = max(0, int((kickoff - now).total_seconds() // 60))
    local_kickoff = kickoff.astimezone(get_timezone(local_tz))
    time_label = local_kickoff.strftime("%I:%M %p %Z").lstrip("0")
    return (
        f"{minutes} min to kickoff - {fixture.match_label} at {time_label}. "
        "You have NOT submitted."
    )
