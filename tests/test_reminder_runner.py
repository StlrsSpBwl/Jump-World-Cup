from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import reminder_runner
from rbp_lab.db import initialize_database, list_fixtures, save_fixture
from rbp_lab.models import FixtureRecord


def test_runner_records_reminder_and_is_idempotent(tmp_path, monkeypatch):
    database = tmp_path / "rbp.db"
    initialize_database(database, seed=False)
    now = datetime(2026, 6, 15, 16, 0, tzinfo=ZoneInfo("UTC"))
    save_fixture(
        FixtureRecord(
            match_label="Belgium vs Egypt",
            kickoff_utc=now + timedelta(minutes=15),
        ),
        database,
    )
    delivered = []
    monkeypatch.setattr(reminder_runner, "utc_now", lambda: now)
    monkeypatch.setattr(
        reminder_runner,
        "dispatch",
        lambda fixture, message, channel, lead: delivered.append(fixture.id) or True,
    )

    first, _ = reminder_runner.run_once(30, "stdout", database)
    second, _ = reminder_runner.run_once(30, "stdout", database)

    assert first == 1
    assert second == 0
    assert delivered == [1]
    assert list_fixtures(database)[0].reminded_at == now
