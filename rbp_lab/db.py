from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterator
from zoneinfo import ZoneInfo

import pandas as pd

from .classification import classify_question
from .config import DATABASE_PATH, DashboardSettings, utc_now
from .metrics import compute_question_metrics
from .models import FixtureRecord, MatchRecord, QuestionRecord, SubmissionStatus


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS matches (
    id INTEGER PRIMARY KEY,
    match_label TEXT NOT NULL,
    competition_stage TEXT NOT NULL DEFAULT '',
    match_date TEXT NOT NULL,
    official_rbp_model REAL,
    official_rbp_claude REAL,
    notes TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS question_records (
    id INTEGER PRIMARY KEY,
    match_id INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    question_text TEXT NOT NULL,
    category TEXT NOT NULL,
    p_model REAL NOT NULL CHECK (p_model BETWEEN 0 AND 1),
    p_claude REAL NOT NULL CHECK (p_claude BETWEEN 0 AND 1),
    p_crowd REAL NOT NULL CHECK (p_crowd BETWEEN 0 AND 1),
    outcome INTEGER CHECK (outcome IN (0, 1) OR outcome IS NULL),
    weight REAL NOT NULL DEFAULT 1.0 CHECK (weight > 0),
    brier_model REAL,
    brier_claude REAL,
    brier_crowd REAL,
    rbp_model REAL,
    rbp_claude REAL,
    model_vs_llm REAL
);

CREATE INDEX IF NOT EXISTS idx_rbp_match_date ON matches(match_date);
CREATE INDEX IF NOT EXISTS idx_rbp_records_match ON question_records(match_id);
CREATE INDEX IF NOT EXISTS idx_rbp_records_category ON question_records(category);

CREATE TABLE IF NOT EXISTS fixtures (
    id INTEGER PRIMARY KEY,
    match_label TEXT NOT NULL COLLATE NOCASE UNIQUE,
    competition_stage TEXT NOT NULL DEFAULT '',
    kickoff_utc TEXT NOT NULL,
    submission_status TEXT NOT NULL DEFAULT 'pending'
        CHECK (submission_status IN ('pending', 'submitted', 'skipped', 'missed')),
    submitted_at TEXT,
    linked_match_id INTEGER REFERENCES matches(id) ON DELETE SET NULL,
    reminded_at TEXT,
    notes TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_fixtures_kickoff ON fixtures(kickoff_utc);
CREATE INDEX IF NOT EXISTS idx_fixtures_status ON fixtures(submission_status);
"""


def connect(path: str | Path = DATABASE_PATH) -> sqlite3.Connection:
    connection = sqlite3.connect(Path(path))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


@contextmanager
def transaction(path: str | Path = DATABASE_PATH) -> Iterator[sqlite3.Connection]:
    connection = connect(path)
    try:
        connection.execute("BEGIN")
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_database(path: str | Path = DATABASE_PATH, seed: bool = True) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with connect(destination) as connection:
        connection.executescript(SCHEMA)
        _add_column(connection, "matches", "official_rbp_model", "REAL")
        _add_column(connection, "matches", "official_rbp_claude", "REAL")
    if seed and not list_matches(path).shape[0]:
        seed_database(path)
    if seed and not list_fixtures(path):
        seed_fixtures(path)


def list_matches(path: str | Path = DATABASE_PATH) -> pd.DataFrame:
    with connect(path) as connection:
        return pd.read_sql_query(
            "SELECT * FROM matches ORDER BY match_date, id", connection, parse_dates=["match_date", "created_at"]
        )


def _add_column(
    connection: sqlite3.Connection,
    table: str,
    column: str,
    declaration: str,
) -> None:
    existing = {
        str(row["name"])
        for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
    }
    if column not in existing:
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")


def load_match(match_id: int, path: str | Path = DATABASE_PATH) -> tuple[dict, pd.DataFrame]:
    with connect(path) as connection:
        match = connection.execute("SELECT * FROM matches WHERE id=?", (match_id,)).fetchone()
        if match is None:
            raise KeyError(f"Match {match_id} does not exist")
        records = pd.read_sql_query(
            "SELECT * FROM question_records WHERE match_id=? ORDER BY id",
            connection,
            params=(match_id,),
        )
    return dict(match), records


def load_all_records(path: str | Path = DATABASE_PATH) -> pd.DataFrame:
    with connect(path) as connection:
        return pd.read_sql_query(
            """
            SELECT q.*, m.match_label, m.competition_stage, m.match_date,
                   m.official_rbp_model, m.official_rbp_claude, m.notes
            FROM question_records q
            JOIN matches m ON m.id = q.match_id
            ORDER BY m.match_date, m.id, q.id
            """,
            connection,
            parse_dates=["match_date"],
        )


def _serialize_datetime(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")
    return value.astimezone(ZoneInfo("UTC")).isoformat()


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ZoneInfo("UTC"))
    return parsed.astimezone(ZoneInfo("UTC"))


def normalize_match_label(value: str) -> str:
    return " ".join(value.strip().split())


def list_fixtures(path: str | Path = DATABASE_PATH) -> list[FixtureRecord]:
    with connect(path) as connection:
        rows = connection.execute(
            "SELECT * FROM fixtures ORDER BY kickoff_utc, id"
        ).fetchall()
    return [
        FixtureRecord(
            **{
                **dict(row),
                "kickoff_utc": _parse_datetime(row["kickoff_utc"]),
                "submitted_at": _parse_datetime(row["submitted_at"]),
                "reminded_at": _parse_datetime(row["reminded_at"]),
                "created_at": _parse_datetime(row["created_at"]),
            }
        )
        for row in rows
    ]


def load_fixture(
    fixture_id: int, path: str | Path = DATABASE_PATH
) -> FixtureRecord:
    with connect(path) as connection:
        row = connection.execute(
            "SELECT * FROM fixtures WHERE id=?", (fixture_id,)
        ).fetchone()
    if row is None:
        raise KeyError(f"Fixture {fixture_id} does not exist")
    values = dict(row)
    for column in ("kickoff_utc", "submitted_at", "reminded_at", "created_at"):
        values[column] = _parse_datetime(values[column])
    return FixtureRecord(**values)


def save_fixture(
    fixture: FixtureRecord, path: str | Path = DATABASE_PATH
) -> int:
    label = normalize_match_label(fixture.match_label)
    submitted_at = fixture.submitted_at
    if fixture.submission_status == SubmissionStatus.SUBMITTED and submitted_at is None:
        submitted_at = utc_now()
    if fixture.submission_status != SubmissionStatus.SUBMITTED:
        submitted_at = None

    with transaction(path) as connection:
        if fixture.id is None:
            existing = connection.execute(
                "SELECT id FROM fixtures WHERE match_label=? COLLATE NOCASE",
                (label,),
            ).fetchone()
            if existing:
                fixture_id = int(existing["id"])
                connection.execute(
                    """
                    UPDATE fixtures
                    SET match_label=?, competition_stage=?, kickoff_utc=?,
                        submission_status=?, submitted_at=?, linked_match_id=?,
                        reminded_at=?, notes=?
                    WHERE id=?
                    """,
                    (
                        label,
                        fixture.competition_stage,
                        _serialize_datetime(fixture.kickoff_utc),
                        fixture.submission_status.value,
                        _serialize_datetime(submitted_at),
                        fixture.linked_match_id,
                        _serialize_datetime(fixture.reminded_at),
                        fixture.notes,
                        fixture_id,
                    ),
                )
            else:
                cursor = connection.execute(
                    """
                    INSERT INTO fixtures (
                        match_label, competition_stage, kickoff_utc,
                        submission_status, submitted_at, linked_match_id,
                        reminded_at, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        label,
                        fixture.competition_stage,
                        _serialize_datetime(fixture.kickoff_utc),
                        fixture.submission_status.value,
                        _serialize_datetime(submitted_at),
                        fixture.linked_match_id,
                        _serialize_datetime(fixture.reminded_at),
                        fixture.notes,
                    ),
                )
                fixture_id = int(cursor.lastrowid)
        else:
            fixture_id = fixture.id
            connection.execute(
                """
                UPDATE fixtures
                SET match_label=?, competition_stage=?, kickoff_utc=?,
                    submission_status=?, submitted_at=?, linked_match_id=?,
                    reminded_at=?, notes=?
                WHERE id=?
                """,
                (
                    label,
                    fixture.competition_stage,
                    _serialize_datetime(fixture.kickoff_utc),
                    fixture.submission_status.value,
                    _serialize_datetime(submitted_at),
                    fixture.linked_match_id,
                    _serialize_datetime(fixture.reminded_at),
                    fixture.notes,
                    fixture_id,
                ),
            )
    return fixture_id


def update_fixture_status(
    fixture_id: int,
    status: SubmissionStatus | str,
    path: str | Path = DATABASE_PATH,
) -> None:
    status_value = SubmissionStatus(status)
    submitted_at = _serialize_datetime(utc_now()) if status_value == SubmissionStatus.SUBMITTED else None
    with transaction(path) as connection:
        connection.execute(
            "UPDATE fixtures SET submission_status=?, submitted_at=? WHERE id=?",
            (status_value.value, submitted_at, fixture_id),
        )


def mark_fixture_reminded(
    fixture_id: int,
    reminded_at: datetime,
    path: str | Path = DATABASE_PATH,
) -> None:
    with transaction(path) as connection:
        connection.execute(
            "UPDATE fixtures SET reminded_at=? WHERE id=?",
            (_serialize_datetime(reminded_at), fixture_id),
        )


def mark_passed_fixtures_missed(
    now_utc: datetime | None = None,
    path: str | Path = DATABASE_PATH,
) -> int:
    now_utc = now_utc or utc_now()
    with transaction(path) as connection:
        cursor = connection.execute(
            """
            UPDATE fixtures
            SET submission_status='missed'
            WHERE submission_status='pending' AND kickoff_utc <= ?
            """,
            (_serialize_datetime(now_utc),),
        )
        return int(cursor.rowcount)


def fixtures_for_match(
    match_id: int, path: str | Path = DATABASE_PATH
) -> list[FixtureRecord]:
    return [
        fixture for fixture in list_fixtures(path) if fixture.linked_match_id == match_id
    ]


def save_match(
    match: MatchRecord,
    records: pd.DataFrame,
    settings: DashboardSettings,
    path: str | Path = DATABASE_PATH,
) -> int:
    # Persist one canonical direction. The UI sign toggle is a reversible view concern.
    calculated = compute_question_metrics(records, DashboardSettings(settings.rbp_agg))
    validated = [
        QuestionRecord(
            **{
                column: (None if pd.isna(row[column]) else row[column])
                for column in QuestionRecord.model_fields
                if column in calculated.columns and column not in {"id", "match_id"}
            }
        )
        for _, row in calculated.iterrows()
    ]

    with transaction(path) as connection:
        if match.id is None:
            cursor = connection.execute(
                """
                INSERT INTO matches (
                    match_label, competition_stage, match_date,
                    official_rbp_model, official_rbp_claude, notes
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    match.match_label,
                    match.competition_stage,
                    match.match_date.isoformat(),
                    match.official_rbp_model,
                    match.official_rbp_claude,
                    match.notes,
                ),
            )
            match_id = int(cursor.lastrowid)
        else:
            match_id = match.id
            connection.execute(
                """
                UPDATE matches
                SET match_label=?, competition_stage=?, match_date=?,
                    official_rbp_model=COALESCE(?, official_rbp_model),
                    official_rbp_claude=COALESCE(?, official_rbp_claude),
                    notes=?
                WHERE id=?
                """,
                (
                    match.match_label,
                    match.competition_stage,
                    match.match_date.isoformat(),
                    match.official_rbp_model,
                    match.official_rbp_claude,
                    match.notes,
                    match_id,
                ),
            )
            connection.execute("DELETE FROM question_records WHERE match_id=?", (match_id,))

        for record in validated:
            values = record.model_dump()
            connection.execute(
                """
                INSERT INTO question_records (
                    match_id, question_text, category, p_model, p_claude, p_crowd,
                    outcome, weight, brier_model, brier_claude, brier_crowd,
                    rbp_model, rbp_claude, model_vs_llm
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    match_id,
                    record.question_text,
                    record.category.value,
                    record.p_model,
                    record.p_claude,
                    record.p_crowd,
                    record.outcome,
                    record.weight,
                    record.brier_model,
                    record.brier_claude,
                    record.brier_crowd,
                    record.rbp_model,
                    record.rbp_claude,
                    record.model_vs_llm,
                ),
            )
    return match_id


def seed_database(path: str | Path = DATABASE_PATH) -> None:
    today = date.today()
    examples = [
        ("USA vs Mexico", "Group stage", today - timedelta(days=14)),
        ("Brazil vs Japan", "Group stage", today - timedelta(days=8)),
        ("France vs Senegal", "Round of 16", today - timedelta(days=2)),
    ]
    questions = [
        ("Over 2.5 total goals", 0.62, 0.58, 0.55, 1, 1.0),
        ("Both teams to score", 0.57, 0.61, 0.59, 0, 1.0),
        ("Home team to win", 0.54, 0.49, 0.52, 1, 1.2),
        ("Away team +1.5 handicap", 0.68, 0.72, 0.70, 1, 0.8),
        ("Over 9.5 corners", 0.51, 0.46, 0.48, 1, 1.0),
        ("Over 3.5 total cards", 0.66, 0.58, 0.61, 1, 0.7),
        ("Home team 4+ shots on target", 0.48, 0.55, 0.52, 0, 1.1),
        ("Penalty awarded in the match", 0.26, 0.31, 0.29, 0, 0.6),
    ]
    for match_index, (label, stage, match_date) in enumerate(examples):
        rows = []
        for question_index, (text, model, claude, crowd, outcome, weight) in enumerate(questions):
            shift = (match_index - 1) * 0.025 + (question_index % 3 - 1) * 0.01
            rows.append(
                {
                    "question_text": text,
                    "category": classify_question(text),
                    "p_model": min(0.95, max(0.05, model + shift)),
                    "p_claude": min(0.95, max(0.05, claude - shift / 2)),
                    "p_crowd": crowd,
                    "outcome": outcome if not (match_index == 1 and question_index == 7) else None,
                    "weight": weight,
                }
            )
        save_match(
            MatchRecord(
                match_label=label,
                competition_stage=stage,
                match_date=match_date,
                notes="Seed data for exploring RBP Lab.",
            ),
            pd.DataFrame(rows),
            DashboardSettings(),
            path,
        )


def seed_fixtures(path: str | Path = DATABASE_PATH) -> None:
    now = utc_now().replace(second=0, microsecond=0)
    examples = [
        FixtureRecord(
            match_label="Belgium vs Egypt",
            competition_stage="Group stage",
            kickoff_utc=now - timedelta(hours=2),
            submission_status=SubmissionStatus.MISSED,
            notes="Example missed fixture.",
        ),
        FixtureRecord(
            match_label="Canada vs Bosnia",
            competition_stage="Group stage",
            kickoff_utc=now + timedelta(minutes=20),
            notes="Example fixture inside the reminder window.",
        ),
        FixtureRecord(
            match_label="Argentina vs Morocco",
            competition_stage="Group stage",
            kickoff_utc=now + timedelta(hours=8),
            notes="Example upcoming fixture.",
        ),
    ]
    for fixture in examples:
        save_fixture(fixture, path)
