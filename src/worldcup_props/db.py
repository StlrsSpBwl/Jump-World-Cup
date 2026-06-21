from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS matches (
    id INTEGER PRIMARY KEY,
    source TEXT NOT NULL,
    source_match_id TEXT NOT NULL,
    match_date TEXT NOT NULL,
    competition TEXT NOT NULL,
    competition_type TEXT NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    home_confederation TEXT,
    away_confederation TEXT,
    neutral INTEGER NOT NULL DEFAULT 1,
    home_elo REAL,
    away_elo REAL,
    referee_name TEXT,
    UNIQUE(source, source_match_id)
);

CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(match_date);
CREATE INDEX IF NOT EXISTS idx_matches_teams ON matches(home_team, away_team);

CREATE TABLE IF NOT EXISTS team_match_stats (
    match_id INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    team TEXT NOT NULL,
    opponent TEXT NOT NULL,
    is_home INTEGER NOT NULL,
    confederation TEXT,
    fouls INTEGER,
    corners INTEGER,
    offsides INTEGER,
    shots_on_target INTEGER,
    cards INTEGER,
    yellow_cards INTEGER,
    red_cards INTEGER,
    possession REAL,
    first_half_fouls INTEGER,
    first_half_corners INTEGER,
    first_half_offsides INTEGER,
    first_half_shots_on_target INTEGER,
    first_half_cards INTEGER,
    first_half_yellow_cards INTEGER,
    first_half_red_cards INTEGER,
    pressing_proxy REAL,
    PRIMARY KEY (match_id, team)
);

CREATE INDEX IF NOT EXISTS idx_stats_team ON team_match_stats(team);

CREATE TABLE IF NOT EXISTS referees (
    referee_name TEXT PRIMARY KEY,
    matches INTEGER NOT NULL,
    fouls_per_match REAL,
    cards_per_match REAL,
    source TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS player_profiles (
    player_name TEXT NOT NULL,
    team TEXT NOT NULL,
    shots_per90 REAL NOT NULL,
    shots_on_target_per90 REAL NOT NULL,
    goals_per90 REAL NOT NULL,
    assists_per90 REAL NOT NULL,
    box_touches_per90 REAL,
    penalty_taker INTEGER NOT NULL DEFAULT 0,
    set_piece_role REAL NOT NULL DEFAULT 0,
    player_role TEXT NOT NULL DEFAULT 'unknown',
    effective_matches REAL NOT NULL DEFAULT 0,
    source TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (player_name, team)
);

CREATE TABLE IF NOT EXISTS player_club_profiles (
    player_name TEXT NOT NULL,
    national_team TEXT NOT NULL,
    club TEXT NOT NULL,
    competition TEXT NOT NULL,
    season TEXT NOT NULL,
    minutes REAL NOT NULL,
    position TEXT,
    national_role TEXT,
    expected_minutes REAL,
    likely_starter INTEGER NOT NULL DEFAULT 0,
    shots_per90 REAL,
    shots_on_target_per90 REAL,
    goals_per90 REAL,
    assists_per90 REAL,
    fouls_committed_per90 REAL,
    fouls_drawn_per90 REAL,
    yellow_cards_per90 REAL,
    penalty_taker INTEGER NOT NULL DEFAULT 0,
    set_piece_role REAL NOT NULL DEFAULT 0,
    source TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (player_name, national_team, club, competition, season)
);

CREATE INDEX IF NOT EXISTS idx_player_club_team
    ON player_club_profiles(national_team, player_name);

CREATE TABLE IF NOT EXISTS lineup_entries (
    match_key TEXT NOT NULL,
    team TEXT NOT NULL,
    player_name TEXT NOT NULL,
    status TEXT NOT NULL,
    start_probability REAL NOT NULL,
    expected_start_minutes REAL NOT NULL,
    sub_probability REAL NOT NULL,
    expected_sub_minutes REAL NOT NULL,
    confirmed INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (match_key, team, player_name)
);

CREATE INDEX IF NOT EXISTS idx_lineup_match ON lineup_entries(match_key, team);

CREATE TABLE IF NOT EXISTS market_quotes (
    id INTEGER PRIMARY KEY,
    question_key TEXT NOT NULL,
    provider TEXT NOT NULL,
    outcome TEXT NOT NULL,
    decimal_odds REAL NOT NULL,
    observed_at TEXT NOT NULL,
    definition TEXT,
    UNIQUE(question_key, provider, outcome, observed_at)
);

CREATE TABLE IF NOT EXISTS match_market_quotes (
    id INTEGER PRIMARY KEY,
    match_key TEXT NOT NULL,
    market_type TEXT NOT NULL,
    selection TEXT NOT NULL,
    line REAL,
    decimal_odds REAL NOT NULL,
    book TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    source TEXT NOT NULL,
    raw_implied_probability REAL NOT NULL,
    devigged_probability REAL,
    devig_method TEXT,
    definition TEXT,
    UNIQUE(match_key, market_type, selection, line, book, observed_at)
);

CREATE INDEX IF NOT EXISTS idx_match_market_lookup
    ON match_market_quotes(match_key, market_type, line, observed_at);

CREATE TABLE IF NOT EXISTS tournament_context (
    match_key TEXT NOT NULL,
    team TEXT NOT NULL,
    points REAL,
    goal_difference REAL,
    group_rank INTEGER,
    qualification_probability REAL,
    qualified INTEGER NOT NULL DEFAULT 0,
    eliminated INTEGER NOT NULL DEFAULT 0,
    must_win INTEGER NOT NULL DEFAULT 0,
    goal_difference_priority INTEGER NOT NULL DEFAULT 0,
    coast_if_leading INTEGER NOT NULL DEFAULT 0,
    damage_limitation INTEGER NOT NULL DEFAULT 0,
    tactical_style TEXT,
    notes TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (match_key, team)
);

CREATE INDEX IF NOT EXISTS idx_tournament_context_match
    ON tournament_context(match_key);

CREATE TABLE IF NOT EXISTS forecast_results (
    id INTEGER PRIMARY KEY,
    match_key TEXT NOT NULL,
    question_key TEXT NOT NULL,
    question_type TEXT NOT NULL,
    submitted_probability REAL NOT NULL,
    crowd_probability REAL,
    outcome INTEGER NOT NULL,
    market_blended_probability REAL,
    weight REAL NOT NULL DEFAULT 1.0,
    observed_at TEXT NOT NULL,
    UNIQUE(match_key, question_key, observed_at)
);

CREATE INDEX IF NOT EXISTS idx_forecast_results_type
    ON forecast_results(question_type);

CREATE TABLE IF NOT EXISTS data_quality_issues (
    id INTEGER PRIMARY KEY,
    match_id INTEGER REFERENCES matches(id) ON DELETE CASCADE,
    severity TEXT NOT NULL,
    code TEXT NOT NULL,
    message TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(destination)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize(path: str | Path) -> None:
    with connect(path) as connection:
        connection.executescript(SCHEMA)
        _add_column(connection, "team_match_stats", "shots_on_target", "INTEGER")
        _add_column(
            connection, "team_match_stats", "first_half_shots_on_target", "INTEGER"
        )
        _add_column(connection, "team_match_stats", "cards", "INTEGER")
        _add_column(connection, "team_match_stats", "first_half_cards", "INTEGER")
        _add_column(connection, "team_match_stats", "yellow_cards", "INTEGER")
        _add_column(connection, "team_match_stats", "red_cards", "INTEGER")
        _add_column(connection, "team_match_stats", "first_half_yellow_cards", "INTEGER")
        _add_column(connection, "team_match_stats", "first_half_red_cards", "INTEGER")
        _add_column(
            connection,
            "player_profiles",
            "player_role",
            "TEXT NOT NULL DEFAULT 'unknown'",
        )
        _add_column(
            connection,
            "player_profiles",
            "effective_matches",
            "REAL NOT NULL DEFAULT 0",
        )
        _add_column(connection, "player_club_profiles", "national_role", "TEXT")
        _add_column(connection, "player_club_profiles", "expected_minutes", "REAL")
        _add_column(
            connection,
            "player_club_profiles",
            "likely_starter",
            "INTEGER NOT NULL DEFAULT 0",
        )
        _add_column(
            connection,
            "player_club_profiles",
            "penalty_taker",
            "INTEGER NOT NULL DEFAULT 0",
        )
        _add_column(
            connection,
            "player_club_profiles",
            "set_piece_role",
            "REAL NOT NULL DEFAULT 0",
        )
        _add_column(connection, "tournament_context", "tactical_style", "TEXT")


def _add_column(
    connection: sqlite3.Connection, table: str, column: str, declaration: str
) -> None:
    existing = {
        str(row["name"])
        for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
    }
    if column not in existing:
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")


@contextmanager
def transaction(path: str | Path) -> Iterator[sqlite3.Connection]:
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
