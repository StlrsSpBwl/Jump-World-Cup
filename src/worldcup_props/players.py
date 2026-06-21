from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .db import connect, initialize, transaction
from .market import match_key
from .strength import CompetitionStrengthModel


@dataclass(frozen=True)
class PlayerProfile:
    player_name: str
    team: str
    shots_per90: float
    shots_on_target_per90: float
    goals_per90: float
    assists_per90: float
    box_touches_per90: float | None
    penalty_taker: bool
    set_piece_role: float
    player_role: str = "unknown"
    effective_matches: float = 0.0


@dataclass(frozen=True)
class LineupEntry:
    match_key: str
    team: str
    player_name: str
    status: str
    start_probability: float
    expected_start_minutes: float
    sub_probability: float
    expected_sub_minutes: float
    confirmed: bool


@dataclass(frozen=True)
class PlayerClubProfile:
    player_name: str
    national_team: str
    club: str
    competition: str
    season: str
    minutes: float
    position: str
    national_role: str
    expected_minutes: float | None
    likely_starter: bool
    shots_per90: float | None
    shots_on_target_per90: float | None
    goals_per90: float | None
    assists_per90: float | None
    fouls_committed_per90: float | None
    fouls_drawn_per90: float | None
    yellow_cards_per90: float | None
    penalty_taker: bool
    set_piece_role: float
    source: str
    strength_multiplier: float
    strength_source: str


def ingest_player_profiles_csv(
    database_path: str | Path, csv_path: str | Path
) -> int:
    initialize(database_path)
    now = datetime.now(timezone.utc).isoformat()
    count = 0
    with Path(csv_path).open(newline="", encoding="utf-8-sig") as handle:
        with transaction(database_path) as connection:
            for row in csv.DictReader(handle):
                connection.execute(
                    """
                    INSERT INTO player_profiles (
                        player_name, team, shots_per90, shots_on_target_per90,
                        goals_per90, assists_per90, box_touches_per90,
                        penalty_taker, set_piece_role, player_role,
                        effective_matches, source, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(player_name, team) DO UPDATE SET
                        shots_per90=excluded.shots_per90,
                        shots_on_target_per90=excluded.shots_on_target_per90,
                        goals_per90=excluded.goals_per90,
                        assists_per90=excluded.assists_per90,
                        box_touches_per90=excluded.box_touches_per90,
                        penalty_taker=excluded.penalty_taker,
                        set_piece_role=excluded.set_piece_role,
                        player_role=excluded.player_role,
                        effective_matches=excluded.effective_matches,
                        source=excluded.source,
                        updated_at=excluded.updated_at
                    """,
                    (
                        row["player_name"].strip(),
                        row["team"].strip(),
                        float(row["shots_per90"]),
                        float(row["shots_on_target_per90"]),
                        float(row["goals_per90"]),
                        float(row["assists_per90"]),
                        _optional_float(row.get("box_touches_per90")),
                        int(_as_bool(row.get("penalty_taker"))),
                        float(row.get("set_piece_role") or 0.0),
                        _normalize_role(row.get("player_role")),
                        float(
                            row.get("effective_matches")
                            or row.get("profile_effective_matches")
                            or 0.0
                        ),
                        row.get("source") or "manual_csv",
                        now,
                    ),
                )
                count += 1
    return count


def ingest_player_club_profiles_csv(
    database_path: str | Path, csv_path: str | Path
) -> int:
    initialize(database_path)
    now = datetime.now(timezone.utc).isoformat()
    count = 0
    with Path(csv_path).open(newline="", encoding="utf-8-sig") as handle:
        with transaction(database_path) as connection:
            for row in csv.DictReader(handle):
                connection.execute(
                    """
                    INSERT INTO player_club_profiles (
                        player_name, national_team, club, competition, season,
                        minutes, position, national_role, expected_minutes,
                        likely_starter, shots_per90, shots_on_target_per90,
                        goals_per90, assists_per90, fouls_committed_per90,
                        fouls_drawn_per90, yellow_cards_per90, penalty_taker,
                        set_piece_role, source, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(player_name, national_team, club, competition, season)
                    DO UPDATE SET
                        minutes=excluded.minutes,
                        position=excluded.position,
                        national_role=excluded.national_role,
                        expected_minutes=excluded.expected_minutes,
                        likely_starter=excluded.likely_starter,
                        shots_per90=excluded.shots_per90,
                        shots_on_target_per90=excluded.shots_on_target_per90,
                        goals_per90=excluded.goals_per90,
                        assists_per90=excluded.assists_per90,
                        fouls_committed_per90=excluded.fouls_committed_per90,
                        fouls_drawn_per90=excluded.fouls_drawn_per90,
                        yellow_cards_per90=excluded.yellow_cards_per90,
                        penalty_taker=excluded.penalty_taker,
                        set_piece_role=excluded.set_piece_role,
                        source=excluded.source,
                        updated_at=excluded.updated_at
                    """,
                    (
                        row["player_name"].strip(),
                        row["national_team"].strip(),
                        row.get("club", "").strip(),
                        row["competition"].strip(),
                        row.get("season", "").strip() or "unknown",
                        float(row.get("minutes") or 0.0),
                        _normalize_role(row.get("position")),
                        _normalize_role(row.get("national_role")),
                        _optional_float(row.get("expected_minutes")),
                        int(_as_bool(row.get("likely_starter"))),
                        _optional_float(row.get("shots_per90")),
                        _optional_float(row.get("shots_on_target_per90")),
                        _optional_float(row.get("goals_per90")),
                        _optional_float(row.get("assists_per90")),
                        _optional_float(row.get("fouls_committed_per90")),
                        _optional_float(row.get("fouls_drawn_per90")),
                        _optional_float(row.get("yellow_cards_per90")),
                        int(_as_bool(row.get("penalty_taker"))),
                        float(row.get("set_piece_role") or 0.0),
                        row.get("source") or "manual_club_csv",
                        now,
                    ),
                )
                count += 1
    return count


def ingest_player_club_profile_rows(
    database_path: str | Path, rows: list[dict[str, object]]
) -> int:
    initialize(database_path)
    now = datetime.now(timezone.utc).isoformat()
    count = 0
    with transaction(database_path) as connection:
        for row in rows:
            connection.execute(
                """
                INSERT INTO player_club_profiles (
                    player_name, national_team, club, competition, season,
                    minutes, position, national_role, expected_minutes,
                    likely_starter, shots_per90, shots_on_target_per90,
                    goals_per90, assists_per90, fouls_committed_per90,
                    fouls_drawn_per90, yellow_cards_per90, penalty_taker,
                    set_piece_role, source, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(player_name, national_team, club, competition, season)
                DO UPDATE SET
                    minutes=excluded.minutes,
                    position=excluded.position,
                    national_role=excluded.national_role,
                    expected_minutes=excluded.expected_minutes,
                    likely_starter=excluded.likely_starter,
                    shots_per90=excluded.shots_per90,
                    shots_on_target_per90=excluded.shots_on_target_per90,
                    goals_per90=excluded.goals_per90,
                    assists_per90=excluded.assists_per90,
                    fouls_committed_per90=excluded.fouls_committed_per90,
                    fouls_drawn_per90=excluded.fouls_drawn_per90,
                    yellow_cards_per90=excluded.yellow_cards_per90,
                    penalty_taker=excluded.penalty_taker,
                    set_piece_role=excluded.set_piece_role,
                    source=excluded.source,
                    updated_at=excluded.updated_at
                """,
                (
                    str(row["player_name"]).strip(),
                    str(row["national_team"]).strip(),
                    str(row.get("club") or "").strip(),
                    str(row["competition"]).strip(),
                    str(row.get("season") or "unknown").strip() or "unknown",
                    float(row.get("minutes") or 0.0),
                    _normalize_role(row.get("position")),
                    _normalize_role(row.get("national_role")),
                    _optional_float(row.get("expected_minutes")),
                    int(_as_bool(row.get("likely_starter"))),
                    _optional_float(row.get("shots_per90")),
                    _optional_float(row.get("shots_on_target_per90")),
                    _optional_float(row.get("goals_per90")),
                    _optional_float(row.get("assists_per90")),
                    _optional_float(row.get("fouls_committed_per90")),
                    _optional_float(row.get("fouls_drawn_per90")),
                    _optional_float(row.get("yellow_cards_per90")),
                    int(_as_bool(row.get("penalty_taker"))),
                    float(row.get("set_piece_role") or 0.0),
                    str(row.get("source") or "manual_club_rows"),
                    now,
                ),
            )
            count += 1
    return count


def ingest_lineups_csv(database_path: str | Path, csv_path: str | Path) -> int:
    initialize(database_path)
    now = datetime.now(timezone.utc).isoformat()
    count = 0
    with Path(csv_path).open(newline="", encoding="utf-8-sig") as handle:
        with transaction(database_path) as connection:
            for row in csv.DictReader(handle):
                key = _lineup_match_key(row)
                status = (row.get("status") or "uncertain").strip().casefold()
                confirmed = _as_bool(row.get("confirmed"))
                start_probability, sub_probability = _lineup_probabilities(
                    status,
                    confirmed,
                    row.get("start_probability"),
                    row.get("sub_probability"),
                )
                connection.execute(
                    """
                    INSERT INTO lineup_entries (
                        match_key, team, player_name, status, start_probability,
                        expected_start_minutes, sub_probability,
                        expected_sub_minutes, confirmed, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(match_key, team, player_name) DO UPDATE SET
                        status=excluded.status,
                        start_probability=excluded.start_probability,
                        expected_start_minutes=excluded.expected_start_minutes,
                        sub_probability=excluded.sub_probability,
                        expected_sub_minutes=excluded.expected_sub_minutes,
                        confirmed=excluded.confirmed,
                        updated_at=excluded.updated_at
                    """,
                    (
                        key,
                        row["team"].strip(),
                        row["player_name"].strip(),
                        status,
                        start_probability,
                        float(row.get("expected_start_minutes") or 75.0),
                        sub_probability,
                        float(row.get("expected_sub_minutes") or 22.0),
                        int(confirmed),
                        row.get("timestamp") or now,
                    ),
                )
                count += 1
    return count


def load_match_players(
    database_path: str | Path, home: str, away: str
) -> tuple[dict[str, list[PlayerProfile]], dict[str, list[LineupEntry]]]:
    initialize(database_path)
    key = match_key(home, away)
    with connect(database_path) as connection:
        profiles = [
            PlayerProfile(
                player_name=str(row["player_name"]),
                team=str(row["team"]),
                shots_per90=float(row["shots_per90"]),
                shots_on_target_per90=float(row["shots_on_target_per90"]),
                goals_per90=float(row["goals_per90"]),
                assists_per90=float(row["assists_per90"]),
                box_touches_per90=(
                    float(row["box_touches_per90"])
                    if row["box_touches_per90"] is not None
                    else None
                ),
                penalty_taker=bool(row["penalty_taker"]),
                set_piece_role=float(row["set_piece_role"]),
                player_role=str(row["player_role"] or "unknown"),
                effective_matches=float(row["effective_matches"] or 0.0),
            )
            for row in connection.execute(
                "SELECT * FROM player_profiles WHERE team IN (?, ?)",
                (home, away),
            )
        ]
        existing_profile_keys = {(profile.player_name, profile.team) for profile in profiles}
        strength_model = CompetitionStrengthModel.load()
        club_profiles = [
            _club_row_to_player_profile(row, strength_model)
            for row in connection.execute(
                "SELECT * FROM player_club_profiles WHERE national_team IN (?, ?)",
                (home, away),
            )
            if (str(row["player_name"]), str(row["national_team"])) not in existing_profile_keys
        ]
        profiles.extend(club_profiles)
        lineups = [
            LineupEntry(
                match_key=str(row["match_key"]),
                team=str(row["team"]),
                player_name=str(row["player_name"]),
                status=str(row["status"]),
                start_probability=float(row["start_probability"]),
                expected_start_minutes=float(row["expected_start_minutes"]),
                sub_probability=float(row["sub_probability"]),
                expected_sub_minutes=float(row["expected_sub_minutes"]),
                confirmed=bool(row["confirmed"]),
            )
            for row in connection.execute(
                "SELECT * FROM lineup_entries WHERE match_key=?",
                (key,),
            )
        ]
    return (
        {team: [profile for profile in profiles if profile.team == team] for team in (home, away)},
        {team: [entry for entry in lineups if entry.team == team] for team in (home, away)},
    )


def load_club_profiles(
    database_path: str | Path,
    strength_model: CompetitionStrengthModel | None = None,
) -> dict[str, list[PlayerClubProfile]]:
    initialize(database_path)
    strength_model = strength_model or CompetitionStrengthModel.load()
    grouped: dict[str, list[PlayerClubProfile]] = {}
    with connect(database_path) as connection:
        for row in connection.execute("SELECT * FROM player_club_profiles"):
            adjustment = strength_model.adjustment(str(row["competition"]))
            profile = PlayerClubProfile(
                player_name=str(row["player_name"]),
                national_team=str(row["national_team"]),
                club=str(row["club"] or ""),
                competition=str(row["competition"]),
                season=str(row["season"] or ""),
                minutes=float(row["minutes"] or 0.0),
                position=str(row["position"] or "unknown"),
                national_role=str(row["national_role"] or row["position"] or "unknown"),
                expected_minutes=(
                    float(row["expected_minutes"])
                    if row["expected_minutes"] is not None
                    else None
                ),
                likely_starter=bool(row["likely_starter"]),
                shots_per90=_row_float(row, "shots_per90"),
                shots_on_target_per90=_row_float(row, "shots_on_target_per90"),
                goals_per90=_row_float(row, "goals_per90"),
                assists_per90=_row_float(row, "assists_per90"),
                fouls_committed_per90=_row_float(row, "fouls_committed_per90"),
                fouls_drawn_per90=_row_float(row, "fouls_drawn_per90"),
                yellow_cards_per90=_row_float(row, "yellow_cards_per90"),
                penalty_taker=bool(row["penalty_taker"]),
                set_piece_role=float(row["set_piece_role"] or 0.0),
                source=str(row["source"]),
                strength_multiplier=adjustment.multiplier,
                strength_source=adjustment.source,
            )
            grouped.setdefault(profile.national_team, []).append(profile)
    return grouped


def _club_row_to_player_profile(
    row: object, strength_model: CompetitionStrengthModel
) -> PlayerProfile:
    competition = str(row["competition"])
    multiplier = strength_model.adjustment(competition).multiplier
    minutes = float(row["minutes"] or 0.0)
    effective_matches = max(minutes / 90.0, 0.0)
    return PlayerProfile(
        player_name=str(row["player_name"]),
        team=str(row["national_team"]),
        shots_per90=float(row["shots_per90"] or 0.0) * multiplier,
        shots_on_target_per90=float(row["shots_on_target_per90"] or 0.0) * multiplier,
        goals_per90=float(row["goals_per90"] or 0.0) * multiplier,
        assists_per90=float(row["assists_per90"] or 0.0) * multiplier,
        box_touches_per90=None,
        penalty_taker=bool(row["penalty_taker"]),
        set_piece_role=float(row["set_piece_role"] or 0.0),
        player_role=_normalize_role(row["national_role"] or row["position"]),
        effective_matches=effective_matches,
    )


def _lineup_match_key(row: dict[str, str]) -> str:
    if row.get("match"):
        from .market import normalize_match_key

        return normalize_match_key(row["match"])
    return match_key(row["home"], row["away"])


def _lineup_probabilities(
    status: str,
    confirmed: bool,
    start_value: str | None,
    sub_value: str | None,
) -> tuple[float, float]:
    if confirmed:
        if status in {"start", "starter", "starting"}:
            return 1.0, 0.0
        if status in {"sub", "bench", "substitute"}:
            return 0.0, 1.0
        if status in {"out", "not_in_squad", "absent"}:
            return 0.0, 0.0
    start = float(start_value) if start_value not in {None, ""} else 0.65
    sub = float(sub_value) if sub_value not in {None, ""} else 0.25
    if start < 0 or sub < 0 or start + sub > 1.0 + 1e-9:
        raise ValueError("lineup start/sub probabilities must be non-negative and sum <= 1")
    return start, sub


def _optional_float(value: object | None) -> float | None:
    return None if value is None or str(value).strip() == "" else float(value)


def _row_float(row: object, key: str) -> float | None:
    return float(row[key]) if row[key] is not None else None


def _as_bool(value: str | None) -> bool:
    return str(value or "").strip().casefold() in {"1", "true", "yes", "y"}


def _normalize_role(value: str | None) -> str:
    role = str(value or "unknown").strip().casefold().replace(" ", "_")
    aliases = {
        "f": "forward",
        "fw": "forward",
        "cf": "forward",
        "striker": "forward",
        "st": "forward",
        "lst": "forward",
        "rst": "forward",
        "wing": "winger",
        "wide_forward": "winger",
        "lw": "winger",
        "rw": "winger",
        "lm": "winger",
        "rm": "winger",
        "am": "attacking_mid",
        "cam": "attacking_mid",
        "lam": "attacking_mid",
        "ram": "attacking_mid",
        "attacking_midfielder": "attacking_mid",
        "m": "central_mid",
        "cm": "central_mid",
        "lcm": "central_mid",
        "rcm": "central_mid",
        "cdm": "central_mid",
        "ldm": "central_mid",
        "rdm": "central_mid",
        "rcdm": "central_mid",
        "lcdm": "central_mid",
        "central_midfielder": "central_mid",
        "dm": "central_mid",
        "defensive_mid": "central_mid",
        "fb": "fullback",
        "full_back": "fullback",
        "lb": "fullback",
        "rb": "fullback",
        "rwb": "fullback",
        "lwb": "fullback",
        "d": "center_back",
        "cb": "center_back",
        "lcb": "center_back",
        "rcb": "center_back",
        "centre_back": "center_back",
    }
    return aliases.get(role, role or "unknown")
