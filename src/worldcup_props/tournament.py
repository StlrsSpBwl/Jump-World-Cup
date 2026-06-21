from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .db import connect, initialize, transaction
from .market import match_key


@dataclass(frozen=True)
class TeamTournamentContext:
    team: str
    points: float | None = None
    goal_difference: float | None = None
    group_rank: int | None = None
    qualification_probability: float | None = None
    qualified: bool = False
    eliminated: bool = False
    must_win: bool = False
    goal_difference_priority: bool = False
    coast_if_leading: bool = False
    damage_limitation: bool = False
    tactical_style: str | None = None
    notes: str | None = None


@dataclass(frozen=True)
class MatchTournamentContext:
    home: TeamTournamentContext | None
    away: TeamTournamentContext | None

    @property
    def available(self) -> bool:
        return self.home is not None or self.away is not None


def ingest_tournament_context_csv(
    database_path: str | Path, csv_path: str | Path
) -> int:
    initialize(database_path)
    now = datetime.now(timezone.utc).isoformat()
    count = 0
    with Path(csv_path).open(newline="", encoding="utf-8-sig") as handle:
        with transaction(database_path) as connection:
            for row in csv.DictReader(handle):
                key = (row.get("match_key") or row.get("match") or "").strip()
                if not key:
                    home = (row.get("home") or row.get("home_team") or "").strip()
                    away = (row.get("away") or row.get("away_team") or "").strip()
                    if not home or not away:
                        raise ValueError(
                            "tournament context rows require match/match_key or home+away"
                        )
                    key = match_key(home, away)
                team = (row.get("team") or "").strip()
                if not team:
                    raise ValueError("tournament context rows require team")
                connection.execute(
                    """
                    INSERT INTO tournament_context (
                        match_key, team, points, goal_difference, group_rank,
                        qualification_probability, qualified, eliminated,
                        must_win, goal_difference_priority, coast_if_leading,
                        damage_limitation, tactical_style, notes, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(match_key, team) DO UPDATE SET
                        points=excluded.points,
                        goal_difference=excluded.goal_difference,
                        group_rank=excluded.group_rank,
                        qualification_probability=excluded.qualification_probability,
                        qualified=excluded.qualified,
                        eliminated=excluded.eliminated,
                        must_win=excluded.must_win,
                        goal_difference_priority=excluded.goal_difference_priority,
                        coast_if_leading=excluded.coast_if_leading,
                        damage_limitation=excluded.damage_limitation,
                        tactical_style=excluded.tactical_style,
                        notes=excluded.notes,
                        updated_at=excluded.updated_at
                    """,
                    (
                        key,
                        team,
                        _optional_float(row.get("points")),
                        _optional_float(row.get("goal_difference")),
                        _optional_int(row.get("group_rank")),
                        _optional_probability(row.get("qualification_probability")),
                        int(_as_bool(row.get("qualified"))),
                        int(_as_bool(row.get("eliminated"))),
                        int(_as_bool(row.get("must_win"))),
                        int(_as_bool(row.get("goal_difference_priority"))),
                        int(_as_bool(row.get("coast_if_leading"))),
                        int(_as_bool(row.get("damage_limitation"))),
                        (row.get("tactical_style") or "").strip() or None,
                        (row.get("notes") or "").strip() or None,
                        now,
                    ),
                )
                count += 1
    return count


def load_tournament_context(
    database_path: str | Path, home: str, away: str
) -> MatchTournamentContext | None:
    initialize(database_path)
    key = match_key(home, away)
    with connect(database_path) as connection:
        rows = {
            str(row["team"]): _row_to_context(row)
            for row in connection.execute(
                "SELECT * FROM tournament_context WHERE match_key=?",
                (key,),
            )
        }
    context = MatchTournamentContext(home=rows.get(home), away=rows.get(away))
    return context if context.available else None


def _row_to_context(row: object) -> TeamTournamentContext:
    return TeamTournamentContext(
        team=str(row["team"]),
        points=_row_float(row, "points"),
        goal_difference=_row_float(row, "goal_difference"),
        group_rank=(
            int(row["group_rank"]) if row["group_rank"] is not None else None
        ),
        qualification_probability=_row_float(row, "qualification_probability"),
        qualified=bool(row["qualified"]),
        eliminated=bool(row["eliminated"]),
        must_win=bool(row["must_win"]),
        goal_difference_priority=bool(row["goal_difference_priority"]),
        coast_if_leading=bool(row["coast_if_leading"]),
        damage_limitation=bool(row["damage_limitation"]),
        tactical_style=str(row["tactical_style"]) if row["tactical_style"] else None,
        notes=str(row["notes"]) if row["notes"] else None,
    )


def _row_float(row: object, key: str) -> float | None:
    value = row[key]
    if value is None or value == "":
        return None
    return float(value)


def _optional_float(value: str | None) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def _optional_int(value: str | None) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    return int(float(value))


def _optional_probability(value: str | None) -> float | None:
    parsed = _optional_float(value)
    if parsed is None:
        return None
    if parsed > 1.0:
        parsed /= 100.0
    return max(0.0, min(parsed, 1.0))


def _as_bool(value: object) -> bool:
    if value is None:
        return False
    return str(value).strip().casefold() in {"1", "true", "yes", "y", "on"}
