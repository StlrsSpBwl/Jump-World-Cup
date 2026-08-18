from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .data import MatchRow, TeamStatsRow, _optional_float, _optional_int, ingest_matches
from .db import connect, initialize, transaction
from .market import match_key


TEAM_NAME_OVERRIDES = {
    "Cabo Verde": "Cape Verde",
    "IR Iran": "Iran",
    "Congo DR": "DR Congo",
}


@dataclass(frozen=True)
class FifaDatasetImportResult:
    matches_ingested: int
    referees_ingested: int
    tournament_context_rows_ingested: int
    completed_matches: int
    team_match_rows: int
    rows_with_shots_on_target: int
    rows_with_corners: int
    rows_with_fouls: int
    rows_with_offsides: int
    rows_with_cards: int
    rows_with_red_cards: int


def ingest_fifa_world_cup_dataset(
    database_path: str | Path,
    dataset_dir: str | Path,
) -> FifaDatasetImportResult:
    dataset = Path(dataset_dir)
    rows = list(match_rows_from_fifa_dataset(dataset))
    matches_ingested = ingest_matches(database_path, rows)
    referees_ingested = ingest_referees_from_fifa_dataset(database_path, dataset)
    context_rows = ingest_tournament_context_from_fifa_dataset(database_path, dataset)
    coverage = _coverage(database_path, source="fifa_world_cup_2026_dataset")
    return FifaDatasetImportResult(
        matches_ingested=matches_ingested,
        referees_ingested=referees_ingested,
        tournament_context_rows_ingested=context_rows,
        completed_matches=len(rows),
        **coverage,
    )


def match_rows_from_fifa_dataset(dataset_dir: str | Path) -> Iterable[MatchRow]:
    dataset = Path(dataset_dir)
    teams = _teams(dataset)
    matches = _read_csv_by_id(dataset / "matches.csv", "match_id")
    referees = _read_csv_by_id(dataset / "referees.csv", "referee_id")
    stats = _team_stats(dataset)
    cards = _card_events(dataset)
    for match_id, raw in sorted(matches.items(), key=lambda item: int(item[0])):
        if raw.get("status") != "Completed":
            continue
        home_team = teams[raw["home_team_id"]]
        away_team = teams[raw["away_team_id"]]
        home_stats = stats[match_id].get(raw["home_team_id"], {})
        away_stats = stats[match_id].get(raw["away_team_id"], {})
        home_cards = cards[match_id][raw["home_team_id"]]
        away_cards = cards[match_id][raw["away_team_id"]]
        referee = referees.get(raw.get("referee_id", ""), {})
        yield MatchRow(
            source="fifa_world_cup_2026_dataset",
            source_match_id=match_id,
            match_date=raw["date"],
            competition="FIFA World Cup 2026",
            competition_type="world_cup",
            home_team=home_team["team_name"],
            away_team=away_team["team_name"],
            home_confederation=home_team["confederation"],
            away_confederation=away_team["confederation"],
            neutral=True,
            home_elo=_optional_float(home_team.get("elo_rating")),
            away_elo=_optional_float(away_team.get("elo_rating")),
            referee_name=referee.get("name") or None,
            home_stats=_team_stats_row(
                home_team,
                away_team,
                home_stats,
                home_cards,
                is_home=True,
            ),
            away_stats=_team_stats_row(
                away_team,
                home_team,
                away_stats,
                away_cards,
                is_home=False,
            ),
        )


def ingest_referees_from_fifa_dataset(
    database_path: str | Path,
    dataset_dir: str | Path,
) -> int:
    initialize(database_path)
    now = datetime.now(timezone.utc).isoformat()
    count = 0
    with Path(dataset_dir, "referees.csv").open(newline="", encoding="utf-8-sig") as handle:
        with transaction(database_path) as connection:
            for row in csv.DictReader(handle):
                name = (row.get("name") or "").strip()
                if not name:
                    continue
                connection.execute(
                    """
                    INSERT INTO referees (
                        referee_name, matches, fouls_per_match, cards_per_match,
                        source, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(referee_name) DO UPDATE SET
                        matches=excluded.matches,
                        fouls_per_match=excluded.fouls_per_match,
                        cards_per_match=excluded.cards_per_match,
                        source=excluded.source,
                        updated_at=excluded.updated_at
                    """,
                    (
                        name,
                        1,
                        None,
                        _optional_float(row.get("avg_cards_per_game")),
                        "fifa_world_cup_2026_dataset",
                        now,
                    ),
                )
                count += 1
    return count


def ingest_tournament_context_from_fifa_dataset(
    database_path: str | Path,
    dataset_dir: str | Path,
) -> int:
    initialize(database_path)
    dataset = Path(dataset_dir)
    teams = _teams(dataset)
    matches = _read_csv_by_id(dataset / "matches.csv", "match_id")
    standings = _standings(teams, matches)
    team_event_summary = _team_event_summary(dataset, teams)
    now = datetime.now(timezone.utc).isoformat()
    count = 0
    with transaction(database_path) as connection:
        for raw in sorted(matches.values(), key=lambda row: int(row["match_id"])):
            if raw.get("status") == "Completed":
                continue
            home = teams[raw["home_team_id"]]["team_name"]
            away = teams[raw["away_team_id"]]["team_name"]
            key = match_key(home, away)
            for team in (home, away):
                row = standings[team]
                summary = team_event_summary[team]
                played = int(row["played"])
                points = int(row["points"])
                rank = int(row["rank"])
                must_win = played >= 1 and points <= 1 and rank >= 3
                coast_if_leading = played >= 1 and points >= 3 and rank <= 2
                goal_difference_priority = played >= 1 and rank in {2, 3, 4}
                notes = (
                    f"auto from mominullptr/FIFA-World-Cup-2026-Dataset: "
                    f"group {row['group']}, {points} pts, GD {row['gd']}, "
                    f"rank {rank}; {summary['goals_2h']} 2H goals, "
                    f"{summary['cards_2h']} 2H cards so far"
                )
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
                        points,
                        row["gd"],
                        rank,
                        None,
                        0,
                        0,
                        int(must_win),
                        int(goal_difference_priority),
                        int(coast_if_leading),
                        0,
                        None,
                        notes,
                        now,
                    ),
                )
                count += 1
    return count


def _team_stats_row(
    team: dict[str, str],
    opponent: dict[str, str],
    stats: dict[str, str],
    cards: dict[str, int],
    *,
    is_home: bool,
) -> TeamStatsRow:
    return TeamStatsRow(
        team=team["team_name"],
        opponent=opponent["team_name"],
        is_home=is_home,
        confederation=team["confederation"],
        fouls=_optional_int(stats.get("fouls")),
        corners=_optional_int(stats.get("corners")),
        offsides=_optional_int(stats.get("offsides")),
        shots_on_target=_optional_int(stats.get("shots_on_target")),
        cards=cards["cards"] or None,
        yellow_cards=cards["yellow_cards"] or None,
        red_cards=cards["red_cards"] or None,
        possession=_optional_float(stats.get("possession_pct")),
        first_half_cards=cards["first_half_cards"] or None,
        first_half_yellow_cards=cards["first_half_yellow_cards"] or None,
        first_half_red_cards=cards["first_half_red_cards"] or None,
    )


def _teams(dataset: Path) -> dict[str, dict[str, str]]:
    teams = _read_csv_by_id(dataset / "teams.csv", "team_id")
    for row in teams.values():
        row["team_name"] = TEAM_NAME_OVERRIDES.get(row["team_name"], row["team_name"])
    return teams


def _team_stats(dataset: Path) -> dict[str, dict[str, dict[str, str]]]:
    grouped: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    with (dataset / "match_team_stats.csv").open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            grouped[row["match_id"]][row["team_id"]] = row
    return grouped


def _card_events(dataset: Path) -> defaultdict[str, defaultdict[str, dict[str, int]]]:
    cards: defaultdict[str, defaultdict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(
            lambda: {
                "cards": 0,
                "yellow_cards": 0,
                "red_cards": 0,
                "first_half_cards": 0,
                "first_half_yellow_cards": 0,
                "first_half_red_cards": 0,
            }
        )
    )
    with (dataset / "match_events.csv").open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            event = (row.get("event_type") or "").strip().casefold()
            if event not in {"yellow card", "red card"}:
                continue
            minute = int(float(row["minute"]))
            bucket = cards[row["match_id"]][row["team_id"]]
            bucket["cards"] += 1
            if event == "yellow card":
                bucket["yellow_cards"] += 1
            else:
                bucket["red_cards"] += 1
            if minute <= 45:
                bucket["first_half_cards"] += 1
                if event == "yellow card":
                    bucket["first_half_yellow_cards"] += 1
                else:
                    bucket["first_half_red_cards"] += 1
    return cards


def _standings(
    teams: dict[str, dict[str, str]],
    matches: dict[str, dict[str, str]],
) -> dict[str, dict[str, Any]]:
    standings = {
        team["team_name"]: {
            "team": team["team_name"],
            "group": team["group_letter"],
            "played": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "gf": 0,
            "ga": 0,
            "gd": 0,
            "points": 0,
            "rank": None,
        }
        for team in teams.values()
    }
    for raw in matches.values():
        if raw.get("status") != "Completed":
            continue
        home = teams[raw["home_team_id"]]["team_name"]
        away = teams[raw["away_team_id"]]["team_name"]
        home_score = int(raw["home_score"])
        away_score = int(raw["away_score"])
        for team, goals_for, goals_against in (
            (home, home_score, away_score),
            (away, away_score, home_score),
        ):
            row = standings[team]
            row["played"] += 1
            row["gf"] += goals_for
            row["ga"] += goals_against
            row["gd"] = row["gf"] - row["ga"]
            if goals_for > goals_against:
                row["wins"] += 1
                row["points"] += 3
            elif goals_for == goals_against:
                row["draws"] += 1
                row["points"] += 1
            else:
                row["losses"] += 1
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in standings.values():
        by_group[row["group"]].append(row)
    for rows in by_group.values():
        rows.sort(key=lambda row: (-row["points"], -row["gd"], -row["gf"], row["team"]))
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
    return standings


def _team_event_summary(
    dataset: Path, teams: dict[str, dict[str, str]]
) -> defaultdict[str, dict[str, int]]:
    summary: defaultdict[str, dict[str, int]] = defaultdict(
        lambda: {"goals_1h": 0, "goals_2h": 0, "cards_1h": 0, "cards_2h": 0}
    )
    with (dataset / "match_events.csv").open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            team = teams[row["team_id"]]["team_name"]
            minute = int(float(row["minute"]))
            event = (row.get("event_type") or "").strip().casefold()
            if event == "goal":
                summary[team]["goals_1h" if minute <= 45 else "goals_2h"] += 1
            elif event in {"yellow card", "red card"}:
                summary[team]["cards_1h" if minute <= 45 else "cards_2h"] += 1
    return summary


def _read_csv_by_id(path: Path, key: str) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {row[key]: row for row in csv.DictReader(handle)}


def _coverage(database_path: str | Path, *, source: str) -> dict[str, int]:
    initialize(database_path)

    with connect(database_path) as connection:
        row = connection.execute(
            """
            SELECT
                COUNT(*) AS team_match_rows,
                SUM(shots_on_target IS NOT NULL) AS rows_with_shots_on_target,
                SUM(corners IS NOT NULL) AS rows_with_corners,
                SUM(fouls IS NOT NULL) AS rows_with_fouls,
                SUM(offsides IS NOT NULL) AS rows_with_offsides,
                SUM(cards IS NOT NULL) AS rows_with_cards,
                SUM(red_cards IS NOT NULL) AS rows_with_red_cards
            FROM team_match_stats AS stats
            JOIN matches AS matches ON matches.id = stats.match_id
            WHERE matches.source=?
            """,
            (source,),
        ).fetchone()
    return {key: int(row[key] or 0) for key in row.keys()}
