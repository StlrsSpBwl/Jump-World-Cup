from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .db import connect, initialize, transaction


@dataclass
class ValidationReport:
    issues: list[dict[str, Any]]
    coverage: list[dict[str, Any]]

    @property
    def error_count(self) -> int:
        return sum(issue["severity"] == "error" for issue in self.issues)


def validate_database(database_path: str | Path, persist: bool = True) -> ValidationReport:
    initialize(database_path)
    issues: list[dict[str, Any]] = []
    with connect(database_path) as connection:
        rows = connection.execute(
            """
            SELECT m.id, m.home_team, m.away_team, COUNT(s.team) AS stat_rows
            FROM matches m LEFT JOIN team_match_stats s ON s.match_id = m.id
            GROUP BY m.id
            """
        ).fetchall()
        for row in rows:
            if row["stat_rows"] != 2:
                issues.append(
                    _issue(row["id"], "error", "missing_team_row", "Match must have two stat rows")
                )

        stat_rows = connection.execute(
            """
            SELECT match_id, team, fouls, corners, offsides, shots_on_target, cards,
                   possession, first_half_fouls, first_half_corners,
                   first_half_offsides, first_half_shots_on_target, first_half_cards
            FROM team_match_stats
            """
        ).fetchall()
        maxima = {
            "fouls": 60,
            "corners": 35,
            "offsides": 20,
            "shots_on_target": 30,
            "cards": 15,
            "possession": 100,
        }
        for row in stat_rows:
            missing = [name for name in ("fouls", "corners", "offsides") if row[name] is None]
            if missing:
                issues.append(
                    _issue(
                        row["match_id"],
                        "warning",
                        "missing_stats",
                        f"{row['team']} missing {', '.join(missing)}",
                    )
                )
            for name, maximum in maxima.items():
                value = row[name]
                if value is not None and (value < 0 or value > maximum):
                    issues.append(
                        _issue(
                            row["match_id"],
                            "error",
                            "implausible_value",
                            f"{row['team']} {name}={value}",
                        )
                    )
            for name in ("fouls", "corners", "offsides", "shots_on_target", "cards"):
                first_half = row[f"first_half_{name}"]
                full = row[name]
                if first_half is not None and full is not None and first_half > full:
                    issues.append(
                        _issue(
                            row["match_id"],
                            "error",
                            "halftime_exceeds_fulltime",
                            f"{row['team']} first_half_{name}={first_half} > {full}",
                        )
                    )

        duplicates = connection.execute(
            """
            SELECT match_date, home_team, away_team, COUNT(*) AS n, GROUP_CONCAT(id) AS ids
            FROM matches
            GROUP BY match_date, home_team, away_team
            HAVING COUNT(*) > 1
            """
        ).fetchall()
        for row in duplicates:
            issues.append(
                _issue(
                    None,
                    "warning",
                    "possible_duplicate",
                    f"{row['match_date']} {row['home_team']}-{row['away_team']}: ids {row['ids']}",
                )
            )

        coverage = [
            dict(row)
            for row in connection.execute(
                """
                SELECT team,
                       COUNT(*) AS matches,
                       SUM(fouls IS NOT NULL) AS fouls_matches,
                       SUM(corners IS NOT NULL) AS corners_matches,
                       SUM(offsides IS NOT NULL) AS offsides_matches,
                       SUM(shots_on_target IS NOT NULL) AS shots_on_target_matches,
                       SUM(cards IS NOT NULL) AS cards_matches,
                       SUM(first_half_corners IS NOT NULL) AS first_half_corners_matches
                FROM team_match_stats
                GROUP BY team
                ORDER BY matches DESC, team
                """
            )
        ]

    if persist:
        with transaction(database_path) as connection:
            connection.execute("DELETE FROM data_quality_issues")
            connection.executemany(
                """
                INSERT INTO data_quality_issues (match_id, severity, code, message)
                VALUES (:match_id, :severity, :code, :message)
                """,
                issues,
            )
    return ValidationReport(issues=issues, coverage=coverage)


def _issue(
    match_id: int | None, severity: str, code: str, message: str
) -> dict[str, Any]:
    return {
        "match_id": match_id,
        "severity": severity,
        "code": code,
        "message": message,
    }
