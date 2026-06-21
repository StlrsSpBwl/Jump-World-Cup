from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import requests
from bs4 import BeautifulSoup, Comment

from .db import initialize, transaction
from .registry import clean_confederation, load_team_confederation_registry


STAT_COLUMNS = (
    "fouls",
    "corners",
    "offsides",
    "shots_on_target",
    "cards",
    "yellow_cards",
    "red_cards",
    "possession",
    "first_half_fouls",
    "first_half_corners",
    "first_half_offsides",
    "first_half_shots_on_target",
    "first_half_cards",
    "first_half_yellow_cards",
    "first_half_red_cards",
    "pressing_proxy",
)

FBREF_MANIFEST_COLUMNS = (
    "url",
    "source_match_id",
    "match_date",
    "competition",
    "competition_type",
    "home_team",
    "away_team",
    "home_confederation",
    "away_confederation",
    "neutral",
    "home_elo",
    "away_elo",
    "referee_name",
)


@dataclass
class TeamStatsRow:
    team: str
    opponent: str
    is_home: bool
    confederation: str | None = None
    fouls: int | None = None
    corners: int | None = None
    offsides: int | None = None
    shots_on_target: int | None = None
    cards: int | None = None
    yellow_cards: int | None = None
    red_cards: int | None = None
    possession: float | None = None
    first_half_fouls: int | None = None
    first_half_corners: int | None = None
    first_half_offsides: int | None = None
    first_half_shots_on_target: int | None = None
    first_half_cards: int | None = None
    first_half_yellow_cards: int | None = None
    first_half_red_cards: int | None = None
    pressing_proxy: float | None = None


@dataclass
class MatchRow:
    source: str
    source_match_id: str
    match_date: str
    competition: str
    competition_type: str
    home_team: str
    away_team: str
    home_stats: TeamStatsRow
    away_stats: TeamStatsRow
    home_confederation: str | None = None
    away_confederation: str | None = None
    neutral: bool = True
    home_elo: float | None = None
    away_elo: float | None = None
    referee_name: str | None = None


class DataProvider:
    def fetch(self, start_date: str) -> Iterable[MatchRow]:
        ...


class CachedHttpClient:
    def __init__(
        self,
        cache_dir: str | Path,
        user_agent: str = "worldcup-props/0.1 research contact: local-user",
        min_delay_seconds: float = 3.0,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.user_agent = user_agent
        self.min_delay_seconds = min_delay_seconds
        self._last_request = 0.0

    def get_text(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        suffix: str = ".html",
        force_refresh: bool = False,
    ) -> str:
        key = hashlib.sha256(
            json.dumps([url, sorted((params or {}).items())], sort_keys=True).encode()
        ).hexdigest()
        cached = self.cache_dir / f"{key}{suffix}"
        if cached.exists() and not force_refresh:
            return cached.read_text(encoding="utf-8")
        delay = self.min_delay_seconds - (time.monotonic() - self._last_request)
        if delay > 0:
            time.sleep(delay)
        request_headers = {"User-Agent": self.user_agent}
        request_headers.update(headers or {})
        response = requests.get(url, params=params, headers=request_headers, timeout=30)
        self._last_request = time.monotonic()
        response.raise_for_status()
        cached.write_text(response.text, encoding="utf-8")
        return response.text

    def get_json(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        force_refresh: bool = False,
    ) -> Any:
        return json.loads(
            self.get_text(
                url,
                params=params,
                headers=headers,
                suffix=".json",
                force_refresh=force_refresh,
            )
        )


def _optional_int(value: Any) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    return int(float(str(value).replace("%", "").strip()))


def _optional_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(str(value).replace("%", "").strip())


def ingest_matches(database_path: str | Path, rows: Iterable[MatchRow]) -> int:
    initialize(database_path)
    count = 0
    registry = load_team_confederation_registry(missing_ok=True)
    with transaction(database_path) as connection:
        for row in rows:
            home_confederation = (
                registry.confederation_for(row.home_team)
                or clean_confederation(row.home_confederation)
            )
            away_confederation = (
                registry.confederation_for(row.away_team)
                or clean_confederation(row.away_confederation)
            )
            connection.execute(
                """
                INSERT INTO matches (
                    source, source_match_id, match_date, competition, competition_type,
                    home_team, away_team, home_confederation, away_confederation,
                    neutral, home_elo, away_elo, referee_name
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source, source_match_id) DO UPDATE SET
                    match_date=excluded.match_date,
                    competition=excluded.competition,
                    competition_type=excluded.competition_type,
                    home_team=excluded.home_team,
                    away_team=excluded.away_team,
                    home_confederation=excluded.home_confederation,
                    away_confederation=excluded.away_confederation,
                    neutral=excluded.neutral,
                    home_elo=excluded.home_elo,
                    away_elo=excluded.away_elo,
                    referee_name=excluded.referee_name
                """,
                (
                    row.source,
                    row.source_match_id,
                    row.match_date,
                    row.competition,
                    row.competition_type,
                    row.home_team,
                    row.away_team,
                    home_confederation,
                    away_confederation,
                    int(row.neutral),
                    row.home_elo,
                    row.away_elo,
                    row.referee_name,
                ),
            )
            match_id = int(
                connection.execute(
                    "SELECT id FROM matches WHERE source=? AND source_match_id=?",
                    (row.source, row.source_match_id),
                ).fetchone()[0]
            )
            for stats in (row.home_stats, row.away_stats):
                stats_confederation = (
                    registry.confederation_for(stats.team)
                    or clean_confederation(stats.confederation)
                )
                connection.execute(
                    """
                    INSERT INTO team_match_stats (
                        match_id, team, opponent, is_home, confederation, fouls, corners,
                        offsides, shots_on_target, cards, yellow_cards, red_cards,
                        possession, first_half_fouls,
                        first_half_corners, first_half_offsides,
                        first_half_shots_on_target, first_half_cards,
                        first_half_yellow_cards, first_half_red_cards, pressing_proxy
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(match_id, team) DO UPDATE SET
                        opponent=excluded.opponent,
                        is_home=excluded.is_home,
                        confederation=excluded.confederation,
                        fouls=excluded.fouls,
                        corners=excluded.corners,
                        offsides=excluded.offsides,
                        shots_on_target=excluded.shots_on_target,
                        cards=excluded.cards,
                        yellow_cards=excluded.yellow_cards,
                        red_cards=excluded.red_cards,
                        possession=excluded.possession,
                        first_half_fouls=excluded.first_half_fouls,
                        first_half_corners=excluded.first_half_corners,
                        first_half_offsides=excluded.first_half_offsides,
                        first_half_shots_on_target=excluded.first_half_shots_on_target,
                        first_half_cards=excluded.first_half_cards,
                        first_half_yellow_cards=excluded.first_half_yellow_cards,
                        first_half_red_cards=excluded.first_half_red_cards,
                        pressing_proxy=excluded.pressing_proxy
                    """,
                    (
                        match_id,
                        stats.team,
                        stats.opponent,
                        int(stats.is_home),
                        stats_confederation,
                        stats.fouls,
                        stats.corners,
                        stats.offsides,
                        stats.shots_on_target,
                        stats.cards,
                        stats.yellow_cards,
                        stats.red_cards,
                        stats.possession,
                        stats.first_half_fouls,
                        stats.first_half_corners,
                        stats.first_half_offsides,
                        stats.first_half_shots_on_target,
                        stats.first_half_cards,
                        stats.first_half_yellow_cards,
                        stats.first_half_red_cards,
                        stats.pressing_proxy,
                    ),
                )
            count += 1
    return count


def rows_from_csv(path: str | Path, source: str = "csv") -> Iterable[MatchRow]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        for index, raw in enumerate(csv.DictReader(handle), start=1):
            home = raw["home_team"].strip()
            away = raw["away_team"].strip()

            def team_stats(prefix: str, team: str, opponent: str, is_home: bool) -> TeamStatsRow:
                return TeamStatsRow(
                    team=team,
                    opponent=opponent,
                    is_home=is_home,
                    confederation=raw.get(f"{prefix}_confederation") or None,
                    **{
                        column: (
                            _optional_float(raw.get(f"{prefix}_{column}"))
                            if column in {"possession", "pressing_proxy"}
                            else _optional_int(raw.get(f"{prefix}_{column}"))
                        )
                        for column in STAT_COLUMNS
                    },
                )

            yield MatchRow(
                source=raw.get("source") or source,
                source_match_id=raw.get("source_match_id") or f"{source}-{index}",
                match_date=raw["match_date"],
                competition=raw.get("competition") or "Unknown",
                competition_type=raw.get("competition_type") or "other",
                home_team=home,
                away_team=away,
                home_stats=team_stats("home", home, away, True),
                away_stats=team_stats("away", away, home, False),
                home_confederation=raw.get("home_confederation") or None,
                away_confederation=raw.get("away_confederation") or None,
                neutral=str(raw.get("neutral", "1")).strip().casefold()
                not in {"0", "false", "no"},
                home_elo=_optional_float(raw.get("home_elo")),
                away_elo=_optional_float(raw.get("away_elo")),
                referee_name=raw.get("referee_name") or None,
            )


def ingest_referees_csv(database_path: str | Path, path: str | Path) -> int:
    initialize(database_path)
    now = datetime.now(timezone.utc).isoformat()
    count = 0
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        with transaction(database_path) as connection:
            for row in csv.DictReader(handle):
                connection.execute(
                    """
                    INSERT INTO referees (
                        referee_name, matches, fouls_per_match, cards_per_match, source, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(referee_name) DO UPDATE SET
                        matches=excluded.matches,
                        fouls_per_match=excluded.fouls_per_match,
                        cards_per_match=excluded.cards_per_match,
                        source=excluded.source,
                        updated_at=excluded.updated_at
                    """,
                    (
                        row["referee_name"].strip(),
                        int(row["matches"]),
                        _optional_float(row.get("fouls_per_match")),
                        _optional_float(row.get("cards_per_match")),
                        row.get("source") or "manual_csv",
                        now,
                    ),
                )
                count += 1
    return count


def ingest_statshub_referees(
    database_path: str | Path, rows: Iterable[Mapping[str, Any]]
) -> int:
    initialize(database_path)
    now = datetime.now(timezone.utc).isoformat()
    count = 0
    with transaction(database_path) as connection:
        for row in rows:
            if not row.get("name") or row.get("games") is None:
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
                    str(row["name"]).strip(),
                    int(row["games"]),
                    _optional_float(row.get("avgFoulsPerGame")),
                    _optional_float(row.get("avgCardsPerGame")),
                    "statshub",
                    now,
                ),
            )
            count += 1
    return count


class APIFootballProvider:
    """Optional API-Football provider for fixtures with statistics."""

    BASE_URL = "https://v3.football.api-sports.io"

    def __init__(self, cache_dir: str | Path, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("API_FOOTBALL_KEY")
        if not self.api_key:
            raise ValueError("API_FOOTBALL_KEY is required")
        self.http = CachedHttpClient(Path(cache_dir) / "api_football", min_delay_seconds=1.0)

    def fixture_statistics(self, fixture_id: int) -> Any:
        return self.http.get_json(
            f"{self.BASE_URL}/fixtures/statistics",
            params={"fixture": fixture_id},
            headers={"x-apisports-key": self.api_key},
        )

    def fixtures(self, start_date: str, end_date: str) -> Any:
        return self.http.get_json(
            f"{self.BASE_URL}/fixtures",
            params={"from": start_date, "to": end_date},
            headers={"x-apisports-key": self.api_key},
        )


class StatsHubProvider:
    """Cached adapter for StatsHub's public team-statistics endpoints."""

    BASE_URL = "https://www.statshub.com"
    STAT_KEYS = {
        "fouls": "fouls",
        "corners": "cornerKicks",
        "offsides": "offsides",
        "shots_on_target": "shotsOnGoal",
        "cards": "cards",
        "yellow_cards": "yellowCards",
        "red_cards": "redCards",
        "possession": "ballPossession",
    }

    def __init__(
        self,
        teams_path: str | Path,
        cache_dir: str | Path,
        *,
        limit: int = 500,
    ) -> None:
        self.teams_path = Path(teams_path)
        self.limit = limit
        self.http = CachedHttpClient(
            Path(cache_dir) / "statshub",
            user_agent="worldcup-props/0.1 research importer",
            min_delay_seconds=1.0,
        )
        self.teams = self._load_teams()

    def fetch(self, start_date: str) -> Iterable[MatchRow]:
        matches: dict[int, dict[str, Any]] = {}
        for team_id in self.teams:
            for stat, statistic_key in self.STAT_KEYS.items():
                self._merge_stat(matches, team_id, stat, statistic_key, "ALL")
                if stat != "possession":
                    self._merge_stat(
                        matches,
                        team_id,
                        f"first_half_{stat}",
                        statistic_key,
                        "1ST",
                    )
        for event_id, values in sorted(
            matches.items(), key=lambda item: item[1]["timestamp"]
        ):
            match_date = datetime.fromtimestamp(
                values["timestamp"], tz=timezone.utc
            ).date().isoformat()
            if match_date < start_date:
                continue
            yield self._to_match_row(event_id, match_date, values)

    def fetch_referees(self) -> list[dict[str, Any]]:
        response = self.http.get_json(
            f"{self.BASE_URL}/api/world-cup/referees",
            params={"scope": "career"},
        )
        return list(response.get("data", []))

    def discover_world_cup_teams(
        self,
        season_id: int = 58210,
        unique_tournament_id: int = 16,
    ) -> list[dict[str, Any]]:
        response = self.http.get_json(
            f"{self.BASE_URL}/api/unique-tournament/"
            f"{unique_tournament_id}/{season_id}/events"
        )
        found: dict[int, dict[str, Any]] = {}
        for item in response.get("data", []):
            for side in ("homeTeam", "awayTeam"):
                team = item.get(side) or {}
                if team.get("id") is not None:
                    found[int(team["id"])] = team
        return sorted(found.values(), key=lambda team: str(team.get("name", "")))

    def fetch_player_profile_rows(
        self,
        *,
        limit: int = 50,
        min_minutes: float = 90.0,
        fixture_id: int | None = None,
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for team_id, team in self.teams.items():
            params: dict[str, object] = {
                "limit": limit,
                "location": "both",
            }
            if fixture_id is not None:
                params["fixtureId"] = fixture_id
            response = self.http.get_json(
                f"{self.BASE_URL}/api/team/{team_id}/players/performance",
                params=params,
            )
            team_name = team["team"]
            confederation = clean_confederation(team.get("confederation")) or "UNK"
            for player in response.get("data", []):
                row = self._player_performance_row(
                    player,
                    team_name=team_name,
                    confederation=confederation,
                    min_minutes=min_minutes,
                )
                if row is not None:
                    rows.append(row)
        return rows

    def _load_teams(self) -> dict[int, dict[str, str]]:
        teams: dict[int, dict[str, str]] = {}
        with self.teams_path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                team_id = int(row["statshub_team_id"])
                teams[team_id] = {
                    "team": row["team"].strip(),
                    "confederation": (row.get("confederation") or "UNK").strip(),
                }
        if not teams:
            raise ValueError("StatsHub team registry contains no teams")
        return teams

    def _player_performance_row(
        self,
        player: Mapping[str, Any],
        *,
        team_name: str,
        confederation: str,
        min_minutes: float,
    ) -> dict[str, object] | None:
        totals = {
            "minutes": 0.0,
            "shots": 0.0,
            "shots_on_target": 0.0,
            "goals": 0.0,
            "assists": 0.0,
            "fouls_committed": 0.0,
            "fouls_drawn": 0.0,
            "yellow_cards": 0.0,
        }
        positions: dict[str, int] = {}
        starts = 0
        appearances = 0
        stats = player.get("stats") or {}
        if not isinstance(stats, Mapping):
            return None
        player_name = str(player.get("name") or "").strip()
        if not player_name:
            return None
        for item in stats.values():
            if not isinstance(item, Mapping):
                continue
            minutes = _optional_float(item.get("minutesPlayed")) or 0.0
            if minutes <= 0:
                continue
            appearances += 1
            if minutes >= 45:
                starts += 1
            totals["minutes"] += minutes
            totals["shots"] += _optional_float(item.get("shots")) or 0.0
            totals["shots_on_target"] += (
                _optional_float(item.get("onTargetScoringAttempt")) or 0.0
            )
            totals["goals"] += _optional_float(item.get("goals")) or 0.0
            totals["assists"] += _optional_float(item.get("goalAssist")) or 0.0
            totals["fouls_committed"] += _optional_float(item.get("fouls")) or 0.0
            totals["fouls_drawn"] += _optional_float(item.get("wasFouled")) or 0.0
            totals["yellow_cards"] += _optional_float(item.get("yellowCard")) or 0.0
            position = str(item.get("position") or player.get("position") or "").strip()
            if position:
                positions[position] = positions.get(position, 0) + 1
        minutes = totals["minutes"]
        if minutes < min_minutes:
            return None
        position = max(positions.items(), key=lambda item: item[1])[0] if positions else str(
            player.get("position") or "unknown"
        )
        factor = 90.0 / minutes
        competition = f"International {confederation}" if confederation != "UNK" else "International"
        return {
            "player_name": player_name,
            "national_team": team_name,
            "club": "StatsHub recent",
            "competition": competition,
            "season": "recent",
            "minutes": minutes,
            "position": position,
            "national_role": position,
            "expected_minutes": 0.0,
            "likely_starter": False,
            "shots_per90": totals["shots"] * factor,
            "shots_on_target_per90": totals["shots_on_target"] * factor,
            "goals_per90": totals["goals"] * factor,
            "assists_per90": totals["assists"] * factor,
            "fouls_committed_per90": totals["fouls_committed"] * factor,
            "fouls_drawn_per90": totals["fouls_drawn"] * factor,
            "yellow_cards_per90": totals["yellow_cards"] * factor,
            "penalty_taker": False,
            "set_piece_role": 0.0,
            "source": "statshub_player_performance",
        }

    def _merge_stat(
        self,
        matches: dict[int, dict[str, Any]],
        team_id: int,
        target: str,
        statistic_key: str,
        half: str,
    ) -> None:
        response = self.http.get_json(
            f"{self.BASE_URL}/api/team/{team_id}/event-statistics",
            params={
                "eventType": "all",
                "statisticKey": statistic_key,
                "eventHalf": half,
                "limit": self.limit,
            },
        )
        for item in response.get("data", []):
            event_id = int(item["event_id"])
            match = matches.setdefault(
                event_id,
                {
                    "timestamp": int(item["time_start_timestamp"]),
                    "home_team_id": int(item["home_team_id"]),
                    "away_team_id": int(item["away_team_id"]),
                    "home_team": item["home_team_name"],
                    "away_team": item["away_team_name"],
                    "competition": item.get("league_name") or "Unknown",
                    "home": {},
                    "away": {},
                },
            )
            match["home"][target] = _optional_float(item.get("home_value"))
            match["away"][target] = _optional_float(item.get("away_value"))

    def _to_match_row(
        self, event_id: int, match_date: str, values: dict[str, Any]
    ) -> MatchRow:
        home_id = int(values["home_team_id"])
        away_id = int(values["away_team_id"])
        home_registry = self.teams.get(home_id, {})
        away_registry = self.teams.get(away_id, {})
        home_name = home_registry.get("team") or values["home_team"]
        away_name = away_registry.get("team") or values["away_team"]
        home_confederation = home_registry.get("confederation") or None
        away_confederation = away_registry.get("confederation") or None
        competition = str(values["competition"])

        def stats(side: str, team: str, opponent: str, is_home: bool) -> TeamStatsRow:
            data = values[side]
            cards = _optional_int(data.get("cards"))
            yellow_cards = _optional_int(data.get("yellow_cards"))
            red_cards = _optional_int(data.get("red_cards"))
            if red_cards is None and cards is not None and yellow_cards is not None:
                red_cards = max(cards - yellow_cards, 0)
            if cards is None and yellow_cards is not None and red_cards is not None:
                cards = yellow_cards + red_cards
            first_half_cards = _optional_int(data.get("first_half_cards"))
            first_half_yellow_cards = _optional_int(
                data.get("first_half_yellow_cards")
            )
            first_half_red_cards = _optional_int(data.get("first_half_red_cards"))
            if (
                first_half_red_cards is None
                and first_half_cards is not None
                and first_half_yellow_cards is not None
            ):
                first_half_red_cards = max(first_half_cards - first_half_yellow_cards, 0)
            if (
                first_half_cards is None
                and first_half_yellow_cards is not None
                and first_half_red_cards is not None
            ):
                first_half_cards = first_half_yellow_cards + first_half_red_cards
            return TeamStatsRow(
                team=team,
                opponent=opponent,
                is_home=is_home,
                confederation=(
                    home_confederation if is_home else away_confederation
                ),
                fouls=_optional_int(data.get("fouls")),
                corners=_optional_int(data.get("corners")),
                offsides=_optional_int(data.get("offsides")),
                shots_on_target=_optional_int(data.get("shots_on_target")),
                cards=cards,
                yellow_cards=yellow_cards,
                red_cards=red_cards,
                possession=_optional_float(data.get("possession")),
                first_half_fouls=_optional_int(data.get("first_half_fouls")),
                first_half_corners=_optional_int(data.get("first_half_corners")),
                first_half_offsides=_optional_int(data.get("first_half_offsides")),
                first_half_shots_on_target=_optional_int(
                    data.get("first_half_shots_on_target")
                ),
                first_half_cards=first_half_cards,
                first_half_yellow_cards=first_half_yellow_cards,
                first_half_red_cards=first_half_red_cards,
            )

        return MatchRow(
            source="statshub",
            source_match_id=str(event_id),
            match_date=match_date,
            competition=competition,
            competition_type=_competition_type(competition),
            home_team=home_name,
            away_team=away_name,
            home_stats=stats("home", home_name, away_name, True),
            away_stats=stats("away", away_name, home_name, False),
            home_confederation=home_confederation,
            away_confederation=away_confederation,
            neutral=True,
        )


class SoccerdataFBrefProvider:
    """Thin adapter around soccerdata's FBref reader.

    The returned DataFrames vary somewhat by soccerdata release, so this adapter
    exports normalized CSV input rather than silently guessing column meanings.
    """

    def __init__(self, leagues: list[str], seasons: list[str], cache_dir: str | Path) -> None:
        try:
            import soccerdata as sd
        except ImportError as exc:
            raise RuntimeError(
                "Install the optional dependency with: pip install '.[soccerdata]'"
            ) from exc
        self.reader = sd.FBref(
            leagues=leagues,
            seasons=seasons,
            data_dir=Path(cache_dir) / "soccerdata",
        )

    def export_raw(self, output_dir: str | Path) -> list[Path]:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        outputs: list[Path] = []
        methods = {
            "schedule": self.reader.read_schedule,
            "misc": lambda: self.reader.read_team_match_stats(stat_type="misc"),
            "possession": lambda: self.reader.read_team_match_stats(stat_type="possession"),
        }
        for name, method in methods.items():
            frame = method()
            path = destination / f"fbref_{name}.csv"
            frame.to_csv(path)
            outputs.append(path)
        return outputs


class FBrefManifestProvider:
    """Parse cached FBref match-report team-stat tables from a CSV manifest.

    Required manifest columns are url, match_date, competition, home_team, and
    away_team. Team names in the manifest are authoritative, which avoids
    brittle name extraction from FBref presentation markup.
    """

    STAT_LABELS = {
        "fouls": ("fouls",),
        "corners": ("corners", "corner kicks"),
        "offsides": ("offsides",),
        "possession": ("possession",),
    }

    def __init__(self, manifest_path: str | Path, cache_dir: str | Path) -> None:
        self.manifest_path = Path(manifest_path)
        self.http = CachedHttpClient(Path(cache_dir) / "fbref", min_delay_seconds=3.5)

    def fetch(self, start_date: str) -> Iterable[MatchRow]:
        with self.manifest_path.open(newline="", encoding="utf-8-sig") as handle:
            for index, metadata in enumerate(csv.DictReader(handle), start=1):
                if metadata["match_date"] < start_date:
                    continue
                html = self.http.get_text(metadata["url"])
                values = self._parse_team_stats(html)
                home = metadata["home_team"].strip()
                away = metadata["away_team"].strip()
                yield MatchRow(
                    source="fbref",
                    source_match_id=metadata.get("source_match_id")
                    or hashlib.sha256(metadata["url"].encode()).hexdigest()[:20],
                    match_date=metadata["match_date"],
                    competition=metadata["competition"],
                    competition_type=metadata.get("competition_type") or "other",
                    home_team=home,
                    away_team=away,
                    home_confederation=metadata.get("home_confederation") or None,
                    away_confederation=metadata.get("away_confederation") or None,
                    neutral=str(metadata.get("neutral", "1")).casefold()
                    not in {"0", "false", "no"},
                    home_elo=_optional_float(metadata.get("home_elo")),
                    away_elo=_optional_float(metadata.get("away_elo")),
                    referee_name=metadata.get("referee_name") or None,
                    home_stats=TeamStatsRow(
                        team=home,
                        opponent=away,
                        is_home=True,
                        confederation=metadata.get("home_confederation") or None,
                        fouls=_optional_int(values.get("fouls", [None, None])[0]),
                        corners=_optional_int(values.get("corners", [None, None])[0]),
                        offsides=_optional_int(values.get("offsides", [None, None])[0]),
                        possession=_optional_float(
                            values.get("possession", [None, None])[0]
                        ),
                    ),
                    away_stats=TeamStatsRow(
                        team=away,
                        opponent=home,
                        is_home=False,
                        confederation=metadata.get("away_confederation") or None,
                        fouls=_optional_int(values.get("fouls", [None, None])[1]),
                        corners=_optional_int(values.get("corners", [None, None])[1]),
                        offsides=_optional_int(values.get("offsides", [None, None])[1]),
                        possession=_optional_float(
                            values.get("possession", [None, None])[1]
                        ),
                    ),
                )

    def _parse_team_stats(self, html: str) -> dict[str, list[str]]:
        soup = BeautifulSoup(html, "html.parser")
        fragments = [html]
        fragments.extend(
            str(comment)
            for comment in soup.find_all(string=lambda value: isinstance(value, Comment))
            if "<table" in str(comment)
        )
        result: dict[str, list[str]] = {}
        for fragment in fragments:
            fragment_soup = BeautifulSoup(fragment, "html.parser")
            for row in fragment_soup.select("table tr"):
                cells = row.find_all(["th", "td"])
                if len(cells) < 3:
                    continue
                label = cells[0].get_text(" ", strip=True).casefold().rstrip(":")
                for stat, labels in self.STAT_LABELS.items():
                    if label not in labels:
                        continue
                    numeric = [
                        cell.get_text(" ", strip=True)
                        for cell in cells[1:]
                        if cell.get_text(" ", strip=True)
                    ]
                    if len(numeric) >= 2:
                        result[stat] = numeric[:2]
        if not any(stat in result for stat in ("fouls", "corners", "offsides")):
            raise ValueError("No supported team statistics found in FBref match report")
        return result


def _competition_type(name: str) -> str:
    normalized = name.casefold()
    if "friendly" in normalized:
        return "friendly"
    if "qualif" in normalized:
        return "qualifier"
    if "world cup" in normalized:
        return "world_cup"
    if "euro" in normalized or "copa am" in normalized:
        return "tournament"
    if "nations league" in normalized:
        return "nations_league"
    return "other"


def build_fbref_manifest(
    schedule_url: str,
    output_path: str | Path,
    cache_dir: str | Path,
    *,
    competition: str,
    competition_type: str,
    neutral: bool = True,
) -> int:
    """Build an ingestible manifest from an FBref competition schedule page."""
    http = CachedHttpClient(Path(cache_dir) / "fbref", min_delay_seconds=3.5)
    html = http.get_text(schedule_url)
    rows = parse_fbref_schedule(
        html,
        competition=competition,
        competition_type=competition_type,
        neutral=neutral,
    )
    if not rows:
        raise ValueError(
            "No completed FBref match-report links found on the schedule page"
        )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FBREF_MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def parse_fbref_schedule(
    html: str,
    *,
    competition: str,
    competition_type: str,
    neutral: bool = True,
) -> list[dict[str, str]]:
    """Extract completed match-report links from an FBref schedule document."""
    soup = BeautifulSoup(html, "html.parser")
    fragments = [html]
    fragments.extend(
        str(comment)
        for comment in soup.find_all(string=lambda value: isinstance(value, Comment))
        if "<table" in str(comment)
    )
    matches: dict[str, dict[str, str]] = {}
    for fragment in fragments:
        fragment_soup = BeautifulSoup(fragment, "html.parser")
        for table_row in fragment_soup.select("tr"):
            date_cell = table_row.select_one('[data-stat="date"]')
            home_cell = table_row.select_one('[data-stat="home_team"]')
            away_cell = table_row.select_one('[data-stat="away_team"]')
            report_cell = table_row.select_one('[data-stat="match_report"] a[href]')
            if not all((date_cell, home_cell, away_cell, report_cell)):
                continue
            match_date = date_cell.get("csk") or date_cell.get_text(" ", strip=True)
            home_team = home_cell.get_text(" ", strip=True)
            away_team = away_cell.get_text(" ", strip=True)
            href = report_cell.get("href", "")
            if not match_date or not home_team or not away_team or "/matches/" not in href:
                continue
            url = href if href.startswith("http") else f"https://fbref.com{href}"
            source_match_id = _fbref_match_id(href)
            matches[url] = {
                "url": url,
                "source_match_id": source_match_id,
                "match_date": match_date[:10],
                "competition": competition,
                "competition_type": competition_type,
                "home_team": home_team,
                "away_team": away_team,
                "home_confederation": "",
                "away_confederation": "",
                "neutral": "1" if neutral else "0",
                "home_elo": "",
                "away_elo": "",
                "referee_name": "",
            }
    return sorted(matches.values(), key=lambda row: (row["match_date"], row["source_match_id"]))


def _fbref_match_id(href: str) -> str:
    parts = [part for part in href.split("/") if part]
    try:
        return parts[parts.index("matches") + 1]
    except (ValueError, IndexError):
        return hashlib.sha256(href.encode()).hexdigest()[:20]
