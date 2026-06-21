from __future__ import annotations

import csv
import json
import math
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .data import CachedHttpClient
from .db import connect, initialize, transaction
from .domain import Question


LIQUID_MARKETS = {"h2h", "totals", "asian_handicap"}
MARKET_ALIASES = {
    "1x2": "h2h",
    "match_winner": "h2h",
    "moneyline": "h2h",
    "over_under": "totals",
    "total_goals": "totals",
    "spreads": "asian_handicap",
    "handicap": "asian_handicap",
    "both_teams_to_score": "btts",
    "both teams to score": "btts",
    "anytime_goalscorer": "player_anytime_goalscorer",
    "player_goal_scorer": "player_anytime_goalscorer",
}


def devig_proportional(decimal_odds: Iterable[float]) -> list[float]:
    inverse = np.array([1.0 / float(odd) for odd in decimal_odds], dtype=float)
    if np.any(inverse <= 0) or not np.all(np.isfinite(inverse)):
        raise ValueError("decimal odds must be finite and positive")
    return (inverse / inverse.sum()).tolist()


def devig_shin(decimal_odds: Iterable[float]) -> list[float]:
    odds = list(decimal_odds)
    inverse = np.array([1.0 / float(odd) for odd in odds], dtype=float)
    if len(inverse) < 2 or inverse.sum() <= 1.0:
        return devig_proportional(odds)
    inverse_sum = float(inverse.sum())

    def probabilities(z: float) -> np.ndarray:
        if z >= 1.0 - 1e-10:
            return inverse / inverse_sum
        return (
            np.sqrt(z * z + 4.0 * (1.0 - z) * inverse * inverse / inverse_sum) - z
        ) / (2.0 * (1.0 - z))

    low, high = 0.0, 0.999999
    for _ in range(100):
        middle = (low + high) / 2.0
        if probabilities(middle).sum() > 1.0:
            low = middle
        else:
            high = middle
    result = probabilities((low + high) / 2.0)
    return (result / result.sum()).tolist()


def normalize_market_type(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().casefold()).strip("_")
    return MARKET_ALIASES.get(normalized, normalized)


def normalize_match_key(value: str, away: str | None = None) -> str:
    if away is not None:
        teams = (value, away)
    else:
        parts = re.split(r"\s+(?:vs?\.?|at)\s+|\s*[|–—]\s*", value.strip(), maxsplit=1)
        if len(parts) != 2:
            raise ValueError(
                f"Could not parse match {value!r}; use 'Home vs Away' or 'Home|Away'"
            )
        teams = (parts[0], parts[1])
    return "|".join(re.sub(r"\s+", " ", team.strip()).casefold() for team in teams)


def match_key(home: str, away: str) -> str:
    return normalize_match_key(home, away)


def ingest_market_csv(database_path: str | Path, csv_path: str | Path) -> int:
    """Ingest either the match-market CSV or the legacy question-market CSV.

    Match-market rows normally provide decimal odds and are de-vigged by book.
    They may also provide already fair probabilities in a ``probability`` or
    ``devigged_probability`` column.  That path is useful when the source is a
    de-vigged consensus/benchmark rather than raw book odds.
    """
    initialize(database_path)
    with Path(csv_path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return 0
    if "question_key" in rows[0]:
        return _ingest_legacy_market_rows(database_path, rows)
    has_decimal_odds = "decimal_odds" in rows[0]
    probability_column = (
        "probability"
        if "probability" in rows[0]
        else "devigged_probability"
        if "devigged_probability" in rows[0]
        else None
    )
    required = {"match", "market", "selection", "book", "timestamp"}
    if probability_column is None:
        required.add("decimal_odds")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Market CSV is missing columns: {', '.join(sorted(missing))}")
    odds_records = []
    fair_records = []
    for row in rows:
        event_key = normalize_match_key(row["match"])
        market_type = normalize_market_type(row["market"])
        record = {
            "match_key": event_key,
            "market_type": market_type,
            "selection": _manual_selection(row["selection"], event_key, market_type),
            "line": _line_from_row(row),
            "book": row["book"].strip() or "manual",
            "observed_at": row["timestamp"].strip()
            or datetime.now(timezone.utc).isoformat(),
            "source": row.get("source", "").strip() or "manual_csv",
            "definition": row.get("definition", "").strip() or None,
        }
        probability = _fair_probability_from_row(row, probability_column)
        decimal_odds = row.get("decimal_odds", "").strip() if has_decimal_odds else ""
        if probability is not None:
            fair_records.append({**record, "probability": probability})
        elif decimal_odds:
            odds_records.append({**record, "decimal_odds": float(decimal_odds)})
        else:
            raise ValueError(
                "Market CSV row must include either decimal_odds or probability"
            )
    return _store_market_records(database_path, odds_records) + _store_fair_market_records(
        database_path, fair_records
    )


def ingest_odds_api(
    database_path: str | Path,
    payload: Sequence[Mapping[str, Any]],
    source: str = "the_odds_api",
) -> int:
    records: list[dict[str, Any]] = []
    for event in payload:
        event_key = match_key(str(event["home_team"]), str(event["away_team"]))
        for bookmaker in event.get("bookmakers", []):
            book = str(bookmaker.get("key") or bookmaker.get("title") or "unknown")
            observed_at = str(
                bookmaker.get("last_update")
                or event.get("commence_time")
                or datetime.now(timezone.utc).isoformat()
            )
            for market in bookmaker.get("markets", []):
                market_type = normalize_market_type(str(market.get("key", "")))
                outcomes = list(market.get("outcomes", []))
                for outcome in outcomes:
                    records.append(
                        {
                            "match_key": event_key,
                            "market_type": market_type,
                            "selection": _api_selection(
                                market_type,
                                str(outcome.get("name", "")),
                                str(event["home_team"]),
                                str(event["away_team"]),
                            ),
                            "line": (
                                float(outcome["point"])
                                if outcome.get("point") is not None
                                else None
                            ),
                            "decimal_odds": float(outcome["price"]),
                            "book": book,
                            "observed_at": observed_at,
                            "source": source,
                            "definition": None,
                        }
                    )
    return _store_market_records(database_path, records)


def aggregate_match_markets(
    database_path: str | Path,
    home: str,
    away: str,
    as_of: str | None = None,
) -> list[dict[str, Any]]:
    initialize(database_path)
    key = match_key(home, away)
    with connect(database_path) as connection:
        cutoff = as_of or "9999-12-31T23:59:59Z"
        rows = [
            dict(row)
            for row in connection.execute(
                """
                WITH latest AS (
                    SELECT match_key, market_type, selection, line, book,
                           MAX(observed_at) AS observed_at
                    FROM match_market_quotes
                    WHERE match_key = ? AND observed_at <= ?
                    GROUP BY match_key, market_type, selection, line, book
                )
                SELECT q.*
                FROM match_market_quotes q
                JOIN latest l
                  ON q.match_key=l.match_key
                 AND q.market_type=l.market_type
                 AND q.selection=l.selection
                 AND q.book=l.book
                 AND q.observed_at=l.observed_at
                 AND (q.line=l.line OR (q.line IS NULL AND l.line IS NULL))
                ORDER BY q.market_type, q.line, q.selection, q.book
                """,
                (key, cutoff),
            )
        ]
    grouped: dict[tuple[str, str, float | None], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["market_type"], row["selection"], row["line"])].append(row)
    output: list[dict[str, Any]] = []
    for (market_type, selection, line), quotes in grouped.items():
        pinnacle = [row for row in quotes if _is_pinnacle(str(row["book"]))]
        chosen = pinnacle or quotes
        probabilities = [
            float(row["devigged_probability"])
            for row in chosen
            if row["devigged_probability"] is not None
        ]
        if not probabilities:
            continue
        output.append(
            {
                "match_key": key,
                "market_type": market_type,
                "selection": selection,
                "line": line,
                "probability": float(np.median(probabilities)),
                "books": sorted({str(row["book"]) for row in chosen}),
                "preferred_pinnacle": bool(pinnacle),
                "definition": next(
                    (row["definition"] for row in chosen if row["definition"]), None
                ),
            }
        )
    return output


def lookup_direct_market_probability(
    database_path: str | Path,
    definition_path: str | Path,
    *,
    home: str,
    away: str,
    question_type: str,
    selection: str | None = None,
) -> tuple[float | None, dict[str, Any]]:
    mapping = json.loads(Path(definition_path).read_text(encoding="utf-8"))
    definition = mapping.get(question_type)
    if not definition:
        return None, {"definition_match": False, "warning": "No market mapping exists"}
    if not definition.get("definition_match"):
        return None, {
            "definition_match": False,
            "warning": definition.get("notes") or "Resolution definitions do not match",
        }
    market_type = definition.get("market_type")
    target_selection = selection or definition.get("selection")
    target_line = definition.get("line")
    quotes = aggregate_match_markets(database_path, home, away)
    candidates = [
        quote
        for quote in quotes
        if quote["market_type"] == market_type
        and (target_selection is None or quote["selection"] == target_selection.casefold())
        and (
            target_line is None
            or quote["line"] is not None
            and abs(float(quote["line"]) - float(target_line)) < 1e-6
        )
    ]
    if not candidates:
        return None, {"definition_match": True, "warning": "No matching quote found"}
    quote = candidates[0]
    return float(quote["probability"]), {
        "definition_match": True,
        "market_type": market_type,
        "selection": quote["selection"],
        "line": quote["line"],
        "books": quote["books"],
        "preferred_pinnacle": quote["preferred_pinnacle"],
        "market_weight": (
            0.70 if market_type in LIQUID_MARKETS else 0.50
        ),
    }


def lookup_market_probability(
    database_path: str | Path,
    question: Question,
    method: str = "shin",
) -> tuple[float | None, dict[str, Any]]:
    """Legacy direct-question lookup retained for backward compatibility."""
    initialize(database_path)
    with connect(database_path) as connection:
        rows = connection.execute(
            """
            WITH latest AS (
                SELECT provider, MAX(observed_at) AS observed_at
                FROM market_quotes WHERE question_key = ? GROUP BY provider
            )
            SELECT q.provider, q.outcome, q.decimal_odds, q.observed_at, q.definition
            FROM market_quotes q
            JOIN latest l ON l.provider=q.provider AND l.observed_at=q.observed_at
            WHERE q.question_key=? ORDER BY q.provider, q.outcome
            """,
            (question.key, question.key),
        ).fetchall()
    by_provider: dict[str, list[Any]] = defaultdict(list)
    for row in rows:
        by_provider[str(row["provider"])].append(row)
    probabilities: list[float] = []
    details: dict[str, Any] = {"method": method, "providers": {}}
    for provider, quotes in by_provider.items():
        outcomes = [str(row["outcome"]).casefold() for row in quotes]
        target_name = "yes" if "yes" in outcomes else "home" if "home" in outcomes else None
        if target_name is None:
            continue
        odds = [float(row["decimal_odds"]) for row in quotes]
        fair = devig_shin(odds) if method == "shin" else devig_proportional(odds)
        target = outcomes.index(target_name)
        probabilities.append(fair[target])
        details["providers"][provider] = {
            "probability": fair[target],
            "observed_at": quotes[target]["observed_at"],
            "definition": quotes[target]["definition"],
        }
    if not probabilities:
        return None, details
    return float(np.median(probabilities)), details


def blend_probabilities(
    model_probability: float,
    market_probability: float | None,
    market_weight: float,
) -> float:
    """Blend probabilities in log-odds space."""
    if market_probability is None:
        return model_probability
    weight = min(max(market_weight, 0.0), 1.0)
    model_logit = _logit(model_probability)
    market_logit = _logit(market_probability)
    return float(1.0 / (1.0 + math.exp(-((1.0 - weight) * model_logit + weight * market_logit))))


def _store_market_records(
    database_path: str | Path, records: Sequence[Mapping[str, Any]]
) -> int:
    if not records:
        return 0
    groups: dict[tuple[str, str, float | None, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if float(record["decimal_odds"]) <= 1.0:
            raise ValueError("decimal_odds must be greater than 1.0")
        line = record.get("line")
        group_line = (
            abs(float(line))
            if str(record["market_type"]) == "asian_handicap" and line is not None
            else line
        )
        key = (
            str(record["match_key"]),
            str(record["market_type"]),
            group_line,
            str(record["book"]),
            str(record["observed_at"]),
        )
        groups[key].append(record)
    count = 0
    with transaction(database_path) as connection:
        for (_, market_type, _, _, _), quotes in groups.items():
            odds = [float(quote["decimal_odds"]) for quote in quotes]
            method = "shin" if len(quotes) == 2 and market_type != "h2h" else "proportional"
            fair = devig_shin(odds) if method == "shin" else devig_proportional(odds)
            for quote, probability in zip(quotes, fair):
                connection.execute(
                    """
                    INSERT OR REPLACE INTO match_market_quotes (
                        match_key, market_type, selection, line, decimal_odds, book,
                        observed_at, source, raw_implied_probability,
                        devigged_probability, devig_method, definition
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        quote["match_key"],
                        market_type,
                        quote["selection"],
                        quote.get("line"),
                        quote["decimal_odds"],
                        quote["book"],
                        quote["observed_at"],
                        quote["source"],
                        1.0 / float(quote["decimal_odds"]),
                        probability,
                        method,
                        quote.get("definition"),
                    ),
                )
                count += 1
    return count


def _store_fair_market_records(
    database_path: str | Path, records: Sequence[Mapping[str, Any]]
) -> int:
    if not records:
        return 0
    count = 0
    with transaction(database_path) as connection:
        for quote in records:
            probability = float(quote["probability"])
            if probability <= 0.0 or probability >= 1.0:
                raise ValueError("probability must be between 0 and 1")
            synthetic_decimal = 1.0 / probability
            connection.execute(
                """
                INSERT OR REPLACE INTO match_market_quotes (
                    match_key, market_type, selection, line, decimal_odds, book,
                    observed_at, source, raw_implied_probability,
                    devigged_probability, devig_method, definition
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    quote["match_key"],
                    quote["market_type"],
                    quote["selection"],
                    quote.get("line"),
                    synthetic_decimal,
                    quote["book"],
                    quote["observed_at"],
                    quote["source"],
                    probability,
                    probability,
                    "already_devigged",
                    quote.get("definition"),
                ),
            )
            count += 1
    return count


def _ingest_legacy_market_rows(
    database_path: str | Path, rows: Sequence[Mapping[str, str]]
) -> int:
    with transaction(database_path) as connection:
        for row in rows:
            connection.execute(
                """
                INSERT OR REPLACE INTO market_quotes (
                    question_key, provider, outcome, decimal_odds, observed_at, definition
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    row["question_key"].strip().casefold(),
                    row.get("provider") or "manual",
                    row["outcome"].strip().casefold(),
                    float(row["decimal_odds"]),
                    row.get("observed_at") or datetime.now(timezone.utc).isoformat(),
                    row.get("definition") or None,
                ),
            )
    return len(rows)


def _line_from_row(row: Mapping[str, str]) -> float | None:
    value = row.get("line", "").strip()
    if value:
        return float(value)
    selection = row["selection"].strip()
    match = re.search(r"(?<!\w)([+-]?\d+(?:\.\d+)?)\s*$", selection)
    return float(match.group(1)) if match else None


def _fair_probability_from_row(
    row: Mapping[str, str], probability_column: str | None
) -> float | None:
    if probability_column is None:
        return None
    value = row.get(probability_column, "").strip()
    if not value:
        return None
    probability = float(value)
    if probability > 1.0:
        probability /= 100.0
    if probability <= 0.0 or probability >= 1.0:
        raise ValueError("probability must be between 0 and 1")
    return probability


def _normalize_selection(value: str) -> str:
    normalized = re.sub(r"\s+", " ", value.strip()).casefold()
    normalized = re.sub(r"\s+[+-]?\d+(?:\.\d+)?\s*$", "", normalized)
    aliases = {"1": "home", "x": "draw", "2": "away", "yes": "yes", "no": "no"}
    return aliases.get(normalized, normalized)


def _manual_selection(value: str, event_key: str, market_type: str) -> str:
    normalized = _normalize_selection(value)
    if market_type in {"h2h", "asian_handicap"}:
        home, away = event_key.split("|", 1)
        if normalized == home:
            return "home"
        if normalized == away:
            return "away"
    return normalized


def _api_selection(market_type: str, value: str, home: str, away: str) -> str:
    normalized = value.strip().casefold()
    if market_type in {"h2h", "asian_handicap"}:
        if normalized == home.casefold():
            return "home"
        if normalized == away.casefold():
            return "away"
        if normalized == "draw":
            return "draw"
    if market_type == "totals":
        return "over" if normalized.startswith("over") else "under"
    return _normalize_selection(value)


def _is_pinnacle(book: str) -> bool:
    return book.casefold() in {"pinnacle", "pinnacle sports", "pinny"}


def _logit(probability: float) -> float:
    clipped = min(max(float(probability), 1e-6), 1.0 - 1e-6)
    return math.log(clipped / (1.0 - clipped))


class OddsAPIClient:
    BASE_URL = "https://api.the-odds-api.com/v4"

    def __init__(self, cache_dir: str | Path, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("THE_ODDS_API_KEY")
        if not self.api_key:
            raise ValueError("THE_ODDS_API_KEY is required")
        self.http = CachedHttpClient(Path(cache_dir) / "odds_api", min_delay_seconds=1.0)

    def sports(self) -> Any:
        return self.http.get_json(f"{self.BASE_URL}/sports", params={"apiKey": self.api_key})

    def odds(
        self,
        sport_key: str,
        markets: list[str],
        regions: str = "us,uk,eu",
        odds_format: str = "decimal",
        force_refresh: bool = False,
    ) -> Any:
        return self.http.get_json(
            f"{self.BASE_URL}/sports/{sport_key}/odds",
            params={
                "apiKey": self.api_key,
                "regions": regions,
                "markets": ",".join(markets),
                "oddsFormat": odds_format,
            },
            force_refresh=force_refresh,
        )

    def event_odds(
        self,
        sport_key: str,
        event_id: str,
        markets: list[str],
        regions: str = "us,uk,eu",
        odds_format: str = "decimal",
    ) -> Any:
        """Fetch event-level markets, including player props where offered."""
        return self.http.get_json(
            f"{self.BASE_URL}/sports/{sport_key}/events/{event_id}/odds",
            params={
                "apiKey": self.api_key,
                "regions": regions,
                "markets": ",".join(markets),
                "oddsFormat": odds_format,
            },
        )
