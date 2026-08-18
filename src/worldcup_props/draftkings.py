"""DraftKings sportsbook adapter.

Parses a DraftKings sportsbook capture (a HAR exported from the browser, or the
already-extracted ``{markets, selections}`` / event JSON payloads) into the
match-market CSV rows that :func:`worldcup_props.market.ingest_market_csv`
already de-vigs and stores.

The adapter only emits markets it can map with confidence. Anything it does not
recognise is reported in ``DKParseResult.skipped`` rather than guessed at, so a
DraftKings schema change surfaces loudly instead of silently fabricating odds.
"""

from __future__ import annotations

import base64
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# DraftKings market name -> (our market_type, kind). ``kind`` drives how the
# row is emitted downstream:
#   "priced_multiway" -> decimal odds, de-vigged together by the CSV importer
#   "player_one_sided" -> single "yes" price; stored as raw implied probability
MARKET_MAP: dict[str, tuple[str, str]] = {
    "moneyline": ("h2h", "priced_multiway"),
    "total goals": ("totals", "priced_multiway"),
    "both teams to score": ("btts", "priced_multiway"),
    "anytime goalscorer": ("player_anytime_goalscorer", "player_one_sided"),
}

# DraftKings outcomeType -> our selection token, for the multi-way markets.
OUTCOME_SELECTION: dict[str, str] = {
    "home": "home",
    "away": "away",
    "draw": "draw",
    "tie": "draw",
    "over": "over",
    "under": "under",
    "yes": "yes",
    "no": "no",
}

_MARKETS_ENDPOINT = "eventSubcategory/v1/markets"
_EVENT_ENDPOINT = "pagedata/event/v1/events"


@dataclass
class DKEvent:
    name: str
    home: str
    away: str
    start: str | None
    event_id: str


@dataclass
class DKParseResult:
    event: DKEvent
    rows: list[dict[str, Any]]
    skipped: list[str] = field(default_factory=list)


def _decimal(odds: Mapping[str, Any] | None) -> float | None:
    if not odds:
        return None
    raw = odds.get("decimal")
    if raw in (None, ""):
        return None
    try:
        value = float(str(raw))
    except (TypeError, ValueError):
        return None
    return value if value > 1.0 else None


def _har_bodies(har: Mapping[str, Any]) -> Iterable[tuple[str, str]]:
    for entry in har.get("log", {}).get("entries", []):
        url = entry.get("request", {}).get("url", "")
        content = entry.get("response", {}).get("content", {})
        text = content.get("text", "")
        if not text:
            continue
        if content.get("encoding") == "base64":
            try:
                text = base64.b64decode(text).decode("utf-8", "replace")
            except (ValueError, UnicodeDecodeError):
                continue
        yield url, text


def extract_from_har(
    har: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[dict[str, Any]]]:
    """Pull (event_payload, markets, selections) out of a DraftKings HAR."""
    event_payload: dict[str, Any] | None = None
    markets: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    for url, text in _har_bodies(har):
        if _MARKETS_ENDPOINT in url:
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                continue
            markets.extend(data.get("markets", []))
            selections.extend(data.get("selections", []))
        elif _EVENT_ENDPOINT in url and event_payload is None:
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                continue
            if data.get("events"):
                event_payload = data
    return event_payload, markets, selections


def _event_from_payload(
    payload: Mapping[str, Any] | None, markets: Sequence[Mapping[str, Any]]
) -> DKEvent:
    if payload and payload.get("events"):
        event = payload["events"][0]
        name = str(event.get("name", "")).strip()
        home = away = ""
        for participant in event.get("participants", []):
            role = str(participant.get("venueRole", "")).casefold()
            if role == "home":
                home = str(participant.get("name", "")).strip()
            elif role == "away":
                away = str(participant.get("name", "")).strip()
        if (not home or not away) and " vs " in name.casefold():
            parts = name.split(" vs ") if " vs " in name else name.split(" VS ")
            if len(parts) == 2:
                home = home or parts[0].strip()
                away = away or parts[1].strip()
        return DKEvent(
            name=name or f"{home} vs {away}",
            home=home,
            away=away,
            start=str(event.get("startEventDate")) if event.get("startEventDate") else None,
            event_id=str(event.get("id", "")),
        )
    event_id = str(markets[0].get("eventId", "")) if markets else ""
    return DKEvent(name="", home="", away="", start=None, event_id=event_id)


def build_rows(
    event_payload: Mapping[str, Any] | None,
    markets: Sequence[Mapping[str, Any]],
    selections: Sequence[Mapping[str, Any]],
) -> DKParseResult:
    event = _event_from_payload(event_payload, markets)
    if not event.home or not event.away:
        raise ValueError(
            "DraftKings capture is missing event participants; cannot build match key"
        )
    match = f"{event.home} vs {event.away}"
    timestamp = event.start or ""
    selections_by_market: dict[str, list[Mapping[str, Any]]] = {}
    for selection in selections:
        selections_by_market.setdefault(str(selection.get("marketId")), []).append(selection)

    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    for market in markets:
        name = str(market.get("name", "")).strip()
        mapped = MARKET_MAP.get(name.casefold())
        if mapped is None:
            skipped.append(f"unmapped market: {name!r}")
            continue
        market_type, kind = mapped
        market_selections = selections_by_market.get(str(market.get("id")), [])
        for selection in market_selections:
            decimal_odds = _decimal(selection.get("displayOdds"))
            if decimal_odds is None:
                skipped.append(
                    f"{name}: selection {selection.get('label')!r} missing decimal odds"
                )
                continue
            if kind == "priced_multiway":
                token = OUTCOME_SELECTION.get(str(selection.get("outcomeType", "")).casefold())
                if token is None:
                    skipped.append(
                        f"{name}: unknown outcomeType {selection.get('outcomeType')!r}"
                    )
                    continue
                rows.append(
                    {
                        "match": match,
                        "market": market_type,
                        "selection": token,
                        "decimal_odds": f"{decimal_odds}",
                        "probability": "",
                        "book": "draftkings",
                        "timestamp": timestamp,
                        "line": _format_line(selection.get("points")),
                        "source": "draftkings_har",
                        "definition": "",
                    }
                )
            else:  # player_one_sided: store raw implied probability for the player
                rows.append(
                    {
                        "match": match,
                        "market": market_type,
                        "selection": str(selection.get("label", "")).strip(),
                        "decimal_odds": "",
                        "probability": f"{1.0 / decimal_odds:.6f}",
                        "book": "draftkings",
                        "timestamp": timestamp,
                        "line": "",
                        "source": "draftkings_har",
                        "definition": "raw implied (one-sided market, not de-vigged)",
                    }
                )
    return DKParseResult(event=event, rows=rows, skipped=skipped)


def _format_line(points: Any) -> str:
    if points in (None, ""):
        return ""
    try:
        return f"{float(points):g}"
    except (TypeError, ValueError):
        return ""


def parse_har_file(path: str | Path) -> DKParseResult:
    har = json.loads(Path(path).read_text(encoding="utf-8"))
    event_payload, markets, selections = extract_from_har(har)
    if not markets:
        raise ValueError(
            f"No DraftKings market responses ({_MARKETS_ENDPOINT}) found in {path}"
        )
    return build_rows(event_payload, markets, selections)


CSV_COLUMNS = [
    "match",
    "market",
    "selection",
    "decimal_odds",
    "probability",
    "book",
    "timestamp",
    "line",
    "source",
    "definition",
]


def write_market_csv(rows: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})
    return destination
