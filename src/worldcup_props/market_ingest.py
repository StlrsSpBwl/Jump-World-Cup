from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable, Protocol, Sequence

from pydantic import BaseModel, Field
from scipy.optimize import brentq, minimize
from scipy.stats import poisson

from .market import devig_proportional, devig_shin


class AnchorQuality(str, Enum):
    liquid = "liquid"
    aggregator = "aggregator"
    none = "none"


class SearchResult(BaseModel):
    title: str = ""
    url: str
    snippet: str = ""


class SearchClient(Protocol):
    def search(self, query: str) -> list[SearchResult]:
        ...


class Line(BaseModel):
    market: str
    point: float | None = None
    raw_implied: dict[str, float]
    devig_prob: dict[str, float] | None
    source: str
    tier: AnchorQuality
    captured_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MarketProfile(BaseModel):
    match_id: str
    one_x_two: Line | None = None
    total_goals: Line | None = None
    team_total_home: Line | None = None
    team_total_away: Line | None = None
    prop_lines: dict[str, Line] = Field(default_factory=dict)
    derived: dict[str, float] = Field(default_factory=dict)
    unpriced: list[str] = Field(default_factory=list)
    reconciliation_warnings: list[str] = Field(default_factory=list)

    def require_priced(self, market: str) -> None:
        if market in self.unpriced:
            raise ValueError(f"{market} is unpriced; do not emit a market anchor")
        line = _line_for_market(self, market)
        if line is None or line.devig_prob is None or line.tier == AnchorQuality.none:
            raise ValueError(f"{market} has no usable market anchor")


QUERY_TEMPLATES = [
    "{home} {away} world cup moneyline draw odds",
    "{home} {away} total goals over under",
    "{home} {away} team total goals over under",
    "{home} {away} corners over under odds",
    "{home} {away} shots on target prop odds",
    "{home} {away} cards bookings prop odds",
]

SOURCE_RANKS = {
    "pinnacle": 0,
    "kalshi": 0,
    "fanduel": 1,
    "bet365": 1,
    "draftkings": 1,
    "espnbet": 1,
    "espn bet": 1,
    "oddschecker": 2,
    "tips.gg": 2,
    "tipsgg": 2,
}

CORE_MARKETS = (
    "one_x_two",
    "total_goals",
    "team_total_home",
    "team_total_away",
)


def to_implied(value: str | float | int, fmt: str | None = None) -> float:
    text = str(value).strip()
    if not text:
        raise ValueError("empty odds value")
    if fmt:
        fmt = fmt.casefold().strip()
    if fmt == "percent" or text.endswith("%"):
        number = float(text.rstrip("%").strip())
        probability = number / 100.0
    elif fmt == "fractional" or "/" in text:
        numerator, denominator = text.split("/", 1)
        decimal = 1.0 + float(numerator) / float(denominator)
        probability = 1.0 / decimal
    elif fmt == "american" or re.fullmatch(r"[+-]\d+(?:\.\d+)?", text):
        american = float(text)
        if american > 0:
            probability = 100.0 / (american + 100.0)
        else:
            probability = abs(american) / (abs(american) + 100.0)
    elif fmt == "decimal" or re.fullmatch(r"\d+(?:\.\d+)?", text):
        decimal = float(text)
        if decimal <= 1.0:
            raise ValueError(f"decimal odds must be > 1.0: {value!r}")
        probability = 1.0 / decimal
    else:
        raise ValueError(f"ambiguous odds value: {value!r}")
    if not 0.0 < probability < 1.0:
        raise ValueError(f"odds imply invalid probability: {value!r}")
    return probability


def ingest_market_profile(
    home: str,
    away: str,
    search_client: SearchClient,
    *,
    flagged_players: Sequence[str] = (),
    required_markets: Sequence[str] = ("team_sot",),
    captured_at: datetime | None = None,
) -> MarketProfile:
    captured_at = captured_at or datetime.now(timezone.utc)
    results = _run_query_ladder(home, away, search_client, flagged_players)
    ranked = sorted(results.values(), key=lambda result: (_source_rank(result.url), result.url))
    profile = MarketProfile(match_id=f"{home}|{away}".casefold())
    profile.one_x_two = _best_line(
        _parse_one_x_two(ranked, home, away, captured_at), "one_x_two"
    )
    profile.total_goals = _best_line(
        _parse_match_total(ranked, captured_at), "total_goals"
    )
    profile.team_total_home = _best_line(
        _parse_team_total(ranked, home, "team_total_home", captured_at),
        "team_total_home",
    )
    profile.team_total_away = _best_line(
        _parse_team_total(ranked, away, "team_total_away", captured_at),
        "team_total_away",
    )
    for market, lines in {
        "corners": _parse_generic_total(ranked, "corners", captured_at),
        "cards": _parse_generic_total(ranked, "cards", captured_at),
        "team_sot": _parse_generic_total(ranked, "shots_on_target", captured_at),
    }.items():
        line = _best_line(lines, market)
        if line is not None:
            profile.prop_lines[market] = line
    for player in flagged_players:
        player_line = _best_line(
            _parse_player_anytime_goal(ranked, player, captured_at),
            f"player_anytime_goalscorer:{player}",
        )
        if player_line is not None:
            profile.prop_lines[f"player_anytime_goalscorer:{player.casefold()}"] = player_line

    missing = [
        market
        for market in [*CORE_MARKETS, *required_markets]
        if _line_for_market(profile, market) is None
    ]
    profile.unpriced = sorted(set(missing))
    profile.derived = _derive_lambdas(profile)
    profile.reconciliation_warnings = _reconcile(profile)
    _assert_no_fabricated_anchors(profile)
    return profile


def summarize_market_profile(profile: MarketProfile) -> str:
    lines: list[str] = [f"Market profile: {profile.match_id}"]
    for label in [
        "one_x_two",
        "total_goals",
        "team_total_home",
        "team_total_away",
    ]:
        line = _line_for_market(profile, label)
        lines.append(_format_line(label, line))
    for label, line in sorted(profile.prop_lines.items()):
        lines.append(_format_line(label, line))
    if profile.derived:
        derived = ", ".join(
            f"{key}={value:.3f}" for key, value in sorted(profile.derived.items())
        )
        lines.append(f"DERIVED: {derived}")
    lines.append("UNPRICED: " + (", ".join(profile.unpriced) if profile.unpriced else "none"))
    if profile.reconciliation_warnings:
        lines.append("WARNINGS:")
        lines.extend(f"- {warning}" for warning in profile.reconciliation_warnings)
    return "\n".join(lines)


def _run_query_ladder(
    home: str,
    away: str,
    search_client: SearchClient,
    flagged_players: Sequence[str],
) -> dict[str, SearchResult]:
    results: dict[str, SearchResult] = {}
    queries = [template.format(home=home, away=away) for template in QUERY_TEMPLATES]
    queries.extend(f"{player} anytime goalscorer odds" for player in flagged_players)
    for query in queries:
        for result in search_client.search(query):
            parsed = result if isinstance(result, SearchResult) else SearchResult.model_validate(result)
            results.setdefault(parsed.url, parsed)
    return results


def _best_line(lines: Iterable[Line], market: str) -> Line | None:
    usable = [line for line in lines if line.devig_prob is not None]
    if not usable:
        return None
    return sorted(usable, key=lambda line: (_source_rank(line.source), line.source, market))[0]


def _parse_one_x_two(
    results: Sequence[SearchResult], home: str, away: str, captured_at: datetime
) -> list[Line]:
    lines: list[Line] = []
    for result in results:
        text = _result_text(result)
        home_odd = _odds_after_team(text, home)
        away_odd = _odds_after_team(text, away)
        draw_odd = _odds_after_label(text, "draw")
        if home_odd and away_odd and draw_odd:
            lines.append(
                _make_line(
                    "one_x_two",
                    None,
                    {"home": home_odd, "draw": draw_odd, "away": away_odd},
                    result,
                    captured_at,
                    method="proportional",
                )
            )
    return lines


def _parse_match_total(results: Sequence[SearchResult], captured_at: datetime) -> list[Line]:
    lines: list[Line] = []
    for result in results:
        text = _result_text(result)
        if not re.search(r"\b(total goals|over/?under|o/u)\b", text, re.I):
            continue
        point = _line_point(text, default=2.5)
        over = _odds_after_label(text, "over") or _odds_after_label(text, "o")
        under = _odds_after_label(text, "under") or _odds_after_label(text, "u")
        if over and under:
            lines.append(
                _make_line(
                    "total_goals",
                    point,
                    {"over": over, "under": under},
                    result,
                    captured_at,
                )
            )
    return lines


def _parse_team_total(
    results: Sequence[SearchResult],
    team: str,
    market: str,
    captured_at: datetime,
) -> list[Line]:
    lines: list[Line] = []
    team_pattern = re.escape(team)
    for result in results:
        text = _result_text(result)
        if not re.search(team_pattern, text, re.I) or not re.search(r"team total", text, re.I):
            continue
        segments = _team_segments(text, team)
        for segment in segments:
            point = _line_point(segment, default=1.5)
            over = _odds_after_label(segment, "over") or _odds_after_label(segment, "o")
            under = _odds_after_label(segment, "under") or _odds_after_label(segment, "u")
            if over and under:
                lines.append(
                    _make_line(
                        market,
                        point,
                        {"over": over, "under": under},
                        result,
                        captured_at,
                    )
                )
    return lines


def _parse_generic_total(
    results: Sequence[SearchResult], market: str, captured_at: datetime
) -> list[Line]:
    keyword = {
        "corners": r"corner",
        "cards": r"cards?|bookings?",
        "shots_on_target": r"shots? on target|sot",
    }[market]
    lines: list[Line] = []
    for result in results:
        text = _result_text(result)
        if not re.search(keyword, text, re.I):
            continue
        if market == "shots_on_target" and not re.search(
            r"(?:team|total)\s+(?:shots? on target|sot)|(?:shots? on target|sot)\s+(?:team|total)",
            text,
            re.I,
        ):
            continue
        point = _line_point(text)
        over = _odds_after_label(text, "over") or _odds_after_label(text, "o")
        under = _odds_after_label(text, "under") or _odds_after_label(text, "u")
        if point is not None and over and under:
            lines.append(
                _make_line(
                    market,
                    point,
                    {"over": over, "under": under},
                    result,
                    captured_at,
                )
            )
    return lines


def _parse_player_anytime_goal(
    results: Sequence[SearchResult], player: str, captured_at: datetime
) -> list[Line]:
    lines: list[Line] = []
    for result in results:
        text = _result_text(result)
        if not re.search(re.escape(player), text, re.I):
            continue
        odd = _odds_after_team(text, player)
        if odd:
            raw = {"yes": to_implied(odd)}
            lines.append(
                Line(
                    market=f"player_anytime_goalscorer:{player}",
                    raw_implied=raw,
                    devig_prob=raw,
                    source=result.url,
                    tier=_tier_for_source(result.url),
                    captured_at=captured_at,
                )
            )
    return lines


def _make_line(
    market: str,
    point: float | None,
    odds: dict[str, str],
    result: SearchResult,
    captured_at: datetime,
    *,
    method: str = "shin",
) -> Line:
    raw = {selection: to_implied(value) for selection, value in odds.items()}
    decimal_odds = [1.0 / value for value in raw.values()]
    fair_values = (
        devig_proportional(decimal_odds)
        if method == "proportional" or len(raw) == 3
        else devig_shin(decimal_odds)
    )
    fair = dict(zip(raw.keys(), fair_values))
    total = sum(fair.values())
    fair = {key: value / total for key, value in fair.items()}
    return Line(
        market=market,
        point=point,
        raw_implied=raw,
        devig_prob=fair,
        source=result.url,
        tier=_tier_for_source(result.url),
        captured_at=captured_at,
    )


def _derive_lambdas(profile: MarketProfile) -> dict[str, float]:
    home_lambda = _lambda_from_team_total(profile.team_total_home)
    away_lambda = _lambda_from_team_total(profile.team_total_away)
    if home_lambda is not None and away_lambda is not None:
        return {
            "lambda_home": home_lambda,
            "lambda_away": away_lambda,
            "total": home_lambda + away_lambda,
            "supremacy": home_lambda - away_lambda,
        }
    total_lambda = _lambda_from_total(profile.total_goals)
    if total_lambda is not None and profile.one_x_two and profile.one_x_two.devig_prob:
        home_lambda, away_lambda = _split_total_from_1x2(total_lambda, profile.one_x_two)
        return {
            "lambda_home": home_lambda,
            "lambda_away": away_lambda,
            "total": total_lambda,
            "supremacy": home_lambda - away_lambda,
        }
    if total_lambda is not None:
        return {"total": total_lambda}
    return {}


def _lambda_from_team_total(line: Line | None) -> float | None:
    if not line or not line.devig_prob or line.point is None:
        return None
    if abs(line.point - 1.5) > 1e-6:
        return None
    over = line.devig_prob.get("over")
    if over is None:
        return None
    return float(brentq(lambda lam: 1.0 - poisson.cdf(1, lam) - over, 0.01, 8.0))


def _lambda_from_total(line: Line | None) -> float | None:
    if not line or not line.devig_prob or line.point is None:
        return None
    over = line.devig_prob.get("over")
    if over is None:
        return None
    threshold = math.floor(line.point)
    return float(brentq(lambda lam: 1.0 - poisson.cdf(threshold, lam) - over, 0.01, 8.0))


def _split_total_from_1x2(total_lambda: float, line: Line) -> tuple[float, float]:
    assert line.devig_prob is not None
    target_home = line.devig_prob["home"]
    target_away = line.devig_prob["away"]

    def objective(x: list[float]) -> float:
        share = 1.0 / (1.0 + math.exp(-float(x[0])))
        home_lambda = total_lambda * share
        away_lambda = total_lambda - home_lambda
        home, _, away = _one_x_two_from_lambdas(home_lambda, away_lambda)
        return (home - target_home) ** 2 + (away - target_away) ** 2

    fitted = minimize(objective, [0.0], method="Nelder-Mead")
    share = 1.0 / (1.0 + math.exp(-float(fitted.x[0])))
    return total_lambda * share, total_lambda * (1.0 - share)


def _one_x_two_from_lambdas(home_lambda: float, away_lambda: float, max_goals: int = 12) -> tuple[float, float, float]:
    home = draw = away = 0.0
    for h in range(max_goals + 1):
        hp = poisson.pmf(h, home_lambda)
        for a in range(max_goals + 1):
            probability = hp * poisson.pmf(a, away_lambda)
            if h > a:
                home += probability
            elif h == a:
                draw += probability
            else:
                away += probability
    total = home + draw + away
    return home / total, draw / total, away / total


def _reconcile(profile: MarketProfile) -> list[str]:
    warnings: list[str] = []
    if "lambda_home" not in profile.derived or "lambda_away" not in profile.derived:
        return warnings
    for label, lambda_key, line in [
        ("home", "lambda_home", profile.team_total_home),
        ("away", "lambda_away", profile.team_total_away),
    ]:
        if not line or not line.devig_prob or line.point is None:
            continue
        if abs(line.point - 1.5) > 1e-6:
            continue
        market = float(line.devig_prob["over"])
        model = float(1.0 - poisson.cdf(1, profile.derived[lambda_key]))
        if abs(model - market) > 0.03:
            warnings.append(
                f"{label} team total mismatch: model P(2+)={model:.3f}, market={market:.3f}"
            )
    return warnings


def _assert_no_fabricated_anchors(profile: MarketProfile) -> None:
    for line in [
        profile.one_x_two,
        profile.total_goals,
        profile.team_total_home,
        profile.team_total_away,
        *profile.prop_lines.values(),
    ]:
        if line is not None and line.tier == AnchorQuality.none and line.devig_prob is not None:
            raise AssertionError("unpriced markets must not carry de-vigged probabilities")


def _line_for_market(profile: MarketProfile, market: str) -> Line | None:
    return {
        "one_x_two": profile.one_x_two,
        "1x2": profile.one_x_two,
        "total_goals": profile.total_goals,
        "team_total_home": profile.team_total_home,
        "team_total_away": profile.team_total_away,
    }.get(market) or profile.prop_lines.get(market)


def _format_line(label: str, line: Line | None) -> str:
    if line is None:
        return f"{label}: UNPRICED"
    point = "" if line.point is None else f" point={line.point:g}"
    devig = (
        "none"
        if line.devig_prob is None
        else ", ".join(f"{key}={value:.3f}" for key, value in line.devig_prob.items())
    )
    return f"{label}:{point} {devig} tier={line.tier.value} source={line.source}"


def _result_text(result: SearchResult) -> str:
    return f"{result.title} {result.snippet}"


def _source_rank(url: str) -> int:
    normalized = url.casefold().replace("-", "").replace("_", "")
    for source, rank in SOURCE_RANKS.items():
        if source.replace(".", "").replace(" ", "") in normalized:
            return rank
    return 3


def _tier_for_source(url: str) -> AnchorQuality:
    rank = _source_rank(url)
    if rank <= 1:
        return AnchorQuality.liquid
    if rank == 2:
        return AnchorQuality.aggregator
    return AnchorQuality.aggregator


def _odds_after_team(text: str, label: str) -> str | None:
    return _odds_after_label(text, label)


def _odds_after_label(text: str, label: str) -> str | None:
    pattern = re.compile(
        rf"{re.escape(label)}\s*(?:[:=]|at|odds)?\s*({ODDS_PATTERN})",
        re.I,
    )
    match = pattern.search(text)
    return match.group(1) if match else None


ODDS_PATTERN = r"(?:[+-]\d{2,4}|\d+/\d+|\d+(?:\.\d+)?)"


def _line_point(text: str, default: float | None = None) -> float | None:
    patterns = [
        r"(?:o/u|line|total|team total|point)\s*[:=]?\s*([0-9]+(?:\.5)?)",
        r"(?:over|under|o|u)\s*([0-9]+(?:\.5)?)",
        r"([0-9]+(?:\.5)?)\s*(?:goals?|corners?|cards?|bookings?|shots? on target|sot)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            return float(match.group(1))
    return default


def _team_segments(text: str, team: str) -> list[str]:
    chunks = re.split(r";|\n|(?<=[A-Za-z0-9])\.\s+(?=[A-Z])", text)
    return [
        chunk
        for chunk in chunks
        if re.search(re.escape(team), chunk, re.I)
        and re.search(r"team total", chunk, re.I)
    ]
