from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from .config import Settings
from .domain import Question, QuestionType, Stat

if TYPE_CHECKING:
    from .goals import GoalCalibration
    from .tournament import MatchTournamentContext, TeamTournamentContext


@dataclass(frozen=True)
class ContestAgentResult:
    probability: float
    metadata: dict[str, Any]


def apply_question_agent(
    probability: float,
    question: Question,
    settings: Settings,
    *,
    goal_calibration: "GoalCalibration",
    tournament_context: "MatchTournamentContext | None",
    market_probability: float | None,
    crowd_anchor: dict[str, Any] | None,
) -> ContestAgentResult:
    before = _clip(probability)
    if not settings.use_contest_agent:
        return ContestAgentResult(
            before,
            {
                "enabled": False,
                "before": before,
                "after": before,
                "rules": [],
            },
        )

    favorite = _favorite_profile(goal_calibration)
    prop_key = _question_prop_key(question)
    rules: list[dict[str, Any]] = []
    after = before

    after = _apply_liquid_market_guard(
        after,
        market_probability,
        settings,
        rules,
        reason="direct_question_market",
    )

    after = _apply_structured_favorite_dominance_guard(
        after,
        question,
        settings,
        favorite=favorite,
        tournament_context=tournament_context,
        prop_key=prop_key,
        rules=rules,
    )

    metadata = {
        "enabled": True,
        "before": before,
        "after": after,
        "prop_key": prop_key,
        "favorite": favorite,
        "crowd_anchor": {
            "key": crowd_anchor.get("key") if crowd_anchor else None,
            "applied": crowd_anchor.get("applied") if crowd_anchor else False,
            "after": crowd_anchor.get("after") if crowd_anchor else None,
        },
        "rules": rules,
    }
    return ContestAgentResult(after, metadata)


def apply_match_event_agent(
    probability: float,
    event: str,
    settings: Settings,
    *,
    goal_calibration: "GoalCalibration",
    market_probability: float | None,
    crowd_anchor: dict[str, Any] | None,
) -> ContestAgentResult:
    before = _clip(probability)
    if not settings.use_contest_agent:
        return ContestAgentResult(
            before,
            {
                "enabled": False,
                "before": before,
                "after": before,
                "rules": [],
            },
        )
    favorite = _favorite_profile(goal_calibration)
    rules: list[dict[str, Any]] = []
    after = before
    if event in {"home_win", "away_win", "under_2_5_goals"}:
        after = _apply_liquid_market_guard(
            after,
            market_probability,
            settings,
            rules,
            reason=f"liquid_match_event:{event}",
        )
    return ContestAgentResult(
        after,
        {
            "enabled": True,
            "before": before,
            "after": after,
            "event": event,
            "favorite": favorite,
            "crowd_anchor": {
                "key": crowd_anchor.get("key") if crowd_anchor else None,
                "applied": crowd_anchor.get("applied") if crowd_anchor else False,
                "after": crowd_anchor.get("after") if crowd_anchor else None,
            },
            "rules": rules,
        },
    )


def apply_player_event_agent(
    probability: float,
    event: str,
    settings: Settings,
    *,
    market_probability: float | None,
    lineup: Any | None,
    crowd_anchor: dict[str, Any] | None,
) -> ContestAgentResult:
    before = _clip(probability)
    if not settings.use_contest_agent:
        return ContestAgentResult(
            before,
            {
                "enabled": False,
                "before": before,
                "after": before,
                "rules": [],
            },
        )
    rules: list[dict[str, Any]] = []
    after = before
    if lineup is not None and _lineup_is_out(lineup):
        rules.append(
            {
                "name": "confirmed_player_unavailable",
                "before": after,
                "after": 0.0,
                "reason": "Confirmed unavailable players cannot record an event.",
            }
        )
        after = 0.0
    else:
        after = _apply_liquid_market_guard(
            after,
            market_probability,
            settings,
            rules,
            reason=f"direct_player_market:{event}",
        )
    return ContestAgentResult(
        after,
        {
            "enabled": True,
            "before": before,
            "after": after,
            "event": event,
            "crowd_anchor": {
                "key": crowd_anchor.get("key") if crowd_anchor else None,
                "applied": crowd_anchor.get("applied") if crowd_anchor else False,
                "after": crowd_anchor.get("after") if crowd_anchor else None,
            },
            "lineup_status": getattr(lineup, "status", None),
            "rules": rules,
        },
    )


def _apply_liquid_market_guard(
    probability: float,
    market_probability: float | None,
    settings: Settings,
    rules: list[dict[str, Any]],
    *,
    reason: str,
) -> float:
    if market_probability is None:
        return probability
    delta = abs(probability - market_probability)
    if delta < settings.contest_agent_market_disagreement_trigger:
        return probability
    weight = float(settings.contest_agent_market_copy_weight)
    after = _clip((1.0 - weight) * probability + weight * market_probability)
    rules.append(
        {
            "name": "market_disagreement_guard",
            "reason": reason,
            "market_probability": market_probability,
            "trigger": settings.contest_agent_market_disagreement_trigger,
            "weight": weight,
            "before": probability,
            "after": after,
        }
    )
    return after


def _apply_structured_favorite_dominance_guard(
    probability: float,
    question: Question,
    settings: Settings,
    *,
    favorite: dict[str, Any],
    tournament_context: MatchTournamentContext | None,
    prop_key: str,
    rules: list[dict[str, Any]],
) -> float:
    if question.question_type != QuestionType.SECOND_HALF_MORE_THAN:
        return probability
    if question.stat not in {Stat.CORNERS, Stat.SHOTS_ON_TARGET}:
        return probability
    if favorite["win_probability"] < settings.contest_agent_extreme_favorite_win_probability:
        return probability
    favorite_side = favorite["side"]
    if favorite_side not in {"home", "away"}:
        return probability
    favorite_context = (
        tournament_context.home
        if favorite_side == "home" and tournament_context is not None
        else tournament_context.away
        if favorite_side == "away" and tournament_context is not None
        else None
    )
    if not _is_structured_possession(favorite_context, settings):
        return probability

    if favorite_side == "home":
        floor = settings.contest_agent_structured_dominance_floor.get(prop_key)
        if floor is None or probability >= floor:
            return probability
        after = _clip(floor)
        rules.append(
            {
                "name": "structured_possession_extreme_favorite_floor",
                "prop_key": prop_key,
                "favorite_side": favorite_side,
                "favorite_win_probability": favorite["win_probability"],
                "tactical_style": favorite_context.tactical_style,
                "before": probability,
                "after": after,
            }
        )
        return after

    cap = settings.contest_agent_structured_dominance_cap.get(prop_key)
    if cap is None or probability <= cap:
        return probability
    after = _clip(cap)
    rules.append(
        {
            "name": "structured_possession_extreme_favorite_underdog_cap",
            "prop_key": prop_key,
            "favorite_side": favorite_side,
            "favorite_win_probability": favorite["win_probability"],
            "tactical_style": favorite_context.tactical_style,
            "before": probability,
            "after": after,
        }
    )
    return after


def _question_prop_key(question: Question) -> str:
    if question.question_type == QuestionType.THRESHOLD:
        return f"{question.stat.value}:threshold:{question.k}"
    return f"{question.stat.value}:{question.question_type.value}"


def _favorite_profile(goal_calibration: "GoalCalibration") -> dict[str, Any]:
    home_win, draw, away_win = _win_draw_loss(
        goal_calibration.lambda_home, goal_calibration.lambda_away
    )
    if home_win > away_win:
        side = "home"
        probability = home_win
    elif away_win > home_win:
        side = "away"
        probability = away_win
    else:
        side = "even"
        probability = home_win
    return {
        "side": side,
        "win_probability": probability,
        "home_win_probability": home_win,
        "draw_probability": draw,
        "away_win_probability": away_win,
        "lambda_home": goal_calibration.lambda_home,
        "lambda_away": goal_calibration.lambda_away,
        "source": goal_calibration.source,
    }


def _win_draw_loss(lambda_home: float, lambda_away: float, max_goals: int = 12) -> tuple[float, float, float]:
    home_probs = _poisson_probs(lambda_home, max_goals)
    away_probs = _poisson_probs(lambda_away, max_goals)
    home_win = draw = away_win = 0.0
    for home_goals, home_prob in enumerate(home_probs):
        for away_goals, away_prob in enumerate(away_probs):
            prob = home_prob * away_prob
            if home_goals > away_goals:
                home_win += prob
            elif home_goals == away_goals:
                draw += prob
            else:
                away_win += prob
    return float(home_win), float(draw), float(away_win)


def _poisson_probs(lam: float, max_goals: int) -> list[float]:
    lam = max(float(lam), 0.001)
    return [math.exp(-lam) * lam**goals / math.factorial(goals) for goals in range(max_goals + 1)]


def _is_structured_possession(
    context: "TeamTournamentContext | None", settings: Settings
) -> bool:
    if context is None or not context.tactical_style:
        return False
    style = context.tactical_style.strip().lower().replace("-", "_").replace(" ", "_")
    configured = {
        item.strip().lower().replace("-", "_").replace(" ", "_")
        for item in settings.structured_possession_tactical_styles
    }
    return style in configured


def _lineup_is_out(lineup: Any) -> bool:
    status = str(getattr(lineup, "status", "")).strip().lower()
    return status in {"out", "injured", "unavailable", "not_in_squad"} or (
        float(getattr(lineup, "start_probability", 0.0)) <= 0.0
        and float(getattr(lineup, "sub_probability", 0.0)) <= 0.0
    )


def _clip(value: float) -> float:
    return float(max(0.0, min(1.0, value)))
