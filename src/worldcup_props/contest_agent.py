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
    crowd_probability: float | None = None,
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

    after = _apply_favorite_sot2h_dominance(
        after,
        settings,
        favorite=favorite,
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
        "crowd_probability": crowd_probability,
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
    crowd_probability: float | None = None,
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
            "crowd_probability": crowd_probability,
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
    player_profile: Any | None = None,
    crowd_anchor: dict[str, Any] | None,
    crowd_probability: float | None = None,
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
        after = _apply_high_usage_bench_floor(
            after,
            event,
            settings,
            lineup=lineup,
            player_profile=player_profile,
            market_probability=market_probability,
            rules=rules,
        )
        after = _apply_liquid_market_guard(
            after,
            market_probability,
            settings,
            rules,
            reason=f"direct_player_market:{event}",
        )
        _record_missing_player_market(
            event,
            settings,
            market_probability=market_probability,
            rules=rules,
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
            "crowd_probability": crowd_probability,
            "lineup_status": getattr(lineup, "status", None),
            "player_profile": _player_profile_metadata(player_profile),
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


def _favorite_sot2h_target(win_probability: float) -> float | None:
    """Empirical P(favorite has more 2nd-half SOT) by supremacy.

    Measured over 2,408 historical team-matches keyed on pre-match SOT
    supremacy: ~even 0.56, clear favorite 0.62, strong favorite 0.79. Returns
    None when the match is too close for the effect to apply.
    """
    if win_probability < 0.50:
        return None
    if win_probability < 0.56:
        return 0.57
    if win_probability < 0.68:
        return 0.63
    return 0.76


def _apply_favorite_sot2h_dominance(
    probability: float,
    settings: Settings,
    *,
    favorite: dict[str, Any],
    prop_key: str,
    rules: list[dict[str, Any]],
) -> float:
    if not settings.contest_agent_favorite_sot2h_dominance:
        return probability
    if prop_key != "shots_on_target:second_half_more_than":
        return probability
    side = favorite.get("side")
    # The win-probability threshold below already filters unreliable/near-even
    # supremacy (a symmetric model_only fit resolves to side "even"), so this
    # fires on any clear favorite whether the lambdas came from market or model.
    if side not in {"home", "away"}:
        return probability
    target_favorite = _favorite_sot2h_target(float(favorite["win_probability"]))
    if target_favorite is None:
        return probability
    # The subject of a "more_than" question is the home team.
    target = target_favorite if side == "home" else 1.0 - target_favorite
    weight = float(settings.contest_agent_favorite_sot2h_weight)
    after = _clip((1.0 - weight) * probability + weight * target)
    if abs(after - probability) <= 1e-12:
        return probability
    rules.append(
        {
            "name": "favorite_second_half_sot_dominance",
            "reason": (
                "Favorites win the 2nd-half SOT battle far more than the flat "
                "simulator predicts (empirical 0.62 clear / 0.79 strong over "
                "2,408 matches)."
            ),
            "favorite_side": side,
            "favorite_win_probability": favorite["win_probability"],
            "empirical_target": target,
            "weight": weight,
            "before": probability,
            "after": after,
        }
    )
    return after


def _apply_high_usage_bench_floor(
    probability: float,
    event: str,
    settings: Settings,
    *,
    lineup: Any | None,
    player_profile: Any | None,
    market_probability: float | None,
    rules: list[dict[str, Any]],
) -> float:
    if market_probability is not None:
        return probability
    floor = settings.contest_agent_high_usage_bench_floor.get(event)
    if floor is None or probability >= floor:
        return probability
    if not _lineup_is_confirmed_bench(lineup):
        return probability
    if not _is_high_usage_attacker(player_profile, settings):
        return probability
    after = _clip(floor)
    rules.append(
        {
            "name": "high_usage_bench_player_floor",
            "event": event,
            "reason": (
                "Confirmed bench attackers with high shot/goal involvement still "
                "retain meaningful sub appearance upside; do not collapse them "
                "toward a generic low-minute prior without a direct market."
            ),
            "floor": floor,
            "before": probability,
            "after": after,
            "lineup": _lineup_metadata(lineup),
            "player_profile": _player_profile_metadata(player_profile),
        }
    )
    return after


def _record_missing_player_market(
    event: str,
    settings: Settings,
    *,
    market_probability: float | None,
    rules: list[dict[str, Any]],
) -> None:
    if market_probability is not None:
        return
    if event not in set(settings.contest_agent_require_player_market_events):
        return
    rules.append(
        {
            "name": "direct_player_market_missing",
            "event": event,
            "reason": (
                "Player props are high-error without sportsbook or prediction-market "
                "prices. Fetch DraftKings/FanDuel/bet365/ESPN BET or enter a manual "
                "market row before trusting the final number."
            ),
        }
    )


def _lineup_is_confirmed_bench(lineup: Any | None) -> bool:
    if lineup is None:
        return False
    status = str(getattr(lineup, "status", "")).strip().lower()
    start_probability = float(getattr(lineup, "start_probability", 0.0))
    sub_probability = float(getattr(lineup, "sub_probability", 0.0))
    confirmed = bool(getattr(lineup, "confirmed", True))
    return (
        confirmed
        and status in {"bench", "sub", "substitute", "available"}
        and start_probability <= 0.05
        and sub_probability >= 0.20
    )


def _is_high_usage_attacker(player_profile: Any | None, settings: Settings) -> bool:
    if player_profile is None:
        return False
    shots = float(getattr(player_profile, "shots_per90", 0.0) or 0.0)
    goals_assists = float(getattr(player_profile, "goals_per90", 0.0) or 0.0) + float(
        getattr(player_profile, "assists_per90", 0.0) or 0.0
    )
    set_piece = float(getattr(player_profile, "set_piece_role", 0.0) or 0.0)
    role = str(getattr(player_profile, "player_role", "") or "").lower()
    return (
        shots >= settings.contest_agent_high_usage_shots_per90
        or goals_assists >= settings.contest_agent_high_usage_goal_assist_per90
        or set_piece >= settings.contest_agent_high_usage_set_piece_role
        or any(token in role for token in ("forward", "wing", "attacker", "striker"))
        and shots >= 1.0
    )


def _lineup_metadata(lineup: Any | None) -> dict[str, Any] | None:
    if lineup is None:
        return None
    return {
        "status": getattr(lineup, "status", None),
        "start_probability": getattr(lineup, "start_probability", None),
        "sub_probability": getattr(lineup, "sub_probability", None),
        "confirmed": getattr(lineup, "confirmed", None),
    }


def _player_profile_metadata(player_profile: Any | None) -> dict[str, Any] | None:
    if player_profile is None:
        return None
    return {
        "shots_per90": getattr(player_profile, "shots_per90", None),
        "shots_on_target_per90": getattr(player_profile, "shots_on_target_per90", None),
        "goals_per90": getattr(player_profile, "goals_per90", None),
        "assists_per90": getattr(player_profile, "assists_per90", None),
        "penalty_taker": getattr(player_profile, "penalty_taker", None),
        "set_piece_role": getattr(player_profile, "set_piece_role", None),
        "player_role": getattr(player_profile, "player_role", None),
    }


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
