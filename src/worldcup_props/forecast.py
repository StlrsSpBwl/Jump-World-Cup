from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .calibration import apply_calibration_map, apply_shrinkage
from .config import Settings
from .contest_agent import (
    apply_match_event_agent,
    apply_player_event_agent,
    apply_question_agent,
)
from .coherent import simulate_coherent_match
from .crowd import apply_crowd_anchor, crowd_key_for_player_event
from .db import connect
from .domain import Forecast, Question
from .evaluation import crowd_bias_for_type
from .goals import GoalCalibration, calibrate_match_goals, fallback_goal_lambdas
from .market import (
    blend_probabilities,
    lookup_direct_market_probability,
    lookup_market_probability,
)
from .model import ModelArtifact
from .players import LineupEntry, PlayerProfile, load_match_players
from .simulation import simulate
from .tournament import load_tournament_context


def forecast_question(
    artifact: ModelArtifact,
    question: Question,
    settings: Settings,
    database_path: str | Path | None = None,
    simulations: int | None = None,
    seed: int | None = None,
    use_market: bool = True,
) -> Forecast:
    fallback_home, fallback_away, fallback_details = _model_fallback_goal_lambdas(
        artifact, question, settings
    )
    home_parameters = artifact.teams.get(question.home)
    away_parameters = artifact.teams.get(question.away)
    home_ess = home_parameters.effective_matches if home_parameters else 0.0
    away_ess = away_parameters.effective_matches if away_parameters else 0.0
    goal_calibration = calibrate_match_goals(
        database_path,
        question.home,
        question.away,
        fallback_home,
        fallback_away,
        use_market=use_market,
        rho=settings.dixon_coles_rho,
        prior_precision=settings.market_goal_prior_precision,
        use_supremacy_weighted_market_fusion=(
            settings.use_supremacy_weighted_market_fusion
        ),
        supremacy_market_fusion_slope=settings.supremacy_market_fusion_slope,
        supremacy_market_fusion_max_extra=settings.supremacy_market_fusion_max_extra,
        supremacy_market_fusion_ess_threshold=(
            settings.supremacy_market_fusion_ess_threshold
        ),
        min_team_effective_matches=min(home_ess, away_ess),
    )
    if goal_calibration.source == "model_only":
        goal_calibration.fusion_details.update(fallback_details)
    referee_cards_per_match = _referee_cards_per_match(
        database_path, question.referee
    )
    tournament_context = (
        load_tournament_context(database_path, question.home, question.away)
        if database_path is not None
        else None
    )
    result = simulate(
        artifact,
        question,
        settings,
        simulations=simulations,
        seed=seed,
        goal_calibration=goal_calibration,
        referee_cards_per_match=referee_cards_per_match,
        tournament_context=tournament_context,
    )
    calibration_key = _calibration_key(question)
    coefficient = artifact.calibration_lambda.get(calibration_key, 1.0)
    base_rate = artifact.base_rates.get(calibration_key, 0.5)
    calibration_map = artifact.calibration_maps.get(calibration_key)
    model_probability = (
        apply_calibration_map(result.raw_probability, calibration_map)
        if calibration_map
        else apply_shrinkage(result.raw_probability, coefficient, base_rate)
    )
    model_only_probability = model_probability
    if goal_calibration.source == "market_calibrated":
        fallback_calibration = GoalCalibration(
            lambda_home=fallback_home,
            lambda_away=fallback_away,
            source="model_only",
            objective=0.0,
            residuals=[],
            rho=settings.dixon_coles_rho,
            targets_used=0,
            prior_precision=0.0,
            fusion_details=fallback_details,
        )
        fallback_result = simulate(
            artifact,
            question,
            settings,
            simulations=simulations,
            seed=seed,
            goal_calibration=fallback_calibration,
            referee_cards_per_match=referee_cards_per_match,
            tournament_context=tournament_context,
        )
        model_only_probability = (
            apply_calibration_map(fallback_result.raw_probability, calibration_map)
            if calibration_map
            else apply_shrinkage(
                fallback_result.raw_probability, coefficient, base_rate
            )
        )
    model_probability, coverage_guard = _apply_coverage_guard(
        model_probability,
        base_rate,
        home_ess,
        away_ess,
        settings,
        market_credit=(
            settings.market_coverage_ess_credit
            if goal_calibration.source == "market_calibrated"
            and question.stat.value
            in {"corners", "offsides", "shots_on_target"}
            else 0.0
        ),
    )
    model_only_probability, _ = _apply_coverage_guard(
        model_only_probability,
        base_rate,
        home_ess,
        away_ess,
        settings,
        market_credit=0.0,
    )
    market_probability = None
    market_details = {}
    if database_path is not None and use_market:
        market_probability, market_details = lookup_market_probability(
            database_path, question
        )
    definition_match = _legacy_definition_matches(market_details)
    warnings = []
    if market_probability is not None and not definition_match:
        warnings.append(
            "Direct market found but not blended because its resolution definition "
            "is not explicitly marked exact"
        )
    if coverage_guard["low_coverage"]:
        missing = [
            team
            for team, parameters in (
                (question.home, home_parameters),
                (question.away, away_parameters),
            )
            if parameters is None
        ]
        warnings.append(
            "Coverage safeguard applied because matchup history is sparse"
            + (f"; missing fitted teams: {', '.join(missing)}" if missing else "")
        )
    direct_market = market_probability if definition_match else None
    market_weight = (
        settings.market_blend_weight
        if settings.market_blend_weight is not None
        else settings.thin_market_blend_weight
    )
    probability = blend_probabilities(
        model_probability, direct_market, market_weight
    )
    ess = min(home_ess, away_ess)
    ess_weight = 1.0
    pre_ess_probability = probability
    if settings.use_ess_gating:
        ess_weight = ess / max(ess + settings.ess_probability_prior_matches, 1e-9)
        probability = base_rate + ess_weight * (probability - base_rate)
    field_bias = None
    field_bias_rows = 0
    pre_field_probability = probability
    if settings.use_field_bias and database_path is not None:
        field_bias, field_bias_rows = crowd_bias_for_type(
            database_path, calibration_key
        )
        if field_bias is not None:
            shade = float(
                max(
                    -settings.field_bias_max_deviation,
                    min(settings.field_bias_max_deviation, -field_bias),
                )
            )
            probability = float(max(0.0, min(1.0, probability + shade)))
    pre_crowd_anchor_probability = probability
    probability, crowd_anchor = apply_crowd_anchor(
        probability,
        _crowd_anchor_key(question),
        settings,
        database_path=database_path,
    )
    contest_agent = apply_question_agent(
        probability,
        question,
        settings,
        goal_calibration=goal_calibration,
        tournament_context=tournament_context,
        market_probability=market_probability if definition_match else None,
        crowd_anchor=crowd_anchor,
    )
    probability = contest_agent.probability
    metadata = dict(result.metadata)
    combined_warnings = [*warnings, *metadata.get("warnings", [])]
    metadata.update(
        {
            "calibration_key": calibration_key,
            "calibration_lambda": coefficient,
            "calibration_base_rate": base_rate,
            "calibration_map": calibration_map,
            "market": market_details,
            "market_enabled": use_market,
            "market_definition_match": definition_match,
            "market_blend_weight": market_weight if direct_market is not None else 0.0,
            "goal_market_calibration": goal_calibration.as_dict(),
            "warnings": combined_warnings,
            "coverage_guard": {
                **coverage_guard,
                "home_team_in_model": home_parameters is not None,
                "away_team_in_model": away_parameters is not None,
            },
            "ess_gate": {
                "enabled": settings.use_ess_gating,
                "effective_sample_size": ess,
                "weight": ess_weight,
                "before": pre_ess_probability,
                "after": probability if not settings.use_field_bias else pre_field_probability,
            },
            "field_bias": {
                "enabled": settings.use_field_bias,
                "estimated_bias": field_bias,
                "rows": field_bias_rows,
                "before": pre_field_probability,
                "after": pre_crowd_anchor_probability,
            },
            "crowd_anchor": crowd_anchor,
            "contest_agent": contest_agent.metadata,
            "simulations": int(simulations or settings.simulations),
        }
    )
    return Forecast(
        probability=probability,
        model_probability=model_probability,
        model_only_probability=model_only_probability,
        market_probability=market_probability,
        p_home_more=result.p_home_more,
        p_tie=result.p_tie,
        p_away_more=result.p_away_more,
        interval_80=result.interval_80,
        effective_sample_size_home=(
            home_ess
        ),
        effective_sample_size_away=(
            away_ess
        ),
        raw_model_probability=result.raw_probability,
        metadata=metadata,
    )


def forecast_player_event(
    artifact: ModelArtifact,
    question: Question,
    player: str,
    team: str,
    event: str,
    settings: Settings,
    database_path: str | Path,
    k: int = 1,
    simulations: int | None = None,
    seed: int | None = None,
    use_market: bool = True,
) -> dict[str, Any]:
    if team not in {question.home, question.away}:
        raise ValueError("player team must be either the home or away team")
    profiles, lineups = load_match_players(database_path, question.home, question.away)
    matching_profiles = [
        profile for profile in profiles.get(team, []) if profile.player_name == player
    ]
    if not matching_profiles:
        raise ValueError(
            f"No player profile found for {player!r} on {team!r}; "
            "ingest player profiles before forecasting this prop"
        )
    calibration = _goal_calibration(
        database_path, artifact, question, settings, use_market
    )
    referee_cards = _referee_cards_per_match(database_path, question.referee)
    tournament_context = load_tournament_context(
        database_path, question.home, question.away
    )
    result = simulate_coherent_match(
        artifact,
        question,
        settings,
        profiles=profiles,
        lineups=lineups,
        simulations=simulations,
        seed=seed,
        goal_calibration=calibration,
        referee_cards_per_match=referee_cards,
        tournament_context=tournament_context,
    )
    values = _player_event_values(result.player_events[player], event)
    raw_probability = float(np.mean(values >= k))
    home_ess, away_ess = _team_effective_samples(
        artifact, question.home, question.away
    )
    profile = matching_profiles[0]
    player_lineup = next(
        (
            entry
            for entry in lineups.get(team, [])
            if entry.player_name == player and entry.confirmed
        ),
        None,
    )
    player_prior = _player_prop_prior(
        profile,
        player_lineup,
        event,
        result,
        question,
        team,
        settings,
    )
    player_prior = _apply_player_profile_shrinkage(
        raw_probability, player_prior, settings
    )
    if event == "goal_or_assist":
        goal_assist_pathway = _player_goal_assist_probability(
            profile,
            profiles.get(team, []),
            lineups.get(team, []),
            result,
            question,
            team,
            settings,
        )
        player_prior["goal_assist_pathway"] = goal_assist_pathway
        model_input_probability = goal_assist_pathway["probability"]
    else:
        model_input_probability = player_prior["player_shrunk_probability"]
    model_probability = model_input_probability
    coverage_guard = {
        "enabled": False,
        "reason": "player props use role/profile prior shrinkage instead of team coverage guard",
        "home_effective_matches": home_ess,
        "away_effective_matches": away_ess,
        "market_ess_credit": 0.0,
        "effective_coverage": min(home_ess, away_ess),
        "weight": 1.0,
        "base_rate": player_prior["base_rate"],
        "before": raw_probability,
        "after": model_probability,
        "low_coverage": False,
    }
    model_only_probability = model_probability
    if calibration.source == "market_calibrated":
        fallback_result = simulate_coherent_match(
            artifact,
            question,
            settings,
            profiles=profiles,
            lineups=lineups,
            simulations=simulations,
            seed=seed,
            goal_calibration=_fallback_calibration(artifact, question, settings),
            referee_cards_per_match=referee_cards,
            tournament_context=tournament_context,
        )
        fallback_values = _player_event_values(
            fallback_result.player_events[player], event
        )
        fallback_raw_probability = float(np.mean(fallback_values >= k))
        if event == "goal_or_assist":
            model_only_probability = _player_goal_assist_probability(
                profile,
                profiles.get(team, []),
                lineups.get(team, []),
                fallback_result,
                question,
                team,
                settings,
            )["probability"]
        else:
            model_only_probability = _apply_player_profile_shrinkage(
                fallback_raw_probability, player_prior, settings
            )["player_shrunk_probability"]
    pre_lineup_probability = None
    lineup_delta = None
    if player_lineup is not None:
        pre_lineups = {
            side: list(entries) for side, entries in lineups.items()
        }
        pre_lineups[team] = [
            _pre_lineup_entry(entry, settings)
            if entry.player_name == player
            else entry
            for entry in pre_lineups[team]
        ]
        pre_result = simulate_coherent_match(
            artifact,
            question,
            settings,
            profiles=profiles,
            lineups=pre_lineups,
            simulations=simulations,
            seed=seed,
            goal_calibration=calibration,
            referee_cards_per_match=referee_cards,
            tournament_context=tournament_context,
        )
        pre_values = _player_event_values(pre_result.player_events[player], event)
        pre_lineup_raw_probability = float(np.mean(pre_values >= k))
        if event == "goal_or_assist":
            pre_lineup_probability = _player_goal_assist_probability(
                profile,
                profiles.get(team, []),
                pre_lineups.get(team, []),
                pre_result,
                question,
                team,
                settings,
            )["probability"]
        else:
            pre_lineup_probability = _apply_player_profile_shrinkage(
                pre_lineup_raw_probability, player_prior, settings
            )["player_shrunk_probability"]
        lineup_delta = model_probability - pre_lineup_probability
    floor_details = _player_antizero_floor(
        profile,
        player_lineup,
        event,
        player_prior["minutes_fraction"],
        settings,
    )
    pre_floor_probability = model_probability
    if floor_details["applied"]:
        model_probability = max(model_probability, floor_details["floor"])
        probability_floor = max(model_only_probability, floor_details["floor"])
        model_only_probability = probability_floor
    market_probability = None
    market_details: dict[str, Any] = {}
    direct_market_question_type = None
    if event == "goals" and k == 1:
        direct_market_question_type = "anytime_goalscorer"
    elif event == "goal_or_assist" and k == 1:
        direct_market_question_type = "player_goal_or_assist"
    if use_market and direct_market_question_type:
        market_probability, market_details = lookup_direct_market_probability(
            database_path,
            settings.market_definition_path,
            home=question.home,
            away=question.away,
            question_type=direct_market_question_type,
            selection=player,
        )
    market_weight = settings.thin_market_blend_weight
    probability = blend_probabilities(
        model_probability, market_probability, market_weight
    )
    probability, crowd_anchor = apply_crowd_anchor(
        probability,
        crowd_key_for_player_event(event),
        settings,
        database_path=database_path,
    )
    contest_agent = apply_player_event_agent(
        probability,
        event,
        settings,
        market_probability=market_probability,
        lineup=player_lineup,
        crowd_anchor=crowd_anchor,
    )
    probability = contest_agent.probability
    return {
        "probability": probability,
        "model_probability": model_probability,
        "model_only_probability": model_only_probability,
        "raw_model_probability": raw_probability,
        "market_probability": market_probability,
        "event": event,
        "threshold": k,
        "player": player,
        "team": team,
        "interval_80": [
            float(value) for value in np.quantile(values, [0.1, 0.9])
        ],
        "pre_lineup_probability": pre_lineup_probability,
        "lineup_delta": lineup_delta,
        "metadata": {
            **result.metadata,
            "market": market_details,
            "market_blend_weight": (
                market_weight if market_probability is not None else 0.0
            ),
            "crowd_anchor": crowd_anchor,
            "contest_agent": contest_agent.metadata,
            "lineup_news_odds_refresh_required": True,
            "coverage_guard": {
                **coverage_guard,
                "pre_player_prior_raw_probability": raw_probability,
                "post_player_prior_probability": model_input_probability,
                "home_team_in_model": question.home in artifact.teams,
                "away_team_in_model": question.away in artifact.teams,
            },
            "player_prior": {
                **player_prior,
                "anti_zero_floor": floor_details,
                "pre_floor_probability": pre_floor_probability,
                "post_floor_probability": model_probability,
            },
            "warnings": (
                ["Coverage safeguard applied because matchup history is sparse"]
                if coverage_guard["low_coverage"]
                else []
            ),
            "simulations": int(simulations or settings.simulations),
        },
    }


def forecast_match_event(
    artifact: ModelArtifact,
    question: Question,
    event: str,
    settings: Settings,
    database_path: str | Path,
    simulations: int | None = None,
    seed: int | None = None,
    use_market: bool = True,
) -> dict[str, Any]:
    profiles, lineups = load_match_players(database_path, question.home, question.away)
    calibration = _goal_calibration(
        database_path, artifact, question, settings, use_market
    )
    tournament_context = load_tournament_context(
        database_path, question.home, question.away
    )
    result = simulate_coherent_match(
        artifact,
        question,
        settings,
        profiles=profiles,
        lineups=lineups,
        simulations=simulations,
        seed=seed,
        goal_calibration=calibration,
        referee_cards_per_match=_referee_cards_per_match(
            database_path, question.referee
        ),
        tournament_context=tournament_context,
    )
    if event == "penalty_awarded":
        outcomes = result.penalty_awarded
        mapping_key = "penalty_awarded"
        market_selection = None
        market_weight = settings.thin_market_blend_weight
    elif event == "red_card_shown":
        outcomes = result.red_card_shown
        mapping_key = None
        market_selection = None
        market_weight = settings.thin_market_blend_weight
    elif event == "penalty_or_red":
        outcomes = result.penalty_awarded | result.red_card_shown
        mapping_key = None
        market_selection = None
        market_weight = settings.thin_market_blend_weight
    elif event == "home_win":
        outcomes = result.home_segments["goals"].sum(axis=1) > result.away_segments[
            "goals"
        ].sum(axis=1)
        mapping_key = "match_winner"
        market_selection = "home"
        market_weight = 1.0
    elif event == "away_win":
        outcomes = result.away_segments["goals"].sum(axis=1) > result.home_segments[
            "goals"
        ].sum(axis=1)
        mapping_key = "match_winner"
        market_selection = "away"
        market_weight = 1.0
    elif event == "draw":
        outcomes = result.home_segments["goals"].sum(axis=1) == result.away_segments[
            "goals"
        ].sum(axis=1)
        mapping_key = "match_winner"
        market_selection = "draw"
        market_weight = 1.0
    elif event == "under_2_5_goals":
        outcomes = (
            result.home_segments["goals"].sum(axis=1)
            + result.away_segments["goals"].sum(axis=1)
            <= 2
        )
        mapping_key = "total_goals_2_or_fewer"
        market_selection = None
        market_weight = 1.0
    else:
        raise ValueError(f"Unsupported match event: {event}")
    model_probability = float(np.mean(outcomes))
    model_only_probability = model_probability
    if calibration.source == "market_calibrated":
        fallback_result = simulate_coherent_match(
            artifact,
            question,
            settings,
            profiles=profiles,
            lineups=lineups,
            simulations=simulations,
            seed=seed,
            goal_calibration=_fallback_calibration(artifact, question, settings),
            referee_cards_per_match=_referee_cards_per_match(
                database_path, question.referee
            ),
            tournament_context=tournament_context,
        )
        if event == "penalty_awarded":
            fallback_outcomes = fallback_result.penalty_awarded
        elif event == "red_card_shown":
            fallback_outcomes = fallback_result.red_card_shown
        else:
            fallback_outcomes = (
                fallback_result.penalty_awarded | fallback_result.red_card_shown
            )
        if event == "home_win":
            fallback_outcomes = fallback_result.home_segments["goals"].sum(axis=1) > (
                fallback_result.away_segments["goals"].sum(axis=1)
            )
        elif event == "away_win":
            fallback_outcomes = fallback_result.away_segments["goals"].sum(axis=1) > (
                fallback_result.home_segments["goals"].sum(axis=1)
            )
        elif event == "draw":
            fallback_outcomes = fallback_result.home_segments["goals"].sum(axis=1) == (
                fallback_result.away_segments["goals"].sum(axis=1)
            )
        elif event == "under_2_5_goals":
            fallback_outcomes = (
                fallback_result.home_segments["goals"].sum(axis=1)
                + fallback_result.away_segments["goals"].sum(axis=1)
                <= 2
            )
        model_only_probability = float(np.mean(fallback_outcomes))
    market_probability = None
    market_details: dict[str, Any] = {}
    if use_market and mapping_key:
        market_probability, market_details = lookup_direct_market_probability(
            database_path,
            settings.market_definition_path,
            home=question.home,
            away=question.away,
            question_type=mapping_key,
            selection=market_selection,
        )
    probability = blend_probabilities(
        model_probability,
        market_probability,
        market_weight,
    )
    probability, crowd_anchor = apply_crowd_anchor(
        probability,
        _crowd_anchor_key_for_match_event(event),
        settings,
        database_path=database_path,
    )
    contest_agent = apply_match_event_agent(
        probability,
        event,
        settings,
        goal_calibration=calibration,
        market_probability=market_probability,
        crowd_anchor=crowd_anchor,
    )
    probability = contest_agent.probability
    return {
        "probability": probability,
        "model_probability": model_probability,
        "model_only_probability": model_only_probability,
        "market_probability": market_probability,
        "event": event,
        "metadata": {
            **result.metadata,
            "market": market_details,
            "market_blend_weight": (
                market_weight
                if market_probability is not None
                else 0.0
            ),
            "crowd_anchor": crowd_anchor,
            "contest_agent": contest_agent.metadata,
            "simulations": int(simulations or settings.simulations),
        },
    }


def _goal_calibration(
    database_path: str | Path,
    artifact: ModelArtifact,
    question: Question,
    settings: Settings,
    use_market: bool,
) -> GoalCalibration:
    fallback_home, fallback_away, fallback_details = _model_fallback_goal_lambdas(
        artifact, question, settings
    )
    calibration = calibrate_match_goals(
        database_path,
        question.home,
        question.away,
        fallback_home,
        fallback_away,
        use_market=use_market,
        rho=settings.dixon_coles_rho,
        prior_precision=settings.market_goal_prior_precision,
        use_supremacy_weighted_market_fusion=(
            settings.use_supremacy_weighted_market_fusion
        ),
        supremacy_market_fusion_slope=settings.supremacy_market_fusion_slope,
        supremacy_market_fusion_max_extra=settings.supremacy_market_fusion_max_extra,
        supremacy_market_fusion_ess_threshold=(
            settings.supremacy_market_fusion_ess_threshold
        ),
        min_team_effective_matches=min(
            *_team_effective_samples(artifact, question.home, question.away)
        ),
    )
    if calibration.source == "model_only":
        calibration.fusion_details.update(fallback_details)
    return calibration


def _fallback_calibration(
    artifact: ModelArtifact, question: Question, settings: Settings
) -> GoalCalibration:
    fallback_home, fallback_away, fallback_details = _model_fallback_goal_lambdas(
        artifact, question, settings
    )
    return GoalCalibration(
        lambda_home=fallback_home,
        lambda_away=fallback_away,
        source="model_only",
        objective=0.0,
        residuals=[],
        rho=settings.dixon_coles_rho,
        targets_used=0,
        prior_precision=0.0,
        fusion_details=fallback_details,
    )


def _model_fallback_goal_lambdas(
    artifact: ModelArtifact, question: Question, settings: Settings
) -> tuple[float, float, dict[str, float | bool | str | None]]:
    elo_home, elo_away = fallback_goal_lambdas(
        question.home_elo,
        question.away_elo,
        question.neutral,
        settings.fallback_total_goals,
    )
    if question.home_elo is not None or question.away_elo is not None:
        return elo_home, elo_away, {
            "enabled": False,
            "fallback_source": "explicit_elo",
        }
    home = artifact.teams.get(question.home)
    away = artifact.teams.get(question.away)
    min_ess = min(
        home.effective_matches if home else 0.0,
        away.effective_matches if away else 0.0,
    )
    if (
        not settings.use_team_strength_goal_fallback
        or home is None
        or away is None
        or min_ess < settings.goal_fallback_min_team_ess
    ):
        return elo_home, elo_away, {
            "enabled": False,
            "fallback_source": "neutral_elo",
            "reason": (
                "disabled_or_insufficient_team_coverage"
                if settings.use_team_strength_goal_fallback
                else "disabled"
            ),
            "min_team_effective_matches": float(min_ess),
        }
    stat = "shots_on_target"
    home_sot = float(
        np.sqrt(max(home.rates[stat] * away.conceded_rates[stat], 0.01))
    )
    away_sot = float(
        np.sqrt(max(away.rates[stat] * home.conceded_rates[stat], 0.01))
    )
    exponent = float(settings.goal_fallback_sot_share_exponent)
    home_power = home_sot**exponent
    away_power = away_sot**exponent
    home_share = home_power / max(home_power + away_power, 1e-9)
    global_sot_total = max(2.0 * artifact.global_rates.get(stat, 4.0), 0.01)
    sot_total_ratio = (home_sot + away_sot) / global_sot_total
    total = settings.fallback_total_goals * (
        sot_total_ratio ** float(settings.goal_fallback_sot_total_elasticity)
    )
    lower, upper = settings.goal_fallback_total_bounds
    total = float(np.clip(total, lower, upper))
    home_lambda = float(np.clip(total * home_share, 0.15, 4.5))
    away_lambda = float(np.clip(total * (1.0 - home_share), 0.15, 4.5))
    return home_lambda, away_lambda, {
        "enabled": True,
        "fallback_source": "team_sot_attack_concession",
        "home_projected_sot": home_sot,
        "away_projected_sot": away_sot,
        "home_goal_share": float(home_share),
        "total_goal_prior": total,
        "min_team_effective_matches": float(min_ess),
    }


def _team_effective_samples(
    artifact: ModelArtifact, home: str, away: str
) -> tuple[float, float]:
    home_parameters = artifact.teams.get(home)
    away_parameters = artifact.teams.get(away)
    return (
        home_parameters.effective_matches if home_parameters else 0.0,
        away_parameters.effective_matches if away_parameters else 0.0,
    )


def _apply_coverage_guard(
    probability: float,
    base_rate: float,
    home_ess: float,
    away_ess: float,
    settings: Settings,
    *,
    market_credit: float,
) -> tuple[float, dict[str, Any]]:
    effective_coverage = min(home_ess, away_ess) + market_credit
    if not settings.use_coverage_safeguards:
        weight = 1.0
    elif min(home_ess, away_ess) >= settings.coverage_probability_prior_matches:
        weight = 1.0
    else:
        weight = min(
            1.0,
            effective_coverage
            / max(settings.coverage_probability_prior_matches, 1e-9),
        )
    guarded = float(base_rate + weight * (probability - base_rate))
    return guarded, {
        "enabled": settings.use_coverage_safeguards,
        "home_effective_matches": home_ess,
        "away_effective_matches": away_ess,
        "market_ess_credit": market_credit,
        "effective_coverage": effective_coverage,
        "weight": weight,
        "base_rate": base_rate,
        "before": probability,
        "after": guarded,
        "low_coverage": min(home_ess, away_ess) < 3.0,
    }


def _player_goal_assist_probability(
    profile: PlayerProfile,
    team_profiles: list[PlayerProfile],
    team_lineups: list[LineupEntry],
    result: Any,
    question: Question,
    team: str,
    settings: Settings,
) -> dict[str, Any]:
    minutes_fraction = _player_minutes_fraction(
        profile.player_name,
        _lineup_for_player(team_lineups, profile.player_name),
        result,
    )
    goal_lambda = _team_goal_lambda(result, question, team)
    goal_share_raw = _player_rate_share(
        profile,
        team_profiles,
        team_lineups,
        result,
        "goals_per90",
    )
    assist_share_raw = _player_rate_share(
        profile,
        team_profiles,
        team_lineups,
        result,
        "assists_per90",
    )
    focus = _goal_assist_focus_bucket(profile, goal_share_raw, assist_share_raw, settings)
    goal_share = _bounded_share(
        goal_share_raw,
        settings.player_ga_goal_share_bounds,
        focus,
    )
    assist_share = _bounded_share(
        assist_share_raw,
        settings.player_ga_assist_share_bounds,
        focus,
        apply_lower=(
            assist_share_raw
            >= settings.player_ga_assist_share_bounds.get(
                focus, settings.player_ga_assist_share_bounds["other"]
            )[0]
            * 0.85
            or profile.set_piece_role >= 0.3
        ),
    )
    team_penalty_probability = float(
        np.mean(result.home_penalty if team == question.home else result.away_penalty)
    )
    penalty_lambda = (
        team_penalty_probability
        * settings.player_ga_penalty_conversion
        * minutes_fraction
        if profile.penalty_taker
        else 0.0
    )
    set_piece_lambda = (
        settings.player_ga_set_piece_assist_lambda
        * max(profile.set_piece_role, 0.0)
        * minutes_fraction
    )
    lam_goal = goal_lambda * goal_share * minutes_fraction + penalty_lambda
    lam_assist = (
        settings.player_ga_assist_rate
        * goal_lambda
        * assist_share
        * minutes_fraction
        + set_piece_lambda
    )
    p_goal = 1.0 - math.exp(-max(lam_goal, 0.0))
    p_assist = 1.0 - math.exp(-max(lam_assist, 0.0))
    probability = 1.0 - (1.0 - p_goal) * (1.0 - p_assist)
    return {
        "probability": float(np.clip(probability, 0.0, 0.95)),
        "team_goal_lambda": goal_lambda,
        "minutes_fraction": minutes_fraction,
        "focus_bucket": focus,
        "goal_share_raw": goal_share_raw,
        "assist_share_raw": assist_share_raw,
        "goal_share": goal_share,
        "assist_share": assist_share,
        "lam_goal": lam_goal,
        "lam_assist": lam_assist,
        "p_goal": p_goal,
        "p_assist": p_assist,
        "assist_rate": settings.player_ga_assist_rate,
        "penalty_taker": profile.penalty_taker,
        "team_penalty_probability": team_penalty_probability,
        "penalty_lambda": penalty_lambda,
        "set_piece_role": profile.set_piece_role,
        "set_piece_lambda": set_piece_lambda,
    }


def _team_goal_lambda(result: Any, question: Question, team: str) -> float:
    calibration = result.metadata.get("goal_calibration", {})
    key = "lambda_home" if team == question.home else "lambda_away"
    if key in calibration:
        return float(calibration[key])
    segments = result.home_segments if team == question.home else result.away_segments
    return float(np.mean(segments["goals"].sum(axis=1)))


def _player_rate_share(
    profile: PlayerProfile,
    team_profiles: list[PlayerProfile],
    team_lineups: list[LineupEntry],
    result: Any,
    rate_field: str,
) -> float:
    lineup_by_player = {entry.player_name: entry for entry in team_lineups}
    confirmed_lineup_known = any(entry.confirmed for entry in team_lineups)
    weights = []
    target_weight = 0.0
    for candidate in team_profiles:
        lineup = lineup_by_player.get(candidate.player_name)
        if lineup is None and confirmed_lineup_known:
            minutes_fraction = 0.0
        else:
            minutes_fraction = _player_minutes_fraction(
                candidate.player_name,
                lineup,
                result,
            )
        weight = max(float(getattr(candidate, rate_field)), 0.0) * minutes_fraction
        weights.append(weight)
        if candidate.player_name == profile.player_name:
            target_weight = weight
    total = sum(weights)
    if total <= 1e-9:
        return 0.0
    return float(np.clip(target_weight / total, 0.0, 1.0))


def _goal_assist_focus_bucket(
    profile: PlayerProfile,
    goal_share_raw: float,
    assist_share_raw: float,
    settings: Settings,
) -> str:
    role = _player_role(profile)
    profile_weight = profile.effective_matches / max(
        profile.effective_matches + settings.player_prior_effective_matches,
        1e-9,
    )
    attacking_role = role in {"forward", "winger", "attacking_mid"}
    high_usage = (
        goal_share_raw >= 0.16
        or assist_share_raw >= 0.14
        or profile.goals_per90 + profile.assists_per90 >= 0.45
    )
    if (
        attacking_role
        and high_usage
        and profile_weight >= settings.player_ga_min_profile_weight_for_focal
    ):
        return "focal"
    if attacking_role:
        return "attacker"
    return "other"


def _bounded_share(
    raw_share: float,
    bounds_by_bucket: dict[str, list[float]],
    bucket: str,
    *,
    apply_lower: bool = True,
) -> float:
    lower, upper = bounds_by_bucket.get(bucket, bounds_by_bucket["other"])
    if not apply_lower:
        lower = 0.0
    return float(np.clip(raw_share, lower, upper))


def _lineup_for_player(
    lineups: list[LineupEntry], player: str
) -> LineupEntry | None:
    return next((entry for entry in lineups if entry.player_name == player), None)


def _player_prop_prior(
    profile: PlayerProfile,
    lineup: LineupEntry | None,
    event: str,
    result: Any,
    question: Question,
    team: str,
    settings: Settings,
) -> dict[str, Any]:
    role = _player_role(profile)
    event_rates = settings.player_role_base_rates.get(event, {})
    full_minutes_base = float(
        event_rates.get(role, event_rates.get("unknown", settings.player_event_base_rates.get(event, 0.25)))
    )
    minutes_fraction = _player_minutes_fraction(profile.player_name, lineup, result)
    team_context_multiplier = _team_context_multiplier(
        result, question, team, event, settings
    )
    base_lambda = -np.log(max(1.0 - full_minutes_base, 1e-6))
    base_rate = float(
        1.0 - np.exp(-base_lambda * minutes_fraction * team_context_multiplier)
    )
    data_weight = profile.effective_matches / max(
        profile.effective_matches + settings.player_prior_effective_matches, 1e-9
    )
    return {
        "role": role,
        "role_full_minutes_base_rate": full_minutes_base,
        "minutes_fraction": minutes_fraction,
        "team_context_multiplier": team_context_multiplier,
        "base_rate": base_rate,
        "profile_effective_matches": profile.effective_matches,
        "profile_data_weight": data_weight,
    }


def _apply_player_profile_shrinkage(
    raw_probability: float, prior: dict[str, Any], settings: Settings
) -> dict[str, Any]:
    data_weight = float(prior["profile_data_weight"])
    base_rate = float(prior["base_rate"])
    prior = dict(prior)
    if float(prior["profile_effective_matches"]) <= 0.0:
        adjustment = settings.player_thin_profile_upside_weight * max(
            raw_probability - base_rate, 0.0
        )
    else:
        adjustment = data_weight * (raw_probability - base_rate)
    prior["player_shrunk_probability"] = float(base_rate + adjustment)
    return prior


def _player_antizero_floor(
    profile: PlayerProfile,
    lineup: LineupEntry | None,
    event: str,
    minutes_fraction: float,
    settings: Settings,
) -> dict[str, Any]:
    role = _player_role(profile)
    floor = float(settings.player_role_antizero_floors.get(event, {}).get(role, 0.0))
    likely_starter = _is_likely_starter(lineup, minutes_fraction)
    applied = bool(floor > 0.0 and likely_starter)
    return {
        "role": role,
        "floor": floor if applied else 0.0,
        "configured_floor": floor,
        "likely_starter": likely_starter,
        "applied": applied,
    }


def _player_role(profile: PlayerProfile) -> str:
    role = (profile.player_role or "unknown").strip().casefold()
    if role != "unknown":
        return role
    if profile.shots_per90 >= 2.2 or profile.goals_per90 >= 0.25:
        return "forward"
    if profile.shots_per90 >= 1.5 or profile.assists_per90 >= 0.18:
        return "winger"
    if profile.shots_per90 >= 0.8:
        return "central_mid"
    return "unknown"


def _player_minutes_fraction(
    player: str,
    lineup: LineupEntry | None,
    result: Any,
) -> float:
    if lineup is not None and lineup.confirmed and lineup.start_probability >= 0.95:
        return 1.0
    events = result.player_events.get(player)
    if events is not None:
        return float(np.clip(np.mean(events["minutes"]) / 90.0, 0.0, 1.0))
    if lineup is None:
        return 0.75
    expected_minutes = (
        lineup.start_probability * lineup.expected_start_minutes
        + lineup.sub_probability * lineup.expected_sub_minutes
    )
    return float(np.clip(expected_minutes / 90.0, 0.0, 1.0))


def _is_likely_starter(lineup: LineupEntry | None, minutes_fraction: float) -> bool:
    if lineup is None:
        return minutes_fraction >= 0.65
    if lineup.confirmed:
        return lineup.start_probability >= 0.95 and lineup.expected_start_minutes >= 45.0
    return lineup.start_probability >= 0.60 or minutes_fraction >= 0.65


def _team_context_multiplier(
    result: Any,
    question: Question,
    team: str,
    event: str,
    settings: Settings,
) -> float:
    if team == question.home:
        segments = result.home_segments
    else:
        segments = result.away_segments
    if event == "second_half_shots_on_target":
        observed = float(np.mean(segments["shots_on_target"][:, 3:].sum(axis=1)))
        baseline = 4.0 * 0.55
    elif event in {"goals", "assists", "goal_or_assist"}:
        observed = float(np.mean(segments["goals"].sum(axis=1)))
        baseline = settings.fallback_total_goals / 2.0
    else:
        observed = float(np.mean(segments["shots_on_target"].sum(axis=1)))
        baseline = 4.0
    ratio = max(observed / max(baseline, 1e-6), 0.05)
    raw_multiplier = ratio ** settings.player_team_context_exponent
    lower, upper = settings.player_team_context_bounds
    return float(np.clip(raw_multiplier, lower, upper))


def _referee_cards_per_match(
    database_path: str | Path | None, referee: str | None
) -> float | None:
    if database_path is None or not referee:
        return None
    with connect(database_path) as connection:
        row = connection.execute(
            "SELECT cards_per_match FROM referees WHERE referee_name=?",
            (referee,),
        ).fetchone()
    return (
        float(row["cards_per_match"])
        if row is not None and row["cards_per_match"] is not None
        else None
    )


def _player_event_values(events: dict[str, np.ndarray], event: str) -> np.ndarray:
    if event == "goal_or_assist":
        return ((events["goals"] + events["assists"]) >= 1).astype(int)
    if event not in events:
        raise ValueError(f"Unsupported player event: {event}")
    return events[event]


def _pre_lineup_entry(entry: LineupEntry, settings: Settings) -> LineupEntry:
    return LineupEntry(
        match_key=entry.match_key,
        team=entry.team,
        player_name=entry.player_name,
        status="pre_lineup_baseline",
        start_probability=settings.default_player_start_probability,
        expected_start_minutes=settings.default_start_minutes,
        sub_probability=settings.default_player_sub_probability,
        expected_sub_minutes=settings.default_sub_minutes,
        confirmed=False,
    )


def _legacy_definition_matches(details: dict[str, object]) -> bool:
    providers = details.get("providers")
    if not isinstance(providers, dict) or not providers:
        return False
    accepted = {"exact", "true", "definition_match", "matched"}
    definitions = [
        str(value.get("definition") or "").strip().casefold()
        for value in providers.values()
        if isinstance(value, dict)
    ]
    return bool(definitions) and all(value in accepted for value in definitions)


def _calibration_key(question: Question) -> str:
    if question.k is None:
        return f"{question.stat.value}:{question.question_type.value}"
    return f"{question.stat.value}:{question.question_type.value}:{question.k}"


def _crowd_anchor_key(question: Question) -> str:
    return _calibration_key(question)


def _crowd_anchor_key_for_match_event(event: str) -> str:
    return {
        "penalty_awarded": "penalty_awarded",
        "penalty_or_red": "penalty_or_red",
        "home_win": "match_winner",
        "away_win": "match_winner",
        "draw": "match_winner",
        "under_2_5_goals": "total_goals_2_or_fewer",
    }.get(event, event)
