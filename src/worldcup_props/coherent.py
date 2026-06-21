from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import Settings
from .domain import Question, QuestionType, Stat
from .goals import GoalCalibration, fallback_goal_lambdas
from .model import ModelArtifact
from .players import LineupEntry, PlayerProfile
from .simulation import (
    _draw_match_flow,
    _simulate_card_segments,
    _simulate_foul_segments,
    _simulate_score_states,
    _simulate_territory_segments,
    _team_or_prior,
    _tournament_incentive_summary,
)
from .tournament import MatchTournamentContext


@dataclass
class CoherentMatchResult:
    home_segments: dict[str, np.ndarray]
    away_segments: dict[str, np.ndarray]
    player_events: dict[str, dict[str, np.ndarray]]
    penalty_awarded: np.ndarray
    red_card_shown: np.ndarray
    home_penalty: np.ndarray
    away_penalty: np.ndarray
    metadata: dict[str, Any]


def simulate_coherent_match(
    artifact: ModelArtifact,
    question: Question,
    settings: Settings,
    profiles: dict[str, list[PlayerProfile]] | None = None,
    lineups: dict[str, list[LineupEntry]] | None = None,
    simulations: int | None = None,
    seed: int | None = None,
    goal_calibration: GoalCalibration | None = None,
    referee_cards_per_match: float | None = None,
    tournament_context: MatchTournamentContext | None = None,
) -> CoherentMatchResult:
    n = simulations or settings.simulations
    rng = np.random.default_rng(settings.random_seed if seed is None else seed)
    home = _team_or_prior(artifact, question.home)
    away = _team_or_prior(artifact, question.away)
    if goal_calibration is None:
        lambda_home, lambda_away = fallback_goal_lambdas(
            question.home_elo,
            question.away_elo,
            question.neutral,
            settings.fallback_total_goals,
        )
        goal_calibration = GoalCalibration(
            lambda_home,
            lambda_away,
            "model_only",
            0.0,
            [],
            settings.dixon_coles_rho,
            0,
        )
    flow = _draw_match_flow(goal_calibration, n, rng, settings)
    (
        score_states,
        home_goals,
        away_goals,
        home_goal_segments,
        away_goal_segments,
    ) = _simulate_score_states(
        goal_calibration, n, rng, flow, settings, tournament_context
    )
    base_question = Question(
        home=question.home,
        away=question.away,
        stat=Stat.FOULS,
        question_type=QuestionType.MORE_THAN,
        referee=question.referee,
        competition_type=question.competition_type,
        home_elo=question.home_elo,
        away_elo=question.away_elo,
        neutral=question.neutral,
    )
    home_fouls, away_fouls, foul_details = _simulate_foul_segments(
        artifact,
        base_question,
        home,
        away,
        n,
        rng,
        score_states,
        settings,
        tournament_context,
    )
    home_segments: dict[str, np.ndarray] = {
        "goals": home_goal_segments,
        "fouls": home_fouls,
    }
    away_segments: dict[str, np.ndarray] = {
        "goals": away_goal_segments,
        "fouls": away_fouls,
    }
    territory_details: dict[str, Any] = {}
    for stat in (Stat.CORNERS, Stat.OFFSIDES, Stat.SHOTS_ON_TARGET):
        stat_question = Question(
            home=question.home,
            away=question.away,
            stat=stat,
            question_type=QuestionType.MORE_THAN,
            referee=question.referee,
            competition_type=question.competition_type,
            home_elo=question.home_elo,
            away_elo=question.away_elo,
            neutral=question.neutral,
        )
        home_stat, away_stat, details = _simulate_territory_segments(
            artifact,
            stat_question,
            home,
            away,
            n,
            rng,
            score_states,
            goal_calibration,
            settings,
            flow,
            tournament_context,
        )
        home_segments[stat.value] = home_stat
        away_segments[stat.value] = away_stat
        territory_details[stat.value] = details
    home_cards, away_cards, card_details = _simulate_card_segments(
        artifact,
        base_question,
        home,
        away,
        home_fouls,
        away_fouls,
        rng,
        settings,
        referee_cards_per_match,
        tournament_context,
    )
    home_segments["cards"] = home_cards
    away_segments["cards"] = away_cards

    (
        penalty_awarded,
        red_card_shown,
        home_penalty,
        away_penalty,
        event_details,
    ) = _simulate_rare_events(
        home_segments,
        away_segments,
        goal_calibration,
        rng,
        settings,
        referee_cards_per_match,
        red_card_base_probability=_red_card_base_probability(
            artifact, home.confederation, away.confederation
        ),
    )
    profiles = profiles or {question.home: [], question.away: []}
    lineups = lineups or {question.home: [], question.away: []}
    player_events: dict[str, dict[str, np.ndarray]] = {}
    lineup_details: dict[str, Any] = {}
    for team, segments, penalty_side in (
        (question.home, home_segments, home_penalty),
        (question.away, away_segments, away_penalty),
    ):
        events, details = _allocate_player_events(
            team,
            profiles.get(team, []),
            lineups.get(team, []),
            segments,
            penalty_side,
            n,
            rng,
            settings,
        )
        player_events.update(events)
        lineup_details[team] = details

    total_fouls = home_fouls.sum(axis=1) + away_fouls.sum(axis=1)
    total_cards = home_cards.sum(axis=1) + away_cards.sum(axis=1)
    rare_union = penalty_awarded | red_card_shown
    return CoherentMatchResult(
        home_segments=home_segments,
        away_segments=away_segments,
        player_events=player_events,
        penalty_awarded=penalty_awarded,
        red_card_shown=red_card_shown,
        home_penalty=home_penalty,
        away_penalty=away_penalty,
        metadata={
            "goal_calibration": goal_calibration.as_dict(),
            "latent_match_flow": flow.summary(),
            "tournament_incentives": _tournament_incentive_summary(
                score_states, tournament_context, settings
            ),
            "fouls": foul_details,
            "cards": card_details,
            "territory": territory_details,
            "rare_events": event_details,
            "lineups": lineup_details,
            "accounting": {
                "home_goal_total": int(home_goals.sum()),
                "away_goal_total": int(away_goals.sum()),
                "penalty_or_red_probability": float(np.mean(rare_union)),
                "penalty_or_red_foul_correlation": _safe_correlation(
                    total_fouls, rare_union.astype(float)
                ),
                "cards_foul_correlation": _safe_correlation(
                    total_fouls, total_cards
                ),
            },
        },
    )


def _simulate_rare_events(
    home_segments: dict[str, np.ndarray],
    away_segments: dict[str, np.ndarray],
    calibration: GoalCalibration,
    rng: np.random.Generator,
    settings: Settings,
    referee_cards_per_match: float | None,
    red_card_base_probability: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    home_fouls = home_segments["fouls"].sum(axis=1)
    away_fouls = away_segments["fouls"].sum(axis=1)
    total_fouls = home_fouls + away_fouls
    home_sot = home_segments["shots_on_target"].sum(axis=1)
    away_sot = away_segments["shots_on_target"].sum(axis=1)
    total_sot = home_sot + away_sot
    expected_fouls = max(float(np.mean(total_fouls)), 1.0)
    expected_sot = max(float(np.mean(total_sot)), 1.0)
    foul_intensity = np.maximum(total_fouls, 1.0) / expected_fouls
    box_pressure = np.maximum(total_sot, 1.0) / expected_sot
    supremacy = abs(
        math.tanh((calibration.lambda_home - calibration.lambda_away) / 1.5)
    )
    referee_multiplier = (
        (referee_cards_per_match / settings.global_cards_per_match) ** 0.35
        if referee_cards_per_match
        else 1.0
    )
    penalty_hazard = (
        settings.penalty_base_probability
        * foul_intensity**0.20
        * box_pressure**settings.penalty_box_pressure_elasticity
        * (1.0 + 0.15 * supremacy)
        * referee_multiplier
    )
    red_hazard = (
        (red_card_base_probability or settings.red_card_base_probability)
        * foul_intensity**settings.red_card_foul_elasticity
        * (1.0 + 0.10 * supremacy)
        * referee_multiplier
    )
    penalty_probability = 1.0 - np.exp(-penalty_hazard)
    red_probability = 1.0 - np.exp(-red_hazard)
    penalty_awarded = rng.random(len(total_fouls)) < penalty_probability
    red_card_shown = rng.random(len(total_fouls)) < red_probability
    home_pressure = home_sot + 0.35 * home_segments["corners"].sum(axis=1) + 0.5
    away_pressure = away_sot + 0.35 * away_segments["corners"].sum(axis=1) + 0.5
    home_award_share = home_pressure / (home_pressure + away_pressure)
    home_penalty = penalty_awarded & (rng.random(len(total_fouls)) < home_award_share)
    away_penalty = penalty_awarded & ~home_penalty
    return (
        penalty_awarded,
        red_card_shown,
        home_penalty,
        away_penalty,
        {
            "penalty_probability": float(np.mean(penalty_awarded)),
            "red_card_probability": float(np.mean(red_card_shown)),
            "red_card_base_probability": float(
                red_card_base_probability or settings.red_card_base_probability
            ),
            "penalty_or_red_probability": float(
                np.mean(penalty_awarded | red_card_shown)
            ),
            "absolute_supremacy": supremacy,
        },
    )


def _red_card_base_probability(
    artifact: ModelArtifact,
    home_confederation: str,
    away_confederation: str,
) -> float | None:
    rates = artifact.red_card_rates or {}
    global_probability = rates.get("global_match_probability")
    global_team_rate = float(rates.get("global_team_rate") or 0.0)
    confed_rates = rates.get("confederation_team_rates") or {}
    if global_probability is None or global_team_rate <= 0.0:
        return None
    home_rate = float(confed_rates.get(home_confederation, global_team_rate))
    away_rate = float(confed_rates.get(away_confederation, global_team_rate))
    multiplier = (home_rate + away_rate) / max(2.0 * global_team_rate, 1e-9)
    hazard = -math.log(max(1.0 - float(global_probability), 1e-9))
    probability = 1.0 - math.exp(-hazard * multiplier)
    return float(np.clip(probability, 0.01, 0.35))


def _allocate_player_events(
    team: str,
    profiles: list[PlayerProfile],
    lineups: list[LineupEntry],
    team_segments: dict[str, np.ndarray],
    penalty_awarded: np.ndarray,
    n: int,
    rng: np.random.Generator,
    settings: Settings,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    lineup_by_player = {entry.player_name: entry for entry in lineups}
    confirmed_lineup_known = any(entry.confirmed for entry in lineups)
    events: dict[str, dict[str, np.ndarray]] = {}
    minute_weights: dict[str, np.ndarray] = {}
    lineup_summary: dict[str, Any] = {}
    for profile in profiles:
        entry = lineup_by_player.get(profile.player_name)
        if entry is None and confirmed_lineup_known:
            entry = LineupEntry(
                match_key="",
                team=team,
                player_name=profile.player_name,
                status="unlisted_confirmed_lineup",
                start_probability=0.0,
                expected_start_minutes=0.0,
                sub_probability=0.0,
                expected_sub_minutes=0.0,
                confirmed=True,
            )
        minutes, segment_availability, role = _simulate_minutes(
            entry, n, rng, settings
        )
        minute_weights[profile.player_name] = segment_availability
        events[profile.player_name] = {
            "minutes": minutes,
            "shots": np.zeros(n, dtype=int),
            "shots_on_target": np.zeros(n, dtype=int),
            "second_half_shots_on_target": np.zeros(n, dtype=int),
            "goals": np.zeros(n, dtype=int),
            "assists": np.zeros(n, dtype=int),
        }
        lineup_summary[profile.player_name] = {
            "confirmed": bool(entry.confirmed) if entry else False,
            "status": entry.status if entry else "profile_default",
            "expected_minutes": float(np.mean(minutes)),
            "start_probability": (
                entry.start_probability
                if entry
                else settings.default_player_start_probability
            ),
            "sub_probability": (
                entry.sub_probability
                if entry
                else settings.default_player_sub_probability
            ),
            "simulated_start_rate": float(np.mean(role == 1)),
        }
    if not profiles:
        return events, lineup_summary

    sot_allocations = _allocate_team_counts_to_players(
        team_segments["shots_on_target"],
        profiles,
        minute_weights,
        "shots_on_target_per90",
        settings.player_other_share_prior,
        rng,
    )
    expected_extra_ratio = max(
        np.mean(
            [
                max(profile.shots_per90 - profile.shots_on_target_per90, 0.0)
                / max(profile.shots_on_target_per90, 0.25)
                for profile in profiles
            ]
        ),
        0.5,
    )
    extra_shots = rng.poisson(
        team_segments["shots_on_target"] * expected_extra_ratio
    )
    extra_allocations = _allocate_team_counts_to_players(
        extra_shots,
        profiles,
        minute_weights,
        "shots_per90",
        settings.player_other_share_prior,
        rng,
    )
    goal_allocations = _allocate_team_counts_to_players(
        team_segments["goals"],
        profiles,
        minute_weights,
        "goals_per90",
        settings.player_other_share_prior,
        rng,
        penalty_awarded=penalty_awarded,
    )
    assisted_goals = rng.binomial(team_segments["goals"], 0.72)
    assist_allocations = _allocate_team_counts_to_players(
        assisted_goals,
        profiles,
        minute_weights,
        "assists_per90",
        settings.player_other_share_prior * 1.5,
        rng,
    )
    for profile in profiles:
        name = profile.player_name
        sot_segments = sot_allocations[name]
        events[name]["shots_on_target"] = sot_segments.sum(axis=1)
        events[name]["second_half_shots_on_target"] = sot_segments[:, 3:].sum(
            axis=1
        )
        events[name]["shots"] = (
            sot_segments + extra_allocations[name]
        ).sum(axis=1)
        events[name]["goals"] = goal_allocations[name].sum(axis=1)
        events[name]["assists"] = assist_allocations[name].sum(axis=1)
    events[f"{team}::__other__"] = {
        "minutes": np.full(n, 90.0),
        "shots": (
            sot_allocations["__other__"] + extra_allocations["__other__"]
        ).sum(axis=1),
        "shots_on_target": sot_allocations["__other__"].sum(axis=1),
        "second_half_shots_on_target": sot_allocations["__other__"][:, 3:].sum(
            axis=1
        ),
        "goals": goal_allocations["__other__"].sum(axis=1),
        "assists": assist_allocations["__other__"].sum(axis=1),
    }
    return events, lineup_summary


def _simulate_minutes(
    entry: LineupEntry | None,
    n: int,
    rng: np.random.Generator,
    settings: Settings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    start_probability = (
        entry.start_probability
        if entry
        else settings.default_player_start_probability
    )
    sub_probability = (
        entry.sub_probability
        if entry
        else settings.default_player_sub_probability
    )
    start_minutes = (
        entry.expected_start_minutes if entry else settings.default_start_minutes
    )
    sub_minutes = (
        entry.expected_sub_minutes if entry else settings.default_sub_minutes
    )
    draw = rng.random(n)
    role = np.where(
        draw < start_probability,
        1,
        np.where(draw < start_probability + sub_probability, 2, 0),
    )
    minutes = np.where(role == 1, start_minutes, np.where(role == 2, sub_minutes, 0.0))
    availability = np.zeros((n, 6), dtype=float)
    for segment in range(6):
        start_minute = segment * 15.0
        end_minute = (segment + 1) * 15.0
        starter_overlap = np.clip(
            np.minimum(minutes, end_minute) - start_minute, 0.0, 15.0
        ) / 15.0
        sub_start = 90.0 - minutes
        sub_overlap = np.clip(
            end_minute - np.maximum(sub_start, start_minute), 0.0, 15.0
        ) / 15.0
        availability[:, segment] = np.where(
            role == 1, starter_overlap, np.where(role == 2, sub_overlap, 0.0)
        )
    return minutes, availability, role


def _allocate_team_counts_to_players(
    team_counts: np.ndarray,
    profiles: list[PlayerProfile],
    availability: dict[str, np.ndarray],
    rate_field: str,
    other_weight: float,
    rng: np.random.Generator,
    penalty_awarded: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    allocations = {
        profile.player_name: np.zeros_like(team_counts, dtype=int)
        for profile in profiles
    }
    allocations["__other__"] = np.zeros_like(team_counts, dtype=int)
    for segment in range(team_counts.shape[1]):
        remaining = team_counts[:, segment].astype(int).copy()
        weights = []
        for profile in profiles:
            rate = max(float(getattr(profile, rate_field)), 0.01)
            weight = rate * availability[profile.player_name][:, segment]
            if penalty_awarded is not None and profile.penalty_taker:
                on_field = availability[profile.player_name][:, segment] > 0
                weight = weight + 1.5 * penalty_awarded.astype(float) * on_field
            weights.append(weight)
        remaining_weight = np.full(len(remaining), other_weight, dtype=float)
        for weight in weights:
            remaining_weight += weight
        for profile, weight in zip(profiles, weights):
            probability = np.divide(
                weight,
                remaining_weight,
                out=np.zeros(len(remaining)),
                where=remaining_weight > 0,
            )
            draw = rng.binomial(remaining, np.clip(probability, 0.0, 1.0))
            allocations[profile.player_name][:, segment] = draw
            remaining -= draw
            remaining_weight -= weight
        allocations["__other__"][:, segment] = remaining
    return allocations


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) < 1e-9 or np.std(right) < 1e-9:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])
