from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import Settings
from .domain import Question, QuestionType, Stat, TieHandling
from .goals import GoalCalibration, fallback_goal_lambdas
from .model import ModelArtifact, TeamParameters
from .registry import normalize_team_name
from .tournament import MatchTournamentContext, TeamTournamentContext


@dataclass
class SimulationResult:
    raw_probability: float
    p_home_more: float | None
    p_tie: float | None
    p_away_more: float | None
    interval_80: tuple[float, float]
    home_counts: np.ndarray
    away_counts: np.ndarray
    metadata: dict[str, Any]


@dataclass
class MatchFlow:
    values: np.ndarray
    centered: np.ndarray
    supremacy: float
    method: str

    def summary(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "mean": float(np.mean(self.values)),
            "std": float(np.std(self.values)),
            "supremacy_shift": self.supremacy,
            "p10": float(np.quantile(self.values, 0.1)),
            "p90": float(np.quantile(self.values, 0.9)),
        }


@dataclass
class JointSimulationResult:
    home_counts: dict[str, np.ndarray]
    away_counts: dict[str, np.ndarray]
    home_goals: np.ndarray
    away_goals: np.ndarray
    metadata: dict[str, Any]


def simulate(
    artifact: ModelArtifact,
    question: Question,
    settings: Settings,
    simulations: int | None = None,
    seed: int | None = None,
    goal_calibration: GoalCalibration | None = None,
    referee_cards_per_match: float | None = None,
    tournament_context: MatchTournamentContext | None = None,
) -> SimulationResult:
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
            lambda_home=lambda_home,
            lambda_away=lambda_away,
            source="model_only",
            objective=0.0,
            residuals=[],
            rho=settings.dixon_coles_rho,
            targets_used=0,
        )
    flow = _draw_match_flow(goal_calibration, n, rng, settings)
    score_states, _, _, _, _ = _simulate_score_states(
        goal_calibration, n, rng, flow, settings, tournament_context
    )
    if question.stat == Stat.FOULS:
        home_counts, away_counts, details = _simulate_fouls(
            artifact,
            question,
            home,
            away,
            n,
            rng,
            score_states,
            settings,
            tournament_context,
        )
    elif question.stat == Stat.CARDS:
        foul_home, foul_away, foul_details = _simulate_foul_segments(
            artifact,
            question,
            home,
            away,
            n,
            rng,
            score_states,
            settings,
            tournament_context,
        )
        home_segments, away_segments, details = _simulate_card_segments(
            artifact,
            question,
            home,
            away,
            foul_home,
            foul_away,
            rng,
            settings,
            referee_cards_per_match=referee_cards_per_match,
            tournament_context=tournament_context,
        )
        home_counts, away_counts = _select_period(
            home_segments, away_segments, question.question_type
        )
        details["foul_process"] = foul_details
    else:
        home_counts, away_counts, details = _simulate_territory_stat(
            artifact,
            question,
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
    details["count_prior_sources"] = _count_prior_sources(
        artifact, home, away, question.stat.value
    )
    details["goal_calibration"] = goal_calibration.as_dict()
    details["latent_match_flow"] = flow.summary()
    details["tournament_incentives"] = _tournament_incentive_summary(
        score_states, tournament_context, settings
    )

    if question.question_type == QuestionType.THRESHOLD:
        assert question.k is not None
        raw_probability = float(np.mean(home_counts >= question.k))
        interval = tuple(float(value) for value in np.quantile(home_counts, [0.1, 0.9]))
        return SimulationResult(
            raw_probability=raw_probability,
            p_home_more=None,
            p_tie=None,
            p_away_more=None,
            interval_80=interval,
            home_counts=home_counts,
            away_counts=away_counts,
            metadata=details,
        )

    p_home = float(np.mean(home_counts > away_counts))
    p_tie = float(np.mean(home_counts == away_counts))
    p_away = float(np.mean(home_counts < away_counts))
    tie_handling = TieHandling(settings.tie_handling)
    if tie_handling == TieHandling.STRICT:
        raw_probability = p_home
    elif tie_handling == TieHandling.HALF:
        raw_probability = p_home + 0.5 * p_tie
    elif tie_handling == TieHandling.HOME:
        raw_probability = p_home + p_tie
    else:
        raw_probability = p_home
    difference = home_counts - away_counts
    interval = tuple(float(value) for value in np.quantile(difference, [0.1, 0.9]))
    details["tie_handling"] = tie_handling.value
    return SimulationResult(
        raw_probability=raw_probability,
        p_home_more=p_home,
        p_tie=p_tie,
        p_away_more=p_away,
        interval_80=interval,
        home_counts=home_counts,
        away_counts=away_counts,
        metadata=details,
    )


def _simulate_fouls(
    artifact: ModelArtifact,
    question: Question,
    home: TeamParameters,
    away: TeamParameters,
    n: int,
    rng: np.random.Generator,
    score_states: np.ndarray,
    settings: Settings,
    tournament_context: MatchTournamentContext | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    home_segments, away_segments, details = _simulate_foul_segments(
        artifact,
        question,
        home,
        away,
        n,
        rng,
        score_states,
        settings,
        tournament_context,
    )
    home_counts, away_counts = _select_period(
        home_segments, away_segments, question.question_type
    )
    return home_counts, away_counts, details


def _simulate_foul_segments(
    artifact: ModelArtifact,
    question: Question,
    home: TeamParameters,
    away: TeamParameters,
    n: int,
    rng: np.random.Generator,
    score_states: np.ndarray,
    settings: Settings,
    tournament_context: MatchTournamentContext | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    elo_gap = abs((question.home_elo or 1500.0) - (question.away_elo or 1500.0))
    log_mu = math.log(max(artifact.foul_total_mean, 0.1))
    log_mu += artifact.context_effects.get(
        question.competition_type, artifact.context_effects.get("other", 0.0)
    )
    log_mu += artifact.foul_elo_gap_coefficient * elo_gap / 400.0
    referee_mode = "integrated"
    if question.referee and question.referee in artifact.referee_effects:
        referee_effect = np.full(n, artifact.referee_effects[question.referee])
        referee_mode = "known"
    elif artifact.referee_effects:
        names = list(artifact.referee_effects)
        probabilities = np.array(
            [artifact.referee_weights.get(name, 0.0) for name in names], dtype=float
        )
        probabilities = (
            probabilities / probabilities.sum()
            if probabilities.sum() > 0
            else np.full(len(names), 1.0 / len(names))
        )
        sampled = rng.choice(names, size=n, p=probabilities)
        referee_effect = np.array([artifact.referee_effects[name] for name in sampled])
    else:
        referee_effect = np.zeros(n)
        referee_mode = "global"
    total_mu = np.exp(log_mu + referee_effect)
    totals = _draw_negative_binomial(total_mu, artifact.foul_total_dispersion, rng)

    expected_home_possession = _project_possession(home, away, question)
    possession_deficit = (50.0 - expected_home_possession) / 10.0
    historical_logit = 0.5 * (
        _logit(home.foul_share) - _logit(away.foul_share)
    )
    pressing_difference = (home.pressing_proxy or 0.0) - (away.pressing_proxy or 0.0)
    style_interaction = (home.pressing_proxy or 0.0) * (away.pressing_proxy or 0.0)
    split_logit = (
        historical_logit
        + artifact.foul_possession_coefficient * possession_deficit
        + artifact.foul_pressing_coefficient * pressing_difference
        + (
            artifact.foul_style_interaction_coefficient * style_interaction
            if settings.use_foul_style_interaction
            else 0.0
        )
    )
    mean_share = 1.0 / (1.0 + math.exp(-split_logit))
    concentration = max(artifact.foul_split_concentration, 2.0)
    shares = rng.beta(
        max(mean_share * concentration, 0.01),
        max((1.0 - mean_share) * concentration, 0.01),
        size=n,
    )
    home_full = rng.binomial(totals, shares)
    away_full = totals - home_full
    home_weights, away_weights = _segment_weights(
        "fouls",
        artifact.first_half_shares["fouls"],
        score_states,
        settings,
        tournament_context=tournament_context,
    )
    home_segments = _allocate_counts(home_full, home_weights, rng)
    away_segments = _allocate_counts(away_full, away_weights, rng)
    return home_segments, away_segments, {
        "model": "negative_binomial_total_then_beta_binomial_split",
        "expected_total": float(np.mean(total_mu)),
        "expected_home_possession": expected_home_possession,
        "expected_home_foul_share": mean_share,
        "style_interaction": style_interaction,
        "referee_mode": referee_mode,
    }


def _simulate_card_segments(
    artifact: ModelArtifact,
    question: Question,
    home: TeamParameters,
    away: TeamParameters,
    home_fouls: np.ndarray,
    away_fouls: np.ndarray,
    rng: np.random.Generator,
    settings: Settings,
    referee_cards_per_match: float | None,
    tournament_context: MatchTournamentContext | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    total_fouls = home_fouls.sum(axis=1) + away_fouls.sum(axis=1)
    referee_rate = referee_cards_per_match or settings.global_cards_per_match
    historical_total = max(home.rates["cards"] + away.rates["cards"], 0.1)
    global_total = max(2.0 * artifact.global_rates["cards"], 0.1)
    foul_multiplier = (
        np.maximum(total_fouls, 1.0) / max(artifact.foul_total_mean, 1.0)
    ) ** settings.card_foul_elasticity
    total_mean = referee_rate * (historical_total / global_total) ** 0.5 * foul_multiplier
    total_cards = _draw_negative_binomial(
        total_mean, settings.card_dispersion, rng
    )
    home_foul_total = home_fouls.sum(axis=1)
    foul_share = np.divide(
        home_foul_total + 0.5,
        total_fouls + 1.0,
        out=np.full(len(total_fouls), 0.5),
        where=total_fouls >= 0,
    )
    possession = _project_possession(home, away, question)
    share_logit = np.log(foul_share / np.maximum(1.0 - foul_share, 1e-6))
    historical_share = home.rates["cards"] / historical_total
    share_logit = 0.70 * share_logit + 0.30 * _logit(historical_share)
    share_logit += 0.10 * (50.0 - possession) / 10.0
    home_share = 1.0 / (1.0 + np.exp(-share_logit))
    home_total = rng.binomial(total_cards, np.clip(home_share, 0.05, 0.95))
    away_total = total_cards - home_total
    home_weights = home_fouls.astype(float) + 0.25
    away_weights = away_fouls.astype(float) + 0.25
    home_segments = _allocate_counts(home_total, home_weights, rng)
    away_segments = _allocate_counts(away_total, away_weights, rng)
    return home_segments, away_segments, {
        "model": "foul_conditioned_negative_binomial_cards",
        "expected_total_cards": float(np.mean(total_mean)),
        "referee_cards_per_match": referee_rate,
        "historical_home_card_share": historical_share,
        "card_foul_correlation": float(
            np.corrcoef(total_fouls, total_cards)[0, 1]
        ),
    }


def _simulate_territory_stat(
    artifact: ModelArtifact,
    question: Question,
    home: TeamParameters,
    away: TeamParameters,
    n: int,
    rng: np.random.Generator,
    score_states: np.ndarray,
    goal_calibration: GoalCalibration,
    settings: Settings,
    flow: MatchFlow,
    tournament_context: MatchTournamentContext | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    home_segments, away_segments, details = _simulate_territory_segments(
        artifact,
        question,
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
    home_counts, away_counts = _select_period(
        home_segments, away_segments, question.question_type
    )
    return home_counts, away_counts, details


def _simulate_territory_segments(
    artifact: ModelArtifact,
    question: Question,
    home: TeamParameters,
    away: TeamParameters,
    n: int,
    rng: np.random.Generator,
    score_states: np.ndarray,
    goal_calibration: GoalCalibration,
    settings: Settings,
    flow: MatchFlow,
    tournament_context: MatchTournamentContext | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    stat = question.stat.value
    possession_details = _project_possession_details(home, away, question, settings)
    possession_home = possession_details["projected_possession"]
    home_base = math.sqrt(max(home.rates[stat] * away.conceded_rates[stat], 0.01))
    away_base = math.sqrt(max(away.rates[stat] * home.conceded_rates[stat], 0.01))
    coefficients = artifact.dominance_coefficients.get(stat, {})
    base_share = home_base / max(home_base + away_base, 0.01)
    learned_rate_share = home.rates[stat] / max(
        home.rates[stat] + away.rates[stat], 0.01
    )
    supremacy = math.tanh(
        (goal_calibration.lambda_home - goal_calibration.lambda_away) / 1.5
    )
    share_logit = (
        _logit(base_share)
        + float(coefficients.get("intercept", 0.0))
        + float(coefficients.get("possession", 0.6))
        * (possession_home - 50.0)
        / 10.0
        + float(coefficients.get("supremacy", 0.6)) * supremacy
    )
    territory_share = 1.0 / (1.0 + math.exp(-share_logit))
    split_details = _territory_split_details(
        stat,
        learned_rate_share,
        territory_share,
        home,
        away,
        supremacy,
        goal_calibration,
        settings,
    )
    uncapped_dominance_share = split_details["uncapped_final_share"]
    share_bounds = settings.territory_share_bounds.get(stat, [0.05, 0.95])
    dominance_share = float(
        np.clip(uncapped_dominance_share, share_bounds[0], share_bounds[1])
    )
    total_mu = home_base + away_base
    fallback_home, fallback_away = fallback_goal_lambdas(
        question.home_elo,
        question.away_elo,
        question.neutral,
        settings.fallback_total_goals,
    )
    market_total_ratio = (
        (goal_calibration.lambda_home + goal_calibration.lambda_away)
        / max(fallback_home + fallback_away, 0.1)
        if goal_calibration.source == "market_calibrated"
        and settings.use_market_count_propagation
        else 1.0
    )
    uncapped_market_volume_multiplier = (
        market_total_ratio ** settings.market_count_elasticity.get(stat, 0.0)
    )
    market_volume_multiplier = min(
        uncapped_market_volume_multiplier,
        settings.max_market_count_volume_multiplier,
    )
    total_mu *= market_volume_multiplier
    home_mu = total_mu * dominance_share
    away_mu = total_mu * (1.0 - dominance_share)
    rate_bounds = settings.territory_rate_multiplier_bounds.get(
        stat, [0.1, 10.0]
    )
    global_rate = max(artifact.global_rates[stat], 0.01)
    home_mu = float(
        np.clip(
            home_mu,
            global_rate * rate_bounds[0],
            global_rate * rate_bounds[1],
        )
    )
    away_mu = float(
        np.clip(
            away_mu,
            global_rate * rate_bounds[0],
            global_rate * rate_bounds[1],
        )
    )
    if not question.neutral:
        home_mu *= 1.03
        away_mu /= 1.03
    size = artifact.dispersions[stat]
    home_random_effect = (
        np.ones(n)
        if size >= 999.0
        else rng.gamma(shape=size, scale=1.0 / size, size=n)
    )
    away_random_effect = (
        np.ones(n)
        if size >= 999.0
        else rng.gamma(shape=size, scale=1.0 / size, size=n)
    )
    home_weights, away_weights = _segment_weights(
        stat,
        float(
            np.clip(
                home.first_half_rates.get(
                    stat, home.rates[stat] * artifact.first_half_shares[stat]
                )
                / max(home.rates[stat], 0.01),
                0.2,
                0.8,
            )
        ),
        score_states,
        settings,
        away_first_half_share=float(
            np.clip(
                away.first_half_rates.get(
                    stat, away.rates[stat] * artifact.first_half_shares[stat]
                )
                / max(away.rates[stat], 0.01),
                0.2,
                0.8,
            )
        ),
        tournament_context=tournament_context,
    )
    home_flow, away_flow = _flow_multipliers(flow, stat, settings)
    home_segments = rng.poisson(
        home_mu * home_random_effect[:, None] * home_flow[:, None] * home_weights
    )
    away_segments = rng.poisson(
        away_mu * away_random_effect[:, None] * away_flow[:, None] * away_weights
    )
    warnings = []
    if split_details["supremacy_absent"]:
        if settings.use_possession_ess_gating:
            warnings.append(
                "supremacy_absent_model_only_territory_projection: no market-calibrated "
                "goal edge; territory split down-weighted toward learned rates"
            )
        else:
            warnings.append(
                "supremacy_absent_model_only_territory_projection: no market-calibrated "
                "goal edge; possession-driven territory split is diagnostic-only"
            )
    return home_segments, away_segments, {
        "model": "segmented_team_negative_binomial",
        "home_rate": home_mu,
        "away_rate": away_mu,
        "expected_home_possession": possession_home,
        "possession_projection": possession_details,
        "territory_split": {
            **split_details,
            "final_share": dominance_share,
            "final_share_capped": dominance_share != uncapped_dominance_share,
        },
        "dominance_share": dominance_share,
        "uncapped_dominance_share": uncapped_dominance_share,
        "dominance_share_capped": dominance_share != uncapped_dominance_share,
        "supremacy": supremacy,
        "dominance_coefficients": coefficients,
        "market_volume_multiplier": market_volume_multiplier,
        "uncapped_market_volume_multiplier": uncapped_market_volume_multiplier,
        "rate_bounds": [
            global_rate * rate_bounds[0],
            global_rate * rate_bounds[1],
        ],
        "warnings": warnings,
        "tournament_incentives": _tournament_incentive_summary(
            score_states, tournament_context, settings
        ),
    }


def _simulate_score_states(
    calibration: GoalCalibration,
    n: int,
    rng: np.random.Generator,
    flow: MatchFlow,
    settings: Settings,
    tournament_context: MatchTournamentContext | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    states = np.zeros((n, 6), dtype=np.int16)
    home_flow, away_flow = _flow_multipliers(flow, "goals", settings)
    home_lambda = calibration.lambda_home * home_flow
    away_lambda = calibration.lambda_away * away_flow
    home_score, away_score = _draw_dixon_coles_scores(
        home_lambda, away_lambda, calibration.rho, rng
    )
    first_half_share = float(np.clip(settings.goal_first_half_share, 0.2, 0.8))
    segment_template = np.array(
        [first_half_share / 3.0] * 3
        + [(1.0 - first_half_share) / 3.0] * 3,
        dtype=float,
    )
    segment_weights = np.broadcast_to(segment_template, (n, 6))
    home_segments = _allocate_counts(home_score, segment_weights, rng).astype(
        np.int16
    )
    away_segments = _allocate_counts(away_score, segment_weights, rng).astype(
        np.int16
    )
    states = _score_states_from_segments(home_segments, away_segments)
    if tournament_context is not None and settings.use_tournament_incentives:
        home_weights, away_weights = _segment_weights(
            "goals",
            first_half_share,
            states,
            settings,
            tournament_context=tournament_context,
        )
        home_segments = _allocate_counts(home_score, home_weights, rng).astype(
            np.int16
        )
        away_segments = _allocate_counts(away_score, away_weights, rng).astype(
            np.int16
        )
    states = _score_states_from_segments(home_segments, away_segments)
    return states, home_score, away_score, home_segments, away_segments


def _score_states_from_segments(
    home_segments: np.ndarray, away_segments: np.ndarray
) -> np.ndarray:
    n = home_segments.shape[0]
    states = np.zeros((n, home_segments.shape[1]), dtype=np.int16)
    running_home = np.zeros(n, dtype=np.int16)
    running_away = np.zeros(n, dtype=np.int16)
    for segment in range(home_segments.shape[1]):
        states[:, segment] = running_home - running_away
        running_home += home_segments[:, segment]
        running_away += away_segments[:, segment]
    return states


def _draw_dixon_coles_scores(
    home_lambda: np.ndarray,
    away_lambda: np.ndarray,
    rho: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(home_lambda)
    home = np.zeros(n, dtype=np.int16)
    away = np.zeros(n, dtype=np.int16)
    pending = np.arange(n)
    while len(pending):
        candidate_home = rng.poisson(home_lambda[pending]).astype(np.int16)
        candidate_away = rng.poisson(away_lambda[pending]).astype(np.int16)
        tau = _dixon_coles_tau(
            candidate_home,
            candidate_away,
            home_lambda[pending],
            away_lambda[pending],
            rho,
        )
        maximum = np.maximum.reduce(
            [
                np.ones(len(pending)),
                1.0 - home_lambda[pending] * away_lambda[pending] * rho,
                1.0 + home_lambda[pending] * rho,
                1.0 + away_lambda[pending] * rho,
                np.full(len(pending), 1.0 - rho),
            ]
        )
        accepted = rng.random(len(pending)) < np.clip(tau / maximum, 0.0, 1.0)
        accepted_indices = pending[accepted]
        home[accepted_indices] = candidate_home[accepted]
        away[accepted_indices] = candidate_away[accepted]
        pending = pending[~accepted]
    return home, away


def _dixon_coles_tau(
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    home_lambda: np.ndarray,
    away_lambda: np.ndarray,
    rho: float,
) -> np.ndarray:
    tau = np.ones(len(home_goals), dtype=float)
    tau = np.where(
        (home_goals == 0) & (away_goals == 0),
        1.0 - home_lambda * away_lambda * rho,
        tau,
    )
    tau = np.where(
        (home_goals == 0) & (away_goals == 1),
        1.0 + home_lambda * rho,
        tau,
    )
    tau = np.where(
        (home_goals == 1) & (away_goals == 0),
        1.0 + away_lambda * rho,
        tau,
    )
    tau = np.where(
        (home_goals == 1) & (away_goals == 1),
        1.0 - rho,
        tau,
    )
    return np.maximum(tau, 1e-6)


def simulate_joint_match(
    artifact: ModelArtifact,
    question: Question,
    settings: Settings,
    simulations: int | None = None,
    seed: int | None = None,
    goal_calibration: GoalCalibration | None = None,
    tournament_context: MatchTournamentContext | None = None,
) -> JointSimulationResult:
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
    score_states, home_goals, away_goals, _, _ = _simulate_score_states(
        goal_calibration, n, rng, flow, settings, tournament_context
    )
    home_counts: dict[str, np.ndarray] = {}
    away_counts: dict[str, np.ndarray] = {}
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
        home_stat, away_stat, _ = _simulate_territory_stat(
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
        home_counts[stat.value] = home_stat
        away_counts[stat.value] = away_stat
    correlation_inputs = np.column_stack(
        [
            home_goals,
            home_counts["corners"],
            home_counts["shots_on_target"],
            home_counts["offsides"],
        ]
    )
    return JointSimulationResult(
        home_counts=home_counts,
        away_counts=away_counts,
        home_goals=home_goals,
        away_goals=away_goals,
        metadata={
            "goal_calibration": goal_calibration.as_dict(),
            "latent_match_flow": flow.summary(),
            "tournament_incentives": _tournament_incentive_summary(
                score_states, tournament_context, settings
            ),
            "home_stat_correlation": np.corrcoef(
                correlation_inputs, rowvar=False
            ).tolist(),
            "correlation_order": [
                "goals",
                "corners",
                "shots_on_target",
                "offsides",
            ],
        },
    )


def _draw_match_flow(
    calibration: GoalCalibration,
    n: int,
    rng: np.random.Generator,
    settings: Settings,
) -> MatchFlow:
    supremacy = math.tanh(
        (calibration.lambda_home - calibration.lambda_away) / 1.5
    )
    mean = settings.latent_flow_supremacy_shift * supremacy
    if settings.latent_flow_method == "none":
        values = np.full(n, mean)
    elif settings.latent_flow_method == "shared_normal":
        values = rng.normal(mean, settings.latent_flow_std, size=n)
    else:
        raise ValueError(
            f"Unsupported latent_flow_method: {settings.latent_flow_method}"
        )
    return MatchFlow(
        values=values,
        centered=values - mean,
        supremacy=mean,
        method=settings.latent_flow_method,
    )


def _flow_multipliers(
    flow: MatchFlow, stat: str, settings: Settings
) -> tuple[np.ndarray, np.ndarray]:
    loading = float(settings.latent_flow_loadings.get(stat, 0.0))
    variance_correction = 0.5 * loading * loading * settings.latent_flow_std**2
    home = np.exp(loading * flow.centered - variance_correction)
    away = np.exp(-loading * flow.centered - variance_correction)
    return home, away


def _segment_weights(
    stat: str,
    first_half_share: float,
    score_states: np.ndarray,
    settings: Settings,
    away_first_half_share: float | None = None,
    tournament_context: MatchTournamentContext | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    home_base = np.array(
        [first_half_share / 3.0] * 3
        + [(1.0 - first_half_share) / 3.0] * 3,
        dtype=float,
    )
    away_share = (
        first_half_share if away_first_half_share is None else away_first_half_share
    )
    away_base = np.array(
        [away_share / 3.0] * 3 + [(1.0 - away_share) / 3.0] * 3,
        dtype=float,
    )
    home = np.broadcast_to(home_base, score_states.shape).copy()
    away = np.broadcast_to(away_base, score_states.shape).copy()
    chasing = (
        settings.game_state_chasing_multiplier.get(stat, 1.0)
        if settings.use_game_state_dynamics
        else 1.0
    )
    leading = (
        settings.game_state_leading_multiplier.get(stat, 1.0)
        if settings.use_game_state_dynamics
        else 1.0
    )
    home *= np.where(score_states < 0, chasing, np.where(score_states > 0, leading, 1.0))
    away *= np.where(score_states > 0, chasing, np.where(score_states < 0, leading, 1.0))
    if tournament_context is not None and settings.use_tournament_incentives:
        home, away = _apply_tournament_incentive_weights(
            stat, home, away, score_states, settings, tournament_context
        )
    return home, away


def _apply_tournament_incentive_weights(
    stat: str,
    home_weights: np.ndarray,
    away_weights: np.ndarray,
    score_states: np.ndarray,
    settings: Settings,
    context: MatchTournamentContext,
) -> tuple[np.ndarray, np.ndarray]:
    home = home_weights.copy()
    away = away_weights.copy()
    second_half = np.zeros_like(score_states, dtype=bool)
    second_half[:, 3:] = True
    lead_threshold = max(int(settings.tournament_coast_lead_threshold), 1)
    blowout_threshold = max(int(settings.tournament_blowout_lead_threshold), lead_threshold)
    home_leading = second_half & (score_states >= lead_threshold)
    away_leading = second_half & (score_states <= -lead_threshold)
    home_blowout = second_half & (score_states >= blowout_threshold)
    away_blowout = second_half & (score_states <= -blowout_threshold)

    home_coasts = _team_coasts_when_leading(context.home, settings)
    away_coasts = _team_coasts_when_leading(context.away, settings)
    home_damage_limits = bool(context.home and context.home.damage_limitation)
    away_damage_limits = bool(context.away and context.away.damage_limitation)

    if home_coasts:
        leading_multiplier = _coast_leading_multiplier(
            context.home, stat, settings, blowout=False
        )
        blowout_multiplier = _coast_leading_multiplier(
            context.home, stat, settings, blowout=True
        )
        home *= np.where(
            home_blowout,
            blowout_multiplier,
            np.where(
                home_leading,
                leading_multiplier,
                1.0,
            ),
        )
        away *= np.where(
            home_leading,
            (
                settings.tournament_damage_limitation_multiplier
                if away_damage_limits
                else settings.tournament_trailing_multiplier
            ).get(stat, 1.0),
            1.0,
        )
    if away_coasts:
        leading_multiplier = _coast_leading_multiplier(
            context.away, stat, settings, blowout=False
        )
        blowout_multiplier = _coast_leading_multiplier(
            context.away, stat, settings, blowout=True
        )
        away *= np.where(
            away_blowout,
            blowout_multiplier,
            np.where(
                away_leading,
                leading_multiplier,
                1.0,
            ),
        )
        home *= np.where(
            away_leading,
            (
                settings.tournament_damage_limitation_multiplier
                if home_damage_limits
                else settings.tournament_trailing_multiplier
            ).get(stat, 1.0),
            1.0,
        )
    return home, away


def _coast_leading_multiplier(
    context: TeamTournamentContext | None,
    stat: str,
    settings: Settings,
    *,
    blowout: bool,
) -> float:
    if _team_is_structured_possession(context, settings):
        multipliers = (
            settings.structured_possession_blowout_leading_multiplier
            if blowout
            else settings.structured_possession_coast_leading_multiplier
        )
        return float(multipliers.get(stat, 1.0))
    multipliers = (
        settings.tournament_blowout_leading_multiplier
        if blowout
        else settings.tournament_coast_leading_multiplier
    )
    return float(multipliers.get(stat, 1.0))


def _team_is_structured_possession(
    context: TeamTournamentContext | None, settings: Settings
) -> bool:
    if context is None or not context.tactical_style:
        return False
    style = context.tactical_style.strip().lower().replace("-", "_").replace(" ", "_")
    configured = {
        item.strip().lower().replace("-", "_").replace(" ", "_")
        for item in settings.structured_possession_tactical_styles
    }
    return style in configured


def _team_coasts_when_leading(
    context: TeamTournamentContext | None, settings: Settings
) -> bool:
    if context is None:
        return False
    if context.goal_difference_priority or context.must_win:
        return False
    if context.coast_if_leading or context.qualified:
        return True
    probability = context.qualification_probability
    return probability is not None and probability >= settings.tournament_secure_probability


def _tournament_incentive_summary(
    score_states: np.ndarray,
    context: MatchTournamentContext | None,
    settings: Settings,
) -> dict[str, Any]:
    if context is None:
        return {"enabled": False, "reason": "no_tournament_context"}
    if not settings.use_tournament_incentives:
        return {"enabled": False, "reason": "disabled", "context": _context_summary(context)}
    lead_threshold = max(int(settings.tournament_coast_lead_threshold), 1)
    blowout_threshold = max(int(settings.tournament_blowout_lead_threshold), lead_threshold)
    second_half = np.zeros_like(score_states, dtype=bool)
    second_half[:, 3:] = True
    home_big = second_half & (score_states >= lead_threshold)
    away_big = second_half & (score_states <= -lead_threshold)
    home_blowout = second_half & (score_states >= blowout_threshold)
    away_blowout = second_half & (score_states <= -blowout_threshold)
    return {
        "enabled": True,
        "lead_threshold": lead_threshold,
        "blowout_threshold": blowout_threshold,
        "home_coasts_when_leading": _team_coasts_when_leading(context.home, settings),
        "away_coasts_when_leading": _team_coasts_when_leading(context.away, settings),
        "home_structured_possession_coast": _team_is_structured_possession(
            context.home, settings
        ),
        "away_structured_possession_coast": _team_is_structured_possession(
            context.away, settings
        ),
        "home_second_half_big_lead_segment_share": float(np.mean(home_big)),
        "away_second_half_big_lead_segment_share": float(np.mean(away_big)),
        "home_second_half_blowout_segment_share": float(np.mean(home_blowout)),
        "away_second_half_blowout_segment_share": float(np.mean(away_blowout)),
        "context": _context_summary(context),
    }


def _context_summary(context: MatchTournamentContext) -> dict[str, Any]:
    return {
        "home": _team_context_summary(context.home),
        "away": _team_context_summary(context.away),
    }


def _team_context_summary(context: TeamTournamentContext | None) -> dict[str, Any] | None:
    if context is None:
        return None
    return {
        "team": context.team,
        "points": context.points,
        "goal_difference": context.goal_difference,
        "group_rank": context.group_rank,
        "qualification_probability": context.qualification_probability,
        "qualified": context.qualified,
        "eliminated": context.eliminated,
        "must_win": context.must_win,
        "goal_difference_priority": context.goal_difference_priority,
        "coast_if_leading": context.coast_if_leading,
        "damage_limitation": context.damage_limitation,
        "tactical_style": context.tactical_style,
        "notes": context.notes,
    }


def _allocate_counts(
    totals: np.ndarray, weights: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    probabilities = weights / weights.sum(axis=1, keepdims=True)
    remaining = totals.astype(int).copy()
    remaining_probability = np.ones(len(totals), dtype=float)
    output = np.zeros_like(weights, dtype=int)
    for segment in range(weights.shape[1] - 1):
        conditional = np.divide(
            probabilities[:, segment],
            remaining_probability,
            out=np.zeros(len(totals), dtype=float),
            where=remaining_probability > 1e-12,
        )
        conditional = np.clip(conditional, 0.0, 1.0)
        output[:, segment] = rng.binomial(remaining, conditional)
        remaining -= output[:, segment]
        remaining_probability -= probabilities[:, segment]
    output[:, -1] = remaining
    return output


def _select_period(
    home_segments: np.ndarray,
    away_segments: np.ndarray,
    question_type: QuestionType,
) -> tuple[np.ndarray, np.ndarray]:
    if question_type == QuestionType.HALFTIME_MORE_THAN:
        return home_segments[:, :3].sum(axis=1), away_segments[:, :3].sum(axis=1)
    if question_type == QuestionType.SECOND_HALF_MORE_THAN:
        return home_segments[:, 3:].sum(axis=1), away_segments[:, 3:].sum(axis=1)
    return home_segments.sum(axis=1), away_segments.sum(axis=1)


def _draw_negative_binomial(
    mean: np.ndarray, size: float, rng: np.random.Generator
) -> np.ndarray:
    safe_mean = np.maximum(mean, 0.0)
    if size >= 999.0:
        return rng.poisson(safe_mean)
    gamma_rate = rng.gamma(shape=size, scale=safe_mean / size)
    return rng.poisson(gamma_rate)


def _project_possession_details(
    home: TeamParameters,
    away: TeamParameters,
    question: Question,
    settings: Settings,
) -> dict[str, Any]:
    history_projection = 0.5 * (home.possession + (100.0 - away.possession))
    elo_projection = None
    if question.home_elo is not None and question.away_elo is not None:
        elo_projection = 50.0 + 10.0 * math.tanh(
            (question.home_elo - question.away_elo) / 500.0
        )
        projection = 0.65 * history_projection + 0.35 * elo_projection
    else:
        projection = history_projection
    projected = float(np.clip(projection, 30.0, 70.0))
    return {
        "raw_home_possession": home.possession,
        "raw_away_possession": away.possession,
        "history_projection": history_projection,
        "elo_projection": elo_projection,
        "projected_possession": projected,
        "opponent_strength_adjustment": {
            "enabled": settings.use_possession_opponent_strength_adjustment,
            "applied": False,
            "reason": (
                "deferred_no_team_possession_competition_strength_profile"
                if settings.use_possession_opponent_strength_adjustment
                else "disabled"
            ),
        },
    }


def _territory_split_details(
    stat: str,
    learned_rate_share: float,
    territory_share: float,
    home: TeamParameters,
    away: TeamParameters,
    supremacy: float,
    goal_calibration: GoalCalibration,
    settings: Settings,
) -> dict[str, Any]:
    home_ess = float(home.rate_sample_sizes.get(stat, 0.0) or 0.0)
    away_ess = float(away.rate_sample_sizes.get(stat, 0.0) or 0.0)
    split_ess = (
        2.0 * home_ess * away_ess / (home_ess + away_ess)
        if home_ess > 0.0 and away_ess > 0.0
        else min(home_ess, away_ess)
    )
    base_weight = float(settings.possession_rate_split_base_weight.get(stat, 0.35))
    max_weight = float(settings.possession_rate_split_max_weight.get(stat, 0.80))
    max_weight = max(base_weight, min(max_weight, 0.98))
    scale = max(float(settings.possession_ess_scale.get(stat, 12.0)), 1e-6)
    split_conflict = (learned_rate_share - 0.5) * (territory_share - 0.5) < 0.0
    if settings.use_possession_ess_gating:
        rate_weight = base_weight + (max_weight - base_weight) * (
            split_ess / (split_ess + scale)
        )
        if not split_conflict:
            rate_weight = min(rate_weight, base_weight)
    else:
        rate_weight = 0.0
    supremacy_absent = (
        goal_calibration.source != "market_calibrated" and abs(supremacy) < 1e-6
    )
    territory_weight_multiplier = 1.0
    if settings.use_possession_ess_gating and supremacy_absent:
        default_multiplier = (
            settings.possession_supremacy_absent_territory_multiplier
            if split_conflict
            else 0.85
        )
        territory_weight_multiplier = float(
            np.clip(default_multiplier, 0.0, 1.0)
        )
        rate_weight = 1.0 - (1.0 - rate_weight) * territory_weight_multiplier
    rate_weight = float(np.clip(rate_weight, 0.0, 0.98))
    possession_weight = 1.0 - rate_weight
    final_share = rate_weight * learned_rate_share + possession_weight * territory_share
    market_share = None
    market_split_weight = 0.0
    market_split_applied = False
    if goal_calibration.source == "market_calibrated":
        goal_total = goal_calibration.lambda_home + goal_calibration.lambda_away
        if goal_total > 1e-9:
            market_share = goal_calibration.lambda_home / goal_total
            market_split_weight = float(
                np.clip(settings.market_territory_split_weight.get(stat, 0.0), 0.0, 1.0)
            )
            market_conflict = (market_share - 0.5) * (final_share - 0.5) < 0.0
            if market_conflict and market_split_weight > 0.0:
                final_share = (
                    market_split_weight * market_share
                    + (1.0 - market_split_weight) * final_share
                )
                market_split_applied = True
    return {
        "enabled": settings.use_possession_ess_gating,
        "learned_rate_share": learned_rate_share,
        "possession_territory_share": territory_share,
        "uncapped_final_share": final_share,
        "rate_weight": rate_weight,
        "possession_weight": possession_weight,
        "home_stat_ess": home_ess,
        "away_stat_ess": away_ess,
        "split_ess": split_ess,
        "split_conflict": split_conflict,
        "ess_scale": scale,
        "base_rate_weight": base_weight,
        "max_rate_weight": max_weight,
        "supremacy_absent": supremacy_absent,
        "supremacy_absent_territory_multiplier": territory_weight_multiplier,
        "market_supremacy_share": market_share,
        "market_split_weight": market_split_weight,
        "market_split_applied": market_split_applied,
    }


def _project_possession(
    home: TeamParameters, away: TeamParameters, question: Question
) -> float:
    history_projection = 0.5 * (home.possession + (100.0 - away.possession))
    if question.home_elo is not None and question.away_elo is not None:
        elo_projection = 50.0 + 10.0 * math.tanh(
            (question.home_elo - question.away_elo) / 500.0
        )
        projection = 0.65 * history_projection + 0.35 * elo_projection
    else:
        projection = history_projection
    return float(np.clip(projection, 30.0, 70.0))


def _team_or_prior(artifact: ModelArtifact, name: str) -> TeamParameters:
    if name in artifact.teams:
        return artifact.teams[name]
    lookup = artifact.metadata.get("team_confederation_lookup", {})
    confederation = lookup.get(normalize_team_name(name), "UNK")
    rates = dict(artifact.confederation_rates.get(confederation, artifact.global_rates))
    sources = {
        stat: (
            "confederation-fallback"
            if confederation in artifact.confederation_rates and confederation != "UNK"
            else "global-fallback"
        )
        for stat in artifact.global_rates
    }
    return TeamParameters(
        confederation=confederation,
        effective_matches=0.0,
        possession=50.0,
        pressing_proxy=None,
        rates=dict(rates),
        conceded_rates=dict(rates),
        first_half_rates={
            stat: rate * artifact.first_half_shares[stat]
            for stat, rate in rates.items()
        },
        rate_sources=sources,
        conceded_rate_sources=sources,
        foul_share=0.5,
    )


def _count_prior_sources(
    artifact: ModelArtifact,
    home: TeamParameters,
    away: TeamParameters,
    stat: str,
) -> dict[str, Any]:
    def side(parameters: TeamParameters) -> dict[str, Any]:
        confed_prior = artifact.confederation_rates.get(
            parameters.confederation, artifact.global_rates
        ).get(stat, artifact.global_rates[stat])
        return {
            "confederation": parameters.confederation,
            "effective_matches": parameters.effective_matches,
            "rate": parameters.rates.get(stat, artifact.global_rates[stat]),
            "source": parameters.rate_sources.get(stat, "global-fallback"),
            "team_effective_matches_for_stat": parameters.rate_sample_sizes.get(
                stat, 0.0
            ),
            "confederation_prior": confed_prior,
            "global_prior": artifact.global_rates[stat],
            "club_prior": parameters.club_prior_rates.get(stat),
            "club_prior_effective_matches": (
                parameters.club_prior_effective_matches.get(stat)
            ),
            "club_prior_metadata": parameters.club_prior_metadata.get(stat),
        }

    return {
        "stat": stat,
        "home": side(home),
        "away": side(away),
    }


def _logit(probability: float) -> float:
    clipped = min(max(probability, 0.01), 0.99)
    return math.log(clipped / (1.0 - clipped))
