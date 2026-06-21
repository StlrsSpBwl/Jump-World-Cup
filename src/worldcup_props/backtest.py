from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import nbinom

from .calibration import (
    apply_calibration_map,
    apply_shrinkage,
    brier_decomposition,
    fit_platt_brier,
    tune_shrinkage_lambda,
    weighted_brier_score,
)
from .config import Settings
from .db import connect, initialize
from .domain import Question, QuestionType, Stat
from .goals import calibrate_match_goals, fallback_goal_lambdas
from .model import ModelArtifact, fit_model
from .simulation import simulate


TOURNAMENTS = {
    "world_cup_2022": ("2022-11-20", "2022-12-19"),
    "euro_2024": ("2024-06-14", "2024-07-15"),
    "copa_america_2024": ("2024-06-20", "2024-07-15"),
}

TOURNAMENT_ALIASES = {
    "world_cup_2022": ("world cup", "fifa world cup"),
    "euro_2024": ("euro", "european championship"),
    "copa_america_2024": ("copa america", "copa américa"),
}


@dataclass
class BacktestEvent:
    tournament: str
    match_id: int
    key: str
    probability: float
    outcome: int
    baseline_50: float
    baseline_average: float
    weight: float
    market_calibrated_probability: float | None = None


def run_backtest(
    database_path: str | Path,
    settings: Settings,
    tournaments: list[str] | None = None,
    simulations: int = 10_000,
) -> tuple[dict[str, Any], dict[str, float], dict[str, float]]:
    selected = tournaments or list(TOURNAMENTS)
    events: list[BacktestEvent] = []
    tournament_summaries: dict[str, Any] = {}
    for tournament in selected:
        if tournament not in TOURNAMENTS:
            raise ValueError(f"Unknown tournament: {tournament}")
        start, end = TOURNAMENTS[tournament]
        matches = _load_holdout_matches(database_path, tournament, start, end)
        if not matches:
            tournament_summaries[tournament] = {
                "warning": "No matches found in date range",
                "events": 0,
            }
            continue
        artifact = fit_model(database_path, settings, cutoff_date=start)
        tournament_events = _evaluate_matches(
            tournament, matches, artifact, database_path, settings, simulations
        )
        events.extend(tournament_events)
        tournament_summaries[tournament] = {
            "matches": len(matches),
            "events": len(tournament_events),
            "training_matches": artifact.metadata["training_matches"],
        }
    if not events:
        raise ValueError("No held-out tournament events were available for backtesting")

    lambdas: dict[str, float] = {}
    base_rates: dict[str, float] = {}
    prop_reports: dict[str, Any] = {}
    calibration_maps: dict[str, dict[str, float | str]] = {}
    keys = sorted({event.key for event in events})
    for key in keys:
        group = [event for event in events if event.key == key]
        raw = [event.probability for event in group]
        outcomes = [event.outcome for event in group]
        baseline_50 = [event.baseline_50 for event in group]
        baseline_average = [event.baseline_average for event in group]
        weights = [event.weight for event in group]
        coefficient, base_rate, _ = tune_shrinkage_lambda(
            raw, outcomes, weights=weights
        )
        final_linear_map: dict[str, float | str] = {
            "method": "linear_shrinkage",
            "coefficient": coefficient,
            "base_rate": base_rate,
        }
        final_platt_map, _ = fit_platt_brier(raw, outcomes, weights)
        (
            linear_predictions,
            platt_predictions,
            calibration_validation,
        ) = _cross_validated_calibration(group)
        shrinkage_brier = weighted_brier_score(
            linear_predictions, outcomes, weights
        )
        platt_brier = weighted_brier_score(platt_predictions, outcomes, weights)
        if settings.calibration_method == "platt" or (
            settings.calibration_method == "auto"
            and platt_brier + 1e-9 < shrinkage_brier
        ):
            selected_map = final_platt_map
            calibrated = platt_predictions
            model_brier = platt_brier
        else:
            selected_map = final_linear_map
            calibrated = linear_predictions
            model_brier = shrinkage_brier
        brier_50 = weighted_brier_score(baseline_50, outcomes, weights)
        brier_average = weighted_brier_score(baseline_average, outcomes, weights)
        warnings = []
        if model_brier > brier_50:
            warnings.append("Model loses to 50/50-with-tie-adjustment baseline")
        if model_brier > brier_average:
            warnings.append("Model loses to league-average-rate baseline")
        lambdas[key] = coefficient
        base_rates[key] = base_rate
        calibration_maps[key] = selected_map
        prop_reports[key] = {
            "events": len(group),
            "raw_brier": weighted_brier_score(raw, outcomes, weights),
            "calibrated_brier": model_brier,
            "linear_shrinkage_brier": shrinkage_brier,
            "platt_brier": platt_brier,
            "selected_calibration": selected_map,
            "calibration_validation": calibration_validation,
            "baseline_50_brier": brier_50,
            "baseline_average_brier": brier_average,
            "lambda": coefficient,
            "base_rate": base_rate,
            "reliability_before": brier_decomposition(
                raw, outcomes, weights=weights
            ).as_dict(),
            "reliability_after": brier_decomposition(
                calibrated, outcomes, weights=weights
            ).as_dict(),
            "warnings": warnings,
        }
        market_group = [
            event for event in group if event.market_calibrated_probability is not None
        ]
        if market_group:
            market_probabilities = [
                float(event.market_calibrated_probability) for event in market_group
            ]
            market_outcomes = [event.outcome for event in market_group]
            market_weights = [event.weight for event in market_group]
            model_same_matches = [event.probability for event in market_group]
            prop_reports[key]["market_calibration_comparison"] = {
                "events": len(market_group),
                "model_only_brier": weighted_brier_score(
                    model_same_matches, market_outcomes, market_weights
                ),
                "market_calibrated_brier": weighted_brier_score(
                    market_probabilities, market_outcomes, market_weights
                ),
            }
    all_outcomes = [event.outcome for event in events]
    all_calibrated = [
        apply_calibration_map(event.probability, calibration_maps[event.key])
        for event in events
    ]
    all_weights = [event.weight for event in events]
    market_events = [
        event for event in events if event.market_calibrated_probability is not None
    ]
    report = {
        "tournaments": tournament_summaries,
        "overall": {
            "events": len(events),
            "brier_decomposition": brier_decomposition(
                all_calibrated, all_outcomes, weights=all_weights
            ).as_dict(),
        },
        "prop_types": prop_reports,
        "calibration_maps": calibration_maps,
    }
    if market_events:
        report["market_calibration_comparison"] = {
            "events": len(market_events),
            "model_only_brier": weighted_brier_score(
                [event.probability for event in market_events],
                [event.outcome for event in market_events],
                [event.weight for event in market_events],
            ),
            "market_calibrated_brier": weighted_brier_score(
                [float(event.market_calibrated_probability) for event in market_events],
                [event.outcome for event in market_events],
                [event.weight for event in market_events],
            ),
        }
    else:
        report["market_calibration_comparison"] = {
            "events": 0,
            "warning": "No historical match-market quotes matched held-out fixtures",
        }
    return report, lambdas, base_rates


def _cross_validated_calibration(
    group: list[BacktestEvent],
) -> tuple[list[float], list[float], str]:
    tournaments = sorted({event.tournament for event in group})
    raw = [event.probability for event in group]
    outcomes = [event.outcome for event in group]
    weights = [event.weight for event in group]
    if len(tournaments) < 2:
        coefficient, base_rate, _ = tune_shrinkage_lambda(
            raw, outcomes, weights=weights
        )
        platt_map, _ = fit_platt_brier(raw, outcomes, weights)
        return (
            [apply_shrinkage(value, coefficient, base_rate) for value in raw],
            [apply_calibration_map(value, platt_map) for value in raw],
            "in_sample_fallback_single_tournament",
        )
    linear_predictions = [0.5] * len(group)
    platt_predictions = [0.5] * len(group)
    for tournament in tournaments:
        train_indices = [
            index
            for index, event in enumerate(group)
            if event.tournament != tournament
        ]
        test_indices = [
            index
            for index, event in enumerate(group)
            if event.tournament == tournament
        ]
        train_p = [raw[index] for index in train_indices]
        train_y = [outcomes[index] for index in train_indices]
        train_w = [weights[index] for index in train_indices]
        coefficient, base_rate, _ = tune_shrinkage_lambda(
            train_p, train_y, weights=train_w
        )
        platt_map, _ = fit_platt_brier(train_p, train_y, train_w)
        for index in test_indices:
            linear_predictions[index] = apply_shrinkage(
                raw[index], coefficient, base_rate
            )
            platt_predictions[index] = apply_calibration_map(
                raw[index], platt_map
            )
    return (
        linear_predictions,
        platt_predictions,
        "leave_one_tournament_out",
    )


def run_ablation(
    database_path: str | Path,
    settings: Settings,
    tournaments: list[str] | None = None,
    simulations: int = 5_000,
) -> dict[str, Any]:
    baseline = replace(
        settings,
        latent_flow_method="none",
        use_game_state_dynamics=False,
        use_market_count_propagation=False,
        use_ess_gating=False,
        use_possession_ess_gating=False,
        use_foul_style_interaction=False,
        use_supremacy_weighted_market_fusion=False,
        calibration_method="linear",
    )
    components = [
        ("shared_latent_flow", {"latent_flow_method": "shared_normal"}),
        ("platt_recalibration", {"calibration_method": "auto"}),
        ("state_dependent_second_half", {"use_game_state_dynamics": True}),
        ("possession_ess_gating", {"use_possession_ess_gating": True}),
        ("market_count_propagation", {"use_market_count_propagation": True}),
        (
            "supremacy_weighted_market_fusion",
            {"use_supremacy_weighted_market_fusion": True},
        ),
        ("ess_gating", {"use_ess_gating": True}),
        ("foul_style_interaction", {"use_foul_style_interaction": True}),
    ]
    reports: dict[str, Any] = {}
    baseline_report, _, _ = run_backtest(
        database_path,
        baseline,
        tournaments=tournaments,
        simulations=simulations,
    )
    retained_settings = baseline
    retained_scores = {
        key: float(value["calibrated_brier"])
        for key, value in baseline_report["prop_types"].items()
    }
    reports["v1_core"] = {
        "prop_brier": retained_scores,
        "delta_vs_previous": {},
        "mean_delta_vs_previous": None,
        "retained": True,
    }
    reports["coherent_player_cards_penalty"] = {
        "prop_brier": {},
        "delta_vs_previous": {},
        "mean_delta_vs_previous": None,
        "retained": True,
        "brier_evaluable": False,
        "reason": (
            "Historical held-out player, penalty, and red-card outcomes are not "
            "present in the current normalized match table. This component is "
            "retained only to repair joint-distribution accounting; add those "
            "outcomes before treating it as Brier-validated."
        ),
        "structural_validation": [
            "player allocations sum to simulated team totals",
            "card counts share the simulated foul process",
            "penalty/red union conditions on fouls, supremacy, and referee tendency",
            "confirmed lineups change player event probabilities",
        ],
    }
    for name, changes in components:
        variant_settings = replace(retained_settings, **changes)
        report, _, _ = run_backtest(
            database_path,
            variant_settings,
            tournaments=tournaments,
            simulations=simulations,
        )
        scores = {
            key: float(value["calibrated_brier"])
            for key, value in report["prop_types"].items()
        }
        deltas = {
            key: scores[key] - retained_scores[key]
            for key in scores.keys() & retained_scores.keys()
        }
        weighted_delta = float(np.mean(list(deltas.values()))) if deltas else None
        reports[name] = {
            "prop_brier": scores,
            "delta_vs_previous": deltas,
            "mean_delta_vs_previous": weighted_delta,
            "retained": weighted_delta is None or weighted_delta < 0.0,
        }
        if reports[name]["retained"]:
            retained_settings = variant_settings
            retained_scores = scores
    return {
        "order": [
            "v1_core",
            "shared_latent_flow",
            "coherent_player_cards_penalty",
            *[name for name, _ in components if name != "shared_latent_flow"],
        ],
        "variants": reports,
        "retention_rule": "Retain only when mean held-out prop Brier delta is negative",
        "retained_settings": {
            "latent_flow_method": retained_settings.latent_flow_method,
            "calibration_method": retained_settings.calibration_method,
            "use_game_state_dynamics": retained_settings.use_game_state_dynamics,
            "use_market_count_propagation": retained_settings.use_market_count_propagation,
            "use_supremacy_weighted_market_fusion": (
                retained_settings.use_supremacy_weighted_market_fusion
            ),
            "use_ess_gating": retained_settings.use_ess_gating,
            "use_possession_ess_gating": retained_settings.use_possession_ess_gating,
            "use_foul_style_interaction": retained_settings.use_foul_style_interaction,
        },
        "pair_synergy_head_to_head": {
            "shipped": False,
            "baseline": "post_lineup_market_refresh",
            "brier_evaluable": False,
            "reason": (
                "No held-out player-prop history exists to justify pair embeddings "
                "over the post-lineup market-refresh baseline."
            ),
        },
    }


def fit_final_model_with_calibration(
    database_path: str | Path,
    settings: Settings,
    report_path: str | Path,
    tournaments: list[str] | None = None,
    simulations: int = 10_000,
) -> ModelArtifact:
    report, lambdas, base_rates = run_backtest(
        database_path,
        settings,
        tournaments=tournaments,
        simulations=simulations,
    )
    artifact = fit_model(database_path, settings)
    artifact.calibration_lambda = lambdas
    artifact.base_rates = base_rates
    artifact.calibration_maps = {
        str(key): dict(value)
        for key, value in report.get("calibration_maps", {}).items()
    }
    artifact.metadata["backtest_report"] = str(report_path)
    destination = Path(report_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return artifact


def _load_holdout_matches(
    database_path: str | Path, tournament: str, start: str, end: str
) -> list[dict[str, Any]]:
    initialize(database_path)
    query = """
        SELECT m.id, m.match_date, m.competition, m.competition_type,
               m.home_team, m.away_team,
               m.home_elo, m.away_elo, m.referee_name,
               h.fouls AS home_fouls, a.fouls AS away_fouls,
               h.corners AS home_corners, a.corners AS away_corners,
               h.offsides AS home_offsides, a.offsides AS away_offsides,
               h.shots_on_target AS home_shots_on_target,
               a.shots_on_target AS away_shots_on_target,
               h.cards AS home_cards, a.cards AS away_cards,
               h.first_half_corners AS home_first_half_corners,
               a.first_half_corners AS away_first_half_corners
        FROM matches m
        JOIN team_match_stats h ON h.match_id=m.id AND h.is_home=1
        JOIN team_match_stats a ON a.match_id=m.id AND a.is_home=0
        WHERE m.match_date >= ? AND m.match_date < ?
        ORDER BY m.match_date, m.id
    """
    with connect(database_path) as connection:
        candidates = [dict(row) for row in connection.execute(query, (start, end))]
    aliases = TOURNAMENT_ALIASES[tournament]
    expected_type = tournament.rsplit("_", 1)[0]
    return [
        match
        for match in candidates
        if any(alias in match["competition"].casefold() for alias in aliases)
        or match["competition_type"].casefold() == expected_type
    ]


def _evaluate_matches(
    tournament: str,
    matches: list[dict[str, Any]],
    artifact: ModelArtifact,
    database_path: str | Path,
    settings: Settings,
    simulations: int,
) -> list[BacktestEvent]:
    events: list[BacktestEvent] = []
    for match in matches:
        for stat in Stat:
            home_value = match[f"home_{stat.value}"]
            away_value = match[f"away_{stat.value}"]
            if home_value is not None and away_value is not None:
                question = _question(match, stat, QuestionType.MORE_THAN)
                events.append(
                    _event(
                        tournament,
                        match,
                        question,
                        int(home_value > away_value),
                        artifact,
                        database_path,
                        settings,
                        simulations,
                    )
                )
            if home_value is not None:
                for threshold in (1, 2, 3):
                    question = _question(
                        match, stat, QuestionType.THRESHOLD, k=threshold
                    )
                    events.append(
                        _event(
                            tournament,
                            match,
                            question,
                            int(home_value >= threshold),
                            artifact,
                            database_path,
                            settings,
                            simulations,
                        )
                    )
        home_half = match["home_first_half_corners"]
        away_half = match["away_first_half_corners"]
        if home_half is not None and away_half is not None:
            question = _question(
                match, Stat.CORNERS, QuestionType.HALFTIME_MORE_THAN
            )
            events.append(
                _event(
                    tournament,
                    match,
                    question,
                    int(home_half > away_half),
                    artifact,
                    database_path,
                    settings,
                    simulations,
                )
            )
    return events


def _event(
    tournament: str,
    match: dict[str, Any],
    question: Question,
    outcome: int,
    artifact: ModelArtifact,
    database_path: str | Path,
    settings: Settings,
    simulations: int,
) -> BacktestEvent:
    question_seed = int(hashlib.sha256(question.key.encode()).hexdigest()[:8], 16)
    seed = settings.random_seed + int(match["id"]) * 31 + question_seed % 1000
    result = simulate(
        artifact, question, settings, simulations=simulations, seed=seed
    )
    fallback_home, fallback_away = fallback_goal_lambdas(
        question.home_elo,
        question.away_elo,
        question.neutral,
        settings.fallback_total_goals,
    )
    goal_calibration = calibrate_match_goals(
        database_path,
        question.home,
        question.away,
        fallback_home,
        fallback_away,
        use_market=True,
        rho=settings.dixon_coles_rho,
        as_of=f"{match['match_date']}T23:59:59Z",
        prior_precision=settings.market_goal_prior_precision,
    )
    market_probability = None
    if goal_calibration.source == "market_calibrated":
        market_probability = simulate(
            artifact,
            question,
            settings,
            simulations=simulations,
            seed=seed,
            goal_calibration=goal_calibration,
        ).raw_probability
    baseline_50, baseline_average = _baselines(artifact, question)
    probability = result.raw_probability
    if settings.use_ess_gating:
        home_parameters = artifact.teams.get(question.home)
        away_parameters = artifact.teams.get(question.away)
        ess = min(
            home_parameters.effective_matches if home_parameters else 0.0,
            away_parameters.effective_matches if away_parameters else 0.0,
        )
        ess_weight = ess / max(
            ess + settings.ess_probability_prior_matches, 1e-9
        )
        probability = baseline_average + ess_weight * (
            probability - baseline_average
        )
    return BacktestEvent(
        tournament=tournament,
        match_id=int(match["id"]),
        key=_calibration_key(question),
        probability=probability,
        outcome=outcome,
        baseline_50=baseline_50,
        baseline_average=baseline_average,
        weight=float(settings.backtest_prop_weights.get(_calibration_key(question), 1.0)),
        market_calibrated_probability=market_probability,
    )


def _question(
    match: dict[str, Any],
    stat: Stat,
    question_type: QuestionType,
    k: int | None = None,
) -> Question:
    return Question(
        home=match["home_team"],
        away=match["away_team"],
        stat=stat,
        question_type=question_type,
        k=k,
        referee=match["referee_name"],
        competition_type=match["competition_type"],
        home_elo=match["home_elo"],
        away_elo=match["away_elo"],
    )


def _baselines(artifact: ModelArtifact, question: Question) -> tuple[float, float]:
    stat = question.stat.value
    home_parameters = artifact.teams.get(question.home)
    away_parameters = artifact.teams.get(question.away)
    home_confederation = home_parameters.confederation if home_parameters else "UNK"
    away_confederation = away_parameters.confederation if away_parameters else "UNK"
    home_average = artifact.confederation_rates.get(
        home_confederation, artifact.global_rates
    )[stat]
    away_average = artifact.confederation_rates.get(
        away_confederation, artifact.global_rates
    )[stat]
    if question.question_type == QuestionType.THRESHOLD:
        assert question.k is not None
        average = _threshold_probability(
            artifact, stat, question.k, mean_override=home_average
        )
        return 0.5, average
    if stat == "fouls":
        mean = artifact.foul_total_mean / 2.0
        size = max(artifact.foul_total_dispersion / 2.0, 0.25)
    else:
        mean = artifact.global_rates[stat]
        size = artifact.dispersions[stat]
    if question.question_type == QuestionType.HALFTIME_MORE_THAN:
        mean *= artifact.first_half_shares[stat]
        home_average *= artifact.first_half_shares[stat]
        away_average *= artifact.first_half_shares[stat]
    tie = _equal_rate_tie_probability(mean, size)
    tie_adjusted = 0.5 * (1.0 - tie)
    average_probability = _comparison_probability(home_average, away_average, size)
    return tie_adjusted, average_probability


def _threshold_probability(
    artifact: ModelArtifact,
    stat: str,
    threshold: int,
    mean_override: float | None = None,
) -> float:
    if stat == "fouls":
        mean = artifact.foul_total_mean / 2.0
        size = max(artifact.foul_total_dispersion / 2.0, 0.25)
    else:
        mean = artifact.global_rates[stat]
        size = artifact.dispersions[stat]
    if mean_override is not None:
        mean = mean_override
    if size >= 999.0:
        from scipy.stats import poisson

        return float(poisson.sf(threshold - 1, mean))
    probability = size / (size + mean)
    return float(nbinom.sf(threshold - 1, size, probability))


def _equal_rate_tie_probability(mean: float, size: float) -> float:
    maximum = max(30, int(mean + 10.0 * math.sqrt(mean + mean * mean / size)))
    values = np.arange(maximum + 1)
    if size >= 999.0:
        from scipy.stats import poisson

        probabilities = poisson.pmf(values, mean)
    else:
        probability = size / (size + mean)
        probabilities = nbinom.pmf(values, size, probability)
    return float(np.sum(probabilities * probabilities))


def _comparison_probability(mean_home: float, mean_away: float, size: float) -> float:
    maximum = max(
        30,
        int(
            max(mean_home, mean_away)
            + 10.0
            * math.sqrt(
                max(mean_home, mean_away)
                + max(mean_home, mean_away) ** 2 / size
            )
        ),
    )
    values = np.arange(maximum + 1)
    if size >= 999.0:
        from scipy.stats import poisson

        home_pmf = poisson.pmf(values, mean_home)
        away_pmf = poisson.pmf(values, mean_away)
    else:
        home_probability = size / (size + mean_home)
        away_probability = size / (size + mean_away)
        home_pmf = nbinom.pmf(values, size, home_probability)
        away_pmf = nbinom.pmf(values, size, away_probability)
    away_cdf_below = np.concatenate(([0.0], np.cumsum(away_pmf[:-1])))
    return float(np.sum(home_pmf * away_cdf_below))


def _calibration_key(question: Question) -> str:
    if question.k is None:
        return f"{question.stat.value}:{question.question_type.value}"
    return f"{question.stat.value}:{question.question_type.value}:{question.k}"
