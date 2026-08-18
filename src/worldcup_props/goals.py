from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

from .market import aggregate_match_markets


@dataclass(frozen=True)
class GoalMarketTarget:
    name: str
    probability: float
    weight: float
    evaluator: Callable[[np.ndarray], float] = field(repr=False, compare=False)


@dataclass
class GoalCalibration:
    lambda_home: float
    lambda_away: float
    source: str
    objective: float
    residuals: list[dict[str, float | str]]
    rho: float
    targets_used: int
    prior_precision: float = 0.0
    fusion_details: dict[str, float | bool | None] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def calibrate_match_goals(
    database_path: str | Path | None,
    home: str,
    away: str,
    fallback_home: float,
    fallback_away: float,
    use_market: bool = True,
    rho: float = -0.08,
    max_goals: int = 12,
    as_of: str | None = None,
    prior_precision: float = 0.25,
    use_supremacy_weighted_market_fusion: bool = False,
    supremacy_market_fusion_slope: float = 0.35,
    supremacy_market_fusion_max_extra: float = 0.35,
    supremacy_market_fusion_ess_threshold: float = 12.0,
    min_team_effective_matches: float | None = None,
) -> GoalCalibration:
    quotes = (
        aggregate_match_markets(database_path, home, away, as_of=as_of)
        if database_path is not None and use_market
        else []
    )
    targets = _targets_from_quotes(quotes)
    if len(targets) < 2:
        return GoalCalibration(
            lambda_home=float(fallback_home),
            lambda_away=float(fallback_away),
            source="model_only",
            objective=0.0,
            residuals=[],
            rho=rho,
            targets_used=0,
            prior_precision=0.0,
            fusion_details={"enabled": False},
        )
    effective_prior_precision, fusion_details = _supremacy_weighted_prior_precision(
        targets,
        prior_precision,
        use_supremacy_weighted_market_fusion,
        supremacy_market_fusion_slope,
        supremacy_market_fusion_max_extra,
        supremacy_market_fusion_ess_threshold,
        min_team_effective_matches,
    )

    def objective(log_lambdas: np.ndarray) -> float:
        matrix = dixon_coles_matrix(
            float(np.exp(log_lambdas[0])),
            float(np.exp(log_lambdas[1])),
            rho=rho,
            max_goals=max_goals,
        )
        market_loss = float(
            sum(
                target.weight * (target.evaluator(matrix) - target.probability) ** 2
                for target in targets
            )
        )
        prior_loss = effective_prior_precision * float(
            (log_lambdas[0] - math.log(max(fallback_home, 0.08))) ** 2
            + (log_lambdas[1] - math.log(max(fallback_away, 0.08))) ** 2
        )
        return market_loss + prior_loss

    initial = np.log([max(fallback_home, 0.15), max(fallback_away, 0.15)])
    fitted = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        bounds=[(math.log(0.08), math.log(5.5)), (math.log(0.08), math.log(5.5))],
    )
    lambda_home, lambda_away = np.exp(fitted.x)
    matrix = dixon_coles_matrix(
        float(lambda_home), float(lambda_away), rho=rho, max_goals=max_goals
    )
    residuals = []
    for target in targets:
        model_probability = target.evaluator(matrix)
        residuals.append(
            {
                "target": target.name,
                "market_probability": target.probability,
                "model_probability": model_probability,
                "residual": model_probability - target.probability,
                "weight": target.weight,
            }
        )
    return GoalCalibration(
        lambda_home=float(lambda_home),
        lambda_away=float(lambda_away),
        source="market_calibrated",
        objective=float(fitted.fun),
        residuals=residuals,
        rho=rho,
        targets_used=len(targets),
        prior_precision=effective_prior_precision,
        fusion_details=fusion_details,
    )


def _supremacy_weighted_prior_precision(
    targets: list[GoalMarketTarget],
    prior_precision: float,
    enabled: bool,
    slope: float,
    max_extra: float,
    ess_threshold: float,
    min_team_effective_matches: float | None,
) -> tuple[float, dict[str, float | bool | None]]:
    supremacy = _market_supremacy_signal(targets)
    ess_gap = (
        0.0
        if min_team_effective_matches is None
        else max(0.0, 1.0 - min_team_effective_matches / max(ess_threshold, 1e-9))
    )
    normalized_supremacy = min(1.0, supremacy / max(slope, 1e-9))
    extra_market_weight = max_extra * normalized_supremacy * (1.0 + ess_gap)
    effective = (
        prior_precision / (1.0 + extra_market_weight) if enabled else prior_precision
    )
    return effective, {
        "enabled": enabled,
        "market_supremacy_signal": supremacy,
        "normalized_supremacy": normalized_supremacy,
        "min_team_effective_matches": min_team_effective_matches,
        "ess_gap": ess_gap,
        "extra_market_weight": extra_market_weight if enabled else 0.0,
        "original_prior_precision": prior_precision,
        "effective_prior_precision": effective,
    }


def _market_supremacy_signal(targets: list[GoalMarketTarget]) -> float:
    probabilities = {target.name: target.probability for target in targets}
    home = probabilities.get("h2h:home")
    away = probabilities.get("h2h:away")
    if home is not None and away is not None:
        return float(abs(home - away))
    handicaps = [
        abs(float(target.name.rsplit(":", 1)[-1]))
        for target in targets
        if target.name.startswith("asian_handicap:")
    ]
    if handicaps:
        return float(min(1.0, max(handicaps) / 2.0))
    return 0.0


def fallback_goal_lambdas(
    home_elo: float | None,
    away_elo: float | None,
    neutral: bool,
    total_goals: float = 2.55,
) -> tuple[float, float]:
    gap = (home_elo or 1500.0) - (away_elo or 1500.0)
    if not neutral:
        gap += 65.0
    home_share = 1.0 / (1.0 + math.exp(-gap / 300.0))
    home_share = float(np.clip(home_share, 0.22, 0.78))
    return total_goals * home_share, total_goals * (1.0 - home_share)


def dixon_coles_matrix(
    lambda_home: float,
    lambda_away: float,
    rho: float = -0.08,
    max_goals: int = 12,
) -> np.ndarray:
    goals = np.arange(max_goals + 1)
    matrix = np.outer(poisson.pmf(goals, lambda_home), poisson.pmf(goals, lambda_away))
    matrix[0, 0] *= 1.0 - lambda_home * lambda_away * rho
    matrix[0, 1] *= 1.0 + lambda_home * rho
    matrix[1, 0] *= 1.0 + lambda_away * rho
    matrix[1, 1] *= 1.0 - rho
    matrix = np.maximum(matrix, 0.0)
    return matrix / matrix.sum()


def scoreline_probabilities(matrix: np.ndarray) -> dict[str, float]:
    return {
        "home": float(np.tril(matrix, -1).sum()),
        "draw": float(np.trace(matrix)),
        "away": float(np.triu(matrix, 1).sum()),
    }


def total_over_probability(matrix: np.ndarray, line: float) -> float:
    home_goals, away_goals = np.indices(matrix.shape)
    total = home_goals + away_goals
    return _asian_win_equivalent(matrix, total.astype(float) - line)


def both_teams_score_probability(matrix: np.ndarray) -> float:
    """P(both teams score) — both teams record one or more goals."""
    return float(matrix[1:, 1:].sum())


def handicap_probability(matrix: np.ndarray, side: str, line: float) -> float:
    home_goals, away_goals = np.indices(matrix.shape)
    margin = home_goals - away_goals
    signed = margin.astype(float) + line if side == "home" else -margin.astype(float) + line
    return _asian_win_equivalent(matrix, signed)


def _asian_win_equivalent(matrix: np.ndarray, adjusted_margin: np.ndarray) -> float:
    quarter = round(float(np.nanmean(adjusted_margin % 0.5)), 2)
    if quarter in {0.25}:
        return 0.5 * (
            _settlement_probability(matrix, adjusted_margin - 0.25)
            + _settlement_probability(matrix, adjusted_margin + 0.25)
        )
    return _settlement_probability(matrix, adjusted_margin)


def _settlement_probability(matrix: np.ndarray, margin: np.ndarray) -> float:
    wins = float(matrix[margin > 1e-9].sum())
    pushes = float(matrix[np.abs(margin) <= 1e-9].sum())
    return wins + 0.5 * pushes


def _targets_from_quotes(quotes: list[dict[str, Any]]) -> list[GoalMarketTarget]:
    targets: list[GoalMarketTarget] = []
    by_market = {(q["market_type"], q["selection"], q["line"]): q for q in quotes}
    for selection in ("home", "draw", "away"):
        quote = by_market.get(("h2h", selection, None))
        if quote:
            targets.append(
                GoalMarketTarget(
                    f"h2h:{selection}",
                    float(quote["probability"]),
                    2.0,
                    lambda matrix, selection=selection: scoreline_probabilities(matrix)[
                        selection
                    ],
                )
            )
    for quote in quotes:
        market_type = quote["market_type"]
        selection = quote["selection"]
        line = quote["line"]
        if market_type == "totals" and selection == "over" and line is not None:
            weight = 2.0 if abs(float(line) - 2.5) < 1e-6 else 1.0
            targets.append(
                GoalMarketTarget(
                    f"totals:over:{line}",
                    float(quote["probability"]),
                    weight,
                    lambda matrix, line=float(line): total_over_probability(matrix, line),
                )
            )
        elif market_type == "btts" and selection == "yes":
            # BTTS pins the home/away goal split that h2h+totals leave
            # under-determined, so weight it like an h2h leg.
            targets.append(
                GoalMarketTarget(
                    "btts:yes",
                    float(quote["probability"]),
                    2.0,
                    lambda matrix: both_teams_score_probability(matrix),
                )
            )
        elif (
            market_type == "asian_handicap"
            and selection in {"home", "away"}
            and line is not None
        ):
            targets.append(
                GoalMarketTarget(
                    f"asian_handicap:{selection}:{line}",
                    float(quote["probability"]),
                    0.75,
                    lambda matrix, side=selection, line=float(line): handicap_probability(
                        matrix, side, line
                    ),
                )
            )
    return targets
