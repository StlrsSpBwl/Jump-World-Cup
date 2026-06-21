from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import minimize


@dataclass
class BrierDecomposition:
    brier: float
    reliability: float
    resolution: float
    uncertainty: float
    rows: list[dict[str, float | int]]

    def as_dict(self) -> dict[str, object]:
        return {
            "brier": self.brier,
            "reliability": self.reliability,
            "resolution": self.resolution,
            "uncertainty": self.uncertainty,
            "check": self.uncertainty - self.resolution + self.reliability,
            "calibration_table": self.rows,
        }


def brier_score(probabilities: Iterable[float], outcomes: Iterable[int]) -> float:
    return weighted_brier_score(probabilities, outcomes)


def weighted_brier_score(
    probabilities: Iterable[float],
    outcomes: Iterable[int],
    weights: Iterable[float] | None = None,
) -> float:
    p = np.asarray(list(probabilities), dtype=float)
    y = np.asarray(list(outcomes), dtype=float)
    if len(p) == 0 or len(p) != len(y):
        raise ValueError("probabilities and outcomes must have equal non-zero length")
    w = np.ones(len(p)) if weights is None else np.asarray(list(weights), dtype=float)
    if len(w) != len(p) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("weights must be non-negative, positive-sum, and aligned")
    return float(np.average((p - y) ** 2, weights=w))


def tune_shrinkage_lambda(
    probabilities: Iterable[float],
    outcomes: Iterable[int],
    base_rate: float | None = None,
    weights: Iterable[float] | None = None,
) -> tuple[float, float, float]:
    p = np.asarray(list(probabilities), dtype=float)
    y = np.asarray(list(outcomes), dtype=float)
    w = np.ones(len(p)) if weights is None else np.asarray(list(weights), dtype=float)
    if (
        len(p) == 0
        or len(p) != len(y)
        or len(w) != len(p)
        or np.any(w < 0)
        or w.sum() <= 0
    ):
        raise ValueError("probabilities, outcomes, and weights must align")
    base = float(np.average(y, weights=w) if base_rate is None else base_rate)
    direction = p - base
    denominator = float(np.dot(w * direction, direction))
    if denominator <= 1e-12:
        coefficient = 0.0
    else:
        coefficient = float(np.dot(w * direction, y - base) / denominator)
    coefficient = float(np.clip(coefficient, 0.0, 1.0))
    calibrated = base + coefficient * direction
    return coefficient, base, weighted_brier_score(calibrated, y, w)


def apply_shrinkage(probability: float, coefficient: float, base_rate: float) -> float:
    return float(np.clip(base_rate + coefficient * (probability - base_rate), 0.0, 1.0))


def fit_platt_brier(
    probabilities: Iterable[float],
    outcomes: Iterable[int],
    weights: Iterable[float] | None = None,
) -> tuple[dict[str, float | str], float]:
    p = np.clip(np.asarray(list(probabilities), dtype=float), 1e-5, 1.0 - 1e-5)
    y = np.asarray(list(outcomes), dtype=float)
    w = np.ones(len(p)) if weights is None else np.asarray(list(weights), dtype=float)
    if len(p) == 0 or len(p) != len(y) or len(w) != len(p) or w.sum() <= 0:
        raise ValueError("probabilities, outcomes, and weights must align")
    logits = np.log(p / (1.0 - p))

    def objective(parameters: np.ndarray) -> float:
        fitted = 1.0 / (1.0 + np.exp(-(parameters[0] + parameters[1] * logits)))
        return weighted_brier_score(fitted, y, w)

    result = minimize(
        objective,
        np.array([0.0, 1.0]),
        method="L-BFGS-B",
        bounds=[(-6.0, 6.0), (0.0, 4.0)],
    )
    mapping: dict[str, float | str] = {
        "method": "platt",
        "intercept": float(result.x[0]),
        "slope": float(result.x[1]),
    }
    return mapping, float(result.fun)


def apply_calibration_map(
    probability: float, mapping: dict[str, float | str] | None
) -> float:
    if not mapping:
        return float(np.clip(probability, 0.0, 1.0))
    method = str(mapping.get("method", "identity"))
    if method == "platt":
        clipped = float(np.clip(probability, 1e-5, 1.0 - 1e-5))
        logit = np.log(clipped / (1.0 - clipped))
        fitted = float(mapping["intercept"]) + float(mapping["slope"]) * logit
        return float(1.0 / (1.0 + np.exp(-fitted)))
    if method == "linear_shrinkage":
        return apply_shrinkage(
            probability,
            float(mapping["coefficient"]),
            float(mapping["base_rate"]),
        )
    return float(np.clip(probability, 0.0, 1.0))


def brier_decomposition(
    probabilities: Iterable[float],
    outcomes: Iterable[int],
    bins: int = 10,
    weights: Iterable[float] | None = None,
) -> BrierDecomposition:
    p = np.asarray(list(probabilities), dtype=float)
    y = np.asarray(list(outcomes), dtype=float)
    if len(p) == 0 or len(p) != len(y):
        raise ValueError("probabilities and outcomes must have equal non-zero length")
    w = np.ones(len(p)) if weights is None else np.asarray(list(weights), dtype=float)
    if len(w) != len(p) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("weights must be non-negative, positive-sum, and aligned")
    total_weight = float(w.sum())
    base = float(np.average(y, weights=w))
    reliability = 0.0
    resolution = 0.0
    rows: list[dict[str, float | int]] = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.minimum(np.digitize(p, edges[1:-1]), bins - 1)
    for index in range(bins):
        mask = assignments == index
        count = int(mask.sum())
        if count == 0:
            continue
        forecast_mean = float(np.average(p[mask], weights=w[mask]))
        observed_rate = float(np.average(y[mask], weights=w[mask]))
        share = float(w[mask].sum() / total_weight)
        reliability += share * (forecast_mean - observed_rate) ** 2
        resolution += share * (observed_rate - base) ** 2
        rows.append(
            {
                "bin_lower": float(edges[index]),
                "bin_upper": float(edges[index + 1]),
                "count": count,
                "weight": float(w[mask].sum()),
                "mean_forecast": forecast_mean,
                "observed_rate": observed_rate,
            }
        )
    return BrierDecomposition(
        brier=weighted_brier_score(p, y, w),
        reliability=float(reliability),
        resolution=float(resolution),
        uncertainty=float(base * (1.0 - base)),
        rows=rows,
    )
