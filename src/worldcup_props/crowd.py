from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .db import connect, initialize


@dataclass(frozen=True)
class CrowdAnchorSpec:
    anchor: float
    low: float
    high: float
    mode: str
    weight: float = 0.0
    bias_edge: float = 0.0
    description: str = ""


DEFAULT_CROWD_ANCHORS: dict[str, CrowdAnchorSpec] = {
    "penalty_or_red": CrowdAnchorSpec(
        anchor=0.375,
        low=0.34,
        high=0.41,
        mode="fixed",
        weight=0.85,
        description="Penalty/red-card OR props are field-fixated and luck/referee dominated.",
    ),
    "penalty_awarded": CrowdAnchorSpec(
        anchor=0.32,
        low=0.24,
        high=0.40,
        mode="semi",
        weight=0.35,
        description="Penalty-only props are thin and referee dependent.",
    ),
    "btts_and_over_2_5": CrowdAnchorSpec(
        anchor=0.37,
        low=0.31,
        high=0.42,
        mode="fixed",
        weight=0.80,
        description="Compound goal props cluster tightly near the same crowd anchor.",
    ),
    "cards:threshold:4": CrowdAnchorSpec(
        anchor=0.50,
        low=0.45,
        high=0.55,
        mode="fixed",
        weight=0.70,
        description="Full-match 4+ cards is a fixated crowd bucket.",
    ),
    "cards_second_half_threshold_2": CrowdAnchorSpec(
        anchor=0.50,
        low=0.42,
        high=0.56,
        mode="semi",
        weight=0.45,
        bias_edge=-0.06,
        description="Second-half card thresholds have shown crowd yes-inflation.",
    ),
    "offsides:threshold:2": CrowdAnchorSpec(
        anchor=0.48,
        low=0.43,
        high=0.54,
        mode="fixed",
        weight=0.75,
        bias_edge=-0.055,
        description="Offside 2+ is tightly clustered, but the crowd has run high.",
    ),
    "halftime_tied": CrowdAnchorSpec(
        anchor=0.43,
        low=0.40,
        high=0.46,
        mode="fixed",
        weight=0.65,
        bias_edge=-0.02,
        description="Halftime-tied sample is small, so stay near the crowd band.",
    ),
    "fouls:more_than": CrowdAnchorSpec(
        anchor=0.50,
        low=0.38,
        high=0.62,
        mode="semi",
        weight=0.20,
        description="Foul comparisons are semi-anchored and referee/style dependent.",
    ),
    "player_goal_or_assist": CrowdAnchorSpec(
        anchor=0.33,
        low=0.20,
        high=0.48,
        mode="semi",
        weight=0.25,
        bias_edge=-0.04,
        description="Goal-or-assist props carry name-player crowd inflation.",
    ),
    "player_goals": CrowdAnchorSpec(
        anchor=0.30,
        low=0.08,
        high=0.45,
        mode="semi",
        weight=0.30,
        bias_edge=-0.06,
        description="Anytime goal props are name-driven unless a direct market exists.",
    ),
}

LOOSE_CROWD_KEYS = {
    "match_winner",
    "total_goals_2_or_fewer",
    "over_2_5_goals",
    "corners:more_than",
    "corners:halftime_more_than",
    "corners:second_half_more_than",
    "shots_on_target:more_than",
    "shots_on_target:second_half_more_than",
    "shots_on_target:threshold",
    "shots_on_target:threshold:2",
    "shots_on_target:threshold:3",
    "shots_on_target:threshold:4",
    "shots_on_target:threshold:5",
    "player_second_half_shots_on_target",
}


def apply_crowd_anchor(
    probability: float,
    crowd_key: str | None,
    settings: Any,
    *,
    database_path: str | Path | None = None,
) -> tuple[float, dict[str, Any]]:
    before = float(np.clip(probability, 0.0, 1.0))
    enabled = bool(getattr(settings, "use_crowd_anchoring", False))
    if not enabled or not crowd_key:
        return before, {
            "enabled": enabled,
            "key": crowd_key,
            "applied": False,
            "reason": "disabled" if not enabled else "missing_key",
            "before": before,
            "after": before,
        }
    if crowd_key in LOOSE_CROWD_KEYS:
        return before, {
            "enabled": True,
            "key": crowd_key,
            "applied": False,
            "reason": "loose_crowd_bucket",
            "before": before,
            "after": before,
        }
    spec = DEFAULT_CROWD_ANCHORS.get(crowd_key)
    if spec is None:
        return before, {
            "enabled": True,
            "key": crowd_key,
            "applied": False,
            "reason": "no_anchor_spec",
            "before": before,
            "after": before,
        }

    trend = crowd_trend_for_type(database_path, crowd_key, settings)
    drift = 0.0
    if trend["usable"]:
        drift = float(
            np.clip(
                trend["recent_minus_earlier"]
                * float(getattr(settings, "crowd_anchor_drift_weight", 0.5)),
                -float(getattr(settings, "crowd_anchor_max_drift", 0.04)),
                float(getattr(settings, "crowd_anchor_max_drift", 0.04)),
            )
        )
    target = float(np.clip(spec.anchor + spec.bias_edge + drift, spec.low, spec.high))
    if spec.mode == "fixed":
        banded = float(np.clip(before, spec.low, spec.high))
        after = target + (1.0 - spec.weight) * (banded - target)
    elif spec.mode == "semi":
        after = before + spec.weight * (target - before)
        after = float(np.clip(after, spec.low, spec.high))
    else:
        after = before
    after = float(np.clip(after, 0.0, 1.0))
    return after, {
        "enabled": True,
        "key": crowd_key,
        "applied": abs(after - before) > 1e-12,
        "mode": spec.mode,
        "anchor": spec.anchor,
        "bias_edge": spec.bias_edge,
        "drift_adjustment": drift,
        "target": target,
        "band": [spec.low, spec.high],
        "weight": spec.weight,
        "description": spec.description,
        "trend": trend,
        "before": before,
        "after": after,
    }


def crowd_trend_for_type(
    database_path: str | Path | None,
    question_type: str,
    settings: Any,
) -> dict[str, Any]:
    if database_path is None:
        return {"usable": False, "reason": "no_database", "rows": 0}
    initialize(database_path)
    with connect(database_path) as connection:
        rows = [
            dict(row)
            for row in connection.execute(
                """
                SELECT crowd_probability, outcome, observed_at
                FROM forecast_results
                WHERE question_type=? AND crowd_probability IS NOT NULL
                ORDER BY observed_at, id
                """,
                (question_type.casefold(),),
            )
        ]
    values = [float(row["crowd_probability"]) for row in rows]
    n = len(values)
    min_rows = int(getattr(settings, "crowd_anchor_min_drift_rows", 6))
    if n < min_rows:
        return {"usable": False, "reason": "insufficient_rows", "rows": n}
    window = max(1, int(getattr(settings, "crowd_anchor_recent_window", 20)))
    recent = values[-window:]
    earlier = values[:-window] or values[: max(1, n // 2)]
    outcomes = [int(row["outcome"]) for row in rows]
    x = np.arange(n, dtype=float)
    slope = 0.0
    if n > 1 and float(np.std(x)) > 0:
        slope = float(np.polyfit(x, np.asarray(values, dtype=float), 1)[0] * 10.0)
    return {
        "usable": True,
        "rows": n,
        "crowd_mean": float(np.mean(values)),
        "crowd_std": float(np.std(values, ddof=0)),
        "outcome_rate": float(np.mean(outcomes)),
        "bias": float(np.mean(np.asarray(values) - np.asarray(outcomes))),
        "recent_mean": float(np.mean(recent)),
        "earlier_mean": float(np.mean(earlier)),
        "recent_minus_earlier": float(np.mean(recent) - np.mean(earlier)),
        "slope_per_10_events": slope,
    }


def crowd_key_for_player_event(event: str) -> str:
    if event == "second_half_shots_on_target":
        return "player_second_half_shots_on_target"
    if event == "shots_on_target":
        return "player_shots_on_target"
    if event == "goal_or_assist":
        return "player_goal_or_assist"
    if event == "goals":
        return "player_goals"
    return f"player_{event}"
