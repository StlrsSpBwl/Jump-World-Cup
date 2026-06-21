from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import DashboardSettings


METRIC_COLUMNS = [
    "brier_model",
    "brier_claude",
    "brier_crowd",
    "rbp_model",
    "rbp_claude",
    "model_vs_llm",
]


def brier(probability: float, outcome: int) -> float:
    return (float(probability) - int(outcome)) ** 2


def compute_question_metrics(
    frame: pd.DataFrame, settings: DashboardSettings | None = None
) -> pd.DataFrame:
    settings = settings or DashboardSettings()
    result = frame.copy()
    for column in METRIC_COLUMNS:
        result[column] = np.nan

    settled = result["outcome"].notna()
    if not settled.any():
        return result

    y = pd.to_numeric(result.loc[settled, "outcome"]).astype(float)
    weight = pd.to_numeric(result.loc[settled, "weight"]).astype(float)
    model = pd.to_numeric(result.loc[settled, "p_model"]).astype(float)
    claude = pd.to_numeric(result.loc[settled, "p_claude"]).astype(float)
    crowd = pd.to_numeric(result.loc[settled, "p_crowd"]).astype(float)

    model_brier = (model - y) ** 2
    claude_brier = (claude - y) ** 2
    crowd_brier = (crowd - y) ** 2
    sign = settings.sign_multiplier

    result.loc[settled, "brier_model"] = model_brier
    result.loc[settled, "brier_claude"] = claude_brier
    result.loc[settled, "brier_crowd"] = crowd_brier
    result.loc[settled, "rbp_model"] = sign * weight * (crowd_brier - model_brier)
    result.loc[settled, "rbp_claude"] = sign * weight * (crowd_brier - claude_brier)
    result.loc[settled, "model_vs_llm"] = sign * weight * (claude_brier - model_brier)
    return result


def aggregate_metric(
    frame: pd.DataFrame,
    column: str,
    aggregation: str = "weighted_mean",
) -> float:
    settled = frame[column].notna()
    if not settled.any():
        return 0.0
    total = float(frame.loc[settled, column].sum())
    if aggregation == "weighted_sum":
        return total
    if aggregation != "weighted_mean":
        raise ValueError(f"Unknown aggregation: {aggregation}")
    weight_sum = float(frame.loc[settled, "weight"].sum())
    return total / weight_sum if weight_sum else 0.0


def apply_sign_convention(
    frame: pd.DataFrame, settings: DashboardSettings
) -> pd.DataFrame:
    result = frame.copy()
    if settings.sign_multiplier < 0:
        for column in ("rbp_model", "rbp_claude", "model_vs_llm"):
            if column in result:
                result[column] = -result[column]
    return result


def official_match_scores(
    matches: pd.DataFrame,
    settings: DashboardSettings,
) -> pd.DataFrame:
    if matches.empty:
        return pd.DataFrame(
            columns=["id", "match_label", "match_date", "Model", "Claude"]
        )
    result = matches[
        matches["official_rbp_model"].notna()
        | matches["official_rbp_claude"].notna()
    ].copy()
    if result.empty:
        return pd.DataFrame(
            columns=["id", "match_label", "match_date", "Model", "Claude"]
        )
    result = result.sort_values(["match_date", "id"], kind="stable")
    result["Model"] = result["official_rbp_model"] * settings.sign_multiplier
    result["Claude"] = result["official_rbp_claude"] * settings.sign_multiplier
    return result[["id", "match_label", "match_date", "Model", "Claude"]]


def match_summary(
    frame: pd.DataFrame, settings: DashboardSettings | None = None
) -> dict[str, Any]:
    settings = settings or DashboardSettings()
    settled = frame[frame["outcome"].notna()].copy()
    if settled.empty:
        return {
            "rbp_model": 0.0,
            "rbp_claude": 0.0,
            "model_vs_llm": 0.0,
            "questions": 0,
            "model_win_rate_crowd": 0.0,
            "claude_win_rate_crowd": 0.0,
            "model_win_rate_claude": 0.0,
        }
    return {
        "rbp_model": aggregate_metric(settled, "rbp_model", settings.rbp_agg),
        "rbp_claude": aggregate_metric(settled, "rbp_claude", settings.rbp_agg),
        "model_vs_llm": aggregate_metric(settled, "model_vs_llm", settings.rbp_agg),
        "questions": len(settled),
        "model_win_rate_crowd": float((settled["brier_model"] < settled["brier_crowd"]).mean()),
        "claude_win_rate_crowd": float((settled["brier_claude"] < settled["brier_crowd"]).mean()),
        "model_win_rate_claude": float((settled["brier_model"] < settled["brier_claude"]).mean()),
    }
