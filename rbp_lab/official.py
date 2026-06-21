from __future__ import annotations

import pandas as pd

from .config import DashboardSettings


def official_match_scores(
    matches: pd.DataFrame,
    settings: DashboardSettings,
) -> pd.DataFrame:
    columns = ["id", "match_label", "match_date", "Model", "Claude"]
    if matches.empty:
        return pd.DataFrame(columns=columns)
    result = matches[
        matches["official_rbp_model"].notna()
        | matches["official_rbp_claude"].notna()
    ].copy()
    if result.empty:
        return pd.DataFrame(columns=columns)
    result = result.sort_values(["match_date", "id"], kind="stable")
    result["Model"] = result["official_rbp_model"] * settings.sign_multiplier
    result["Claude"] = result["official_rbp_claude"] * settings.sign_multiplier
    return result[columns]
