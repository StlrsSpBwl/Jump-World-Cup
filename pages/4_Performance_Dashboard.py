from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from rbp_lab.config import CLAUDE_COLOR, MODEL_COLOR
from rbp_lab.db import initialize_database, list_matches, load_all_records
from rbp_lab.metrics import (
    aggregate_metric,
    apply_sign_convention,
)
from rbp_lab.official import official_match_scores
from rbp_lab.models import CATEGORY_VALUES
from rbp_lab.ui import (
    category_label,
    configure_page,
    page_header,
    settings_panel,
    show_chart,
    signed_colors,
)


configure_page("Performance Dashboard")
settings = settings_panel()
initialize_database()

page_header(
    "Performance attribution",
    "Where are you gaining ground?",
    "Filter the evidence, compare both forecasters with the crowd, then isolate the categories where the simulator’s structure pays.",
)

records = apply_sign_convention(load_all_records(), settings)
matches = list_matches()
records["category_label"] = records["category"].map(category_label)
match_order = records[["match_id", "match_label", "match_date"]].drop_duplicates().sort_values("match_date")

with st.expander("Global filters", expanded=True):
    filter_columns = st.columns([1.3, 1.3, 2])
    labels = match_order["match_label"].tolist()
    with filter_columns[0]:
        selected_matches = st.multiselect("Matches", labels, default=labels)
    min_date = records["match_date"].min().date()
    max_date = records["match_date"].max().date()
    with filter_columns[1]:
        date_range = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    with filter_columns[2]:
        selected_categories = st.multiselect(
            "Categories",
            CATEGORY_VALUES,
            default=CATEGORY_VALUES,
            format_func=category_label,
        )

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = end_date = date_range
filtered = records[
    records["match_label"].isin(selected_matches)
    & records["category"].isin(selected_categories)
    & records["match_date"].dt.date.between(start_date, end_date)
].copy()
settled = filtered[filtered["outcome"].notna()].copy()

if settled.empty:
    st.warning("No settled questions match these filters.")
    st.stop()

headline = st.columns(4)
official_filtered = official_match_scores(
    matches[
        matches["match_label"].isin(selected_matches)
        & matches["match_date"].dt.date.between(start_date, end_date)
    ],
    settings,
)
headline[0].metric("Questions", len(settled))
headline[1].metric(
    "Model RBP",
    (
        f"{official_filtered['Model'].sum():+.2f}"
        if not official_filtered.empty
        else f"{aggregate_metric(settled, 'rbp_model', settings.rbp_agg):+.3f}"
    ),
)
headline[2].metric(
    "Claude RBP",
    (
        f"{official_filtered['Claude'].sum():+.2f}"
        if not official_filtered.empty
        else f"{aggregate_metric(settled, 'rbp_claude', settings.rbp_agg):+.3f}"
    ),
)
headline[3].metric(
    "Model vs Claude",
    (
        f"{(official_filtered['Model'] - official_filtered['Claude']).sum():+.2f}"
        if not official_filtered.empty
        else f"{aggregate_metric(settled, 'model_vs_llm', settings.rbp_agg):+.3f}"
    ),
)

st.markdown("### Trend")
trend_mode = st.radio("Trend mode", ["Per match", "Cumulative"], horizontal=True)
trend = official_filtered.rename(columns={"id": "match_id"}).copy()
if trend.empty:
    st.info("No official page-one RBP totals match these filters.")
else:
    if trend_mode == "Cumulative":
        trend[["Model", "Claude"]] = trend[["Model", "Claude"]].cumsum()
    trend["plot_label"] = [
        f"{match_date.strftime('%b %d')}<br>{match_label}"
        for match_date, match_label in zip(
            trend["match_date"], trend["match_label"]
        )
    ]
    fig = go.Figure()
    for predictor, color in (("Model", MODEL_COLOR), ("Claude", CLAUDE_COLOR)):
        fig.add_trace(
            go.Scatter(
                x=trend["plot_label"],
                y=trend[predictor],
                mode="lines+markers",
                name=predictor,
                line=dict(color=color, width=3),
                marker=dict(size=9),
                customdata=trend["match_date"].dt.strftime("%Y-%m-%d"),
                text=trend["match_label"],
                hovertemplate="<b>%{text}</b><br>%{customdata}<br>Official RBP: %{y:+.2f}<extra></extra>",
            )
        )
    fig.add_hline(y=0, line_dash="dot", line_color="#718096")
    fig.update_layout(
        title=f"{trend_mode} official RBP across matches",
        yaxis_title="Official RBP",
    )
    fig.update_xaxes(
        categoryorder="array",
        categoryarray=trend["plot_label"].tolist(),
    )
    show_chart(fig, f"trend-{trend_mode.casefold().replace(' ', '-')}")
    st.caption(
        "Uses the official page-one RBP from each PDF and sorts by settled match date. "
        "Category filters apply to the diagnostic charts below, not the official match total."
    )


def category_vs_crowd(data: pd.DataFrame, predictor: str, mode: str) -> pd.DataFrame:
    brier_column = "brier_model" if predictor == "Model" else "brier_claude"
    rbp_column = "rbp_model" if predictor == "Model" else "rbp_claude"
    rows = []
    for category, group in data.groupby("category_label"):
        wins = int((group[brier_column] < group["brier_crowd"]).sum())
        losses = int((group[brier_column] > group["brier_crowd"]).sum())
        if mode == "Net RBP":
            value = aggregate_metric(group, rbp_column, settings.rbp_agg)
        elif mode == "Win rate":
            value = wins / len(group)
        else:
            value = wins - losses
        rows.append({"category": category, "value": value, "wins": wins, "losses": losses})
    return pd.DataFrame(rows).sort_values("value")


st.markdown("### Category performance vs crowd")
category_mode = st.radio("Category metric", ["Net RBP", "Win rate", "Count beaten / lost"], horizontal=True)
chart_columns = st.columns(2)
for chart_column, predictor, color in zip(chart_columns, ("Model", "Claude"), (MODEL_COLOR, CLAUDE_COLOR)):
    data = category_vs_crowd(settled, predictor, category_mode)
    with chart_column:
        fig = go.Figure(
            go.Bar(
                x=data["value"],
                y=data["category"],
                orientation="h",
                marker_color=signed_colors(data["value"]) if category_mode != "Win rate" else color,
                customdata=data[["wins", "losses"]].to_numpy(),
                hovertemplate=(
                    "<b>%{y}</b><br>Value: %{x:.3f}<br>Beaten: %{customdata[0]}"
                    "<br>Lost: %{customdata[1]}<extra></extra>"
                ),
            )
        )
        reference = 0.5 if category_mode == "Win rate" else 0
        fig.add_vline(x=reference, line_dash="dot", line_color="#718096")
        fig.update_layout(title=f"{predictor} vs crowd · {category_mode}", xaxis_title=category_mode)
        show_chart(fig, f"{predictor.casefold()}-category-{category_mode.casefold().replace(' ', '-')}")

st.markdown("### Model vs Claude")
comparison_mode = st.radio("Comparison view", ["Category edge", "Question scatter"], horizontal=True)
if comparison_mode == "Category edge":
    rows = []
    for category, group in settled.groupby("category_label"):
        rows.append(
            {
                "category": category,
                "value": aggregate_metric(group, "model_vs_llm", settings.rbp_agg),
                "questions": len(group),
            }
        )
    comparison = pd.DataFrame(rows).sort_values("value")
    fig = go.Figure(
        go.Bar(
            x=comparison["value"],
            y=comparison["category"],
            orientation="h",
            marker_color=signed_colors(comparison["value"]),
            customdata=comparison["questions"],
            hovertemplate="<b>%{y}</b><br>Model vs Claude: %{x:+.4f}<br>Questions: %{customdata}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_dash="dot", line_color="#718096")
    fig.update_layout(
        title="Where the model gains / loses ground vs Claude",
        xaxis_title="Model vs Claude weighted edge",
    )
    show_chart(fig, "model-vs-claude-category")
else:
    fig = px.scatter(
        settled,
        x="brier_model",
        y="brier_claude",
        color="category_label",
        hover_name="question_text",
        hover_data={"match_label": True, "category_label": False},
        labels={"brier_model": "Model Brier", "brier_claude": "Claude Brier"},
        title="Question-level Brier: points above the line favor the model",
    )
    maximum = float(max(settled["brier_model"].max(), settled["brier_claude"].max(), 0.1))
    fig.add_trace(
        go.Scatter(
            x=[0, maximum],
            y=[0, maximum],
            mode="lines",
            line=dict(color="#94a3b8", dash="dot"),
            name="Parity",
            hoverinfo="skip",
        )
    )
    fig.update_traces(marker=dict(size=10, opacity=0.82), selector=dict(mode="markers"))
    show_chart(fig, "model-vs-claude-scatter")

st.markdown("### Exports")
export_columns = st.columns(2)
with export_columns[0]:
    st.download_button(
        "Download filtered question records",
        filtered.to_csv(index=False).encode(),
        file_name="rbp-question-records.csv",
        mime="text/csv",
    )
with export_columns[1]:
    calibration = settled.copy()
    calibration["probability_bin"] = pd.cut(
        calibration["p_crowd"],
        bins=np.linspace(0, 1, 11),
        include_lowest=True,
    ).astype(str)
    calibration_export = (
        calibration.groupby(["category", "probability_bin"], observed=True)
        .agg(
            questions=("outcome", "size"),
            mean_crowd_probability=("p_crowd", "mean"),
            realized_rate=("outcome", "mean"),
            weighted_realized_rate=(
                "outcome",
                lambda values: np.average(
                    values,
                    weights=calibration.loc[values.index, "weight"],
                ),
            ),
        )
        .reset_index()
    )
    calibration_export["crowd_bias"] = (
        calibration_export["mean_crowd_probability"] - calibration_export["realized_rate"]
    )
    st.download_button(
        "Download crowd calibration",
        calibration_export.to_csv(index=False).encode(),
        file_name="rbp-crowd-calibration.csv",
        mime="text/csv",
    )
