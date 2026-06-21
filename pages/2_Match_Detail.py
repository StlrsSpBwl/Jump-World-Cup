from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from rbp_lab.config import CLAUDE_COLOR, MODEL_COLOR
from rbp_lab.db import fixtures_for_match, initialize_database, list_matches, load_match
from rbp_lab.metrics import apply_sign_convention, match_summary
from rbp_lab.ui import (
    category_label,
    configure_page,
    format_percent,
    page_header,
    safe_filename,
    settings_panel,
    show_chart,
)


configure_page("Match Detail")
settings = settings_panel()
initialize_database()

page_header(
    "Question-level attribution",
    "Match Detail",
    "Open one fixture and trace the score back to the exact forecasts that moved it.",
)

matches = list_matches()
match_map = {
    int(row.id): f"{row.match_date.date()} · {row.match_label}" for row in matches.itertuples()
}
match_id = st.selectbox("Match", list(match_map), format_func=match_map.get)
match, records = load_match(match_id)
records = apply_sign_convention(records, settings)
summary = match_summary(records, settings)
official_model = match.get("official_rbp_model")
official_claude = match.get("official_rbp_claude")

st.caption(f"{match['competition_stage']} · {match['match_date']}")
linked_fixtures = fixtures_for_match(match_id)
if linked_fixtures:
    labels = ", ".join(
        f"{fixture.match_label} ({fixture.submission_status.value})"
        for fixture in linked_fixtures
    )
    st.info(f"Linked fixture: {labels}")
metrics = st.columns(6)
metrics[0].metric(
    "Model RBP",
    f"{official_model * settings.sign_multiplier:+.2f}"
    if official_model is not None
    else f"{summary['rbp_model']:+.3f}",
)
metrics[1].metric(
    "Claude RBP",
    f"{official_claude * settings.sign_multiplier:+.2f}"
    if official_claude is not None
    else f"{summary['rbp_claude']:+.3f}",
)
metrics[2].metric(
    "Model vs Claude",
    f"{(official_model - official_claude) * settings.sign_multiplier:+.2f}"
    if official_model is not None and official_claude is not None
    else f"{summary['model_vs_llm']:+.3f}",
)
metrics[3].metric("Settled", summary["questions"])
metrics[4].metric("Model > crowd", format_percent(summary["model_win_rate_crowd"]))
metrics[5].metric("Model > Claude", format_percent(summary["model_win_rate_claude"]))

st.write("")
predictor = st.radio("Question edge", ["Model", "Claude"], horizontal=True)
column = "rbp_model" if predictor == "Model" else "rbp_claude"
color = MODEL_COLOR if predictor == "Model" else CLAUDE_COLOR
chart_data = records[records["outcome"].notna()].sort_values(column)
fig = go.Figure(
    go.Bar(
        x=chart_data[column],
        y=chart_data["question_text"],
        orientation="h",
        marker_color=[
            color if value >= 0 else "#f06a7b" for value in chart_data[column]
        ],
        customdata=chart_data[["category", "p_crowd", "outcome"]].to_numpy(),
        hovertemplate=(
            "<b>%{y}</b><br>RBP: %{x:.4f}<br>Category: %{customdata[0]}"
            "<br>Crowd: %{customdata[1]:.1%}<br>Outcome: %{customdata[2]}<extra></extra>"
        ),
    )
)
fig.add_vline(x=0, line_dash="dot", line_color="#718096")
fig.update_layout(title=f"{predictor} edge vs crowd by question", xaxis_title="Weighted RBP")
fig.update_yaxes(tickfont=dict(size=10))
show_chart(fig, f"match-{match_id}-{predictor.casefold()}")

display = records.copy()
display["category"] = display["category"].map(category_label)
st.dataframe(
    display[
        [
            "question_text",
            "category",
            "p_model",
            "p_claude",
            "p_crowd",
            "outcome",
            "weight",
            "brier_model",
            "brier_claude",
            "rbp_model",
            "rbp_claude",
            "model_vs_llm",
        ]
    ],
    hide_index=True,
    width="stretch",
    column_config={
        "p_model": st.column_config.NumberColumn(format="%.3f"),
        "p_claude": st.column_config.NumberColumn(format="%.3f"),
        "p_crowd": st.column_config.NumberColumn(format="%.3f"),
        "rbp_model": st.column_config.NumberColumn(format="%+.4f"),
        "rbp_claude": st.column_config.NumberColumn(format="%+.4f"),
        "model_vs_llm": st.column_config.NumberColumn(format="%+.4f"),
    },
)
st.download_button(
    "Download match records CSV",
    records.to_csv(index=False).encode(),
    file_name=f"{safe_filename(match['match_label'])}-rbp.csv",
    mime="text/csv",
)
