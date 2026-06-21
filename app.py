from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from rbp_lab.config import CLAUDE_COLOR, MODEL_COLOR
from rbp_lab.db import initialize_database, list_matches, load_all_records
from rbp_lab.metrics import aggregate_metric, apply_sign_convention
from rbp_lab.official import official_match_scores
from rbp_lab.ui import (
    configure_page,
    page_header,
    render_reminder_banner,
    settings_panel,
    show_chart,
)


configure_page("Home")
settings = settings_panel()
initialize_database()

page_header(
    "Forecasting performance",
    "RBP Lab",
    "See where your simulator earns probability edge, where it gives ground back, "
    "and whether either forecaster is actually beating the crowd.",
)
render_reminder_banner(settings)

matches = list_matches()
official_scores = official_match_scores(matches, settings)
records = apply_sign_convention(load_all_records(), settings)
settled = records[records["outcome"].notna()]

columns = st.columns(4)
columns[0].metric("Matches logged", len(matches))
columns[1].metric("Settled questions", len(settled))
columns[2].metric(
    "Model RBP",
    (
        f"{official_scores['Model'].sum():+.2f}"
        if not official_scores.empty
        else f"{aggregate_metric(settled, 'rbp_model', settings.rbp_agg):+.3f}"
    ),
)
columns[3].metric(
    "Model vs Claude",
    (
        f"{(official_scores['Model'] - official_scores['Claude']).sum():+.2f}"
        if not official_scores.empty
        else f"{aggregate_metric(settled, 'model_vs_llm', settings.rbp_agg):+.3f}"
    ),
)

st.write("")
left, right = st.columns([1.45, 1])
with left:
    st.markdown("### Recent signal")
    if not official_scores.empty:
        labels = [
            f"{match_date.strftime('%b %d')}<br>{match_label}"
            for match_date, match_label in zip(
                official_scores["match_date"], official_scores["match_label"]
            )
        ]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=official_scores["Model"],
                mode="lines+markers",
                name="Model",
                line=dict(color=MODEL_COLOR, width=3),
                marker=dict(size=9),
                customdata=official_scores["match_date"].dt.strftime("%Y-%m-%d"),
                text=official_scores["match_label"],
                hovertemplate="<b>%{text}</b><br>%{customdata}<br>Official RBP: %{y:+.2f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=official_scores["Claude"],
                mode="lines+markers",
                name="Claude",
                line=dict(color=CLAUDE_COLOR, width=3),
                marker=dict(size=9),
                customdata=official_scores["match_date"].dt.strftime("%Y-%m-%d"),
                text=official_scores["match_label"],
                hovertemplate="<b>%{text}</b><br>%{customdata}<br>Official RBP: %{y:+.2f}<extra></extra>",
            )
        )
        fig.add_hline(y=0, line_dash="dot", line_color="#718096")
        fig.update_layout(title="Official RBP by match", yaxis_title="Official RBP")
        fig.update_xaxes(categoryorder="array", categoryarray=labels)
        show_chart(fig, "home-trend")
        st.caption("Official page-one RBP totals, ordered by settled match date.")
    else:
        st.info("Import settled result PDFs to populate official match RBP.")
with right:
    st.markdown("### Workflow")
    st.info(
        "**1. Reconcile** model, Claude, and settlement inputs in one grid.\n\n"
        "**2. Inspect** each match down to the question-level edge.\n\n"
        "**3. Compare** categories and trends on the dashboard.\n\n"
        "**4. Export** records and crowd-calibration evidence."
    )
    st.markdown("### Scoring lens")
    st.caption(
        "Positive RBP means a forecaster reduced weighted Brier loss versus the crowd "
        "under the default sign convention. Change that convention in the sidebar."
    )
