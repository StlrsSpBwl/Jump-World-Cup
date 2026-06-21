from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

from rbp_lab.db import initialize_database, list_matches, load_match, save_match
from rbp_lab.models import CATEGORY_VALUES, MatchRecord
from rbp_lab.parsing import parse_predictions_pdf, parse_settlement_csv, reconcile_sources
from rbp_lab.ui import configure_page, page_header, settings_panel


configure_page("Add / Edit Match")
settings = settings_panel()
initialize_database()

page_header(
    "Ingest and reconcile",
    "Add / Edit Match",
    "Bring three imperfect sources together, fix the joins once, and save a clean scored record.",
)

matches = list_matches()
match_options = {0: "Create a new match"} | {
    int(row.id): f"{row.match_date.date()} · {row.match_label}" for row in matches.itertuples()
}
selected_id = st.selectbox(
    "Mode",
    list(match_options),
    format_func=match_options.get,
    key="match_mode",
)

if st.button("Load selected match", disabled=selected_id == 0):
    match_data, records = load_match(selected_id)
    st.session_state["editor_match_id"] = selected_id
    st.session_state["match_label"] = match_data["match_label"]
    st.session_state["competition_stage"] = match_data["competition_stage"]
    st.session_state["match_date"] = date.fromisoformat(match_data["match_date"])
    st.session_state["match_notes"] = match_data["notes"]
    st.session_state["reconciliation_grid"] = records[
        ["question_text", "category", "p_model", "p_claude", "p_crowd", "outcome", "weight"]
    ]
    st.rerun()

loaded_id = st.session_state.get("editor_match_id")
editing_id = loaded_id if selected_id == loaded_id else None
if selected_id and selected_id != loaded_id:
    st.info("Load the selected match before editing it. Until then, Save creates a new match.")
if selected_id == 0 and st.button("Clear editor for a new match"):
    for key in (
        "editor_match_id",
        "match_label",
        "competition_stage",
        "match_date",
        "match_notes",
        "reconciliation_grid",
    ):
        st.session_state.pop(key, None)
    st.rerun()

meta_columns = st.columns([1.5, 1, 1])
with meta_columns[0]:
    match_label = st.text_input("Match label", key="match_label", placeholder="Canada vs Bosnia")
with meta_columns[1]:
    competition_stage = st.text_input("Competition stage", key="competition_stage", placeholder="Group stage")
with meta_columns[2]:
    match_date = st.date_input("Match date", key="match_date", value=st.session_state.get("match_date", date.today()))
notes = st.text_area("Notes", key="match_notes", height=80)

st.markdown("### Sources")
manual_only = st.toggle("Fully manual entry", help="Skip parsing and start directly in the grid.")
upload_columns = st.columns(3)
with upload_columns[0]:
    model_pdf = st.file_uploader("Model PDF", type=["pdf"])
with upload_columns[1]:
    claude_pdf = st.file_uploader("Claude PDF", type=["pdf"])
with upload_columns[2]:
    settlement_csv = st.file_uploader("Settlement or combined CSV", type=["csv"])

if st.button("Parse and build reconciliation grid", type="primary", disabled=manual_only):
    warnings: list[str] = []
    model_frame = None
    claude_frame = None
    settlement_frame = None
    if model_pdf:
        parsed = parse_predictions_pdf(model_pdf)
        model_frame, warnings = parsed.data, warnings + parsed.warnings
    if claude_pdf:
        parsed = parse_predictions_pdf(claude_pdf)
        claude_frame, warnings = parsed.data, warnings + parsed.warnings
    if settlement_csv:
        parsed = parse_settlement_csv(settlement_csv)
        settlement_frame, warnings = parsed.data, warnings + parsed.warnings
    grid, notices = reconcile_sources(model_frame, claude_frame, settlement_frame)
    st.session_state["reconciliation_grid"] = grid
    st.session_state["parse_messages"] = warnings + notices
    st.rerun()

for message in st.session_state.get("parse_messages", []):
    st.warning(message)

default_grid = pd.DataFrame(
    [
        {
            "question_text": "",
            "category": "other",
            "p_model": np.nan,
            "p_claude": np.nan,
            "p_crowd": np.nan,
            "outcome": np.nan,
            "weight": 1.0,
        }
    ]
)
grid = st.session_state.get("reconciliation_grid", default_grid)
visible_columns = ["question_text", "category", "p_model", "p_claude", "p_crowd", "outcome", "weight"]
for column in visible_columns:
    if column not in grid:
        grid[column] = np.nan

st.markdown("### Reconciliation grid")
st.caption("Use a blank outcome for void. All probabilities must be between 0 and 1.")
edited = st.data_editor(
    grid[visible_columns],
    num_rows="dynamic",
    width="stretch",
    hide_index=True,
    column_config={
        "question_text": st.column_config.TextColumn("Question", width="large", required=True),
        "category": st.column_config.SelectboxColumn("Category", options=CATEGORY_VALUES, required=True),
        "p_model": st.column_config.NumberColumn("Model", min_value=0.0, max_value=1.0, format="%.3f"),
        "p_claude": st.column_config.NumberColumn("Claude", min_value=0.0, max_value=1.0, format="%.3f"),
        "p_crowd": st.column_config.NumberColumn("Crowd", min_value=0.0, max_value=1.0, format="%.3f"),
        "outcome": st.column_config.NumberColumn("Outcome", min_value=0, max_value=1, step=1),
        "weight": st.column_config.NumberColumn("Weight", min_value=0.001, format="%.3f"),
    },
    key="question_editor",
)

if st.button("Save match", type="primary"):
    cleaned = edited.copy()
    cleaned["question_text"] = cleaned["question_text"].fillna("").astype(str).str.strip()
    cleaned = cleaned[cleaned["question_text"] != ""]
    required_probabilities = ["p_model", "p_claude", "p_crowd"]
    missing = [column for column in required_probabilities if cleaned[column].isna().any()]
    invalid_outcomes = cleaned["outcome"].dropna().map(float).isin([0.0, 1.0]).all()
    if not match_label.strip():
        st.error("Enter a match label.")
    elif cleaned.empty:
        st.error("Add at least one question.")
    elif missing:
        st.error(f"Complete all probability columns before saving: {', '.join(missing)}.")
    elif not invalid_outcomes:
        st.error("Outcome must be 0, 1, or blank for void.")
    else:
        cleaned["outcome"] = cleaned["outcome"].map(lambda value: None if pd.isna(value) else int(value))
        match_id = save_match(
            MatchRecord(
                id=editing_id,
                match_label=match_label.strip(),
                competition_stage=competition_stage.strip(),
                match_date=match_date,
                notes=notes.strip(),
            ),
            cleaned,
            settings,
        )
        st.session_state["editor_match_id"] = match_id
        st.session_state["reconciliation_grid"] = cleaned
        st.success(f"Saved {len(cleaned)} questions for match #{match_id}.")
