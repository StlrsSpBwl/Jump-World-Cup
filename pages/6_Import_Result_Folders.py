from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from rbp_lab.classification import classify_question
from rbp_lab.config import PREDICTION_RESULTS_DIR
from rbp_lab.db import initialize_database, list_matches, save_match
from rbp_lab.models import MatchRecord
from rbp_lab.result_folders import (
    combine_result_reports,
    discover_result_pairs,
    parse_sports_predict_results_pdf,
)
from rbp_lab.ui import configure_page, page_header, settings_panel


configure_page("Import Result Folders")
settings = settings_panel()
initialize_database()

page_header(
    "Direct PDF ingestion",
    "Import Result Folders",
    "Scan the Model and Claude folders, pair settled Sports Predict reports, "
    "and send complete question cards directly to the performance dashboard.",
)

folder = Path(
    st.text_input(
        "Prediction results folder",
        value=str(PREDICTION_RESULTS_DIR),
        help="This folder must contain Model and Claude subfolders.",
    )
).expanduser()

pairs, discovery_warnings = discover_result_pairs(folder)
for warning in discovery_warnings:
    st.warning(warning)

if not pairs:
    st.error("No Model/Claude PDF pairs were found.")
    st.stop()

pair_by_label = {pair.label: pair for pair in pairs}
selected_labels = st.multiselect(
    "Result pairs",
    list(pair_by_label),
    default=list(pair_by_label),
)
st.caption(
    f"Found {len(pairs)} paired fixtures. macOS Vision OCR runs locally and is cached, "
    "so the PDFs are not uploaded anywhere."
)

if st.button("Parse selected result PDFs", type="primary", disabled=not selected_labels):
    parsed_results = []
    progress = st.progress(0, text="Preparing result reports...")
    for index, label in enumerate(selected_labels, start=1):
        pair = pair_by_label[label]
        progress.progress(
            (index - 1) / len(selected_labels),
            text=f"OCR and reconcile: {pair.key}",
        )
        try:
            model_report = parse_sports_predict_results_pdf(pair.model_path)
            claude_report = parse_sports_predict_results_pdf(pair.claude_path)
            records, warnings = combine_result_reports(model_report, claude_report)
            if not records.empty:
                records["category"] = records["question_text"].map(classify_question)
            parsed_results.append(
                {
                    "key": pair.key,
                    "match_label": model_report.match_label,
                    "match_date": model_report.match_date,
                    "official_rbp_model": model_report.match_rbp,
                    "official_rbp_claude": claude_report.match_rbp,
                    "model_path": str(pair.model_path),
                    "claude_path": str(pair.claude_path),
                    "records": records,
                    "warnings": warnings,
                }
            )
        except Exception as exc:
            parsed_results.append(
                {
                    "key": pair.key,
                    "match_label": pair.key.replace("_", " vs "),
                    "match_date": None,
                    "official_rbp_model": None,
                    "official_rbp_claude": None,
                    "model_path": str(pair.model_path),
                    "claude_path": str(pair.claude_path),
                    "records": pd.DataFrame(),
                    "warnings": [f"Import failed: {exc}"],
                }
            )
    progress.progress(1.0, text="Result reports parsed.")
    st.session_state["folder_import_results"] = parsed_results

parsed_results = st.session_state.get("folder_import_results", [])
if parsed_results:
    st.markdown("### Import preview")
    overview = pd.DataFrame(
        [
            {
                "Match": item["match_label"],
                "Date": item["match_date"],
                "Complete questions": len(item["records"]),
                "Model official RBP": item["official_rbp_model"],
                "Claude official RBP": item["official_rbp_claude"],
                "Warnings": len(item["warnings"]),
            }
            for item in parsed_results
        ]
    )
    st.dataframe(overview, hide_index=True, width="stretch")

    for item in parsed_results:
        with st.expander(
            f"{item['match_label']} · {len(item['records'])} complete questions"
        ):
            for warning in item["warnings"]:
                st.warning(warning)
            if not item["records"].empty:
                st.dataframe(
                    item["records"][
                        [
                            "question_text",
                            "category",
                            "p_model",
                            "p_claude",
                            "p_crowd",
                            "outcome",
                            "weight",
                        ]
                    ],
                    hide_index=True,
                    width="stretch",
                )

    if st.button("Import preview into RBP Lab", type="primary"):
        existing_matches = list_matches()
        imported = 0
        skipped = 0
        for item in parsed_results:
            records = item["records"]
            if records.empty:
                skipped += 1
                continue
            match_date = item["match_date"] or date.fromtimestamp(
                Path(item["model_path"]).stat().st_mtime
            )
            existing_id = None
            if not existing_matches.empty:
                matching = existing_matches[
                    existing_matches["match_label"].str.casefold()
                    == str(item["match_label"]).casefold()
                ]
                same_date = matching[
                    matching["match_date"].dt.date == match_date
                ]
                if not same_date.empty:
                    existing_id = int(same_date.iloc[0]["id"])
            save_match(
                MatchRecord(
                    id=existing_id,
                    match_label=item["match_label"],
                    competition_stage="Imported settled report",
                    match_date=match_date,
                    official_rbp_model=item["official_rbp_model"],
                    official_rbp_claude=item["official_rbp_claude"],
                    notes=(
                        "Direct folder import.\n"
                        f"Model: {item['model_path']}\nClaude: {item['claude_path']}"
                    ),
                ),
                records,
                settings,
            )
            imported += 1
        st.success(
            f"Imported {imported} matches into the dashboard"
            + (f"; skipped {skipped} empty reports." if skipped else ".")
        )
