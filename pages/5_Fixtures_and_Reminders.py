from __future__ import annotations

import pandas as pd
import streamlit as st

from rbp_lab.config import utc_now
from rbp_lab.db import (
    initialize_database,
    list_fixtures,
    list_matches,
    save_fixture,
    update_fixture_status,
)
from rbp_lab.fixtures import (
    fetch_schedule,
    fixture_from_editor_row,
    fixture_to_local_text,
    parse_fixtures_csv,
)
from rbp_lab.models import (
    FixtureRecord,
    SUBMISSION_STATUS_VALUES,
    SubmissionStatus,
)
from rbp_lab.reminders import to_ics
from rbp_lab.ui import (
    configure_page,
    page_header,
    render_reminder_banner,
    safe_filename,
    settings_panel,
)


configure_page("Fixtures and Reminders")
settings = settings_panel()
initialize_database()

page_header(
    "Submission safety net",
    "Fixtures & Reminders",
    "Put every kickoff on the clock, export native calendar alarms, and track whether the prediction was actually submitted.",
)
render_reminder_banner(settings)

fixtures = list_fixtures()
fixtures_by_id = {fixture.id: fixture for fixture in fixtures}
matches = list_matches()
match_options = {None: "Not linked"} | {
    int(row.id): f"{row.match_date.date()} · {row.match_label}"
    for row in matches.itertuples()
}
match_label_to_id = {label: match_id for match_id, label in match_options.items()}

st.markdown("### Calendar")
pending_upcoming = [
    fixture
    for fixture in fixtures
    if fixture.kickoff_utc > utc_now()
    and fixture.submission_status == SubmissionStatus.PENDING
]
st.download_button(
    "Download .ics for all upcoming fixtures",
    to_ics(pending_upcoming, settings.reminder_lead_minutes, settings.local_tz),
    file_name="rbp-lab-upcoming-fixtures.ics",
    mime="text/calendar",
    disabled=not pending_upcoming,
)
st.caption(
    f"Each event contains a native calendar alarm {settings.reminder_lead_minutes} "
    "minutes before kickoff. Import this into Apple Calendar, Google Calendar, or Outlook."
)

st.markdown("### Status board")
if fixtures:
    overview = pd.DataFrame(
        [
            {
                "Match": fixture.match_label,
                "Kickoff": fixture_to_local_text(fixture, settings.local_tz),
                "Status": fixture.submission_status.value,
                "Stage": fixture.competition_stage,
            }
            for fixture in fixtures
        ]
    )

    def color_status(row: pd.Series) -> list[str]:
        colors = {
            "pending": "background-color: rgba(234,179,8,.18)",
            "submitted": "background-color: rgba(22,199,154,.18)",
            "skipped": "background-color: rgba(113,128,150,.18)",
            "missed": "background-color: rgba(240,106,123,.22)",
        }
        return [colors.get(row["Status"], "")] * len(row)

    st.dataframe(
        overview.style.apply(color_status, axis=1),
        hide_index=True,
        width="stretch",
    )

st.markdown("### Fixture actions")
for fixture in fixtures:
    with st.expander(
        f"{fixture.match_label} · {fixture_to_local_text(fixture, settings.local_tz)} "
        f"· {fixture.submission_status.value.title()}"
    ):
        action_columns = st.columns([1, 1, 1.2, 2])
        if action_columns[0].button(
            "Mark submitted",
            key=f"submitted_{fixture.id}",
            disabled=fixture.submission_status == SubmissionStatus.SUBMITTED,
        ):
            update_fixture_status(fixture.id, SubmissionStatus.SUBMITTED)
            st.rerun()
        if action_columns[1].button(
            "Mark skipped",
            key=f"skipped_{fixture.id}",
            disabled=fixture.submission_status == SubmissionStatus.SKIPPED,
        ):
            update_fixture_status(fixture.id, SubmissionStatus.SKIPPED)
            st.rerun()
        action_columns[2].download_button(
            "Download .ics",
            to_ics([fixture], settings.reminder_lead_minutes, settings.local_tz),
            file_name=f"{safe_filename(fixture.match_label)}.ics",
            mime="text/calendar",
            key=f"ics_{fixture.id}",
        )
        action_columns[3].caption(
            f"Linked match: {match_options.get(fixture.linked_match_id, 'Not linked')}"
        )

st.markdown("### Add or edit fixtures")
editor_rows = [
    {
        "id": fixture.id,
        "match_label": fixture.match_label,
        "competition_stage": fixture.competition_stage,
        "kickoff_local": fixture_to_local_text(fixture, settings.local_tz),
        "submission_status": fixture.submission_status.value,
        "linked_match": match_options.get(fixture.linked_match_id, "Not linked"),
        "notes": fixture.notes,
    }
    for fixture in fixtures
]
if not editor_rows:
    editor_rows = [
        {
            "id": None,
            "match_label": "",
            "competition_stage": "",
            "kickoff_local": "",
            "submission_status": "pending",
            "linked_match": "Not linked",
            "notes": "",
        }
    ]
editor = st.data_editor(
    pd.DataFrame(editor_rows),
    num_rows="dynamic",
    hide_index=True,
    width="stretch",
    column_config={
        "id": st.column_config.NumberColumn("ID", disabled=True),
        "match_label": st.column_config.TextColumn("Match", required=True, width="large"),
        "competition_stage": st.column_config.TextColumn("Stage"),
        "kickoff_local": st.column_config.TextColumn(
            f"Kickoff ({settings.local_tz})",
            help="Use YYYY-MM-DD HH:MM. This is converted to UTC on save.",
            required=True,
        ),
        "submission_status": st.column_config.SelectboxColumn(
            "Status", options=SUBMISSION_STATUS_VALUES, required=True
        ),
        "linked_match": st.column_config.SelectboxColumn(
            "Linked match",
            options=list(match_label_to_id),
        ),
        "notes": st.column_config.TextColumn("Notes", width="large"),
    },
    key="fixtures_editor",
)
if st.button("Save fixture grid", type="primary"):
    errors: list[str] = []
    saved = 0
    for index, row in editor.iterrows():
        if not str(row.get("match_label", "")).strip():
            continue
        try:
            normalized_row = row.copy()
            normalized_row["linked_match_id"] = match_label_to_id.get(
                row.get("linked_match")
            )
            fixture = fixture_from_editor_row(normalized_row, settings.local_tz)
            existing = fixtures_by_id.get(fixture.id)
            if existing:
                fixture = fixture.model_copy(
                    update={
                        "submitted_at": existing.submitted_at,
                        "reminded_at": existing.reminded_at,
                        "created_at": existing.created_at,
                    }
                )
            save_fixture(fixture)
            saved += 1
        except Exception as exc:
            errors.append(f"Row {index + 1}: {exc}")
    if errors:
        for error in errors:
            st.error(error)
    else:
        st.success(f"Saved {saved} fixtures.")
        st.rerun()

st.markdown("### CSV import")
fixture_csv = st.file_uploader(
    "Upload fixtures CSV",
    type=["csv"],
    help="Required: match_label, kickoff. Optional: tz, competition_stage, submission_status, notes.",
)
if fixture_csv:
    preview, warnings = parse_fixtures_csv(fixture_csv, settings.local_tz)
    for warning in warnings:
        st.warning(warning)
    if not preview.empty:
        display_preview = preview.copy()
        display_preview["kickoff_utc"] = display_preview["kickoff_utc"].astype(str)
        st.dataframe(display_preview, hide_index=True, width="stretch")
        if st.button("Import previewed fixtures"):
            for row in preview.to_dict("records"):
                save_fixture(FixtureRecord(**row))
            st.success(f"Imported or updated {len(preview)} fixtures.")
            st.rerun()

st.markdown("### Tournament schedule")
st.button(
    "Import tournament schedule",
    disabled=True,
    help="Schedule provider not wired yet. Manual entry and CSV import are available.",
)
st.caption("Schedule-fetch stub is present; an official source/API still needs to be selected.")
