from __future__ import annotations

from datetime import datetime
from io import BytesIO
from typing import BinaryIO

import pandas as pd
from pydantic import ValidationError

from .config import get_timezone
from .models import FixtureRecord, SubmissionStatus


def local_datetime_to_utc(value: object, local_tz: str) -> datetime:
    parsed = pd.to_datetime(value, errors="raise").to_pydatetime()
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=get_timezone(local_tz))
    return parsed.astimezone(get_timezone("UTC"))


def fixture_to_local_text(fixture: FixtureRecord, local_tz: str) -> str:
    return fixture.kickoff_utc.astimezone(get_timezone(local_tz)).strftime(
        "%Y-%m-%d %H:%M"
    )


def parse_fixtures_csv(
    file: BinaryIO | bytes, default_tz: str
) -> tuple[pd.DataFrame, list[str]]:
    content = file if isinstance(file, bytes) else file.getvalue()
    warnings: list[str] = []
    try:
        frame = pd.read_csv(BytesIO(content))
    except Exception as exc:
        return pd.DataFrame(), [f"Could not read fixtures CSV: {exc}"]

    rename = {
        str(column).strip().casefold().replace(" ", "_"): column
        for column in frame.columns
    }
    label_column = rename.get("match_label") or rename.get("match")
    kickoff_column = rename.get("kickoff") or rename.get("kickoff_utc")
    if label_column is None or kickoff_column is None:
        return pd.DataFrame(), ["CSV requires match_label and kickoff columns."]

    rows: list[dict[str, object]] = []

    def optional_text(value: object) -> str:
        return "" if value is None or pd.isna(value) else str(value).strip()

    for index, row in frame.iterrows():
        label = optional_text(row.get(label_column))
        if not label:
            warnings.append(f"Row {index + 2}: blank match label.")
            continue
        timezone_name = optional_text(row.get(rename.get("tz"))) or default_tz
        try:
            kickoff_utc = local_datetime_to_utc(row[kickoff_column], timezone_name)
        except Exception as exc:
            warnings.append(f"Row {index + 2}: invalid kickoff ({exc}).")
            continue
        status_raw = str(
            row.get(rename.get("submission_status"), SubmissionStatus.PENDING.value)
        ).strip().casefold()
        try:
            status = SubmissionStatus(status_raw)
        except ValueError:
            status = SubmissionStatus.PENDING
            warnings.append(f"Row {index + 2}: unknown status; using pending.")
        rows.append(
            {
                "match_label": label,
                "competition_stage": optional_text(
                    row.get(rename.get("competition_stage"))
                ),
                "kickoff_utc": kickoff_utc,
                "submission_status": status.value,
                "notes": optional_text(row.get(rename.get("notes"))),
            }
        )
    return pd.DataFrame(rows), warnings


def fixture_from_editor_row(row: pd.Series, local_tz: str) -> FixtureRecord:
    linked_match = row.get("linked_match_id")
    if pd.isna(linked_match) or str(linked_match).strip() == "":
        linked_match = None
    else:
        linked_match = int(linked_match)
    try:
        return FixtureRecord(
            id=None if pd.isna(row.get("id")) else int(row["id"]),
            match_label=str(row["match_label"]).strip(),
            competition_stage=str(row.get("competition_stage", "") or "").strip(),
            kickoff_utc=local_datetime_to_utc(row["kickoff_local"], local_tz),
            submission_status=str(row.get("submission_status", "pending")),
            linked_match_id=linked_match,
            notes=str(row.get("notes", "") or "").strip(),
        )
    except ValidationError:
        raise


def fetch_schedule() -> list[FixtureRecord]:
    # TODO: Wire this to the selected official tournament schedule provider.
    raise NotImplementedError("Live tournament schedule import is not configured.")
