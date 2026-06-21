from __future__ import annotations

import io
import re
import string
from dataclasses import dataclass
from typing import BinaryIO

import pandas as pd
from rapidfuzz import fuzz, process

from .classification import classify_question
from .config import FUZZY_MATCH_THRESHOLD


PROBABILITY_PATTERN = re.compile(
    r"^(?P<question>.+?)\s+(?:[.\-–—·]{2,}\s*)?(?P<prob>\d{1,3}(?:\.\d+)?)\s*%?\s*$"
)


@dataclass
class ParseResult:
    data: pd.DataFrame
    warnings: list[str]


def normalize_probability(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().replace("%", "")
    if not text:
        return None
    try:
        probability = float(text)
    except ValueError:
        return None
    if probability > 1:
        probability /= 100
    return probability if 0 <= probability <= 1 else None


def normalize_question(text: object) -> str:
    value = str(text or "").casefold()
    value = value.translate(str.maketrans("", "", string.punctuation))
    return " ".join(value.split())


def _read_bytes(file: BinaryIO | bytes) -> bytes:
    if isinstance(file, bytes):
        return file
    if hasattr(file, "getvalue"):
        return file.getvalue()
    position = file.tell() if hasattr(file, "tell") else None
    content = file.read()
    if position is not None:
        file.seek(position)
    return content


def parse_predictions_pdf(file: BinaryIO | bytes) -> ParseResult:
    content = _read_bytes(file)
    warnings: list[str] = []
    rows: list[dict[str, object]] = []
    if not content:
        return ParseResult(pd.DataFrame(columns=["question_text", "prob", "rationale"]), ["PDF is empty."])

    text_chunks: list[str] = []
    try:
        import pdfplumber

        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                try:
                    for table in page.extract_tables() or []:
                        for cells in table:
                            cleaned = [str(cell).strip() for cell in cells if cell is not None and str(cell).strip()]
                            if len(cleaned) < 2:
                                continue
                            probability = next(
                                (normalize_probability(cell) for cell in reversed(cleaned) if normalize_probability(cell) is not None),
                                None,
                            )
                            if probability is not None:
                                rows.append(
                                    {
                                        "question_text": cleaned[0],
                                        "prob": probability,
                                        "rationale": " | ".join(cleaned[1:-1]),
                                    }
                                )
                    text_chunks.append(page.extract_text() or "")
                except Exception as exc:
                    warnings.append(f"Page {page_number}: {exc}")
    except Exception as exc:
        warnings.append(f"pdfplumber could not read the PDF: {exc}")
        try:
            import fitz

            document = fitz.open(stream=content, filetype="pdf")
            text_chunks = [page.get_text() for page in document]
        except Exception as fallback_exc:
            warnings.append(f"PyMuPDF fallback also failed: {fallback_exc}")

    existing = {normalize_question(row["question_text"]) for row in rows}
    for line in "\n".join(text_chunks).splitlines():
        match = PROBABILITY_PATTERN.match(" ".join(line.split()))
        if not match:
            continue
        question = match.group("question").strip(" .-–—")
        probability = normalize_probability(match.group("prob") + ("%" if "%" in line else ""))
        normalized = normalize_question(question)
        if probability is not None and normalized and normalized not in existing:
            rows.append({"question_text": question, "prob": probability, "rationale": ""})
            existing.add(normalized)

    frame = pd.DataFrame(rows, columns=["question_text", "prob", "rationale"])
    if frame.empty:
        warnings.append("No question/probability rows were recognized.")
    return ParseResult(frame, warnings)


def parse_settlement_csv(file: BinaryIO | bytes) -> ParseResult:
    content = _read_bytes(file)
    warnings: list[str] = []
    if not content:
        return ParseResult(pd.DataFrame(), ["Settlement CSV is empty."])
    try:
        frame = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        return ParseResult(pd.DataFrame(), [f"Could not read settlement CSV: {exc}"])

    aliases = {
        "question": "question_text",
        "questiontext": "question_text",
        "prompt": "question_text",
        "crowdprob": "p_crowd",
        "crowd_prob": "p_crowd",
        "crowd_probability": "p_crowd",
        "p_crowd": "p_crowd",
        "modelprob": "p_model",
        "model_prob": "p_model",
        "p_model": "p_model",
        "claudeprob": "p_claude",
        "claude_prob": "p_claude",
        "p_claude": "p_claude",
        "result": "outcome",
        "settlement": "outcome",
        "outcome": "outcome",
        "weight": "weight",
        "category": "category",
    }
    rename: dict[str, str] = {}
    for column in frame.columns:
        key = re.sub(r"[^a-z0-9_]", "", str(column).casefold().replace(" ", "_"))
        rename[column] = aliases.get(key, key)
    frame = frame.rename(columns=rename)
    if "question_text" not in frame:
        return ParseResult(pd.DataFrame(), ["CSV needs a question_text column."])

    for column in ("p_model", "p_claude", "p_crowd"):
        if column in frame:
            frame[column] = frame[column].map(normalize_probability)
    if "outcome" in frame:
        frame["outcome"] = frame["outcome"].map(parse_outcome)
    else:
        frame["outcome"] = None
    if "weight" not in frame:
        frame["weight"] = 1.0
    frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce").fillna(1.0)
    return ParseResult(frame, warnings)


def parse_outcome(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().casefold()
    if text in {"1", "1.0", "yes", "y", "true", "win", "won"}:
        return 1
    if text in {"0", "0.0", "no", "n", "false", "loss", "lost"}:
        return 0
    return None


def reconcile_sources(
    model: pd.DataFrame | None,
    claude: pd.DataFrame | None,
    settlement: pd.DataFrame | None,
    threshold: int = FUZZY_MATCH_THRESHOLD,
) -> tuple[pd.DataFrame, list[str]]:
    records: list[dict[str, object]] = []
    notices: list[str] = []
    sources = [
        ("settlement", settlement, None),
        ("model", model, "p_model"),
        ("claude", claude, "p_claude"),
    ]

    for source_name, source, probability_column in sources:
        if source is None or source.empty:
            continue
        for _, source_row in source.iterrows():
            question = str(source_row.get("question_text", "")).strip()
            if not question:
                continue
            normalized = normalize_question(question)
            existing_keys = [normalize_question(record["question_text"]) for record in records]
            match_index: int | None = None
            if normalized in existing_keys:
                match_index = existing_keys.index(normalized)
            elif existing_keys:
                best = process.extractOne(normalized, existing_keys, scorer=fuzz.token_sort_ratio)
                if best and best[1] >= threshold:
                    match_index = int(best[2])
                    notices.append(
                        f"Fuzzy matched {source_name}: “{question}” → “{records[match_index]['question_text']}” ({best[1]:.0f})."
                    )
            if match_index is None:
                records.append(
                    {
                        "question_text": question,
                        "category": classify_question(question),
                        "p_model": None,
                        "p_claude": None,
                        "p_crowd": None,
                        "outcome": None,
                        "weight": 1.0,
                        "_sources": source_name,
                    }
                )
                match_index = len(records) - 1
                if records[:-1]:
                    notices.append(f"Unmatched {source_name} row surfaced: “{question}”.")

            target = records[match_index]
            target["_sources"] = ", ".join(dict.fromkeys(str(target["_sources"]).split(", ") + [source_name]))
            if probability_column:
                target[probability_column] = source_row.get("prob")
            else:
                for column in ("p_model", "p_claude", "p_crowd", "outcome", "weight", "category"):
                    if column in source_row and not pd.isna(source_row[column]):
                        target[column] = source_row[column]

    columns = [
        "question_text",
        "category",
        "p_model",
        "p_claude",
        "p_crowd",
        "outcome",
        "weight",
        "_sources",
    ]
    return pd.DataFrame(records, columns=columns), notices
