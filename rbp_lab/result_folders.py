from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

import fitz
import pandas as pd
from rapidfuzz import fuzz, process

from .config import OCR_CACHE_DIR, ROOT
from .parsing import normalize_question


TEAM_CODE_ALIASES = {
    "SPA": "ESP",
    "SPN": "ESP",
    "CV": "CPV",
    "CVD": "CPV",
    "BHN": "BHR",
    "BHI": "BHR",
    "SCO": "SCO",
    "SCT": "SCO",
    "SWD": "SWE",
    "SWE": "SWE",
    "RS": "RSA",
    "RSA": "RSA",
    "MEX": "MEX",
    "MEXICO": "MEX",
}
TEAM_NAMES = {
    "AUS": "Australia",
    "TUR": "Türkiye",
    "BRA": "Brazil",
    "MAR": "Morocco",
    "CAN": "Canada",
    "BHR": "Bahrain",
    "CIV": "Côte d'Ivoire",
    "ECU": "Ecuador",
    "GER": "Germany",
    "CUR": "Curaçao",
    "HAI": "Haiti",
    "SCO": "Scotland",
    "KOR": "Korea Republic",
    "CZA": "Czechia",
    "MEX": "Mexico",
    "RSA": "South Africa",
    "NET": "Netherlands",
    "JPN": "Japan",
    "QAR": "Qatar",
    "SWI": "Switzerland",
    "ESP": "Spain",
    "CPV": "Cape Verde",
    "SWE": "Sweden",
    "TUN": "Tunisia",
    "US": "United States",
    "PAR": "Paraguay",
}


@dataclass(frozen=True)
class ResultPair:
    key: str
    model_path: Path
    claude_path: Path

    @property
    def label(self) -> str:
        return f"{self.key} · {self.model_path.name} + {self.claude_path.name}"


@dataclass
class ResultReport:
    match_label: str
    match_date: date | None
    match_rbp: float | None
    questions: pd.DataFrame
    warnings: list[str]


def fixture_key_from_filename(path: str | Path) -> str:
    stem = Path(path).stem.upper()
    stem = re.sub(r"(?:^|_)(MODEL|CLAUDE)(?:_|$)", "_", stem)
    tokens = [token for token in re.split(r"[^A-Z0-9]+", stem) if token]
    canonical = [TEAM_CODE_ALIASES.get(token, token) for token in tokens[:2]]
    return "_".join(canonical)


def display_label_from_key(key: str) -> str:
    teams = [TEAM_NAMES.get(token, token) for token in key.split("_")]
    return " vs ".join(teams)


def discover_result_pairs(root: str | Path) -> tuple[list[ResultPair], list[str]]:
    root_path = Path(root).expanduser()
    model_dir = root_path / "Model"
    claude_dir = root_path / "Claude"
    warnings: list[str] = []
    if not model_dir.is_dir() or not claude_dir.is_dir():
        return [], [f"Expected Model and Claude folders inside {root_path}."]

    model_files = sorted(model_dir.glob("*.pdf"))
    claude_files = sorted(claude_dir.glob("*.pdf"))
    claude_by_key = {fixture_key_from_filename(path): path for path in claude_files}
    unused_claude = set(claude_files)
    pairs: list[ResultPair] = []

    for model_path in model_files:
        key = fixture_key_from_filename(model_path)
        claude_path = claude_by_key.get(key)
        if claude_path is None and unused_claude:
            choices = {path: fixture_key_from_filename(path) for path in unused_claude}
            best = process.extractOne(
                key,
                list(choices.values()),
                scorer=fuzz.ratio,
            )
            if best and best[1] >= 70:
                claude_path = next(
                    path for path, candidate_key in choices.items() if candidate_key == best[0]
                )
                warnings.append(
                    f"Fuzzy paired {model_path.name} with {claude_path.name} ({best[1]:.0f})."
                )
        if claude_path is None:
            warnings.append(f"No Claude PDF matched {model_path.name}.")
            continue
        unused_claude.discard(claude_path)
        pairs.append(ResultPair(key, model_path, claude_path))

    for path in sorted(unused_claude):
        warnings.append(f"No Model PDF matched {path.name}.")
    return pairs, warnings


def _ocr_binary() -> Path:
    source = ROOT / "tools" / "vision_ocr.swift"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
    binary = Path(tempfile.gettempdir()) / f"rbp-vision-ocr-{digest}"
    if binary.exists():
        return binary
    module_cache = Path(tempfile.gettempdir()) / "rbp-swift-module-cache"
    module_cache.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "swiftc",
            "-module-cache-path",
            str(module_cache),
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return binary


def _ocr_pdf(path: Path) -> list[list[dict[str, Any]]]:
    cache_key = hashlib.sha256(path.read_bytes()).hexdigest()
    OCR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = OCR_CACHE_DIR / f"{cache_key}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory(prefix="rbp-pdf-") as temporary:
        temp_dir = Path(temporary)
        image_paths: list[Path] = []
        document = fitz.open(path)
        for page_index, page in enumerate(document):
            image_path = temp_dir / f"page-{page_index + 1}.png"
            page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).save(image_path)
            image_paths.append(image_path)
        completed = subprocess.run(
            [str(_ocr_binary()), *map(str, image_paths)],
            check=True,
            capture_output=True,
            text=True,
        )
    pages = [
        json.loads(line)
        for line in completed.stdout.splitlines()
        if line.strip().startswith("[")
    ]
    if len(pages) != len(image_paths):
        raise RuntimeError(
            f"Vision OCR returned {len(pages)} pages for a {len(image_paths)}-page PDF."
        )
    cache_path.write_text(json.dumps(pages), encoding="utf-8")
    return pages


def _official_rbp(path: Path) -> float | None:
    cache_key = hashlib.sha256(path.read_bytes()).hexdigest()
    OCR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = OCR_CACHE_DIR / f"{cache_key}.official-rbp.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))["value"]

    with tempfile.TemporaryDirectory(prefix="rbp-header-") as temporary:
        image_path = Path(temporary) / "official-rbp.png"
        document = fitz.open(path)
        document[0].get_pixmap(
            matrix=fitz.Matrix(5, 5),
            clip=fitz.Rect(250, 145, 370, 185),
            alpha=False,
        ).save(image_path)
        completed = subprocess.run(
            [str(_ocr_binary()), str(image_path)],
            check=True,
            capture_output=True,
            text=True,
        )
    value: float | None = None
    for line in completed.stdout.splitlines():
        if not line.strip().startswith("["):
            continue
        for item in json.loads(line):
            match = re.search(
                r"RBP\s*([+-]?\s*\d+(?:\.\d+)?)",
                _clean_ocr_text(item["text"]),
                re.IGNORECASE,
            )
            if match:
                value = float(match.group(1).replace(" ", ""))
                break
    cache_path.write_text(json.dumps({"value": value}), encoding="utf-8")
    return value


def _clean_ocr_text(text: str) -> str:
    return " ".join(text.replace("—", "-").replace("–", "-").split())


def _is_noise(text: str) -> bool:
    normalized = text.casefold()
    return (
        normalized.startswith("http")
        or normalized in {
            "sports predict",
            "match summary",
            "details",
            "match leaderboard",
            "question by question",
            "your forecast vs crowd vs reality",
            "settled",
        }
        or bool(re.fullmatch(r"\d+/\d+", normalized))
        or bool(re.fullmatch(r"\d{1,2}/\d{1,2}/\d{2},.*", normalized))
    )


def _percent_after(lines: list[dict[str, Any]], start: int, stop_labels: set[str]) -> float | None:
    for line in lines[start + 1 :]:
        text = _clean_ocr_text(line["text"])
        if text.upper() in stop_labels:
            return None
        match = re.search(r"(-?\d{1,3})\s*%", text)
        if match:
            value = int(match.group(1))
            return value / 100 if 0 <= value <= 100 else None
    return None


def parse_ocr_lines(
    pages: list[list[dict[str, Any]]],
) -> tuple[dict[str, Any], pd.DataFrame, list[str]]:
    warnings: list[str] = []
    lines: list[dict[str, Any]] = []
    for page_number, page_lines in enumerate(pages, start=1):
        for line in page_lines:
            lines.append({**line, "page": page_number, "text": _clean_ocr_text(line["text"])})

    match_date: date | None = None
    for line in lines:
        for pattern in ("%b %d, %Y", "%B %d, %Y"):
            try:
                match_date = datetime.strptime(line["text"], pattern).date()
                break
            except ValueError:
                continue
        if match_date:
            break

    match_rbp: float | None = None
    for line in lines:
        match = re.fullmatch(r"RBP\s*([+-]?\s*\d+(?:\.\d+)?)", line["text"], re.IGNORECASE)
        if match:
            match_rbp = float(match.group(1).replace(" ", ""))
            break

    match_label = ""
    date_index = next(
        (
            index
            for index, line in enumerate(lines)
            if re.fullmatch(r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{1,2}, \d{4}", line["text"])
        ),
        None,
    )
    if date_index is not None:
        nearby = [
            line
            for line in lines[max(0, date_index - 6) : date_index + 8]
            if line["text"]
            and not _is_noise(line["text"])
            and not re.search(r"\d", line["text"])
            and line["text"].casefold() not in {"yesterday", "winner"}
        ]
        team_names = [
            line["text"]
            for line in nearby
            if line["width"] > 0.05 and line["text"].upper() != line["text"]
        ]
        if len(team_names) >= 2:
            match_label = f"{team_names[0]} vs {team_names[1]}"

    question_indices = [
        index
        for index, line in enumerate(lines)
        if re.fullmatch(r"Q\d+", line["text"].upper())
    ]
    records: list[dict[str, Any]] = []
    for position, start in enumerate(question_indices):
        end = question_indices[position + 1] if position + 1 < len(question_indices) else len(lines)
        segment = lines[start:end]
        you_index = next(
            (index for index, line in enumerate(segment) if line["text"].upper() == "YOU"),
            None,
        )
        crowd_index = next(
            (index for index, line in enumerate(segment) if line["text"].upper() == "CROWD"),
            None,
        )
        result_line = next(
            (line["text"] for line in segment if "RESULT:" in line["text"].upper()),
            "",
        )
        if you_index is None or crowd_index is None:
            warnings.append(
                f"{segment[0]['text']}: YOU/CROWD values are missing, usually because "
                "the browser printout clipped this card at a page break."
            )
            continue

        question_parts = [
            line["text"]
            for line in segment[1:you_index]
            if "RESULT:" not in line["text"].upper()
            and not _is_noise(line["text"])
            and not re.fullmatch(r"[VX✓✕×]\s*", line["text"])
        ]
        question = " ".join(question_parts).strip()
        probability = _percent_after(segment, you_index, {"CROWD"})
        crowd_probability = _percent_after(segment, crowd_index, {"OUTCOME", "RBP"})
        outcome = None
        if re.search(r"RESULT:\s*YES", result_line, re.IGNORECASE):
            outcome = 1
        elif re.search(r"RESULT:\s*NO", result_line, re.IGNORECASE):
            outcome = 0
        if not question or probability is None or crowd_probability is None or outcome is None:
            warnings.append(f"{segment[0]['text']}: incomplete OCR card.")
            continue
        records.append(
            {
                "question_number": segment[0]["text"].upper(),
                "question_text": question,
                "prob": probability,
                "p_crowd": crowd_probability,
                "outcome": outcome,
                "weight": 1.0,
            }
        )
    return {
        "match_label": match_label,
        "match_date": match_date,
        "match_rbp": match_rbp,
    }, pd.DataFrame(records), warnings


def parse_sports_predict_results_pdf(path: str | Path) -> ResultReport:
    source = Path(path)
    metadata, questions, warnings = parse_ocr_lines(_ocr_pdf(source))
    official_rbp = _official_rbp(source)
    if official_rbp is None:
        warnings.append("The official page-one RBP could not be read.")
    if questions.empty:
        warnings.append("No settled question cards were recognized.")
    return ResultReport(
        match_label=metadata["match_label"]
        or display_label_from_key(fixture_key_from_filename(source)),
        match_date=metadata["match_date"],
        match_rbp=official_rbp,
        questions=questions,
        warnings=warnings,
    )


def combine_result_reports(
    model_report: ResultReport,
    claude_report: ResultReport,
    threshold: int = 78,
) -> tuple[pd.DataFrame, list[str]]:
    warnings = [*model_report.warnings, *claude_report.warnings]
    records: list[dict[str, Any]] = []
    claude_questions = claude_report.questions.copy()
    claude_keys = claude_questions["question_text"].map(normalize_question).tolist()
    used: set[int] = set()

    for _, model_row in model_report.questions.iterrows():
        model_key = normalize_question(model_row["question_text"])
        available = [
            (index, key) for index, key in enumerate(claude_keys) if index not in used
        ]
        if not available:
            warnings.append(f"No Claude question matched: {model_row['question_text']}")
            continue
        match = process.extractOne(
            model_key,
            [key for _, key in available],
            scorer=fuzz.token_sort_ratio,
        )
        if match is None or match[1] < threshold:
            warnings.append(f"No Claude question matched: {model_row['question_text']}")
            continue
        claude_index = available[int(match[2])][0]
        used.add(claude_index)
        claude_row = claude_questions.iloc[claude_index]
        if (
            abs(float(model_row["p_crowd"]) - float(claude_row["p_crowd"])) > 0.011
            or int(model_row["outcome"]) != int(claude_row["outcome"])
        ):
            warnings.append(
                f"Settlement mismatch for {model_row['question_text']}; Model report values were used."
            )
        records.append(
            {
                "question_text": model_row["question_text"],
                "p_model": model_row["prob"],
                "p_claude": claude_row["prob"],
                "p_crowd": model_row["p_crowd"],
                "outcome": model_row["outcome"],
                "weight": model_row.get("weight", 1.0),
            }
        )
    for index, row in claude_questions.iterrows():
        if index not in used:
            warnings.append(f"Unmatched Claude question: {row['question_text']}")
    return pd.DataFrame(records), warnings
