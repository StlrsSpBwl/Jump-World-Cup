from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .calibration import brier_decomposition, weighted_brier_score
from .db import connect, initialize, transaction


def log_result(
    database_path: str | Path,
    *,
    match_key: str,
    question_key: str,
    question_type: str,
    submitted_probability: float,
    outcome: int,
    crowd_probability: float | None = None,
    market_blended_probability: float | None = None,
    weight: float = 1.0,
    observed_at: str | None = None,
) -> None:
    for name, value in (
        ("submitted_probability", submitted_probability),
        ("crowd_probability", crowd_probability),
        ("market_blended_probability", market_blended_probability),
    ):
        if value is not None and not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    if outcome not in {0, 1}:
        raise ValueError("outcome must be 0 or 1")
    if weight <= 0:
        raise ValueError("weight must be positive")
    initialize(database_path)
    with transaction(database_path) as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO forecast_results (
                match_key, question_key, question_type, submitted_probability,
                crowd_probability, outcome, market_blended_probability, weight,
                observed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                match_key.strip().casefold(),
                question_key.strip().casefold(),
                question_type.strip().casefold(),
                submitted_probability,
                crowd_probability,
                outcome,
                market_blended_probability,
                weight,
                observed_at or datetime.now(timezone.utc).isoformat(),
            ),
        )


def ingest_results_csv(database_path: str | Path, csv_path: str | Path) -> int:
    with Path(csv_path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        log_result(
            database_path,
            match_key=row["match"],
            question_key=row["question"],
            question_type=row["question_type"],
            submitted_probability=float(row["submitted_probability"]),
            crowd_probability=_optional_float(row.get("crowd_probability")),
            outcome=int(row["outcome"]),
            market_blended_probability=_optional_float(
                row.get("market_blended_probability")
            ),
            weight=float(row.get("weight") or 1.0),
            observed_at=row.get("timestamp") or None,
        )
    return len(rows)


def results_report(database_path: str | Path) -> dict[str, Any]:
    initialize(database_path)
    with connect(database_path) as connection:
        rows = [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM forecast_results ORDER BY observed_at, id"
            )
        ]
    if not rows:
        return {"events": 0, "warning": "No forecast results have been logged"}
    by_type: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_type.setdefault(str(row["question_type"]), []).append(row)
    return {
        "events": len(rows),
        "overall": _summarize(rows),
        "question_types": {
            question_type: _summarize(group)
            for question_type, group in sorted(by_type.items())
        },
    }


def crowd_bias_for_type(
    database_path: str | Path, question_type: str
) -> tuple[float | None, int]:
    initialize(database_path)
    with connect(database_path) as connection:
        row = connection.execute(
            """
            SELECT COUNT(*) AS n, AVG(crowd_probability - outcome) AS bias
            FROM forecast_results
            WHERE question_type=? AND crowd_probability IS NOT NULL
            """,
            (question_type.casefold(),),
        ).fetchone()
    return (
        (float(row["bias"]) if row and row["bias"] is not None else None),
        int(row["n"]) if row else 0,
    )


def _summarize(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    values = list(rows)
    outcomes = [int(row["outcome"]) for row in values]
    weights = [float(row["weight"]) for row in values]
    report: dict[str, Any] = {
        "events": len(values),
        "submitted": _forecast_metrics(
            [float(row["submitted_probability"]) for row in values],
            outcomes,
            weights,
        ),
    }
    crowd_rows = [row for row in values if row["crowd_probability"] is not None]
    if crowd_rows:
        crowd_probabilities = [float(row["crowd_probability"]) for row in crowd_rows]
        crowd_outcomes = [int(row["outcome"]) for row in crowd_rows]
        crowd_weights = [float(row["weight"]) for row in crowd_rows]
        report["crowd"] = _forecast_metrics(
            crowd_probabilities, crowd_outcomes, crowd_weights
        )
        report["crowd_bias"] = float(
            np.average(
                np.asarray(crowd_probabilities) - np.asarray(crowd_outcomes),
                weights=crowd_weights,
            )
        )
    blended_rows = [
        row for row in values if row["market_blended_probability"] is not None
    ]
    if blended_rows:
        report["market_blended"] = _forecast_metrics(
            [float(row["market_blended_probability"]) for row in blended_rows],
            [int(row["outcome"]) for row in blended_rows],
            [float(row["weight"]) for row in blended_rows],
        )
    return report


def _forecast_metrics(
    probabilities: list[float], outcomes: list[int], weights: list[float]
) -> dict[str, Any]:
    return {
        "events": len(probabilities),
        "brier": weighted_brier_score(probabilities, outcomes, weights),
        "decomposition": brier_decomposition(
            probabilities, outcomes, weights=weights
        ).as_dict(),
    }


def _optional_float(value: Any) -> float | None:
    return None if value is None or str(value).strip() == "" else float(value)
