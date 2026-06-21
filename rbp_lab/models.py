from __future__ import annotations

from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Category(str, Enum):
    GOALS_TOTALS = "goals_totals"
    MATCH_RESULT = "match_result"
    SUPREMACY_HANDICAP = "supremacy_handicap"
    BTTS = "btts"
    COMPOUND = "compound"
    CORNERS = "corners"
    CARDS_BOOKINGS = "cards_bookings"
    FOULS = "fouls"
    SHOTS = "shots"
    SHOTS_ON_TARGET = "shots_on_target"
    PLAYER_SCORER = "player_scorer"
    PENALTY = "penalty"
    PERIOD_SPLIT = "period_split"
    OTHER = "other"


CATEGORY_VALUES = [category.value for category in Category]


class MatchRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int | None = None
    match_label: str = Field(min_length=1)
    competition_stage: str = ""
    match_date: date
    official_rbp_model: float | None = None
    official_rbp_claude: float | None = None
    notes: str = ""
    created_at: datetime | None = None


class QuestionRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int | None = None
    match_id: int | None = None
    question_text: str = Field(min_length=1)
    category: Category = Category.OTHER
    p_model: float = Field(ge=0, le=1)
    p_claude: float = Field(ge=0, le=1)
    p_crowd: float = Field(ge=0, le=1)
    outcome: int | None = None
    weight: float = Field(default=1.0, gt=0)
    brier_model: float | None = None
    brier_claude: float | None = None
    brier_crowd: float | None = None
    rbp_model: float | None = None
    rbp_claude: float | None = None
    model_vs_llm: float | None = None

    @field_validator("outcome")
    @classmethod
    def validate_outcome(cls, value: int | None) -> int | None:
        if value not in {None, 0, 1}:
            raise ValueError("outcome must be 0, 1, or void")
        return value


class SubmissionStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    SKIPPED = "skipped"
    MISSED = "missed"


SUBMISSION_STATUS_VALUES = [status.value for status in SubmissionStatus]


class FixtureRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int | None = None
    match_label: str = Field(min_length=1)
    competition_stage: str = ""
    kickoff_utc: datetime
    submission_status: SubmissionStatus = SubmissionStatus.PENDING
    submitted_at: datetime | None = None
    linked_match_id: int | None = None
    reminded_at: datetime | None = None
    notes: str = ""
    created_at: datetime | None = None

    @field_validator("kickoff_utc", "submitted_at", "reminded_at", "created_at")
    @classmethod
    def require_timezone(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("fixture datetimes must be timezone-aware")
        return value
