from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


ROOT = Path(__file__).resolve().parent.parent
DATABASE_PATH = ROOT / "rbp.db"
MODEL_FEATURES_PATH = ROOT / "model_features.yaml"
PREDICTION_RESULTS_DIR = Path(
    os.getenv(
        "PREDICTION_RESULTS_DIR",
        str(Path.home() / "Desktop" / "Prediction results"),
    )
).expanduser()
OCR_CACHE_DIR = ROOT / "data" / "ocr_cache"


def _detect_system_timezone() -> str:
    configured = os.getenv("TZ")
    if configured:
        return configured
    localtime = Path("/etc/localtime")
    try:
        target = localtime.resolve()
        marker = "zoneinfo/"
        if marker in str(target):
            return str(target).split(marker, 1)[1]
    except OSError:
        pass
    zone = getattr(datetime.now().astimezone().tzinfo, "key", None)
    return str(zone or "America/New_York")

RBP_AGG_OPTIONS = ("weighted_mean", "weighted_sum")
SIGN_CONVENTION_OPTIONS = ("positive_beats_crowd", "negative_beats_crowd")
DEFAULT_RBP_AGG = "weighted_mean"
DEFAULT_SIGN_CONVENTION = "positive_beats_crowd"
FUZZY_MATCH_THRESHOLD = 86
REMINDER_LEAD_MINUTES = int(os.getenv("REMINDER_LEAD_MINUTES", "30"))
UPCOMING_WINDOW_HOURS = int(os.getenv("UPCOMING_WINDOW_HOURS", "24"))
LOCAL_TZ = os.getenv("LOCAL_TZ", _detect_system_timezone())
EMAIL_REMINDERS_ENABLED = os.getenv("EMAIL_REMINDERS_ENABLED", "").casefold() in {
    "1",
    "true",
    "yes",
    "on",
}
REMINDER_CHANNEL_OPTIONS = ("desktop", "email", "both", "stdout")
REMINDER_CHANNEL = os.getenv("REMINDER_CHANNEL", "desktop").casefold()
if REMINDER_CHANNEL not in REMINDER_CHANNEL_OPTIONS:
    REMINDER_CHANNEL = "desktop"

MODEL_COLOR = "#16c79a"
CLAUDE_COLOR = "#a78bfa"
POSITIVE_COLOR = "#16c79a"
NEGATIVE_COLOR = "#f06a7b"
NEUTRAL_COLOR = "#718096"


@dataclass(frozen=True)
class DashboardSettings:
    rbp_agg: str = DEFAULT_RBP_AGG
    sign_convention: str = DEFAULT_SIGN_CONVENTION
    reminder_lead_minutes: int = REMINDER_LEAD_MINUTES
    upcoming_window_hours: int = UPCOMING_WINDOW_HOURS
    local_tz: str = LOCAL_TZ
    email_reminders_enabled: bool = EMAIL_REMINDERS_ENABLED
    reminder_channel: str = REMINDER_CHANNEL

    @property
    def sign_multiplier(self) -> float:
        return 1.0 if self.sign_convention == "positive_beats_crowd" else -1.0


def get_timezone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError:
        return ZoneInfo("UTC")


def utc_now() -> datetime:
    return datetime.now(tz=ZoneInfo("UTC"))
