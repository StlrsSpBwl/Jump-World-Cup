from __future__ import annotations

import argparse
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path

from rbp_lab.config import (
    DATABASE_PATH,
    EMAIL_REMINDERS_ENABLED,
    LOCAL_TZ,
    REMINDER_CHANNEL,
    REMINDER_CHANNEL_OPTIONS,
    REMINDER_LEAD_MINUTES,
    utc_now,
)
from rbp_lab.db import (
    initialize_database,
    list_fixtures,
    mark_fixture_reminded,
    mark_passed_fixtures_missed,
)
from rbp_lab.models import FixtureRecord
from rbp_lab.reminders import due_fixtures, format_reminder_text


def notify_desktop(title: str, message: str) -> bool:
    try:
        from plyer import notification

        notification.notify(title=title, message=message, app_name="RBP Lab", timeout=15)
        return True
    except Exception as exc:
        print(f"[RBP Lab desktop fallback] {message} ({exc})")
        return False


def notify_email(fixture: FixtureRecord, message: str, lead_minutes: int) -> bool:
    if not EMAIL_REMINDERS_ENABLED:
        print("[RBP Lab] Email reminders are disabled.")
        return False

    required = {
        "SMTP_HOST": os.getenv("SMTP_HOST"),
        "SMTP_USER": os.getenv("SMTP_USER"),
        "SMTP_PASS": os.getenv("SMTP_PASS"),
        "REMINDER_EMAIL_TO": os.getenv("REMINDER_EMAIL_TO"),
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        print(f"[RBP Lab] Email skipped; missing environment variables: {', '.join(missing)}")
        return False

    port = int(os.getenv("SMTP_PORT", "587"))
    email = EmailMessage()
    email["From"] = required["SMTP_USER"]
    email["To"] = required["REMINDER_EMAIL_TO"]
    email["Subject"] = (
        f"[RBP Lab] Submit {fixture.match_label} - kickoff in {lead_minutes}m"
    )
    email.set_content(message)
    try:
        with smtplib.SMTP(required["SMTP_HOST"], port, timeout=20) as smtp:
            smtp.starttls()
            smtp.login(required["SMTP_USER"], required["SMTP_PASS"])
            smtp.send_message(email)
        return True
    except Exception as exc:
        print(f"[RBP Lab] Email delivery failed for {fixture.match_label}: {exc}")
        return False


def dispatch(
    fixture: FixtureRecord,
    message: str,
    channel: str,
    lead_minutes: int,
) -> bool:
    delivered = False
    if channel in {"desktop", "both"}:
        # The fallback stdout line still counts as a delivered reminder.
        delivered = notify_desktop("Prediction submission due", message) or delivered
        if not delivered:
            delivered = True
    if channel in {"email", "both"}:
        delivered = notify_email(fixture, message, lead_minutes) or delivered
    if channel == "stdout":
        print(f"[RBP Lab reminder] {message}")
        delivered = True
    return delivered


def run_once(
    lead_minutes: int,
    channel: str,
    database_path: str | Path = DATABASE_PATH,
) -> tuple[int, int]:
    now = utc_now()
    initialize_database(database_path, seed=False)
    missed = mark_passed_fixtures_missed(now, database_path)
    fixtures = list_fixtures(database_path)
    due = due_fixtures(fixtures, now, lead_minutes)
    reminded = 0
    for fixture in due:
        message = format_reminder_text(fixture, now, LOCAL_TZ)
        if dispatch(fixture, message, channel, lead_minutes):
            mark_fixture_reminded(fixture.id, now, database_path)
            reminded += 1
    print(
        f"[RBP Lab] checked={len(fixtures)} due={len(due)} "
        f"reminded={reminded} newly_missed={missed}"
    )
    return reminded, missed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run RBP Lab submission reminders once.")
    parser.add_argument(
        "--once",
        action="store_true",
        default=True,
        help="Run one check and exit (default).",
    )
    parser.add_argument("--lead", type=int, default=REMINDER_LEAD_MINUTES)
    parser.add_argument(
        "--channel",
        choices=REMINDER_CHANNEL_OPTIONS,
        default=REMINDER_CHANNEL,
    )
    parser.add_argument("--database", default=str(DATABASE_PATH))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_once(args.lead, args.channel, args.database)


if __name__ == "__main__":
    main()
