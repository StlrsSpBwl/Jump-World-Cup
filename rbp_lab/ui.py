from __future__ import annotations

import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .config import (
    CLAUDE_COLOR,
    DEFAULT_RBP_AGG,
    DEFAULT_SIGN_CONVENTION,
    EMAIL_REMINDERS_ENABLED,
    LOCAL_TZ,
    MODEL_COLOR,
    REMINDER_CHANNEL,
    REMINDER_CHANNEL_OPTIONS,
    REMINDER_LEAD_MINUTES,
    RBP_AGG_OPTIONS,
    SIGN_CONVENTION_OPTIONS,
    UPCOMING_WINDOW_HOURS,
    DashboardSettings,
)


def configure_page(title: str, icon: str = "◫") -> None:
    st.set_page_config(page_title=f"{title} | RBP Lab", page_icon=icon, layout="wide")
    st.markdown(
        """
        <style>
        :root {
          --ink: #edf4f2;
          --muted: #92a8a3;
          --panel: rgba(17, 31, 36, 0.82);
          --line: rgba(125, 211, 192, 0.16);
          --model: #16c79a;
          --claude: #a78bfa;
        }
        .stApp {
          background:
            radial-gradient(circle at 15% 0%, rgba(22,199,154,.10), transparent 28rem),
            radial-gradient(circle at 85% 12%, rgba(167,139,250,.08), transparent 24rem),
            #081116;
          color: var(--ink);
        }
        [data-testid="stSidebar"] { background: #0b171c; border-right: 1px solid var(--line); }
        [data-testid="stMetric"] {
          background: var(--panel);
          border: 1px solid var(--line);
          border-radius: 14px;
          padding: 14px 16px;
        }
        .rbp-kicker {
          color: var(--model);
          font-size: .76rem;
          font-weight: 700;
          letter-spacing: .14em;
          text-transform: uppercase;
        }
        .rbp-title {
          font-size: clamp(2rem, 5vw, 4rem);
          font-weight: 760;
          letter-spacing: -.055em;
          line-height: .98;
          margin: .25rem 0 .75rem;
        }
        .rbp-subtitle { color: var(--muted); font-size: 1.05rem; max-width: 760px; }
        .feature-card {
          background: var(--panel);
          border: 1px solid var(--line);
          border-radius: 16px;
          padding: 18px;
          min-height: 180px;
          margin-bottom: 12px;
        }
        .feature-card h3 { margin-top: 0; font-size: 1.05rem; }
        .feature-tag {
          display: inline-block;
          color: #06130f;
          background: var(--model);
          border-radius: 999px;
          padding: 3px 8px;
          font-size: .7rem;
          font-weight: 700;
        }
        .model-dot, .claude-dot {
          display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 6px;
        }
        .model-dot { background: var(--model); }
        .claude-dot { background: var(--claude); }
        div[data-testid="stDataFrame"], div[data-testid="stDataEditor"] {
          border: 1px solid var(--line); border-radius: 12px; overflow: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(kicker: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="rbp-kicker">{kicker}</div>
        <div class="rbp-title">{title}</div>
        <div class="rbp-subtitle">{subtitle}</div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")


def settings_panel() -> DashboardSettings:
    with st.sidebar:
        st.markdown("### Scoring")
        aggregation = st.selectbox(
            "Match aggregation",
            RBP_AGG_OPTIONS,
            index=RBP_AGG_OPTIONS.index(st.session_state.get("rbp_agg", DEFAULT_RBP_AGG)),
            format_func=lambda value: value.replace("_", " ").title(),
            help="Weighted mean divides the weighted edge by total settled weight.",
        )
        sign = st.selectbox(
            "Sign convention",
            SIGN_CONVENTION_OPTIONS,
            index=SIGN_CONVENTION_OPTIONS.index(
                st.session_state.get("sign_convention", DEFAULT_SIGN_CONVENTION)
            ),
            format_func=lambda value: (
                "Positive = better" if value == "positive_beats_crowd" else "Negative = better"
            ),
        )
        st.session_state["rbp_agg"] = aggregation
        st.session_state["sign_convention"] = sign
        st.caption("Void rows stay in the database and are excluded from scoring.")
        with st.expander("Reminder settings"):
            lead_minutes = st.number_input(
                "Lead time (minutes)",
                min_value=1,
                max_value=1440,
                value=int(st.session_state.get("reminder_lead_minutes", REMINDER_LEAD_MINUTES)),
            )
            upcoming_hours = st.number_input(
                "Upcoming horizon (hours)",
                min_value=1,
                max_value=336,
                value=int(st.session_state.get("upcoming_window_hours", UPCOMING_WINDOW_HOURS)),
            )
            local_tz = st.text_input(
                "Display timezone",
                value=st.session_state.get("local_tz", LOCAL_TZ),
            )
            channel = st.selectbox(
                "Runner channel",
                REMINDER_CHANNEL_OPTIONS,
                index=REMINDER_CHANNEL_OPTIONS.index(
                    st.session_state.get("reminder_channel", REMINDER_CHANNEL)
                ),
                format_func=str.title,
            )
            email_enabled = st.checkbox(
                "Enable email reminders",
                value=bool(
                    st.session_state.get(
                        "email_reminders_enabled", EMAIL_REMINDERS_ENABLED
                    )
                ),
            )
        st.session_state["reminder_lead_minutes"] = int(lead_minutes)
        st.session_state["upcoming_window_hours"] = int(upcoming_hours)
        st.session_state["local_tz"] = local_tz
        st.session_state["reminder_channel"] = channel
        st.session_state["email_reminders_enabled"] = email_enabled
    return DashboardSettings(
        rbp_agg=aggregation,
        sign_convention=sign,
        reminder_lead_minutes=int(lead_minutes),
        upcoming_window_hours=int(upcoming_hours),
        local_tz=local_tz,
        email_reminders_enabled=email_enabled,
        reminder_channel=channel,
    )


def render_reminder_banner(settings: DashboardSettings) -> None:
    from .config import utc_now
    from .config import get_timezone
    from .db import list_fixtures, mark_passed_fixtures_missed
    from .models import SubmissionStatus
    from .reminders import due_fixtures, upcoming_fixtures

    now = utc_now()
    mark_passed_fixtures_missed(now)
    fixtures = list_fixtures()
    missed = [
        fixture
        for fixture in fixtures
        if fixture.submission_status == SubmissionStatus.MISSED
    ]
    due = due_fixtures(fixtures, now, settings.reminder_lead_minutes)
    upcoming = [
        fixture
        for fixture in upcoming_fixtures(
            fixtures, now, settings.upcoming_window_hours
        )
        if fixture not in due
    ]
    if missed:
        labels = ", ".join(fixture.match_label for fixture in missed)
        st.error(f"Missed submission: {labels}. Review the fixture status.")
    if due:
        labels = ", ".join(fixture.match_label for fixture in due)
        st.warning(
            f"Submission due within {settings.reminder_lead_minutes} minutes: {labels}."
        )
    elif upcoming:
        next_fixture = upcoming[0]
        local_time = next_fixture.kickoff_utc.astimezone(get_timezone(settings.local_tz))
        st.info(
            f"Upcoming: {next_fixture.match_label} at "
            f"{local_time.strftime('%Y-%m-%d %H:%M %Z')}."
        )


def category_label(value: str) -> str:
    return str(value).replace("_", " ").title().replace("Btts", "BTTS")


def polish_figure(fig: go.Figure, height: int = 430) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=16, r=18, t=58, b=24),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,17,22,.35)",
        font=dict(family="Inter, ui-sans-serif, system-ui", color="#dce9e6"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hoverlabel=dict(bgcolor="#122229", font_size=13),
    )
    fig.update_xaxes(gridcolor="rgba(125,211,192,.10)", zerolinecolor="rgba(255,255,255,.35)")
    fig.update_yaxes(gridcolor="rgba(125,211,192,.07)", zerolinecolor="rgba(255,255,255,.35)")
    return fig


def show_chart(fig: go.Figure, key: str) -> None:
    st.plotly_chart(
        polish_figure(fig),
        width="stretch",
        config={
            "displaylogo": False,
            "toImageButtonOptions": {"format": "png", "filename": f"rbp-lab-{key}", "scale": 2},
        },
        key=f"chart_{key}",
    )
    image_key = f"png_{key}"
    if st.button("Prepare PNG download", key=f"prepare_{key}"):
        try:
            st.session_state[image_key] = fig.to_image(format="png", scale=2)
        except Exception:
            st.warning("Local PNG rendering is unavailable. Use the camera icon in the chart toolbar.")
    if image_key in st.session_state:
        st.download_button(
            "Download chart PNG",
            st.session_state[image_key],
            file_name=f"rbp-lab-{key}.png",
            mime="image/png",
            key=f"download_{key}",
        )


def signed_colors(values: pd.Series) -> list[str]:
    from .config import NEGATIVE_COLOR, POSITIVE_COLOR

    return [POSITIVE_COLOR if value >= 0 else NEGATIVE_COLOR for value in values]


def format_percent(value: float) -> str:
    return f"{value:.1%}"


def safe_filename(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-") or "export"


def legend_html() -> str:
    return (
        f'<span class="model-dot"></span>Model <span style="margin-left:18px" '
        f'class="claude-dot"></span>Claude'
    )


MODEL_CLAUDE_COLORS = {"Model": MODEL_COLOR, "Claude": CLAUDE_COLOR}
