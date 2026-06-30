"""LLM question parser — free-text contest question -> structured spec.

The rule-based router in `card.py` covers the archetypes seen this tournament,
but novel wording can slip past it. This module sends the raw question list to a
fast, cheap model (Claude Haiku 4.5) and gets back a structured spec per question
that the deterministic spec-router (`card.route_spec`) maps to an engine call.

The LLM only *understands* the question (extracts type + parameters); it never
produces a probability. The number always comes from the closed-form / sub-discount
engines, so the output stays auditable.

Structured output is forced via single-tool use (`tool_choice` pinned to one
tool): the model must return its answer as that tool's validated input. This is
the most SDK-version-robust way to guarantee schema-shaped JSON — it predates the
newer `messages.parse()` / `output_config.format` helpers and works the same on
every anthropic SDK release.

Requires `pip install anthropic` and `ANTHROPIC_API_KEY`. Imported lazily so the
rest of the package (and the default rule-based card path) has no LLM dependency.
"""

from __future__ import annotations

import json
from typing import Optional

from pydantic import BaseModel

DEFAULT_MODEL = "claude-haiku-4-5"  # cheapest fast model; user-overridable

KINDS = [
    "player_prop", "advance", "win", "draw", "ht_lead", "btts", "btts_and_3plus",
    "total_under", "total_over", "second_half_more_goals", "second_half_2plus",
    "team_scores_2h", "team_both_halves", "team_total_goals", "first_goal",
    "half_total_goals", "both_teams_card", "goal_before_break", "goal_after_break",
    "count_threshold", "count_total", "count_compare", "base_rate",
]
EVENTS = ["goals", "shots_on_target", "goal_or_assist", "second_half_shots_on_target"]
STATS = ["fouls", "corners", "offsides", "cards", "shots_on_target"]
BASE_KEYS = [
    "penalty_only", "penalty_or_red", "sub_scores", "any_brace",
    "offside_before_break", "card_after_break", "any_player_2plus_sot",
]


class QuestionSpec(BaseModel):
    """Normalized intent for one contest question — engine-routable."""

    question: str
    kind: str
    subject_team: Optional[str] = None
    player: Optional[str] = None
    event: Optional[str] = None
    stat: Optional[str] = None
    period: str = "full"
    threshold: Optional[int] = None
    base_rate_key: Optional[str] = None


SYSTEM_PROMPT = """You convert football contest questions into structured specs.
For each question, classify its `kind` and extract parameters. Return ALL questions
via the emit_specs tool, one spec each, preserving order and echoing the question text.

kind meanings:
- player_prop: a named player scores / has a shot on target / scores-or-assists.
  Set `player` (the name) and `event` (goals | shots_on_target | goal_or_assist |
  second_half_shots_on_target).
- advance: "will [team] advance / reach the next round". Set subject_team.
- win: "will [team] win the match". Set subject_team.
- draw: "will it end in a tie / draw" (or "tied at halftime").
- ht_lead: "will [team] be ahead/winning at halftime". Set subject_team.
- btts: "will both teams score" (no goal-total condition).
- btts_and_3plus: "both teams score AND 3+ total goals".
- total_under: "2 or fewer total goals".
- total_over: "3 or more total goals".
- second_half_more_goals: "second half has more goals than the first half".
- second_half_2plus: "2 or more goals in the second half".
- team_scores_2h: "will [team] score in the second half". Set subject_team.
- team_both_halves: "will [team] score in BOTH halves". Set subject_team.
- team_total_goals: "will [team] score N or more goals" (a named team's own goal count).
  Set subject_team and threshold. NOT the match total (that's total_over/total_under).
- first_goal: "will [team] score the first goal / open the scoring". Set subject_team.
- half_total_goals: "will the [first/second] half produce N+ goals" (no team). Set period + threshold.
- both_teams_card: "will both teams receive at least one card".
- player_prop with threshold: "[player] has 2+ shots on target" -> set player, event, threshold=2.
- goal_before_break: "a goal before the first hydration break".
- goal_after_break: "a goal after the second hydration break".
- count_threshold: "[team] has X+ [stat]". Set subject_team, stat, threshold, period.
- count_total: "X+ total [stat]" (both teams combined). Set stat, threshold, period.
- count_compare: "[team A] more [stat] than [team B]". Set subject_team=A, stat, period.
- base_rate: anything else with a fixed base rate. Set base_rate_key:
  penalty_only (a penalty is awarded), penalty_or_red (penalty OR red card),
  sub_scores (a substitute scores), any_brace (any player scores 2+ goals),
  offside_before_break (offside before first hydration break),
  card_after_break (a card after a hydration break),
  any_player_2plus_sot (ANY player records 2+ shots on target — not a named player),
  red_card (a red card is shown), fh_stoppage_goal (a goal in first-half stoppage time).

stat ∈ {fouls, corners, offsides, cards, shots_on_target}.
period ∈ {full, first_half, second_half} — default full.
Use the exact team names provided. Leave unused fields null."""


def _tool() -> dict:
    spec_props = {
        "question": {"type": "string"},
        "kind": {"type": "string", "enum": KINDS},
        "subject_team": {"type": ["string", "null"]},
        "player": {"type": ["string", "null"]},
        "event": {"type": ["string", "null"], "enum": EVENTS + [None]},
        "stat": {"type": ["string", "null"], "enum": STATS + [None]},
        "period": {"type": "string", "enum": ["full", "first_half", "second_half"]},
        "threshold": {"type": ["integer", "null"]},
        "base_rate_key": {"type": ["string", "null"], "enum": BASE_KEYS + [None]},
    }
    return {
        "name": "emit_specs",
        "description": "Return the structured spec for every question.",
        "input_schema": {
            "type": "object",
            "properties": {
                "specs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": spec_props,
                        "required": ["question", "kind", "period"],
                    },
                }
            },
            "required": ["specs"],
        },
    }


def parse_questions(
    questions: list[str],
    *,
    home: str,
    away: str,
    model: str = DEFAULT_MODEL,
    client=None,
) -> list[QuestionSpec]:
    """Parse free-text questions into structured specs via a forced tool call.

    Pass `client` to inject a custom/stub Anthropic client (used in tests); when
    omitted, a default `anthropic.Anthropic()` is constructed lazily.
    """
    if client is None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "The LLM parser needs the anthropic SDK: pip install anthropic "
                "(and set ANTHROPIC_API_KEY). Or use the rule-based parser."
            ) from exc
        client = anthropic.Anthropic()

    user = (
        f"Match: {home} (home) vs {away} (away).\n"
        "Questions:\n" + "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions))
    )
    resp = client.messages.create(
        model=model,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        tools=[_tool()],
        tool_choice={"type": "tool", "name": "emit_specs"},
        messages=[{"role": "user", "content": user}],
    )
    block = next(b for b in resp.content if b.type == "tool_use")
    raw = block.input["specs"] if isinstance(block.input, dict) else json.loads(block.input)["specs"]
    return [QuestionSpec(**s) for s in raw]
