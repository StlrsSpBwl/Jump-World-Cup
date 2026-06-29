from pathlib import Path
from types import SimpleNamespace

from worldcup_props.card import forecast_card, route_spec
from worldcup_props.closed_form import goal_props
from worldcup_props.llm_parser import QuestionSpec, _tool, parse_questions

DB = Path(__file__).resolve().parents[1] / "data" / "worldcup_props.sqlite"


def _g():
    return goal_props(1.4, 1.0)


def test_tool_schema_is_well_formed():
    t = _tool()
    assert t["name"] == "emit_specs"
    props = t["input_schema"]["properties"]["specs"]["items"]["properties"]
    assert "kind" in props and "stat" in props and "player" in props


def test_route_spec_player_prop_uses_sub_discount():
    spec = QuestionSpec(question="?", kind="player_prop", player="Iqraam Rayners",
                        event="shots_on_target")
    p, basis = route_spec(spec, "Canada", "South Africa", _g(), 1.4, 1.0,
                          str(DB), {"Iqraam Rayners": "sub"}, None)
    assert basis.startswith("player:sub_discount")
    assert 0.14 < p < 0.24


def test_route_spec_outcome_and_base_rate():
    g = _g()
    btts = route_spec(QuestionSpec(question="?", kind="btts"), "Canada", "South Africa",
                      g, 1.4, 1.0, None, None, None)
    assert btts[1] == "btts" and 0.4 < btts[0] < 0.6
    pen = route_spec(QuestionSpec(question="?", kind="base_rate", base_rate_key="penalty_only"),
                     "Canada", "South Africa", g, 1.4, 1.0, None, None, None)
    assert pen == (0.30, "base:penalty_only")


def test_route_spec_count_total_and_compare():
    g = _g()
    tot = route_spec(QuestionSpec(question="?", kind="count_total", stat="corners", threshold=9),
                     "Canada", "South Africa", g, 1.4, 1.0, str(DB), None, None)
    assert tot[1] == "count_total:corners:full" and tot[0] is not None


def test_parse_questions_with_injected_client_drives_the_card():
    # stub the Anthropic client: forced tool call returns specs as a tool_use block
    specs_payload = {
        "specs": [
            {"question": "Will both teams score?", "kind": "btts", "period": "full"},
            {"question": "Will Rayners have a shot on target?", "kind": "player_prop",
             "player": "Iqraam Rayners", "event": "shots_on_target", "period": "full"},
        ]
    }
    block = SimpleNamespace(type="tool_use", input=specs_payload)
    fake = SimpleNamespace(
        messages=SimpleNamespace(
            create=lambda **kw: SimpleNamespace(content=[block])
        )
    )
    qs = ["Will both teams score?", "Will Rayners have a shot on target?"]
    specs = parse_questions(qs, home="Canada", away="South Africa", client=fake)
    assert [s.kind for s in specs] == ["btts", "player_prop"]

    rows = forecast_card("Canada", "South Africa", 1.4, 1.0, qs, db=str(DB),
                         lineup_status={"Iqraam Rayners": "sub"}, specs=specs)
    assert rows[0]["basis"] == "btts"
    assert rows[1]["basis"].startswith("player:sub_discount")
