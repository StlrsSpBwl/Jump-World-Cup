from dataclasses import dataclass

from worldcup_props.config import Settings
from worldcup_props.contest_agent import (
    apply_player_event_agent,
    apply_question_agent,
)
from worldcup_props.domain import Question, QuestionType, Stat
from worldcup_props.goals import GoalCalibration
from worldcup_props.tournament import MatchTournamentContext, TeamTournamentContext


def test_structured_extreme_favorite_floor_protects_second_half_sot():
    settings = Settings()
    question = Question(
        home="Spain",
        away="Saudi Arabia",
        stat=Stat.SHOTS_ON_TARGET,
        question_type=QuestionType.SECOND_HALF_MORE_THAN,
    )
    calibration = GoalCalibration(
        lambda_home=3.0,
        lambda_away=0.35,
        source="market_calibrated",
        objective=0.0,
        residuals=[],
        rho=-0.08,
        targets_used=4,
    )
    context = MatchTournamentContext(
        home=TeamTournamentContext(
            team="Spain",
            coast_if_leading=True,
            tactical_style="structured_possession",
        ),
        away=TeamTournamentContext(team="Saudi Arabia"),
    )

    result = apply_question_agent(
        0.59,
        question,
        settings,
        goal_calibration=calibration,
        tournament_context=context,
        market_probability=None,
        crowd_anchor={"key": "shots_on_target:second_half_more_than"},
    )

    # structured floor lifts to 0.70, then the data-driven 2H-SOT dominance
    # correction pushes a strong favorite further toward the empirical 0.76.
    assert result.probability == 0.733
    rule_names = [r["name"] for r in result.metadata["rules"]]
    assert "structured_possession_extreme_favorite_floor" in rule_names
    assert "favorite_second_half_sot_dominance" in rule_names


def test_generic_extreme_favorite_does_not_get_structured_floor():
    settings = Settings()
    question = Question(
        home="Brazil",
        away="Haiti",
        stat=Stat.CORNERS,
        question_type=QuestionType.SECOND_HALF_MORE_THAN,
    )
    calibration = GoalCalibration(
        lambda_home=3.0,
        lambda_away=0.35,
        source="market_calibrated",
        objective=0.0,
        residuals=[],
        rho=-0.08,
        targets_used=4,
    )
    context = MatchTournamentContext(
        home=TeamTournamentContext(
            team="Brazil",
            coast_if_leading=True,
            tactical_style="flair_transition",
        ),
        away=TeamTournamentContext(team="Haiti"),
    )

    result = apply_question_agent(
        0.59,
        question,
        settings,
        goal_calibration=calibration,
        tournament_context=context,
        market_probability=None,
        crowd_anchor={"key": "corners:second_half_more_than"},
    )

    assert result.probability == 0.59
    assert result.metadata["rules"] == []


def test_direct_market_blends_via_disagreement_guard():
    settings = Settings()
    question = Question(
        home="Norway",
        away="Senegal",
        stat=Stat.SHOTS_ON_TARGET,
        question_type=QuestionType.THRESHOLD,
        k=6,
    )
    calibration = GoalCalibration(
        lambda_home=1.45,
        lambda_away=1.20,
        source="market_calibrated",
        objective=0.0,
        residuals=[],
        rho=-0.08,
        targets_used=4,
    )

    result = apply_question_agent(
        0.70,
        question,
        settings,
        goal_calibration=calibration,
        tournament_context=None,
        market_probability=0.44,
        crowd_anchor={"key": "shots_on_target:threshold:6"},
    )

    assert result.probability == 0.479
    assert [rule["name"] for rule in result.metadata["rules"]] == [
        "market_disagreement_guard"
    ]


@dataclass
class DummyLineup:
    status: str
    start_probability: float = 0.0
    sub_probability: float = 0.0
    confirmed: bool = True


@dataclass
class DummyPlayerProfile:
    shots_per90: float
    shots_on_target_per90: float = 0.0
    goals_per90: float = 0.0
    assists_per90: float = 0.0
    penalty_taker: bool = False
    set_piece_role: float = 0.0
    player_role: str = "unknown"


def test_confirmed_unavailable_player_is_zeroed():
    result = apply_player_event_agent(
        0.35,
        "shots_on_target",
        Settings(),
        market_probability=None,
        lineup=DummyLineup("out"),
        crowd_anchor={"key": "player_shots_on_target"},
    )

    assert result.probability == 0.0
    assert result.metadata["rules"][0]["name"] == "confirmed_player_unavailable"


def test_high_usage_bench_goal_assist_gets_floor_when_market_missing():
    result = apply_player_event_agent(
        0.12,
        "goal_or_assist",
        Settings(),
        market_probability=None,
        lineup=DummyLineup("bench", start_probability=0.0, sub_probability=0.85),
        player_profile=DummyPlayerProfile(
            shots_per90=2.1,
            goals_per90=0.35,
            assists_per90=0.20,
            player_role="forward",
        ),
        crowd_anchor={"key": "player_goal_or_assist"},
    )

    assert result.probability == 0.22
    assert result.metadata["rules"][0]["name"] == "high_usage_bench_player_floor"
    assert result.metadata["rules"][-1]["name"] == "direct_player_market_missing"


def test_low_usage_bench_player_does_not_get_high_usage_floor():
    result = apply_player_event_agent(
        0.12,
        "goal_or_assist",
        Settings(),
        market_probability=None,
        lineup=DummyLineup("bench", start_probability=0.0, sub_probability=0.85),
        player_profile=DummyPlayerProfile(
            shots_per90=0.25,
            goals_per90=0.02,
            assists_per90=0.03,
            player_role="midfielder",
        ),
        crowd_anchor={"key": "player_goal_or_assist"},
    )

    assert result.probability == 0.12
    assert all(
        rule["name"] != "high_usage_bench_player_floor"
        for rule in result.metadata["rules"]
    )


def test_favorite_second_half_sot_dominance_boosts_strong_favorite():
    settings = Settings()
    question = Question(
        home="Portugal",
        away="Uzbekistan",
        stat=Stat.SHOTS_ON_TARGET,
        question_type=QuestionType.SECOND_HALF_MORE_THAN,
    )
    calibration = GoalCalibration(
        lambda_home=2.3,
        lambda_away=0.7,
        source="market_calibrated",
        objective=0.0,
        residuals=[],
        rho=-0.08,
        targets_used=5,
    )
    result = apply_question_agent(
        0.55,
        question,
        settings,
        goal_calibration=calibration,
        tournament_context=None,
        market_probability=None,
        crowd_anchor={"key": "shots_on_target:second_half_more_than"},
    )
    # strong home favorite -> pushed up toward the empirical 0.76 target
    assert result.probability > 0.62
    assert any(
        r["name"] == "favorite_second_half_sot_dominance"
        for r in result.metadata["rules"]
    )


def test_favorite_sot2h_dominance_skipped_for_even_matchup():
    settings = Settings()
    question = Question(
        home="Portugal",
        away="Uzbekistan",
        stat=Stat.SHOTS_ON_TARGET,
        question_type=QuestionType.SECOND_HALF_MORE_THAN,
    )
    calibration = GoalCalibration(
        lambda_home=1.3,
        lambda_away=1.3,
        source="model_only",
        objective=0.0,
        residuals=[],
        rho=-0.08,
        targets_used=0,
    )
    result = apply_question_agent(
        0.55,
        question,
        settings,
        goal_calibration=calibration,
        tournament_context=None,
        market_probability=None,
        crowd_anchor=None,
    )
    assert result.probability == 0.55
