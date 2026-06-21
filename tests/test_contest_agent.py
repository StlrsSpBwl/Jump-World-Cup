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

    assert result.probability == 0.70
    assert result.metadata["rules"][0]["name"] == (
        "structured_possession_extreme_favorite_floor"
    )


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


@dataclass
class DummyLineup:
    status: str
    start_probability: float = 0.0
    sub_probability: float = 0.0


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
