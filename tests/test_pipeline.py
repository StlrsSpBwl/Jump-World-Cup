from datetime import date, timedelta
from dataclasses import replace

import pytest

from worldcup_props.config import Settings
from worldcup_props.coherent import simulate_coherent_match
from worldcup_props.data import MatchRow, TeamStatsRow, ingest_matches, parse_fbref_schedule
from worldcup_props.db import transaction
from worldcup_props.domain import Question, QuestionType, Stat
from worldcup_props.forecast import forecast_match_event, forecast_player_event, forecast_question
from worldcup_props.goals import GoalCalibration
from worldcup_props.market import ingest_market_csv
from worldcup_props.model import ModelArtifact, fit_model
from worldcup_props.players import (
    LineupEntry,
    PlayerProfile,
    ingest_player_club_profiles_csv,
    load_match_players,
)
from worldcup_props.simulation import simulate, simulate_joint_match
from worldcup_props.tournament import MatchTournamentContext, TeamTournamentContext
from worldcup_props.validation import validate_database


def _synthetic_matches():
    teams = [
        ("USA", "CONCACAF"),
        ("Mexico", "CONCACAF"),
        ("Paraguay", "CONMEBOL"),
        ("South Africa", "CAF"),
    ]
    start = date(2021, 1, 1)
    rows = []
    for index in range(48):
        home_name, home_confed = teams[index % len(teams)]
        away_name, away_confed = teams[(index + 1 + index // 8) % len(teams)]
        if home_name == away_name:
            away_name, away_confed = teams[(index + 2) % len(teams)]
        home_fouls = 9 + index % 6
        away_fouls = 11 + (index * 2) % 7
        home_corners = 3 + index % 6
        away_corners = 2 + (index * 3) % 6
        home_offsides = index % 4
        away_offsides = (index + 2) % 4
        home_possession = 46.0 + index % 10
        rows.append(
            MatchRow(
                source="synthetic",
                source_match_id=str(index),
                match_date=(start + timedelta(days=index * 30)).isoformat(),
                competition="Synthetic Cup",
                competition_type="friendly" if index % 5 == 0 else "qualifier",
                home_team=home_name,
                away_team=away_name,
                home_confederation=home_confed,
                away_confederation=away_confed,
                home_elo=1450 + index % 100,
                away_elo=1500 - index % 80,
                referee_name=f"Ref {index % 5}",
                home_stats=TeamStatsRow(
                    team=home_name,
                    opponent=away_name,
                    is_home=True,
                    confederation=home_confed,
                    fouls=home_fouls,
                    corners=home_corners,
                    offsides=home_offsides,
                    possession=home_possession,
                    first_half_fouls=home_fouls // 2,
                    first_half_corners=home_corners // 2,
                    first_half_offsides=home_offsides // 2,
                    pressing_proxy=10.0 + index % 4,
                ),
                away_stats=TeamStatsRow(
                    team=away_name,
                    opponent=home_name,
                    is_home=False,
                    confederation=away_confed,
                    fouls=away_fouls,
                    corners=away_corners,
                    offsides=away_offsides,
                    possession=100.0 - home_possession,
                    first_half_fouls=away_fouls // 2,
                    first_half_corners=away_corners // 2,
                    first_half_offsides=away_offsides // 2,
                    pressing_proxy=11.0 + index % 3,
                ),
            )
        )
    return rows


def _registry_mapping_matches():
    start = date(2025, 1, 1)
    rows = []
    for index in range(20):
        home_fouls = 10 + index % 3
        away_fouls = 12 + index % 4
        home_corners = 5 + index % 3
        away_corners = 3 + index % 2
        home_offsides = index % 3
        away_offsides = (index + 1) % 3
        rows.append(
            MatchRow(
                source="registry-test",
                source_match_id=str(index),
                match_date=(start + timedelta(days=index * 10)).isoformat(),
                competition="Registry Test Cup",
                competition_type="qualifier",
                home_team="England",
                away_team="Japan",
                home_elo=1780,
                away_elo=1660,
                home_stats=TeamStatsRow(
                    team="England",
                    opponent="Japan",
                    is_home=True,
                    fouls=home_fouls,
                    corners=home_corners,
                    offsides=home_offsides,
                    possession=56.0,
                    first_half_fouls=home_fouls // 2,
                    first_half_corners=home_corners // 2,
                    first_half_offsides=home_offsides // 2,
                ),
                away_stats=TeamStatsRow(
                    team="Japan",
                    opponent="England",
                    is_home=False,
                    fouls=away_fouls,
                    corners=away_corners,
                    offsides=away_offsides,
                    possession=44.0,
                    first_half_fouls=away_fouls // 2,
                    first_half_corners=away_corners // 2,
                    first_half_offsides=away_offsides // 2,
                ),
            )
        )
    return rows


@pytest.fixture()
def fitted(tmp_path):
    database = tmp_path / "props.sqlite"
    ingest_matches(database, _synthetic_matches())
    settings = Settings(
        database_path=str(database),
        raw_cache_dir=str(tmp_path / "raw"),
        artifact_path=str(tmp_path / "model.json"),
        simulations=5_000,
    )
    artifact = fit_model(database, settings, cutoff_date="2026-01-01")
    return database, settings, artifact


def test_registry_confederations_replace_missing_database_labels(tmp_path):
    database = tmp_path / "props.sqlite"
    ingest_matches(database, _registry_mapping_matches())
    settings = Settings(
        database_path=str(database),
        raw_cache_dir=str(tmp_path / "raw"),
        artifact_path=str(tmp_path / "model.json"),
    )
    artifact = fit_model(database, settings, cutoff_date="2026-01-01")

    assert artifact.teams["England"].confederation == "UEFA"
    assert artifact.teams["Japan"].confederation == "AFC"
    assert artifact.metadata["unknown_confederation_share"] == 0.0
    assert artifact.metadata["confederation_audit"]["UEFA"]["training_matches"] == 20
    assert artifact.metadata["confederation_audit"]["AFC"]["training_matches"] == 20
    assert (
        artifact.teams["England"].rate_sources["shots_on_target"]
        == "global-fallback"
    )
    assert (
        artifact.metadata["count_fit_audit"]["shots_on_target"][
            "non_null_training_rows"
        ]
        == 0
    )


def test_unknown_confederation_validation_fails_above_threshold(tmp_path):
    database = tmp_path / "props.sqlite"
    row = MatchRow(
        source="registry-test",
        source_match_id="unknown",
        match_date="2025-01-01",
        competition="Registry Test Cup",
        competition_type="qualifier",
        home_team="Atlantis",
        away_team="El Dorado",
        home_stats=TeamStatsRow(
            team="Atlantis", opponent="El Dorado", is_home=True, fouls=10
        ),
        away_stats=TeamStatsRow(
            team="El Dorado", opponent="Atlantis", is_home=False, fouls=11
        ),
    )
    ingest_matches(database, [row])
    settings = Settings(
        database_path=str(database),
        raw_cache_dir=str(tmp_path / "raw"),
        artifact_path=str(tmp_path / "model.json"),
    )

    with pytest.raises(ValueError, match="Unmapped team confederations"):
        fit_model(database, settings, cutoff_date="2026-01-01")


def test_club_profiles_become_team_rate_priors(tmp_path):
    database = tmp_path / "props.sqlite"
    ingest_matches(database, _registry_mapping_matches())
    club_csv = tmp_path / "club.csv"
    club_csv.write_text(
        "\n".join(
            [
                "player_name,national_team,club,competition,season,minutes,position,national_role,expected_minutes,likely_starter,shots_per90,shots_on_target_per90,goals_per90,assists_per90,fouls_committed_per90,fouls_drawn_per90,yellow_cards_per90,penalty_taker,set_piece_role,source",
                "England Forward,England,Example FC,England Premier League,2025-2026,1800,forward,forward,80,1,3.0,1.2,0.5,0.2,1.1,1.0,0.1,1,0.4,test",
                "England Mid,England,Example FC,England Premier League,2025-2026,1500,central_mid,central_mid,80,1,1.2,0.4,0.1,0.2,1.4,1.2,0.2,0,0.2,test",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert ingest_player_club_profiles_csv(database, club_csv) == 2
    settings = Settings(
        database_path=str(database),
        raw_cache_dir=str(tmp_path / "raw"),
        artifact_path=str(tmp_path / "model.json"),
        club_prior_matches=24.0,
    )

    artifact = fit_model(database, settings, cutoff_date="2026-01-01")

    assert artifact.teams["England"].rate_sources["shots_on_target"] == "club-blended"
    assert artifact.teams["England"].club_prior_rates["shots_on_target"] > 1.0
    assert artifact.metadata["club_prior_audit"]["teams"]["England"]["stats"] == [
        "cards",
        "corners",
        "fouls",
        "offsides",
        "shots_on_target",
    ]
    profiles, _ = load_match_players(database, "England", "Japan")
    profile = next(
        player for player in profiles["England"] if player.player_name == "England Forward"
    )
    assert profile.shots_on_target_per90 == pytest.approx(1.2 * 1.06)
    assert profile.penalty_taker
    assert profile.set_piece_role == pytest.approx(0.4)


def test_validation_and_artifact_round_trip(fitted, tmp_path):
    database, _, artifact = fitted
    report = validate_database(database)
    assert report.error_count == 0
    assert len(report.coverage) == 4
    path = tmp_path / "artifact.json"
    artifact.save(path)
    loaded = ModelArtifact.load(path)
    assert loaded.global_rates == artifact.global_rates
    assert set(loaded.teams) == set(artifact.teams)


def test_more_than_probabilities_partition_one(fitted):
    _, settings, artifact = fitted
    question = Question(
        home="USA",
        away="Paraguay",
        stat=Stat.OFFSIDES,
        question_type=QuestionType.MORE_THAN,
    )
    result = simulate(artifact, question, settings, simulations=20_000, seed=7)
    assert result.p_home_more + result.p_tie + result.p_away_more == pytest.approx(1.0)
    assert 0.0 <= result.raw_probability <= 1.0


def test_second_half_game_state_and_shots_on_target_are_supported(fitted):
    _, settings, artifact = fitted
    question = Question(
        home="USA",
        away="Paraguay",
        stat=Stat.SHOTS_ON_TARGET,
        question_type=QuestionType.SECOND_HALF_MORE_THAN,
    )
    result = simulate(artifact, question, settings, simulations=5_000, seed=9)
    assert result.metadata["model"] == "segmented_team_negative_binomial"
    assert result.metadata["goal_calibration"]["source"] == "model_only"
    assert result.p_home_more + result.p_tie + result.p_away_more == pytest.approx(1.0)


def test_tournament_context_coasts_secure_big_leader_second_half(fitted):
    _, settings, artifact = fitted
    settings = replace(settings, simulations=20_000)
    question = Question(
        home="USA",
        away="Paraguay",
        stat=Stat.SHOTS_ON_TARGET,
        question_type=QuestionType.SECOND_HALF_MORE_THAN,
    )
    calibration = GoalCalibration(
        lambda_home=3.4,
        lambda_away=0.4,
        source="market_calibrated",
        objective=0.0,
        residuals=[],
        rho=settings.dixon_coles_rho,
        targets_used=4,
    )
    context = MatchTournamentContext(
        home=TeamTournamentContext(
            team="USA",
            qualification_probability=0.96,
            coast_if_leading=True,
        ),
        away=TeamTournamentContext(team="Paraguay"),
    )

    baseline = simulate(
        artifact,
        question,
        settings,
        simulations=20_000,
        seed=17,
        goal_calibration=calibration,
    )
    with_context = simulate(
        artifact,
        question,
        settings,
        simulations=20_000,
        seed=17,
        goal_calibration=calibration,
        tournament_context=context,
    )

    assert with_context.home_counts.mean() < baseline.home_counts.mean() * 0.90
    incentives = with_context.metadata["tournament_incentives"]
    assert incentives["enabled"] is True
    assert incentives["home_coasts_when_leading"] is True
    assert incentives["home_second_half_big_lead_segment_share"] > 0.0


def test_structured_possession_favorite_preserves_second_half_control_volume(fitted):
    _, settings, artifact = fitted
    settings = replace(settings, simulations=20_000)
    question = Question(
        home="USA",
        away="Paraguay",
        stat=Stat.SHOTS_ON_TARGET,
        question_type=QuestionType.SECOND_HALF_MORE_THAN,
    )
    calibration = GoalCalibration(
        lambda_home=3.4,
        lambda_away=0.4,
        source="market_calibrated",
        objective=0.0,
        residuals=[],
        rho=settings.dixon_coles_rho,
        targets_used=4,
    )
    generic_context = MatchTournamentContext(
        home=TeamTournamentContext(
            team="USA",
            qualification_probability=0.96,
            coast_if_leading=True,
            tactical_style="flair_transition",
        ),
        away=TeamTournamentContext(team="Paraguay"),
    )
    structured_context = MatchTournamentContext(
        home=TeamTournamentContext(
            team="USA",
            qualification_probability=0.96,
            coast_if_leading=True,
            tactical_style="structured_possession",
        ),
        away=TeamTournamentContext(team="Paraguay"),
    )

    generic = simulate(
        artifact,
        question,
        settings,
        simulations=20_000,
        seed=18,
        goal_calibration=calibration,
        tournament_context=generic_context,
    )
    structured = simulate(
        artifact,
        question,
        settings,
        simulations=20_000,
        seed=18,
        goal_calibration=calibration,
        tournament_context=structured_context,
    )

    assert structured.home_counts.mean() > generic.home_counts.mean() * 1.15
    incentives = structured.metadata["tournament_incentives"]
    assert incentives["home_coasts_when_leading"] is True
    assert incentives["home_structured_possession_coast"] is True


def test_forecast_uses_team_strength_goal_fallback_when_no_elo_or_market(fitted):
    database, settings, artifact = fitted
    home = artifact.teams["USA"]
    away = artifact.teams["Paraguay"]
    home.rates["shots_on_target"] = 6.2
    home.conceded_rates["shots_on_target"] = 2.8
    away.rates["shots_on_target"] = 2.8
    away.conceded_rates["shots_on_target"] = 5.8
    home.effective_matches = 18.0
    away.effective_matches = 18.0
    question = Question(
        home="USA",
        away="Paraguay",
        stat=Stat.SHOTS_ON_TARGET,
        question_type=QuestionType.MORE_THAN,
    )

    forecast = forecast_question(
        artifact,
        question,
        settings,
        database_path=database,
        simulations=8_000,
        seed=99,
        use_market=False,
    )

    goal_details = forecast.metadata["goal_market_calibration"]["fusion_details"]
    assert goal_details["fallback_source"] == "team_sot_attack_concession"
    assert goal_details["home_goal_share"] > 0.60
    assert forecast.metadata["goal_market_calibration"]["lambda_home"] > (
        forecast.metadata["goal_market_calibration"]["lambda_away"]
    )


def test_possession_ess_gate_preserves_learned_corner_edge(fitted):
    _, settings, artifact = fitted
    home = artifact.teams["USA"]
    away = artifact.teams["Paraguay"]
    home.rates["corners"] = 6.0
    away.rates["corners"] = 4.0
    home.conceded_rates["corners"] = 4.8
    away.conceded_rates["corners"] = 4.8
    home.rate_sample_sizes["corners"] = 24.0
    away.rate_sample_sizes["corners"] = 24.0
    home.possession = 50.0
    away.possession = 65.0
    question = Question(
        home="USA",
        away="Paraguay",
        stat=Stat.CORNERS,
        question_type=QuestionType.SECOND_HALF_MORE_THAN,
    )

    ungated = simulate(
        artifact,
        question,
        replace(settings, use_possession_ess_gating=False),
        simulations=2_000,
        seed=101,
    )
    gated = simulate(
        artifact,
        question,
        replace(settings, use_possession_ess_gating=True),
        simulations=2_000,
        seed=101,
    )

    assert ungated.metadata["territory_split"]["possession_territory_share"] < 0.5
    assert ungated.metadata["dominance_share"] < 0.5
    assert gated.metadata["territory_split"]["learned_rate_share"] == pytest.approx(0.6)
    assert gated.metadata["territory_split"]["rate_weight"] > 0.8
    assert gated.metadata["dominance_share"] > 0.5
    assert "supremacy_absent_model_only_territory_projection" in gated.metadata[
        "warnings"
    ][0]


def test_market_supremacy_corrects_conflicting_territory_split(fitted):
    _, settings, artifact = fitted
    home = artifact.teams["USA"]
    away = artifact.teams["Paraguay"]
    home.rates["corners"] = 4.0
    away.rates["corners"] = 6.0
    home.conceded_rates["corners"] = 4.8
    away.conceded_rates["corners"] = 4.8
    home.possession = 50.0
    away.possession = 65.0
    question = Question(
        home="USA",
        away="Paraguay",
        stat=Stat.CORNERS,
        question_type=QuestionType.MORE_THAN,
    )
    calibration = GoalCalibration(
        1.45,
        0.95,
        "market_calibrated",
        0.0,
        [],
        -0.08,
        4,
    )

    result = simulate(
        artifact,
        question,
        settings,
        simulations=2_000,
        seed=101,
        goal_calibration=calibration,
    )

    split = result.metadata["territory_split"]
    assert split["market_split_applied"] is True
    assert split["market_supremacy_share"] > 0.5
    assert result.metadata["dominance_share"] > split["possession_territory_share"]


def test_shared_latent_flow_correlates_dominant_team_counts(fitted):
    _, settings, artifact = fitted
    question = Question(
        home="USA",
        away="Paraguay",
        stat=Stat.CORNERS,
        question_type=QuestionType.MORE_THAN,
        home_elo=1800,
        away_elo=1550,
    )
    joint = simulate_joint_match(
        artifact,
        question,
        replace(settings, latent_flow_method="shared_normal"),
        simulations=20_000,
        seed=19,
        goal_calibration=GoalCalibration(
            1.9, 0.6, "market_calibrated", 0.0, [], -0.08, 4
        ),
    )
    correlations = joint.metadata["home_stat_correlation"]
    assert correlations[1][2] > 0.05
    assert correlations[1][3] > 0.0
    assert correlations[0][2] > 0.0


def test_foul_total_then_split_and_threshold_forecast(fitted):
    database, settings, artifact = fitted
    comparison = Question(
        home="South Africa",
        away="Mexico",
        stat=Stat.FOULS,
        question_type=QuestionType.MORE_THAN,
        referee="Ref 1",
    )
    result = simulate(artifact, comparison, settings, simulations=10_000, seed=11)
    assert result.metadata["referee_mode"] == "known"
    assert (result.home_counts + result.away_counts >= 0).all()

    threshold = Question(
        home="USA",
        away="Paraguay",
        stat=Stat.OFFSIDES,
        question_type=QuestionType.THRESHOLD,
        k=2,
    )
    forecast = forecast_question(
        artifact,
        threshold,
        settings,
        database_path=database,
        simulations=10_000,
        seed=13,
    )
    assert 0.0 <= forecast.probability <= 1.0
    assert forecast.market_probability is None
    assert forecast.effective_sample_size_home > 0


def test_coherent_player_card_and_rare_event_accounting(fitted):
    _, settings, artifact = fitted
    question = Question(
        home="USA",
        away="Paraguay",
        stat=Stat.SHOTS_ON_TARGET,
        question_type=QuestionType.MORE_THAN,
        home_elo=1750,
        away_elo=1550,
    )
    profiles = {
        "USA": [
            PlayerProfile("USA Forward", "USA", 3.0, 1.3, 0.55, 0.18, 6.0, True, 0.0)
        ],
        "Paraguay": [
            PlayerProfile(
                "Paraguay Forward",
                "Paraguay",
                2.4,
                0.9,
                0.35,
                0.12,
                4.5,
                False,
                0.0,
            )
        ],
    }
    lineups = {
        "USA": [],
        "Paraguay": [],
    }
    result = simulate_coherent_match(
        artifact,
        question,
        settings,
        profiles=profiles,
        lineups=lineups,
        simulations=20_000,
        seed=23,
        referee_cards_per_match=5.2,
    )
    for team, player in (("USA", "USA Forward"), ("Paraguay", "Paraguay Forward")):
        allocated_goals = (
            result.player_events[player]["goals"]
            + result.player_events[f"{team}::__other__"]["goals"]
        )
        team_goals = (
            result.home_segments["goals"].sum(axis=1)
            if team == "USA"
            else result.away_segments["goals"].sum(axis=1)
        )
        assert (allocated_goals == team_goals).all()
    assert result.metadata["accounting"]["cards_foul_correlation"] > 0.0
    assert result.metadata["accounting"]["penalty_or_red_foul_correlation"] > 0.0


def test_confirmed_lineup_changes_player_probability(fitted):
    _, settings, artifact = fitted
    question = Question(
        home="USA",
        away="Paraguay",
        stat=Stat.SHOTS_ON_TARGET,
        question_type=QuestionType.MORE_THAN,
    )
    profile = PlayerProfile(
        "USA Forward", "USA", 3.2, 1.4, 0.5, 0.2, 6.0, False, 0.0
    )
    uncertain = LineupEntry(
        "usa|paraguay", "USA", "USA Forward", "uncertain", 0.20, 75.0, 0.20, 22.0, False
    )
    confirmed = LineupEntry(
        "usa|paraguay", "USA", "USA Forward", "starter", 1.0, 82.0, 0.0, 22.0, True
    )
    kwargs = {
        "artifact": artifact,
        "question": question,
        "settings": settings,
        "profiles": {"USA": [profile], "Paraguay": []},
        "simulations": 20_000,
        "seed": 29,
    }
    pre = simulate_coherent_match(
        lineups={"USA": [uncertain], "Paraguay": []}, **kwargs
    )
    post = simulate_coherent_match(
        lineups={"USA": [confirmed], "Paraguay": []}, **kwargs
    )
    pre_probability = (
        pre.player_events["USA Forward"]["shots_on_target"] >= 1
    ).mean()
    post_probability = (
        post.player_events["USA Forward"]["shots_on_target"] >= 1
    ).mean()
    assert post_probability > pre_probability


def test_confirmed_lineup_unlisted_profiles_do_not_steal_player_events(fitted):
    _, settings, artifact = fitted
    question = Question(
        home="USA",
        away="Paraguay",
        stat=Stat.SHOTS_ON_TARGET,
        question_type=QuestionType.MORE_THAN,
    )
    starter = PlayerProfile(
        "USA Forward", "USA", 3.2, 1.4, 0.5, 0.2, 6.0, False, 0.0
    )
    unlisted = PlayerProfile(
        "USA Bench Forward", "USA", 4.5, 2.0, 0.7, 0.1, 6.0, False, 0.0
    )
    confirmed = LineupEntry(
        "usa|paraguay", "USA", "USA Forward", "starter", 1.0, 82.0, 0.0, 22.0, True
    )

    result = simulate_coherent_match(
        artifact,
        question,
        settings,
        profiles={"USA": [starter, unlisted], "Paraguay": []},
        lineups={"USA": [confirmed], "Paraguay": []},
        simulations=12_000,
        seed=31,
    )

    assert result.metadata["lineups"]["USA"]["USA Bench Forward"]["status"] == (
        "unlisted_confirmed_lineup"
    )
    assert result.player_events["USA Bench Forward"]["shots_on_target"].sum() == 0


def test_sparse_matchup_is_automatically_shrunk_toward_base_rate(fitted):
    database, settings, artifact = fitted
    question = Question(
        home="Unknown Elite Team",
        away="Cape Verde",
        stat=Stat.CORNERS,
        question_type=QuestionType.SECOND_HALF_MORE_THAN,
        home_elo=2150,
        away_elo=1550,
    )
    forecast = forecast_question(
        artifact,
        question,
        settings,
        database_path=database,
        simulations=10_000,
        seed=37,
        use_market=False,
    )
    guard = forecast.metadata["coverage_guard"]
    assert guard["enabled"]
    assert guard["low_coverage"]
    assert not guard["home_team_in_model"]
    assert abs(forecast.model_probability - 0.5) < abs(
        forecast.raw_model_probability - 0.5
    )


def test_extreme_supremacy_is_capped_for_count_stats(fitted):
    _, settings, artifact = fitted
    question = Question(
        home="USA",
        away="Paraguay",
        stat=Stat.SHOTS_ON_TARGET,
        question_type=QuestionType.MORE_THAN,
        home_elo=2200,
        away_elo=1400,
    )
    result = simulate(
        artifact,
        question,
        settings,
        simulations=10_000,
        seed=41,
        goal_calibration=GoalCalibration(
            5.2, 0.15, "market_calibrated", 0.0, [], -0.08, 4
        ),
    )
    assert result.metadata["dominance_share"] <= 0.84
    assert result.metadata["home_rate"] <= (
        artifact.global_rates["shots_on_target"] * 2.0
    )
    assert result.metadata["market_volume_multiplier"] <= 1.35


def test_confirmed_attacking_starter_sot_does_not_shrink_to_zero(fitted):
    database, settings, artifact = fitted
    with transaction(database) as connection:
        connection.execute(
            """
            INSERT INTO player_profiles (
                player_name, team, shots_per90, shots_on_target_per90,
                goals_per90, assists_per90, box_touches_per90, penalty_taker,
                set_piece_role, player_role, effective_matches, source, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "Thin Winger",
                "USA",
                0.20,
                0.02,
                0.01,
                0.02,
                None,
                0,
                0.0,
                "winger",
                0.0,
                "thin_manual",
                "2026-01-01T00:00:00+00:00",
            ),
        )
        connection.execute(
            """
            INSERT INTO lineup_entries (
                match_key, team, player_name, status, start_probability,
                expected_start_minutes, sub_probability, expected_sub_minutes,
                confirmed, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "usa|paraguay",
                "USA",
                "Thin Winger",
                "starter",
                1.0,
                82.0,
                0.0,
                0.0,
                1,
                "2026-01-01T00:00:00+00:00",
            ),
        )
    question = Question(
        home="USA",
        away="Paraguay",
        stat=Stat.SHOTS_ON_TARGET,
        question_type=QuestionType.MORE_THAN,
    )
    forecast = forecast_player_event(
        artifact,
        question,
        "Thin Winger",
        "USA",
        "shots_on_target",
        settings,
        database,
        simulations=10_000,
        seed=43,
        use_market=False,
    )
    floor = forecast["metadata"]["player_prior"]["anti_zero_floor"]
    assert floor["applied"]
    assert forecast["model_probability"] >= floor["floor"]
    assert forecast["model_probability"] >= 0.25


def test_confirmed_focal_attacker_goal_assist_uses_team_lambda_pathway(fitted):
    database, settings, artifact = fitted
    with transaction(database) as connection:
        for name, goals, assists, role in (
            ("Focal Winger", 0.55, 0.20, "winger"),
            ("Support Forward", 0.55, 0.20, "forward"),
            ("Creator Mid", 0.30, 0.20, "attacking_mid"),
            ("Runner Mid", 0.20, 0.10, "central_mid"),
            ("Fullback", 0.10, 0.10, "fullback"),
        ):
            connection.execute(
                """
                INSERT INTO player_profiles (
                    player_name, team, shots_per90, shots_on_target_per90,
                    goals_per90, assists_per90, box_touches_per90, penalty_taker,
                    set_piece_role, player_role, effective_matches, source, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    name,
                    "USA",
                    2.8,
                    1.1,
                    goals,
                    assists,
                    None,
                    0,
                    0.0,
                    role,
                    34.0,
                    "test",
                    "2026-01-01T00:00:00+00:00",
                ),
            )
            connection.execute(
                """
                INSERT INTO lineup_entries (
                    match_key, team, player_name, status, start_probability,
                    expected_start_minutes, sub_probability, expected_sub_minutes,
                    confirmed, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "usa|paraguay",
                    "USA",
                    name,
                    "starter",
                    1.0,
                    82.0,
                    0.0,
                    0.0,
                    1,
                    "2026-01-01T00:00:00+00:00",
                ),
            )
    question = Question(
        home="USA",
        away="Paraguay",
        stat=Stat.SHOTS_ON_TARGET,
        question_type=QuestionType.MORE_THAN,
        home_elo=1400,
        away_elo=1555,
    )

    forecast = forecast_player_event(
        artifact,
        question,
        "Focal Winger",
        "USA",
        "goal_or_assist",
        settings,
        database,
        simulations=20_000,
        seed=44,
        use_market=False,
    )

    pathway = forecast["metadata"]["player_prior"]["goal_assist_pathway"]
    assert pathway["focus_bucket"] == "focal"
    assert pathway["minutes_fraction"] == pytest.approx(1.0)
    assert 0.30 <= forecast["model_probability"] <= 0.38


def test_penalty_taker_focal_attacker_gets_goal_assist_lift(fitted):
    database, settings, artifact = fitted
    with transaction(database) as connection:
        for name, penalty_taker in (
            ("Penalty Focal", 1),
            ("Open Play Focal", 0),
            ("Support Forward", 0),
            ("Creator Mid", 0),
        ):
            connection.execute(
                """
                INSERT INTO player_profiles (
                    player_name, team, shots_per90, shots_on_target_per90,
                    goals_per90, assists_per90, box_touches_per90, penalty_taker,
                    set_piece_role, player_role, effective_matches, source, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    name,
                    "USA",
                    2.8,
                    1.1,
                    0.50,
                    0.18,
                    None,
                    penalty_taker,
                    0.0,
                    "forward",
                    30.0,
                    "test",
                    "2026-01-01T00:00:00+00:00",
                ),
            )
            connection.execute(
                """
                INSERT INTO lineup_entries (
                    match_key, team, player_name, status, start_probability,
                    expected_start_minutes, sub_probability, expected_sub_minutes,
                    confirmed, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "usa|paraguay",
                    "USA",
                    name,
                    "starter",
                    1.0,
                    82.0,
                    0.0,
                    0.0,
                    1,
                    "2026-01-01T00:00:00+00:00",
                ),
            )
    question = Question(
        home="USA",
        away="Paraguay",
        stat=Stat.SHOTS_ON_TARGET,
        question_type=QuestionType.MORE_THAN,
        home_elo=1400,
        away_elo=1555,
    )

    penalty = forecast_player_event(
        artifact,
        question,
        "Penalty Focal",
        "USA",
        "goal_or_assist",
        settings,
        database,
        simulations=20_000,
        seed=45,
        use_market=False,
    )
    open_play = forecast_player_event(
        artifact,
        question,
        "Open Play Focal",
        "USA",
        "goal_or_assist",
        settings,
        database,
        simulations=20_000,
        seed=45,
        use_market=False,
    )

    assert penalty["model_probability"] > open_play["model_probability"]
    assert (
        penalty["metadata"]["player_prior"]["goal_assist_pathway"]["penalty_lambda"]
        > 0.0
    )


def test_liquid_match_winner_market_dominates_foundation_prop(fitted, tmp_path):
    database, settings, artifact = fitted
    market_csv = tmp_path / "markets.csv"
    market_csv.write_text(
        "match,market,selection,probability,book,timestamp,line\n"
        "USA vs Paraguay,1X2,USA,0.47,Consensus,2026-01-01T00:00:00Z,\n"
        "USA vs Paraguay,1X2,Draw,0.29,Consensus,2026-01-01T00:00:00Z,\n"
        "USA vs Paraguay,1X2,Paraguay,0.24,Consensus,2026-01-01T00:00:00Z,\n",
        encoding="utf-8",
    )
    assert ingest_market_csv(database, market_csv) == 3
    question = Question(
        home="USA",
        away="Paraguay",
        stat=Stat.SHOTS_ON_TARGET,
        question_type=QuestionType.MORE_THAN,
    )

    forecast = forecast_match_event(
        artifact,
        question,
        "home_win",
        settings,
        database,
        simulations=20_000,
        seed=46,
        use_market=True,
    )

    assert forecast["market_probability"] == pytest.approx(0.47)
    assert forecast["metadata"]["market_blend_weight"] == pytest.approx(1.0)
    assert forecast["probability"] == pytest.approx(0.47)


def test_no_market_fade_player_prop_passes_through_unchanged(fitted):
    database, settings, artifact = fitted
    with transaction(database) as connection:
        connection.execute(
            """
            INSERT INTO player_profiles (
                player_name, team, shots_per90, shots_on_target_per90,
                goals_per90, assists_per90, box_touches_per90, penalty_taker,
                set_piece_role, player_role, effective_matches, source, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "Fade Defender",
                "USA",
                0.10,
                0.02,
                0.01,
                0.01,
                None,
                0,
                0.0,
                "center_back",
                30.0,
                "test",
                "2026-01-01T00:00:00+00:00",
            ),
        )
    question = Question(
        home="USA",
        away="Paraguay",
        stat=Stat.SHOTS_ON_TARGET,
        question_type=QuestionType.MORE_THAN,
    )

    forecast = forecast_player_event(
        artifact,
        question,
        "Fade Defender",
        "USA",
        "shots_on_target",
        settings,
        database,
        simulations=20_000,
        seed=47,
        use_market=True,
    )

    assert forecast["market_probability"] is None
    assert forecast["metadata"]["market_blend_weight"] == 0.0
    assert forecast["probability"] == pytest.approx(forecast["model_probability"])


def test_fbref_schedule_builds_populated_manifest():
    html = """
    <table>
      <tr>
        <th data-stat="date" csk="2022-11-21">2022-11-21</th>
        <td data-stat="home_team">USA</td>
        <td data-stat="away_team">Wales</td>
        <td data-stat="match_report">
          <a href="/en/matches/abc123/USA-Wales-November-21-2022-World-Cup">
            Match Report
          </a>
        </td>
      </tr>
    </table>
    """
    rows = parse_fbref_schedule(
        html,
        competition="FIFA World Cup",
        competition_type="world_cup",
    )
    assert len(rows) == 1
    assert rows[0]["source_match_id"] == "abc123"
    assert rows[0]["home_team"] == "USA"
    assert rows[0]["url"].startswith("https://fbref.com/en/matches/")
