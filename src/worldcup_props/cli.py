from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from .backtest import fit_final_model_with_calibration, run_ablation, run_backtest
from .config import Settings
from .data import (
    FBrefManifestProvider,
    SoccerdataFBrefProvider,
    StatsHubProvider,
    build_fbref_manifest,
    ingest_matches,
    ingest_referees_csv,
    ingest_statshub_referees,
    rows_from_csv,
)
from .db import initialize
from .domain import Question, QuestionType, Stat
from .evaluation import ingest_results_csv, log_result, results_report
from .forecast import forecast_match_event, forecast_player_event, forecast_question
from .fifa_dataset import ingest_fifa_world_cup_dataset
from .market import OddsAPIClient, ingest_market_csv, ingest_odds_api
from .market_ingest import SearchResult, ingest_market_profile, summarize_market_profile
from .model import ModelArtifact, fit_model
from .players import (
    ingest_lineups_csv,
    ingest_player_club_profile_rows,
    ingest_player_club_profiles_csv,
    ingest_player_profiles_csv,
)
from .registry import load_team_confederation_registry
from .tournament import ingest_tournament_context_csv
from .validation import validate_database


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="worldcup-props",
        description="Calibrated forecasts for international football match-stat props",
    )
    parser.add_argument("--config", help="Path to JSON configuration")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init-db", help="Create the SQLite schema")

    ingest = subparsers.add_parser("ingest-csv", help="Ingest normalized match statistics")
    ingest.add_argument("path")
    ingest.add_argument("--source", default="csv")

    refs = subparsers.add_parser("ingest-referees", help="Ingest manual referee statistics")
    refs.add_argument("path")

    player_profiles = subparsers.add_parser(
        "ingest-player-profiles", help="Ingest club-derived player per-90 profiles"
    )
    player_profiles.add_argument("path")

    player_club_profiles = subparsers.add_parser(
        "ingest-player-club-profiles",
        help="Ingest raw club-season player profiles for national-team priors",
    )
    player_club_profiles.add_argument("path")

    statshub_player_profiles = subparsers.add_parser(
        "ingest-statshub-player-club-profiles",
        help="Populate player profile priors from StatsHub player performance logs",
    )
    statshub_player_profiles.add_argument(
        "--teams", default="data/statshub_teams.csv", help="StatsHub team registry CSV"
    )
    statshub_player_profiles.add_argument("--limit", type=int, default=50)
    statshub_player_profiles.add_argument("--min-minutes", type=float, default=90.0)
    statshub_player_profiles.add_argument("--fixture-id", type=int)

    lineups = subparsers.add_parser(
        "ingest-lineups", help="Ingest pre-match probabilities or confirmed lineups"
    )
    lineups.add_argument("path")

    tournament_context = subparsers.add_parser(
        "ingest-tournament-context",
        help="Ingest group-table and incentive context for upcoming matches",
    )
    tournament_context.add_argument("path")

    fifa_dataset = subparsers.add_parser(
        "ingest-fifa-world-cup-dataset",
        help="Ingest mominullptr/FIFA-World-Cup-2026-Dataset match and context data",
    )
    fifa_dataset.add_argument(
        "path",
        help="Local checkout or extracted directory containing matches.csv",
    )

    markets = subparsers.add_parser("ingest-markets", help="Ingest definition-matched odds")
    markets.add_argument("path")

    dk = subparsers.add_parser(
        "ingest-draftkings-har",
        help="Parse a DraftKings sportsbook HAR capture into de-vigged match markets",
    )
    dk.add_argument("har", help="Path to the DraftKings HAR exported from the browser")
    dk.add_argument(
        "--csv-out",
        default=None,
        help="Optional path to also write the normalized match-market CSV",
    )

    odds_api = subparsers.add_parser(
        "ingest-odds-api", help="Fetch and ingest match markets from The Odds API"
    )
    odds_api.add_argument("--sport", default="soccer_fifa_world_cup")
    odds_api.add_argument("--regions", default="us,uk,eu")
    odds_api.add_argument(
        "--markets",
        default="h2h,totals,spreads,btts,player_goal_scorer",
    )
    lineup_refresh = subparsers.add_parser(
        "refresh-lineup-markets",
        help="Re-pull odds after confirmed lineups and ingest the refreshed prices",
    )
    lineup_refresh.add_argument("--sport", default="soccer_fifa_world_cup")
    lineup_refresh.add_argument("--regions", default="us,uk,eu")
    lineup_refresh.add_argument(
        "--markets",
        default="h2h,totals,spreads,btts,player_goal_scorer",
    )

    market_profile = subparsers.add_parser(
        "market-profile",
        help="Build an auditable market profile from captured search results JSON",
    )
    _add_match_arguments(market_profile)
    market_profile.add_argument(
        "--search-results",
        required=True,
        help="JSON list of {title,url,snippet} search results captured by the agent",
    )
    market_profile.add_argument(
        "--player",
        action="append",
        default=[],
        help="Flagged player to search/parse for anytime goalscorer odds",
    )
    market_profile.add_argument("--json", action="store_true", help="Emit JSON instead of text")

    fbref = subparsers.add_parser(
        "fbref-export", help="Export raw soccerdata/FBref tables for normalization"
    )
    fbref.add_argument("--league", action="append", required=True)
    fbref.add_argument("--season", action="append", required=True)
    fbref.add_argument("--output-dir", default="data/raw/fbref_exports")

    scrape_fbref = subparsers.add_parser(
        "ingest-fbref", help="Ingest cached FBref match reports from a manifest"
    )
    scrape_fbref.add_argument("manifest")

    manifest = subparsers.add_parser(
        "build-fbref-manifest",
        help="Generate a populated manifest from an FBref schedule page",
    )
    manifest.add_argument("schedule_url")
    manifest.add_argument("output")
    manifest.add_argument("--competition", required=True)
    manifest.add_argument("--competition-type", required=True)
    manifest.add_argument("--non-neutral", action="store_true")

    statshub = subparsers.add_parser(
        "ingest-statshub", help="Ingest national-team statistics from StatsHub"
    )
    statshub.add_argument(
        "--teams", default="data/statshub_teams.csv", help="StatsHub team registry CSV"
    )
    statshub.add_argument("--limit", type=int, default=500)
    statshub.add_argument(
        "--skip-referees", action="store_true", help="Do not refresh referee aggregates"
    )

    discover = subparsers.add_parser(
        "statshub-discover", help="Discover teams in StatsHub's World Cup fixture feed"
    )
    discover.add_argument("--season-id", type=int, default=58210)
    discover.add_argument("--tournament-id", type=int, default=16)
    discover.add_argument("--output", default="data/statshub_discovered_teams.csv")

    validate = subparsers.add_parser("validate", help="Run data quality checks")
    validate.add_argument("--output")

    train = subparsers.add_parser("train", help="Fit and save the forecasting model")
    train.add_argument("--cutoff")
    train.add_argument("--artifact")
    train.add_argument("--calibrate", action="store_true")
    train.add_argument("--backtest-report", default="artifacts/backtest.json")
    train.add_argument("--backtest-simulations", type=int, default=10_000)

    backtest = subparsers.add_parser("backtest", help="Run temporal tournament backtests")
    backtest.add_argument("--tournament", action="append")
    backtest.add_argument("--simulations", type=int, default=10_000)
    backtest.add_argument("--output", default="artifacts/backtest.json")

    ablation = subparsers.add_parser(
        "ablation", help="Run cumulative held-out Brier feature ablations"
    )
    ablation.add_argument("--tournament", action="append")
    ablation.add_argument("--simulations", type=int, default=5_000)
    ablation.add_argument("--output", default="artifacts/ablation.json")

    predict = subparsers.add_parser("predict", help="Forecast one contest question")
    _add_question_arguments(predict)
    predict.add_argument("--artifact")
    predict.add_argument("--simulations", type=int)
    predict.add_argument("--seed", type=int)
    predict.add_argument("--explain", action="store_true")
    _add_market_toggle(predict)

    predict_player = subparsers.add_parser(
        "predict-player", help="Forecast an in-simulator player event"
    )
    _add_match_arguments(predict_player)
    predict_player.add_argument("--team", required=True)
    predict_player.add_argument("--player", required=True)
    predict_player.add_argument(
        "--event",
        choices=[
            "shots",
            "shots_on_target",
            "second_half_shots_on_target",
            "goals",
            "assists",
            "goal_or_assist",
        ],
        required=True,
    )
    predict_player.add_argument("--k", type=int, default=1)
    predict_player.add_argument("--artifact")
    predict_player.add_argument("--simulations", type=int)
    predict_player.add_argument("--seed", type=int)
    predict_player.add_argument("--explain", action="store_true")
    _add_market_toggle(predict_player)

    predict_event = subparsers.add_parser(
        "predict-event", help="Forecast a coherent penalty/red-card match event"
    )
    _add_match_arguments(predict_event)
    predict_event.add_argument(
        "--event",
        choices=[
            "penalty_awarded",
            "red_card_shown",
            "penalty_or_red",
            "home_win",
            "away_win",
            "draw",
            "under_2_5_goals",
            "second_half_more_goals",
        ],
        required=True,
    )
    predict_event.add_argument("--artifact")
    predict_event.add_argument("--simulations", type=int)
    predict_event.add_argument("--seed", type=int)
    predict_event.add_argument("--explain", action="store_true")
    _add_market_toggle(predict_event)

    batch = subparsers.add_parser("batch", help="Forecast questions from a CSV")
    batch.add_argument("input")
    batch.add_argument("output")
    batch.add_argument("--artifact")
    batch.add_argument("--simulations", type=int)
    batch.add_argument("--seed", type=int)
    _add_market_toggle(batch)

    card = subparsers.add_parser(
        "forecast-card",
        help="Route a full card of questions through the closed-form + sub-discount engines",
    )
    card.add_argument("--home", required=True)
    card.add_argument("--away", required=True)
    card.add_argument("--lambda-home", type=float, required=True, help="Market goal expectancy, home")
    card.add_argument("--lambda-away", type=float, required=True, help="Market goal expectancy, away")
    card.add_argument(
        "--question", action="append", default=[], help="A question (repeatable)"
    )
    card.add_argument("--questions", help="Path to a file with one question per line")
    card.add_argument(
        "--lineup",
        help='JSON of confirmed statuses: {"Player Name": "sub"} or '
        '{"Player Name": {"status": "sub", "role": "forward"}}',
    )
    card.add_argument(
        "--llm",
        action="store_true",
        help="Use the LLM question parser for arbitrary wording (needs anthropic + ANTHROPIC_API_KEY)",
    )
    card.add_argument("--model", default="claude-haiku-4-5", help="Model for the LLM parser")
    card.add_argument("--json", action="store_true", help="Emit JSON instead of a table")

    results = subparsers.add_parser("ingest-results", help="Ingest post-match result log CSV")
    results.add_argument("path")

    result = subparsers.add_parser("log-result", help="Log one resolved contest question")
    result.add_argument("--match", required=True)
    result.add_argument("--question", required=True)
    result.add_argument("--question-type", required=True)
    result.add_argument("--submitted", type=float, required=True)
    result.add_argument("--crowd", type=float)
    result.add_argument("--outcome", type=int, choices=[0, 1], required=True)
    result.add_argument("--market-blended", type=float)
    result.add_argument("--weight", type=float, default=1.0)
    result.add_argument("--timestamp")

    report = subparsers.add_parser("results-report", help="Report Brier results and crowd bias")
    report.add_argument("--output")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = Path.cwd()
    settings = Settings.load(args.config).resolve(root)
    database_path = settings.database_path

    if args.command == "init-db":
        initialize(database_path)
        print(database_path)
        return 0
    if args.command == "ingest-csv":
        count = ingest_matches(database_path, rows_from_csv(args.path, args.source))
        print(json.dumps({"matches_ingested": count, "database": database_path}))
        return 0
    if args.command == "ingest-referees":
        count = ingest_referees_csv(database_path, args.path)
        print(json.dumps({"referees_ingested": count}))
        return 0
    if args.command == "ingest-player-profiles":
        count = ingest_player_profiles_csv(database_path, args.path)
        print(json.dumps({"player_profiles_ingested": count}))
        return 0
    if args.command == "ingest-player-club-profiles":
        count = ingest_player_club_profiles_csv(database_path, args.path)
        print(json.dumps({"player_club_profiles_ingested": count}))
        return 0
    if args.command == "ingest-statshub-player-club-profiles":
        provider = StatsHubProvider(
            args.teams, settings.raw_cache_dir, limit=args.limit
        )
        rows = provider.fetch_player_profile_rows(
            limit=args.limit,
            min_minutes=args.min_minutes,
            fixture_id=args.fixture_id,
        )
        count = ingest_player_club_profile_rows(database_path, rows)
        print(
            json.dumps(
                {
                    "player_club_profiles_ingested": count,
                    "teams": len(provider.teams),
                    "source": "statshub_player_performance",
                }
            )
        )
        return 0
    if args.command == "ingest-lineups":
        count = ingest_lineups_csv(database_path, args.path)
        print(json.dumps({"lineup_entries_ingested": count}))
        return 0
    if args.command == "ingest-tournament-context":
        count = ingest_tournament_context_csv(database_path, args.path)
        print(json.dumps({"tournament_context_rows_ingested": count}))
        return 0
    if args.command == "ingest-fifa-world-cup-dataset":
        result = ingest_fifa_world_cup_dataset(database_path, args.path)
        print(json.dumps(result.__dict__, indent=2, sort_keys=True))
        return 0
    if args.command == "ingest-markets":
        count = ingest_market_csv(database_path, args.path)
        print(json.dumps({"quotes_ingested": count}))
        return 0
    if args.command == "ingest-draftkings-har":
        from .draftkings import parse_har_file, write_market_csv

        result = parse_har_file(args.har)
        csv_path = args.csv_out or str(
            Path(settings.raw_cache_dir) / f"draftkings_{result.event.event_id or 'event'}.csv"
        )
        write_market_csv(result.rows, csv_path)
        count = ingest_market_csv(database_path, csv_path)
        print(
            json.dumps(
                {
                    "match": result.event.name,
                    "quotes_ingested": count,
                    "rows": len(result.rows),
                    "skipped": result.skipped,
                    "csv": csv_path,
                },
                indent=2,
            )
        )
        return 0
    if args.command in {"ingest-odds-api", "refresh-lineup-markets"}:
        client = OddsAPIClient(settings.raw_cache_dir)
        payload = client.odds(
            args.sport,
            [value.strip() for value in args.markets.split(",") if value.strip()],
            regions=args.regions,
            force_refresh=args.command == "refresh-lineup-markets",
        )
        count = ingest_odds_api(database_path, payload)
        print(
            json.dumps(
                {
                    "quotes_ingested": count,
                    "events": len(payload),
                    "lineup_news_refresh": args.command == "refresh-lineup-markets",
                }
            )
        )
        return 0
    if args.command == "market-profile":
        search_results = json.loads(Path(args.search_results).read_text(encoding="utf-8"))

        class CapturedSearchClient:
            def __init__(self, results: list[dict[str, Any]]) -> None:
                self.results = [SearchResult.model_validate(result) for result in results]

            def search(self, query: str) -> list[SearchResult]:
                return self.results

        profile = ingest_market_profile(
            args.home,
            args.away,
            CapturedSearchClient(search_results),
            flagged_players=args.player,
        )
        if args.json:
            print(profile.model_dump_json(indent=2))
        else:
            print(summarize_market_profile(profile))
        return 0
    if args.command == "fbref-export":
        provider = SoccerdataFBrefProvider(
            args.league, args.season, settings.raw_cache_dir
        )
        paths = provider.export_raw(args.output_dir)
        print(json.dumps({"exports": [str(path) for path in paths]}, indent=2))
        return 0
    if args.command == "ingest-fbref":
        provider = FBrefManifestProvider(args.manifest, settings.raw_cache_dir)
        count = ingest_matches(database_path, provider.fetch(settings.start_date))
        print(json.dumps({"matches_ingested": count, "source": "fbref"}))
        return 0
    if args.command == "build-fbref-manifest":
        count = build_fbref_manifest(
            args.schedule_url,
            args.output,
            settings.raw_cache_dir,
            competition=args.competition,
            competition_type=args.competition_type,
            neutral=not args.non_neutral,
        )
        print(json.dumps({"manifest": args.output, "matches": count}))
        return 0
    if args.command == "ingest-statshub":
        provider = StatsHubProvider(
            args.teams, settings.raw_cache_dir, limit=args.limit
        )
        count = ingest_matches(database_path, provider.fetch(settings.start_date))
        referee_count = 0
        if not args.skip_referees:
            referee_count = ingest_statshub_referees(
                database_path, provider.fetch_referees()
            )
        print(
            json.dumps(
                {
                    "matches_ingested": count,
                    "referees_ingested": referee_count,
                    "source": "statshub",
                }
            )
        )
        return 0
    if args.command == "statshub-discover":
        provider = StatsHubProvider(
            "data/statshub_teams.csv", settings.raw_cache_dir
        )
        teams = provider.discover_world_cup_teams(
            season_id=args.season_id,
            unique_tournament_id=args.tournament_id,
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as handle:
            registry = load_team_confederation_registry(settings.team_registry_path)
            writer = csv.DictWriter(
                handle,
                fieldnames=["team", "statshub_team_id", "confederation"],
            )
            writer.writeheader()
            for team in teams:
                team_name = str(team["name"])
                writer.writerow(
                    {
                        "team": team_name,
                        "statshub_team_id": team["id"],
                        "confederation": registry.confederation_for(team_name) or "",
                    }
                )
        print(json.dumps({"teams_discovered": len(teams), "output": str(output)}))
        return 0
    if args.command == "validate":
        report = validate_database(database_path)
        result = {
            "error_count": report.error_count,
            "issues": report.issues,
            "coverage": report.coverage,
        }
        _write_or_print(result, args.output)
        return 1 if report.error_count else 0
    if args.command == "train":
        artifact_path = args.artifact or settings.artifact_path
        if args.calibrate:
            artifact = fit_final_model_with_calibration(
                database_path,
                settings,
                args.backtest_report,
                simulations=args.backtest_simulations,
            )
        else:
            artifact = fit_model(database_path, settings, cutoff_date=args.cutoff)
        artifact.save(artifact_path)
        print(
            json.dumps(
                {
                    "artifact": artifact_path,
                    "fitted_through": artifact.fitted_through,
                    "training_matches": artifact.metadata["training_matches"],
                    "unknown_confederation_share": artifact.metadata.get(
                        "unknown_confederation_share"
                    ),
                    "confederation_audit": artifact.metadata.get(
                        "confederation_audit"
                    ),
                    "count_fit_warnings": artifact.metadata.get(
                        "count_fit_audit", {}
                    ).get("warnings", []),
                    "calibrated": bool(artifact.calibration_lambda),
                }
            )
        )
        return 0
    if args.command == "backtest":
        report, _, _ = run_backtest(
            database_path,
            settings,
            tournaments=args.tournament,
            simulations=args.simulations,
        )
        _write_or_print(report, args.output)
        return 0
    if args.command == "ablation":
        report = run_ablation(
            database_path,
            settings,
            tournaments=args.tournament,
            simulations=args.simulations,
        )
        _write_or_print(report, args.output)
        return 0
    if args.command == "predict":
        artifact = ModelArtifact.load(args.artifact or settings.artifact_path)
        question = _question_from_namespace(args)
        forecast = forecast_question(
            artifact,
            question,
            settings,
            database_path=database_path,
            simulations=args.simulations,
            seed=args.seed,
            use_market=args.use_market,
        )
        output = forecast.as_dict()
        if args.explain:
            output["explanation"] = _explain_forecast(output)
        print(json.dumps(output, indent=2, sort_keys=True))
        return 0
    if args.command == "predict-player":
        artifact = ModelArtifact.load(args.artifact or settings.artifact_path)
        question = _match_question_from_namespace(args)
        output = forecast_player_event(
            artifact,
            question,
            args.player,
            args.team,
            args.event,
            settings,
            database_path,
            k=args.k,
            simulations=args.simulations,
            seed=args.seed,
            use_market=args.use_market,
        )
        if args.explain:
            output["explanation"] = _explain_special_forecast(output)
        print(json.dumps(output, indent=2, sort_keys=True))
        return 0
    if args.command == "predict-event":
        artifact = ModelArtifact.load(args.artifact or settings.artifact_path)
        question = _match_question_from_namespace(args)
        output = forecast_match_event(
            artifact,
            question,
            args.event,
            settings,
            database_path,
            simulations=args.simulations,
            seed=args.seed,
            use_market=args.use_market,
        )
        if args.explain:
            output["explanation"] = _explain_special_forecast(output)
        print(json.dumps(output, indent=2, sort_keys=True))
        return 0
    if args.command == "batch":
        artifact = ModelArtifact.load(args.artifact or settings.artifact_path)
        _run_batch(
            args.input,
            args.output,
            artifact,
            settings,
            database_path,
            args.simulations,
            args.seed,
            args.use_market,
        )
        print(json.dumps({"output": args.output}))
        return 0
    if args.command == "forecast-card":
        from .card import forecast_card

        questions = list(args.question)
        if args.questions:
            questions += [
                line.strip()
                for line in Path(args.questions).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        if not questions:
            parser.error("forecast-card needs --question and/or --questions")
        lineup_status: dict[str, str] = {}
        roles: dict[str, str] = {}
        if args.lineup:
            raw = json.loads(Path(args.lineup).read_text(encoding="utf-8"))
            for name, value in raw.items():
                if isinstance(value, dict):
                    lineup_status[name] = value.get("status", "starter")
                    if value.get("role"):
                        roles[name] = value["role"]
                else:
                    lineup_status[name] = value
        specs = None
        if args.llm:
            from .llm_parser import parse_questions

            specs = parse_questions(
                questions, home=args.home, away=args.away, model=args.model
            )
        rows = forecast_card(
            args.home,
            args.away,
            args.lambda_home,
            args.lambda_away,
            questions,
            db=database_path,
            lineup_status=lineup_status or None,
            roles=roles or None,
            specs=specs,
        )
        if args.json:
            print(json.dumps(rows, indent=2))
        else:
            for row in rows:
                prob = row["probability"]
                shown = f"{prob:.3f}" if prob is not None else "  -  "
                print(f"{shown}  {row['basis']:<28} {row['question']}")
        return 0
    if args.command == "ingest-results":
        count = ingest_results_csv(database_path, args.path)
        print(json.dumps({"results_ingested": count}))
        return 0
    if args.command == "log-result":
        log_result(
            database_path,
            match_key=args.match,
            question_key=args.question,
            question_type=args.question_type,
            submitted_probability=args.submitted,
            crowd_probability=args.crowd,
            outcome=args.outcome,
            market_blended_probability=args.market_blended,
            weight=args.weight,
            observed_at=args.timestamp,
        )
        print(json.dumps({"result_logged": True}))
        return 0
    if args.command == "results-report":
        _write_or_print(results_report(database_path), args.output)
        return 0
    parser.error(f"Unsupported command: {args.command}")
    return 2


def _add_question_arguments(parser: argparse.ArgumentParser) -> None:
    _add_match_arguments(parser)
    parser.add_argument("--stat", choices=[stat.value for stat in Stat], required=True)
    parser.add_argument(
        "--type",
        dest="question_type",
        choices=[kind.value for kind in QuestionType],
        required=True,
    )
    parser.add_argument("--k", type=int)


def _add_match_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--home", required=True)
    parser.add_argument("--away", required=True)
    parser.add_argument("--referee")
    parser.add_argument("--competition-type", default="world_cup")
    parser.add_argument("--home-elo", type=float)
    parser.add_argument("--away-elo", type=float)
    parser.add_argument("--non-neutral", action="store_true")


def _add_market_toggle(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--use-market", dest="use_market", action="store_true")
    group.add_argument("--no-market", dest="use_market", action="store_false")
    parser.set_defaults(use_market=True)


def _question_from_namespace(args: argparse.Namespace) -> Question:
    return Question(
        home=args.home,
        away=args.away,
        stat=Stat(args.stat),
        question_type=QuestionType(args.question_type),
        k=args.k,
        referee=args.referee,
        competition_type=args.competition_type,
        home_elo=args.home_elo,
        away_elo=args.away_elo,
        neutral=not args.non_neutral,
    )


def _match_question_from_namespace(args: argparse.Namespace) -> Question:
    return Question(
        home=args.home,
        away=args.away,
        stat=Stat.SHOTS_ON_TARGET,
        question_type=QuestionType.MORE_THAN,
        referee=args.referee,
        competition_type=args.competition_type,
        home_elo=args.home_elo,
        away_elo=args.away_elo,
        neutral=not args.non_neutral,
    )


def _question_from_row(row: dict[str, str]) -> Question:
    k_value = row.get("k", "").strip()
    return Question(
        home=row["home"].strip(),
        away=row["away"].strip(),
        stat=Stat(row["stat"].strip()),
        question_type=QuestionType(row["type"].strip()),
        k=int(k_value) if k_value else None,
        referee=row.get("referee") or None,
        competition_type=row.get("competition_type") or "world_cup",
        home_elo=float(row["home_elo"]) if row.get("home_elo") else None,
        away_elo=float(row["away_elo"]) if row.get("away_elo") else None,
        neutral=row.get("neutral", "1").casefold() not in {"0", "false", "no"},
    )


def _run_batch(
    input_path: str,
    output_path: str,
    artifact: ModelArtifact,
    settings: Settings,
    database_path: str,
    simulations: int | None,
    seed: int | None,
    use_market: bool,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with Path(input_path).open(newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    fieldnames = list(rows[0].keys()) if rows else []
    result_fields = [
        "probability",
        "model_probability",
        "model_only_probability",
        "market_probability",
        "p_home_more",
        "p_tie",
        "p_away_more",
        "interval_80_low",
        "interval_80_high",
        "effective_sample_size_home",
        "effective_sample_size_away",
        "goal_lambda_home",
        "goal_lambda_away",
        "goal_calibration_source",
        "goal_fit_objective",
    ]
    with output.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames + result_fields)
        writer.writeheader()
        for index, row in enumerate(rows):
            question = _question_from_row(row)
            forecast = forecast_question(
                artifact,
                question,
                settings,
                database_path=database_path,
                simulations=simulations,
                seed=None if seed is None else seed + index,
                use_market=use_market,
            )
            result = forecast.as_dict()
            goal_fit = result["metadata"]["goal_market_calibration"]
            row.update(
                {
                    "probability": result["probability"],
                    "model_probability": result["model_probability"],
                    "model_only_probability": result["model_only_probability"],
                    "market_probability": result["market_probability"],
                    "p_home_more": result["p_home_more"],
                    "p_tie": result["p_tie"],
                    "p_away_more": result["p_away_more"],
                    "interval_80_low": forecast.interval_80[0],
                    "interval_80_high": forecast.interval_80[1],
                    "effective_sample_size_home": forecast.effective_sample_size_home,
                    "effective_sample_size_away": forecast.effective_sample_size_away,
                    "goal_lambda_home": goal_fit["lambda_home"],
                    "goal_lambda_away": goal_fit["lambda_away"],
                    "goal_calibration_source": goal_fit["source"],
                    "goal_fit_objective": goal_fit["objective"],
                }
            )
            writer.writerow(row)


def _write_or_print(value: Any, path: str | None) -> None:
    text = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")
        print(json.dumps({"output": str(destination)}))
    else:
        sys.stdout.write(text)


def _explain_forecast(forecast: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = forecast["metadata"]
    goal_fit = metadata["goal_market_calibration"]
    return [
        {
            "stage": "base_rate",
            "value": metadata["calibration_base_rate"],
        },
        {
            "stage": "raw_simulator",
            "value": forecast["raw_model_probability"],
            "latent_flow": metadata.get("latent_match_flow"),
            "count_prior_sources": metadata.get("count_prior_sources"),
        },
        {
            "stage": "market_goal_fusion",
            "source": goal_fit["source"],
            "lambda_home": goal_fit["lambda_home"],
            "lambda_away": goal_fit["lambda_away"],
            "fit_objective": goal_fit["objective"],
            "residuals": goal_fit["residuals"],
            "count_volume_multiplier": metadata.get("market_volume_multiplier"),
        },
        {
            "stage": "territory_projection",
            "possession_projection": metadata.get("possession_projection"),
            "territory_split": metadata.get("territory_split"),
            "warnings": metadata.get("warnings", []),
        },
        {
            "stage": "tournament_incentives",
            **metadata.get("tournament_incentives", {}),
        },
        {
            "stage": "recalibration",
            "map": metadata.get("calibration_map"),
            "value": forecast["model_probability"],
        },
        {
            "stage": "coverage_guard",
            **metadata.get("coverage_guard", {}),
        },
        {
            "stage": "ess_gate",
            **metadata.get("ess_gate", {}),
        },
        {
            "stage": "field_bias",
            **metadata.get("field_bias", {}),
        },
        {
            "stage": "crowd_anchor",
            **metadata.get("crowd_anchor", {}),
        },
        {
            "stage": "final",
            "value": forecast["probability"],
        },
    ]


def _explain_special_forecast(forecast: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = forecast["metadata"]
    return [
        {
            "stage": "market_goal_fusion",
            **metadata.get("goal_calibration", {}),
            "market_calibrated_simulator_probability": forecast[
                "model_probability"
            ],
            "elo_fallback_model_only_probability": forecast[
                "model_only_probability"
            ],
        },
        {
            "stage": "latent_match_flow",
            **metadata.get("latent_match_flow", {}),
        },
        {
            "stage": "tournament_incentives",
            **metadata.get("tournament_incentives", {}),
        },
        {
            "stage": "lineup",
            "pre_lineup_probability": forecast.get("pre_lineup_probability"),
            "post_lineup_probability": forecast["model_probability"],
            "delta": forecast.get("lineup_delta"),
            "details": metadata.get("lineups"),
        },
        {
            "stage": "player_role_prior",
            **metadata.get("player_prior", {}),
        },
        {
            "stage": "direct_market",
            "probability": forecast.get("market_probability"),
            "details": metadata.get("market"),
        },
        {
            "stage": "crowd_anchor",
            **metadata.get("crowd_anchor", {}),
        },
        {
            "stage": "final",
            "value": forecast["probability"],
        },
    ]


if __name__ == "__main__":
    raise SystemExit(main())
