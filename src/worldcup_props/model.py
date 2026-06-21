from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Settings
from .db import connect, initialize
from .players import PlayerClubProfile, load_club_profiles
from .registry import clean_confederation, load_team_confederation_registry
from .strength import CompetitionStrengthModel


EPSILON = 1e-6
COUNT_STATS = ("fouls", "corners", "offsides", "shots_on_target", "cards")
TERRITORY_STATS = ("corners", "offsides", "shots_on_target")
RATE_PRIORS = {
    "fouls": 12.0,
    "corners": 4.8,
    "offsides": 1.8,
    "shots_on_target": 4.0,
    "cards": 2.1,
}


@dataclass
class TeamParameters:
    confederation: str
    effective_matches: float
    possession: float
    pressing_proxy: float | None
    rates: dict[str, float] = field(default_factory=dict)
    conceded_rates: dict[str, float] = field(default_factory=dict)
    first_half_rates: dict[str, float] = field(default_factory=dict)
    rate_sources: dict[str, str] = field(default_factory=dict)
    conceded_rate_sources: dict[str, str] = field(default_factory=dict)
    rate_sample_sizes: dict[str, float] = field(default_factory=dict)
    conceded_rate_sample_sizes: dict[str, float] = field(default_factory=dict)
    club_prior_rates: dict[str, float] = field(default_factory=dict)
    club_prior_effective_matches: dict[str, float] = field(default_factory=dict)
    club_prior_metadata: dict[str, Any] = field(default_factory=dict)
    foul_share: float = 0.5


@dataclass
class ModelArtifact:
    fitted_through: str
    global_rates: dict[str, float]
    confederation_rates: dict[str, dict[str, float]]
    teams: dict[str, TeamParameters]
    dispersions: dict[str, float]
    first_half_shares: dict[str, float]
    territory_coefficients: dict[str, float]
    dominance_coefficients: dict[str, dict[str, float]]
    foul_total_mean: float
    foul_total_dispersion: float
    referee_effects: dict[str, float]
    referee_weights: dict[str, float]
    context_effects: dict[str, float]
    foul_elo_gap_coefficient: float
    foul_possession_coefficient: float
    foul_pressing_coefficient: float
    foul_style_interaction_coefficient: float
    foul_split_concentration: float
    red_card_rates: dict[str, Any] = field(default_factory=dict)
    calibration_lambda: dict[str, float] = field(default_factory=dict)
    base_rates: dict[str, float] = field(default_factory=dict)
    calibration_maps: dict[str, dict[str, float | str]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "ModelArtifact":
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        values["teams"] = {
            name: TeamParameters(**parameters) for name, parameters in values["teams"].items()
        }
        values.setdefault("dominance_coefficients", _default_dominance_coefficients())
        values.setdefault("calibration_maps", {})
        values.setdefault("red_card_rates", _default_red_card_rates())
        values.setdefault("foul_style_interaction_coefficient", 0.0)
        for stat, prior in RATE_PRIORS.items():
            values["global_rates"].setdefault(stat, prior)
            values["first_half_shares"].setdefault(
                stat, 0.45 if stat == "shots_on_target" else 0.47
            )
            if stat != "fouls":
                values["dispersions"].setdefault(stat, 8.0)
                if stat in TERRITORY_STATS:
                    values["territory_coefficients"].setdefault(stat, 0.08)
        for parameters in values["teams"].values():
            for stat, prior in values["global_rates"].items():
                parameters.rates.setdefault(stat, prior)
                parameters.conceded_rates.setdefault(stat, prior)
                parameters.first_half_rates.setdefault(
                    stat, prior * values["first_half_shares"][stat]
                )
        return cls(**values)


def fit_model(
    database_path: str | Path,
    settings: Settings,
    cutoff_date: str | None = None,
) -> ModelArtifact:
    initialize(database_path)
    cutoff = cutoff_date or date.today().isoformat()
    registry = load_team_confederation_registry(settings.team_registry_path)
    frame = _load_training_frame(
        database_path,
        settings.start_date,
        cutoff,
        registry,
        settings.max_unknown_confederation_share,
    )
    if frame.empty:
        raise ValueError("No training matches found in the requested date range")
    frame = _add_weights(frame, cutoff, settings)
    global_rates = {
        stat: _weighted_mean(frame[stat], frame["weight"], fallback=RATE_PRIORS[stat])
        for stat in COUNT_STATS
    }
    confed_rates, confed_coverage = _fit_confederation_rates(
        frame, global_rates, settings
    )
    strength_model = CompetitionStrengthModel.load(settings.competition_strength_csv_path)
    club_priors, club_audit = _club_team_priors(
        database_path, global_rates, strength_model
    )
    first_half_shares = _first_half_shares(frame, settings)
    territory_coefficients = {
        stat: _territory_coefficient(frame, stat) for stat in TERRITORY_STATS
    }
    dominance_coefficients = _fit_dominance_coefficients(
        frame, settings.club_dominance_csv_path
    )
    teams = _fit_teams(
        frame,
        global_rates,
        confed_rates,
        confed_coverage,
        first_half_shares,
        settings,
        club_priors,
    )
    dispersions = {
        stat: _negative_binomial_size(frame[stat], frame["weight"])
        for stat in (*TERRITORY_STATS, "cards")
    }

    match_frame = _match_level_frame(frame)
    foul_total_mean = _weighted_mean(match_frame["total_fouls"], match_frame["weight"])
    foul_total_dispersion = _negative_binomial_size(
        match_frame["total_fouls"], match_frame["weight"]
    )
    context_effects = _fit_context_effects(match_frame, foul_total_mean, settings)
    referee_effects, referee_weights = _fit_referee_effects(
        database_path, match_frame, foul_total_mean, settings
    )
    foul_elo_gap_coefficient = _weighted_slope(
        match_frame["elo_gap_abs"] / 400.0,
        np.log((match_frame["total_fouls"] + 0.5) / (foul_total_mean + 0.5)),
        match_frame["weight"],
    )
    (
        foul_possession_coefficient,
        foul_pressing_coefficient,
        foul_style_interaction_coefficient,
        foul_split_concentration,
    ) = _fit_foul_split(frame)

    confederation_audit = _confederation_audit(frame)
    count_fit_audit = _count_fit_audit(
        frame, teams, confed_rates, confed_coverage, global_rates, settings
    )
    coverage_table = _coverage_table(teams, confed_rates, global_rates)
    red_card_rates = _fit_red_card_rates(frame, settings)

    return ModelArtifact(
        fitted_through=cutoff,
        global_rates=global_rates,
        confederation_rates=confed_rates,
        teams=teams,
        dispersions=dispersions,
        first_half_shares=first_half_shares,
        territory_coefficients=territory_coefficients,
        dominance_coefficients=dominance_coefficients,
        foul_total_mean=foul_total_mean,
        foul_total_dispersion=foul_total_dispersion,
        referee_effects=referee_effects,
        referee_weights=referee_weights,
        context_effects=context_effects,
        foul_elo_gap_coefficient=float(foul_elo_gap_coefficient),
        foul_possession_coefficient=float(foul_possession_coefficient),
        foul_pressing_coefficient=float(foul_pressing_coefficient),
        foul_style_interaction_coefficient=float(
            foul_style_interaction_coefficient
        ),
        foul_split_concentration=float(foul_split_concentration),
        red_card_rates=red_card_rates,
        calibration_lambda={},
        base_rates={},
        calibration_maps={},
        metadata={
            "training_rows": int(len(frame)),
            "training_matches": int(frame["match_id"].nunique()),
            "start_date": settings.start_date,
            "friendly_weight": settings.friendly_weight,
            "team_prior_matches": settings.team_prior_matches,
            "referee_prior_matches": settings.referee_prior_matches,
            "team_registry_path": str(settings.team_registry_path),
            "unknown_confederation_share": float(
                (frame["confederation"] == "UNK").mean()
            ),
            "unmapped_teams": sorted(
                str(team)
                for team in frame.loc[frame["confederation"] == "UNK", "team"].unique()
            ),
            "confederation_audit": confederation_audit,
            "confederation_rate_coverage": confed_coverage,
            "count_fit_audit": count_fit_audit,
            "red_card_rates": red_card_rates,
            "club_prior_audit": club_audit,
            "competition_strength_source": strength_model.source,
            "count_coverage_table": coverage_table,
            "team_confederation_registry": registry.canonical_metadata(),
            "team_confederation_lookup": registry.lookup_metadata(),
            "dominance_source": (
                "club_csv"
                if Path(settings.club_dominance_csv_path).exists()
                else "international_with_regularized_priors"
            ),
        },
    )


def _load_training_frame(
    database_path: str | Path,
    start_date: str,
    cutoff_date: str,
    registry: Any,
    max_unknown_share: float,
) -> pd.DataFrame:
    query = """
        SELECT m.id AS match_id, m.match_date, m.competition, m.competition_type,
               m.home_team, m.away_team, m.home_elo, m.away_elo, m.referee_name,
               s.team, s.opponent, s.is_home, s.confederation AS db_confederation,
               s.fouls, s.corners, s.offsides, s.shots_on_target, s.cards,
               s.yellow_cards, s.red_cards, s.possession,
               s.first_half_fouls, s.first_half_corners, s.first_half_offsides,
               s.first_half_shots_on_target, s.first_half_cards,
               s.first_half_yellow_cards, s.first_half_red_cards, s.pressing_proxy
        FROM matches m
        JOIN team_match_stats s ON s.match_id = m.id
        WHERE m.match_date >= ? AND m.match_date < ?
        ORDER BY m.match_date, m.id, s.is_home DESC
    """
    with connect(database_path) as connection:
        frame = pd.read_sql_query(query, connection, params=(start_date, cutoff_date))
    for column in (
        "fouls",
        "corners",
        "offsides",
        "shots_on_target",
        "cards",
        "yellow_cards",
        "red_cards",
        "possession",
        "first_half_fouls",
        "first_half_corners",
        "first_half_offsides",
        "first_half_shots_on_target",
        "first_half_cards",
        "first_half_yellow_cards",
        "first_half_red_cards",
        "pressing_proxy",
        "home_elo",
        "away_elo",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame.empty:
        return frame
    mapped_confederations: list[str] = []
    mapping_sources: list[str] = []
    for row in frame.itertuples(index=False):
        registry_confederation = registry.confederation_for(str(row.team))
        db_confederation = clean_confederation(getattr(row, "db_confederation", None))
        if registry_confederation is not None:
            mapped_confederations.append(registry_confederation)
            mapping_sources.append("registry")
        elif db_confederation is not None:
            mapped_confederations.append(db_confederation)
            mapping_sources.append("database")
        else:
            mapped_confederations.append("UNK")
            mapping_sources.append("unknown")
    frame["confederation"] = mapped_confederations
    frame["confederation_source"] = mapping_sources
    unknown = frame["confederation"] == "UNK"
    unknown_share = float(unknown.mean())
    if unknown.any():
        teams = sorted(str(team) for team in frame.loc[unknown, "team"].unique())
        warning = (
            "Unmapped team confederations: "
            + ", ".join(teams)
            + f" ({unknown_share:.1%} of training rows)"
        )
        if unknown_share > max_unknown_share:
            raise ValueError(
                warning
                + f"; maximum allowed is {max_unknown_share:.1%}. "
                + "Add these teams to data/team_confederations.csv."
            )
        print(f"WARNING: {warning}")
    return frame


def _add_weights(frame: pd.DataFrame, cutoff: str, settings: Settings) -> pd.DataFrame:
    result = frame.copy()
    cutoff_timestamp = pd.Timestamp(cutoff)
    age_days = (cutoff_timestamp - pd.to_datetime(result["match_date"])).dt.days.clip(lower=0)
    result["weight"] = np.exp(-math.log(2.0) * age_days / settings.recency_half_life_days)
    friendly = result["competition_type"].str.casefold().isin({"friendly", "friendlies"})
    result.loc[friendly, "weight"] *= settings.friendly_weight
    for team, start_date in settings.manager_regime_start_dates.items():
        regime_start = pd.Timestamp(start_date)
        if regime_start >= cutoff_timestamp:
            continue
        previous_regime = (result["team"] == team) & (
            pd.to_datetime(result["match_date"]) < regime_start
        )
        result.loc[previous_regime, "weight"] *= settings.manager_pre_regime_weight
    return result


def _fit_confederation_rates(
    frame: pd.DataFrame,
    global_rates: dict[str, float],
    settings: Settings,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    output: dict[str, dict[str, float]] = {}
    coverage: dict[str, dict[str, float]] = {}
    for confederation, group in frame.groupby("confederation", dropna=False):
        output[str(confederation)] = {}
        coverage[str(confederation)] = {}
        for stat, global_rate in global_rates.items():
            available = group[stat].notna()
            effective_n = float(group.loc[available, "weight"].sum())
            observed = _weighted_mean(
                group.loc[available, stat],
                group.loc[available, "weight"],
                fallback=global_rate,
            )
            output[str(confederation)][stat] = _shrink(
                observed, effective_n, global_rate, settings.team_prior_matches
            )
            coverage[str(confederation)][stat] = effective_n
    output.setdefault("UNK", dict(global_rates))
    coverage.setdefault("UNK", {stat: 0.0 for stat in global_rates})
    return output, coverage


def _club_team_priors(
    database_path: str | Path,
    global_rates: dict[str, float],
    strength_model: CompetitionStrengthModel,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    grouped = load_club_profiles(database_path, strength_model)
    team_priors: dict[str, dict[str, Any]] = {}
    audit: dict[str, Any] = {
        "teams_with_profiles": len(grouped),
        "players_with_profiles": sum(
            len({profile.player_name for profile in profiles})
            for profiles in grouped.values()
        ),
        "competition_strength_source": strength_model.source,
        "teams": {},
    }
    for team, profiles in grouped.items():
        player_profiles = _combine_club_profiles_by_player(profiles)
        stat_sums = {
            "shots_on_target": 0.0,
            "fouls": 0.0,
            "cards": 0.0,
            "shots": 0.0,
        }
        stat_minutes = {stat: 0.0 for stat in stat_sums}
        competitions: dict[str, int] = {}
        starters_missing_minutes = 0
        for profile in player_profiles:
            competitions[profile["competition"]] = competitions.get(profile["competition"], 0) + 1
            minutes_fraction = profile["expected_minutes_fraction"]
            if minutes_fraction <= 0.0:
                if profile["likely_starter"]:
                    starters_missing_minutes += 1
                continue
            for source_stat, target_stat in (
                ("shots_on_target_per90", "shots_on_target"),
                ("fouls_committed_per90", "fouls"),
                ("yellow_cards_per90", "cards"),
                ("shots_per90", "shots"),
            ):
                value = profile.get(source_stat)
                if value is None:
                    continue
                stat_sums[target_stat] += minutes_fraction * float(value)
                stat_minutes[target_stat] += profile["club_effective_matches"]
        priors: dict[str, Any] = {}
        for stat in ("shots_on_target", "fouls", "cards"):
            if stat_minutes[stat] > 0.0:
                priors[stat] = {
                    "rate": max(stat_sums[stat], 0.01),
                    "effective_matches": stat_minutes[stat],
                    "source": "club_player_minutes_weighted",
                    "players": len(player_profiles),
                    "competitions": competitions,
                }
        if "shots_on_target" in priors:
            sot_rate = float(priors["shots_on_target"]["rate"])
            if global_rates["shots_on_target"] > 0:
                corner_ratio = global_rates["corners"] / global_rates["shots_on_target"]
                offside_ratio = global_rates["offsides"] / global_rates["shots_on_target"]
                priors["corners"] = {
                    "rate": max(sot_rate * corner_ratio, 0.01),
                    "effective_matches": priors["shots_on_target"]["effective_matches"],
                    "source": "club_sot_derived_team_attack_profile",
                    "mapping": "global_corners_per_sot_ratio",
                    "players": len(player_profiles),
                    "competitions": competitions,
                }
                priors["offsides"] = {
                    "rate": max(sot_rate * offside_ratio, 0.01),
                    "effective_matches": priors["shots_on_target"]["effective_matches"],
                    "source": "club_sot_derived_team_attack_profile",
                    "mapping": "global_offsides_per_sot_ratio",
                    "players": len(player_profiles),
                    "competitions": competitions,
                }
        if priors:
            team_priors[team] = priors
        audit["teams"][team] = {
            "players": len(player_profiles),
            "stats": sorted(priors),
            "competitions": competitions,
            "likely_starters_without_expected_minutes": starters_missing_minutes,
        }
    return team_priors, audit


def _combine_club_profiles_by_player(
    profiles: list[PlayerClubProfile],
) -> list[dict[str, Any]]:
    combined: list[dict[str, Any]] = []
    for player_name in sorted({profile.player_name for profile in profiles}):
        player_rows = [profile for profile in profiles if profile.player_name == player_name]
        total_minutes = sum(max(profile.minutes, 0.0) for profile in player_rows)
        if total_minutes <= 0.0:
            continue
        representative = max(player_rows, key=lambda profile: profile.minutes)
        expected_minutes = next(
            (
                profile.expected_minutes
                for profile in player_rows
                if profile.expected_minutes is not None
            ),
            None,
        )
        likely_starter = any(profile.likely_starter for profile in player_rows)
        if expected_minutes is None:
            expected_minutes = 75.0 if likely_starter else 0.0
        record: dict[str, Any] = {
            "player_name": player_name,
            "competition": representative.competition,
            "national_role": representative.national_role,
            "likely_starter": likely_starter,
            "expected_minutes_fraction": float(np.clip(expected_minutes / 90.0, 0.0, 1.0)),
            "club_effective_matches": total_minutes / 90.0,
        }
        for stat in (
            "shots_per90",
            "shots_on_target_per90",
            "goals_per90",
            "assists_per90",
            "fouls_committed_per90",
            "fouls_drawn_per90",
            "yellow_cards_per90",
        ):
            weighted_values = [
                (
                    getattr(profile, stat) * profile.strength_multiplier,
                    max(profile.minutes, 0.0),
                )
                for profile in player_rows
                if getattr(profile, stat) is not None
            ]
            if weighted_values:
                total = sum(minutes for _, minutes in weighted_values)
                record[stat] = sum(value * minutes for value, minutes in weighted_values) / total
            else:
                record[stat] = None
        combined.append(record)
    return combined


def _fit_teams(
    frame: pd.DataFrame,
    global_rates: dict[str, float],
    confed_rates: dict[str, dict[str, float]],
    confed_coverage: dict[str, dict[str, float]],
    first_half_shares: dict[str, float],
    settings: Settings,
    club_priors: dict[str, dict[str, Any]],
) -> dict[str, TeamParameters]:
    opponent = frame[
        [
            "match_id",
            "team",
            "fouls",
            "corners",
            "offsides",
            "shots_on_target",
            "cards",
            "possession",
        ]
    ].rename(
        columns={
            "team": "opponent_row",
            "fouls": "opponent_fouls",
            "corners": "opponent_corners",
            "offsides": "opponent_offsides",
            "shots_on_target": "opponent_shots_on_target",
            "cards": "opponent_cards",
            "possession": "opponent_possession",
        }
    )
    joined = frame.merge(opponent, on="match_id", how="left")
    joined = joined[joined["team"] != joined["opponent_row"]].copy()
    teams: dict[str, TeamParameters] = {}
    for team, group in joined.groupby("team"):
        confederation = str(group["confederation"].mode().iloc[0])
        priors = confed_rates.get(confederation, global_rates)
        effective_n = float(group["weight"].sum())
        rates: dict[str, float] = {}
        conceded: dict[str, float] = {}
        first_half: dict[str, float] = {}
        rate_sources: dict[str, str] = {}
        conceded_sources: dict[str, str] = {}
        rate_sample_sizes: dict[str, float] = {}
        conceded_sample_sizes: dict[str, float] = {}
        club_team_priors = club_priors.get(str(team), {})
        club_prior_rates: dict[str, float] = {}
        club_prior_effective_matches: dict[str, float] = {}
        for stat in COUNT_STATS:
            own_available = group[stat].notna()
            allowed_available = group[f"opponent_{stat}"].notna()
            own_n = float(group.loc[own_available, "weight"].sum())
            allowed_n = float(group.loc[allowed_available, "weight"].sum())
            own = _weighted_mean(
                group.loc[own_available, stat],
                group.loc[own_available, "weight"],
                fallback=priors[stat],
            )
            allowed = _weighted_mean(
                group.loc[allowed_available, f"opponent_{stat}"],
                group.loc[allowed_available, "weight"],
                fallback=priors[stat],
            )
            prior_value = priors[stat]
            prior_matches = settings.team_prior_matches
            if stat in club_team_priors:
                prior_value = float(club_team_priors[stat]["rate"])
                prior_matches = settings.club_prior_matches
                club_prior_rates[stat] = prior_value
                club_prior_effective_matches[stat] = float(
                    club_team_priors[stat].get("effective_matches", 0.0)
                )
            rates[stat] = _shrink(own, own_n, prior_value, prior_matches)
            conceded[stat] = _shrink(
                allowed, allowed_n, priors[stat], settings.team_prior_matches
            )
            rate_sample_sizes[stat] = own_n
            conceded_sample_sizes[stat] = allowed_n
            confed_n = confed_coverage.get(confederation, {}).get(stat, 0.0)
            rate_sources[stat] = _rate_source(
                own_n,
                confed_n,
                settings.team_prior_matches,
                has_club_prior=stat in club_team_priors,
            )
            conceded_sources[stat] = _rate_source(
                allowed_n, confed_n, settings.team_prior_matches
            )
            available = group[f"first_half_{stat}"].notna()
            first_half_n = float(group.loc[available, "weight"].sum())
            first_half_observed = _weighted_mean(
                group.loc[available, f"first_half_{stat}"],
                group.loc[available, "weight"],
                fallback=rates[stat] * first_half_shares[stat],
            )
            first_half[stat] = _shrink(
                first_half_observed,
                first_half_n,
                rates[stat] * first_half_shares[stat],
                settings.team_prior_matches / 2.0,
            )

        total_pair_fouls = group["fouls"] + group["opponent_fouls"]
        valid_share = total_pair_fouls > 0
        observed_share = _weighted_mean(
            group.loc[valid_share, "fouls"] / total_pair_fouls.loc[valid_share],
            group.loc[valid_share, "weight"],
            fallback=0.5,
        )
        foul_share = _shrink(
            observed_share, effective_n, 0.5, settings.team_prior_matches
        )
        teams[str(team)] = TeamParameters(
            confederation=confederation,
            effective_matches=effective_n,
            possession=_weighted_mean(
                group["possession"], group["weight"], fallback=50.0
            ),
            pressing_proxy=(
                _weighted_mean(group["pressing_proxy"], group["weight"])
                if group["pressing_proxy"].notna().any()
                else None
            ),
            rates=rates,
            conceded_rates=conceded,
            first_half_rates=first_half,
            rate_sources=rate_sources,
            conceded_rate_sources=conceded_sources,
            rate_sample_sizes=rate_sample_sizes,
            conceded_rate_sample_sizes=conceded_sample_sizes,
            club_prior_rates=club_prior_rates,
            club_prior_effective_matches=club_prior_effective_matches,
            club_prior_metadata=club_team_priors,
            foul_share=foul_share,
        )
    return teams


def _rate_source(
    team_n: float, confed_n: float, threshold: float, *, has_club_prior: bool = False
) -> str:
    if has_club_prior:
        return "club-blended"
    if team_n >= threshold:
        return "learned"
    if confed_n > 0:
        return "confederation-fallback"
    return "global-fallback"


def _confederation_audit(frame: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    audit: dict[str, dict[str, float | int]] = {}
    for confederation, group in frame.groupby("confederation"):
        audit[str(confederation)] = {
            "teams": int(group["team"].nunique()),
            "training_matches": int(group["match_id"].nunique()),
            "effective_matches": float(group["weight"].sum()),
            "registry_rows": int((group["confederation_source"] == "registry").sum()),
            "database_rows": int((group["confederation_source"] == "database").sum()),
            "unknown_rows": int((group["confederation_source"] == "unknown").sum()),
        }
    return dict(sorted(audit.items()))


def _count_fit_audit(
    frame: pd.DataFrame,
    teams: dict[str, TeamParameters],
    confed_rates: dict[str, dict[str, float]],
    confed_coverage: dict[str, dict[str, float]],
    global_rates: dict[str, float],
    settings: Settings,
) -> dict[str, Any]:
    audit: dict[str, Any] = {}
    warnings: list[str] = []
    for stat in COUNT_STATS:
        non_null_rows = int(frame[stat].notna().sum())
        learned = [
            (team, parameters)
            for team, parameters in teams.items()
            if parameters.rate_sources.get(stat) == "learned"
        ]
        differing = [
            team
            for team, parameters in learned
            if abs(
                parameters.rates.get(stat, global_rates[stat])
                - confed_rates.get(parameters.confederation, global_rates).get(
                    stat, global_rates[stat]
                )
            )
            > 0.05
        ]
        if non_null_rows == 0:
            warnings.append(
                f"{stat}: no non-null training rows; all teams use global fallback"
            )
        elif learned and not differing:
            warnings.append(
                f"{stat}: learned teams are still pinned to the confederation prior"
            )
        audit[stat] = {
            "non_null_training_rows": non_null_rows,
            "learned_teams": len(learned),
            "learned_teams_differing_from_prior": len(differing),
            "team_threshold": settings.team_prior_matches,
            "confederation_effective_matches": {
                confederation: coverage.get(stat, 0.0)
                for confederation, coverage in sorted(confed_coverage.items())
            },
        }
    audit["warnings"] = warnings
    return audit


def _coverage_table(
    teams: dict[str, TeamParameters],
    confed_rates: dict[str, dict[str, float]],
    global_rates: dict[str, float],
) -> dict[str, Any]:
    table: dict[str, Any] = {}
    for team, parameters in sorted(teams.items()):
        priors = confed_rates.get(parameters.confederation, global_rates)
        stats: dict[str, Any] = {}
        for stat in COUNT_STATS:
            stats[stat] = {
                "source": parameters.rate_sources.get(stat, "global-fallback"),
                "rate": parameters.rates.get(stat, global_rates[stat]),
                "team_effective_matches": parameters.rate_sample_sizes.get(stat, 0.0),
                "confederation_prior": priors.get(stat, global_rates[stat]),
                "global_prior": global_rates[stat],
                "club_prior": parameters.club_prior_rates.get(stat),
                "club_prior_effective_matches": (
                    parameters.club_prior_effective_matches.get(stat)
                ),
            }
        table[team] = {
            "confederation": parameters.confederation,
            "effective_matches": parameters.effective_matches,
            "stats": stats,
        }
    return table


def _fit_red_card_rates(frame: pd.DataFrame, settings: Settings) -> dict[str, Any]:
    fallback_match_probability = float(settings.red_card_base_probability)
    fallback_team_rate = -math.log(max(1.0 - fallback_match_probability, EPSILON)) / 2.0
    valid_team = frame["red_cards"].notna()
    observed_team_rate = _weighted_mean(
        frame.loc[valid_team, "red_cards"],
        frame.loc[valid_team, "weight"],
        fallback=fallback_team_rate,
    )
    team_effective_matches = float(frame.loc[valid_team, "weight"].sum())
    global_team_rate = _shrink(
        observed_team_rate,
        team_effective_matches,
        fallback_team_rate,
        settings.red_card_prior_matches,
    )

    match_records: list[dict[str, float]] = []
    for _, group in frame.groupby("match_id"):
        available = group["red_cards"].notna()
        if not available.any():
            continue
        total_reds = float(group.loc[available, "red_cards"].sum())
        match_records.append(
            {
                "any_red": 1.0 if total_reds > 0 else 0.0,
                "total_reds": total_reds,
                "weight": float(group.loc[available, "weight"].mean()),
            }
        )
    if match_records:
        match_frame = pd.DataFrame.from_records(match_records)
        observed_match_probability = _weighted_mean(
            match_frame["any_red"],
            match_frame["weight"],
            fallback=fallback_match_probability,
        )
        observed_match_reds = _weighted_mean(
            match_frame["total_reds"],
            match_frame["weight"],
            fallback=2.0 * fallback_team_rate,
        )
        match_effective_matches = float(match_frame["weight"].sum())
    else:
        observed_match_probability = fallback_match_probability
        observed_match_reds = 2.0 * fallback_team_rate
        match_effective_matches = 0.0
    global_match_probability = _shrink(
        observed_match_probability,
        match_effective_matches,
        fallback_match_probability,
        settings.red_card_prior_matches,
    )
    global_match_reds = _shrink(
        observed_match_reds,
        match_effective_matches,
        2.0 * fallback_team_rate,
        settings.red_card_prior_matches,
    )

    confederation_team_rates: dict[str, float] = {}
    confederation_effective_matches: dict[str, float] = {}
    for confederation, group in frame.groupby("confederation"):
        available = group["red_cards"].notna()
        n = float(group.loc[available, "weight"].sum())
        observed = _weighted_mean(
            group.loc[available, "red_cards"],
            group.loc[available, "weight"],
            fallback=global_team_rate,
        )
        confederation_team_rates[str(confederation)] = _shrink(
            observed, n, global_team_rate, settings.red_card_prior_matches
        )
        confederation_effective_matches[str(confederation)] = n

    return {
        "source": "team_match_stats.red_cards",
        "fallback_match_probability": fallback_match_probability,
        "global_team_rate": float(global_team_rate),
        "global_match_probability": float(global_match_probability),
        "global_match_reds": float(global_match_reds),
        "team_effective_matches": team_effective_matches,
        "match_effective_matches": match_effective_matches,
        "prior_matches": settings.red_card_prior_matches,
        "confederation_team_rates": dict(sorted(confederation_team_rates.items())),
        "confederation_effective_matches": dict(
            sorted(confederation_effective_matches.items())
        ),
    }


def _default_red_card_rates() -> dict[str, Any]:
    return {
        "source": "settings.red_card_base_probability",
        "fallback_match_probability": 0.11,
        "global_team_rate": -math.log(1.0 - 0.11) / 2.0,
        "global_match_probability": 0.11,
        "global_match_reds": -math.log(1.0 - 0.11),
        "team_effective_matches": 0.0,
        "match_effective_matches": 0.0,
        "prior_matches": 0.0,
        "confederation_team_rates": {},
        "confederation_effective_matches": {},
    }


def _match_level_frame(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for match_id, group in frame.groupby("match_id"):
        fouls = group["fouls"].dropna()
        if len(fouls) != 2:
            continue
        first = group.iloc[0]
        home_elo = first["home_elo"]
        away_elo = first["away_elo"]
        gap = abs(home_elo - away_elo) if pd.notna(home_elo) and pd.notna(away_elo) else 0.0
        records.append(
            {
                "match_id": match_id,
                "total_fouls": float(fouls.sum()),
                "weight": float(group["weight"].mean()),
                "competition_type": first["competition_type"],
                "referee_name": first["referee_name"],
                "elo_gap_abs": float(gap),
            }
        )
    return pd.DataFrame.from_records(records)


def _fit_context_effects(
    matches: pd.DataFrame, global_mean: float, settings: Settings
) -> dict[str, float]:
    output: dict[str, float] = {}
    for context, group in matches.groupby("competition_type"):
        n = float(group["weight"].sum())
        observed = _weighted_mean(group["total_fouls"], group["weight"], fallback=global_mean)
        shrunk = _shrink(observed, n, global_mean, settings.context_prior_matches)
        output[str(context)] = float(math.log(max(shrunk, EPSILON) / global_mean))
    output.setdefault("other", 0.0)
    return output


def _fit_referee_effects(
    database_path: str | Path,
    matches: pd.DataFrame,
    global_mean: float,
    settings: Settings,
) -> tuple[dict[str, float], dict[str, float]]:
    effects: dict[str, float] = {}
    counts: dict[str, float] = {}
    known = matches.dropna(subset=["referee_name"])
    for referee, group in known.groupby("referee_name"):
        n = float(group["weight"].sum())
        observed = _weighted_mean(group["total_fouls"], group["weight"], fallback=global_mean)
        shrunk = _shrink(observed, n, global_mean, settings.referee_prior_matches)
        effects[str(referee)] = float(math.log(max(shrunk, EPSILON) / global_mean))
        counts[str(referee)] = n

    with connect(database_path) as connection:
        manual = connection.execute(
            "SELECT referee_name, matches, fouls_per_match FROM referees"
        ).fetchall()
    for row in manual:
        if row["fouls_per_match"] is None:
            continue
        name = str(row["referee_name"])
        n = float(row["matches"])
        shrunk = _shrink(
            float(row["fouls_per_match"]), n, global_mean, settings.referee_prior_matches
        )
        effects[name] = float(math.log(max(shrunk, EPSILON) / global_mean))
        counts[name] = max(counts.get(name, 0.0), n)
    total = sum(counts.values())
    weights = (
        {name: count / total for name, count in counts.items()}
        if total > 0
        else {}
    )
    return effects, weights


def _fit_foul_split(frame: pd.DataFrame) -> tuple[float, float, float, float]:
    opponent = frame[["match_id", "team", "fouls", "possession", "pressing_proxy"]].rename(
        columns={
            "team": "opponent_row",
            "fouls": "opponent_fouls",
            "possession": "opponent_possession",
            "pressing_proxy": "opponent_pressing",
        }
    )
    joined = frame.merge(opponent, on="match_id", how="left")
    joined = joined[joined["team"] != joined["opponent_row"]].copy()
    total = joined["fouls"] + joined["opponent_fouls"]
    valid = (total > 0) & joined["possession"].notna()
    work = joined.loc[valid].copy()
    if len(work) < 8:
        return 0.12, 0.0, 0.0, 18.0
    share = ((work["fouls"] + 0.5) / (total.loc[valid] + 1.0)).clip(0.01, 0.99)
    y = np.log(share / (1.0 - share))
    possession_deficit = (50.0 - work["possession"]) / 10.0
    pressing_diff = (work["pressing_proxy"] - work["opponent_pressing"]).fillna(0.0)
    pressing_scale = float(pressing_diff.std()) or 1.0
    pressing_level = (
        work["pressing_proxy"].fillna(0.0)
        * work["opponent_pressing"].fillna(0.0)
    )
    interaction_scale = float(pressing_level.std()) or 1.0
    x = np.column_stack(
        [
            np.ones(len(work)),
            possession_deficit,
            pressing_diff / pressing_scale,
            pressing_level / interaction_scale,
        ]
    )
    weights = work["weight"].to_numpy(dtype=float)
    ridge = np.diag([0.01, 1.0, 2.0, 4.0])
    beta = np.linalg.solve(x.T @ (weights[:, None] * x) + ridge, x.T @ (weights * y))
    fitted = 1.0 / (1.0 + np.exp(-(x @ beta)))
    residual_variance = float(np.average((share - fitted) ** 2, weights=weights))
    mean_binomial_variance = float(np.average(fitted * (1.0 - fitted), weights=weights))
    concentration = max(3.0, min(100.0, mean_binomial_variance / max(residual_variance, 1e-4) - 1.0))
    return (
        float(beta[1]),
        float(beta[2] / pressing_scale),
        float(beta[3] / interaction_scale),
        concentration,
    )


def _first_half_shares(frame: pd.DataFrame, settings: Settings) -> dict[str, float]:
    output: dict[str, float] = {}
    for stat in COUNT_STATS:
        valid = (
            frame[stat].notna()
            & frame[f"first_half_{stat}"].notna()
            & (frame[stat] > 0)
        )
        ratio = frame.loc[valid, f"first_half_{stat}"] / frame.loc[valid, stat]
        output[stat] = min(
            0.8,
            max(
                0.2,
                _weighted_mean(
                    ratio,
                    frame.loc[valid, "weight"],
                    fallback=settings.first_half_fallback_share[stat],
                ),
            ),
        )
    return output


def _territory_coefficient(frame: pd.DataFrame, stat: str) -> float:
    valid = frame[stat].notna() & frame["possession"].notna()
    if valid.sum() < 8:
        return 0.08 if stat == "corners" else 0.05
    group = frame.loc[valid]
    baseline = _weighted_mean(group[stat], group["weight"])
    x = (group["possession"] - 50.0) / 10.0
    y = np.log((group[stat] + 0.5) / (baseline + 0.5))
    return float(np.clip(_weighted_slope(x, y, group["weight"]), -0.5, 0.5))


def _default_dominance_coefficients() -> dict[str, dict[str, float]]:
    return {
        "corners": {"intercept": 0.0, "possession": 1.10, "supremacy": 0.90},
        "offsides": {"intercept": 0.0, "possession": 0.55, "supremacy": 0.65},
        "shots_on_target": {"intercept": 0.0, "possession": 1.20, "supremacy": 1.00},
    }


def _fit_dominance_coefficients(
    frame: pd.DataFrame, club_csv_path: str | Path
) -> dict[str, dict[str, float]]:
    source = frame
    path = Path(club_csv_path)
    if path.exists():
        club = pd.read_csv(path)
        required = {"possession", *TERRITORY_STATS}
        if required.issubset(club.columns):
            source = club.copy()
    opponent_columns = ["match_id", "team", *TERRITORY_STATS]
    if "match_id" in source.columns and "team" in source.columns:
        opponent = source[opponent_columns].rename(
            columns={
                "team": "opponent_row",
                **{stat: f"opponent_{stat}" for stat in TERRITORY_STATS},
            }
        )
        work = source.merge(opponent, on="match_id", how="left")
        work = work[work["team"] != work["opponent_row"]].copy()
        if "supremacy" not in work:
            if {"is_home", "home_elo", "away_elo"}.issubset(work.columns):
                elo_gap = np.where(
                    work["is_home"].astype(bool),
                    work["home_elo"] - work["away_elo"],
                    work["away_elo"] - work["home_elo"],
                )
                work["supremacy"] = np.tanh(
                    np.nan_to_num(elo_gap, nan=0.0) / 350.0
                )
            else:
                work["supremacy"] = 0.0
    else:
        work = source.copy()
        if "supremacy" not in work:
            work["supremacy"] = 0.0
    defaults = _default_dominance_coefficients()
    output: dict[str, dict[str, float]] = {}
    for stat in TERRITORY_STATS:
        opponent_stat = f"opponent_{stat}"
        if opponent_stat not in work or stat not in work:
            output[stat] = defaults[stat]
            continue
        total = work[stat] + work[opponent_stat]
        valid = (
            (total > 0)
            & work[stat].notna()
            & work[opponent_stat].notna()
            & work["possession"].notna()
        )
        if valid.sum() < 30:
            output[stat] = defaults[stat]
            continue
        group = work.loc[valid]
        share = ((group[stat] + 0.5) / (total.loc[valid] + 1.0)).clip(0.02, 0.98)
        y = np.log(share / (1.0 - share))
        possession = (group["possession"].to_numpy(dtype=float) - 50.0) / 10.0
        supremacy = group["supremacy"].fillna(0.0).to_numpy(dtype=float)
        x = np.column_stack([np.ones(len(group)), possession, supremacy])
        weights = group.get("weight", pd.Series(np.ones(len(group)), index=group.index))
        w = weights.to_numpy(dtype=float)
        ridge = np.diag([0.2, 1.5, 1.5])
        beta = np.linalg.solve(x.T @ (w[:, None] * x) + ridge, x.T @ (w * y))
        output[stat] = {
            "intercept": float(beta[0]),
            "possession": float(max(beta[1], defaults[stat]["possession"] * 0.35)),
            "supremacy": float(max(beta[2], defaults[stat]["supremacy"] * 0.35)),
        }
    return output


def _negative_binomial_size(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna()
    if valid.sum() < 2:
        return 20.0
    x = values.loc[valid].to_numpy(dtype=float)
    w = weights.loc[valid].to_numpy(dtype=float)
    mean = float(np.average(x, weights=w))
    variance = float(np.average((x - mean) ** 2, weights=w))
    if variance <= mean + EPSILON:
        return 1000.0
    return float(np.clip(mean * mean / (variance - mean), 0.25, 1000.0))


def _weighted_slope(x: Any, y: Any, weights: Any) -> float:
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    w_values = np.asarray(weights, dtype=float)
    valid = np.isfinite(x_values) & np.isfinite(y_values) & np.isfinite(w_values)
    if valid.sum() < 3:
        return 0.0
    x_values, y_values, w_values = (
        x_values[valid],
        y_values[valid],
        w_values[valid],
    )
    x_centered = x_values - np.average(x_values, weights=w_values)
    denominator = np.sum(w_values * x_centered * x_centered)
    if denominator <= EPSILON:
        return 0.0
    return float(np.sum(w_values * x_centered * y_values) / denominator)


def _weighted_mean(values: Any, weights: Any, fallback: float = 0.0) -> float:
    values_array = np.asarray(values, dtype=float)
    weights_array = np.asarray(weights, dtype=float)
    valid = np.isfinite(values_array) & np.isfinite(weights_array)
    if not valid.any() or weights_array[valid].sum() <= 0:
        return float(fallback)
    return float(np.average(values_array[valid], weights=weights_array[valid]))


def _shrink(observed: float, n: float, prior: float, prior_n: float) -> float:
    return float((n * observed + prior_n * prior) / max(n + prior_n, EPSILON))
