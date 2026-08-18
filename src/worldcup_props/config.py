from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass
class Settings:
    database_path: str = "data/worldcup_props.sqlite"
    raw_cache_dir: str = "data/raw"
    artifact_path: str = "artifacts/model.json"
    market_csv_path: str = "data/markets.csv"
    market_definition_path: str = "data/market_definitions.json"
    club_dominance_csv_path: str = "data/club_dominance.csv"
    competition_strength_csv_path: str = "data/competition_strength.csv"
    team_registry_path: str = "data/team_confederations.csv"
    start_date: str = "2020-07-01"
    simulations: int = 100_000
    random_seed: int = 2026
    tie_handling: str = "strict"
    friendly_weight: float = 0.45
    recency_half_life_days: float = 730.0
    team_prior_matches: float = 8.0
    club_prior_matches: float = 24.0
    referee_prior_matches: float = 12.0
    context_prior_matches: float = 15.0
    max_unknown_confederation_share: float = 0.05
    liquid_market_blend_weight: float = 0.70
    thin_market_blend_weight: float = 0.50
    market_blend_weight: float | None = None
    fallback_total_goals: float = 2.55
    use_team_strength_goal_fallback: bool = True
    goal_fallback_min_team_ess: float = 6.0
    goal_fallback_sot_share_exponent: float = 1.35
    goal_fallback_sot_total_elasticity: float = 0.35
    goal_fallback_total_bounds: list[float] = field(
        default_factory=lambda: [2.0, 3.35]
    )
    dixon_coles_rho: float = -0.08
    goal_first_half_share: float = 0.44
    market_goal_prior_precision: float = 0.25
    latent_flow_method: str = "shared_normal"
    use_game_state_dynamics: bool = True
    use_market_count_propagation: bool = False
    use_foul_style_interaction: bool = False
    calibration_method: str = "auto"
    latent_flow_std: float = 0.55
    latent_flow_supremacy_shift: float = 0.65
    latent_flow_loadings: dict[str, float] = field(
        default_factory=lambda: {
            "goals": 0.30,
            "corners": 0.42,
            "offsides": 0.28,
            "shots_on_target": 0.38,
        }
    )
    use_supremacy_weighted_market_fusion: bool = False
    supremacy_market_fusion_slope: float = 0.35
    supremacy_market_fusion_max_extra: float = 0.35
    supremacy_market_fusion_ess_threshold: float = 12.0
    market_count_elasticity: dict[str, float] = field(
        default_factory=lambda: {
            "corners": 0.35,
            "offsides": 0.20,
            "shots_on_target": 0.55,
        }
    )
    market_territory_split_weight: dict[str, float] = field(
        default_factory=lambda: {
            "corners": 1.00,
            "shots_on_target": 0.45,
            "offsides": 0.25,
        }
    )
    use_ess_gating: bool = False
    ess_probability_prior_matches: float = 8.0
    use_field_bias: bool = False
    field_bias_max_deviation: float = 0.03
    use_crowd_anchoring: bool = True
    crowd_anchor_min_drift_rows: int = 6
    crowd_anchor_recent_window: int = 20
    crowd_anchor_max_drift: float = 0.04
    crowd_anchor_drift_weight: float = 0.50
    use_contest_agent: bool = True
    contest_agent_market_copy_weight: float = 0.85
    contest_agent_extreme_favorite_win_probability: float = 0.80
    contest_agent_clear_favorite_win_probability: float = 0.62
    contest_agent_structured_dominance_floor: dict[str, float] = field(
        default_factory=lambda: {
            "corners:second_half_more_than": 0.68,
            "shots_on_target:second_half_more_than": 0.70,
        }
    )
    contest_agent_structured_dominance_cap: dict[str, float] = field(
        default_factory=lambda: {
            "corners:second_half_more_than": 0.32,
            "shots_on_target:second_half_more_than": 0.30,
        }
    )
    contest_agent_market_disagreement_trigger: float = 0.08
    contest_agent_require_player_market_events: list[str] = field(
        default_factory=lambda: [
            "goals",
            "goal_or_assist",
            "shots_on_target",
            "second_half_shots_on_target",
        ]
    )
    contest_agent_high_usage_bench_floor: dict[str, float] = field(
        default_factory=lambda: {
            "goals": 0.14,
            "goal_or_assist": 0.22,
            "shots_on_target": 0.18,
            "second_half_shots_on_target": 0.10,
        }
    )
    contest_agent_high_usage_shots_per90: float = 1.50
    contest_agent_high_usage_goal_assist_per90: float = 0.30
    contest_agent_high_usage_set_piece_role: float = 0.35
    # Favorites dominate 2nd-half SOT far more than the flat simulator predicts.
    # Empirical P(favorite more 2H SOT) over 2,408 historical team-matches:
    # ~even 0.56, clear favorite 0.62, strong favorite 0.79. Backtestable toggle.
    contest_agent_favorite_sot2h_dominance: bool = True
    contest_agent_favorite_sot2h_weight: float = 0.55
    # Apply the contest-agent submission layer inside the backtest so its
    # corrections can be gated on held-out Brier (off keeps pure model ablation).
    backtest_apply_contest_agent: bool = False
    use_coverage_safeguards: bool = True
    coverage_probability_prior_matches: float = 12.0
    market_coverage_ess_credit: float = 6.0
    max_market_count_volume_multiplier: float = 1.35
    territory_share_bounds: dict[str, list[float]] = field(
        default_factory=lambda: {
            "corners": [0.18, 0.82],
            "offsides": [0.22, 0.78],
            "shots_on_target": [0.16, 0.84],
        }
    )
    territory_rate_multiplier_bounds: dict[str, list[float]] = field(
        default_factory=lambda: {
            "corners": [0.30, 2.00],
            "offsides": [0.30, 2.00],
            "shots_on_target": [0.30, 2.00],
        }
    )
    use_possession_ess_gating: bool = False
    use_possession_opponent_strength_adjustment: bool = False
    possession_ess_scale: dict[str, float] = field(
        default_factory=lambda: {
            "corners": 10.0,
            "offsides": 12.0,
            "shots_on_target": 12.0,
        }
    )
    possession_rate_split_base_weight: dict[str, float] = field(
        default_factory=lambda: {
            "corners": 0.45,
            "offsides": 0.35,
            "shots_on_target": 0.35,
        }
    )
    possession_rate_split_max_weight: dict[str, float] = field(
        default_factory=lambda: {
            "corners": 0.88,
            "offsides": 0.80,
            "shots_on_target": 0.78,
        }
    )
    possession_supremacy_absent_territory_multiplier: float = 0.55
    player_event_base_rates: dict[str, float] = field(
        default_factory=lambda: {
            "shots": 0.55,
            "shots_on_target": 0.35,
            "second_half_shots_on_target": 0.20,
            "goals": 0.16,
            "assists": 0.12,
            "goal_or_assist": 0.25,
        }
    )
    player_role_base_rates: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "shots_on_target": {
                "forward": 0.55,
                "winger": 0.45,
                "attacking_mid": 0.45,
                "central_mid": 0.30,
                "fullback": 0.20,
                "center_back": 0.12,
                "unknown": 0.35,
            },
            "second_half_shots_on_target": {
                "forward": 0.38,
                "winger": 0.32,
                "attacking_mid": 0.32,
                "central_mid": 0.21,
                "fullback": 0.14,
                "center_back": 0.08,
                "unknown": 0.24,
            },
            "goal_or_assist": {
                "forward": 0.30,
                "winger": 0.22,
                "attacking_mid": 0.22,
                "central_mid": 0.14,
                "fullback": 0.10,
                "center_back": 0.08,
                "unknown": 0.18,
            },
            "goals": {
                "forward": 0.30,
                "winger": 0.20,
                "attacking_mid": 0.18,
                "central_mid": 0.10,
                "fullback": 0.06,
                "center_back": 0.05,
                "unknown": 0.16,
            },
            "assists": {
                "forward": 0.14,
                "winger": 0.18,
                "attacking_mid": 0.20,
                "central_mid": 0.12,
                "fullback": 0.10,
                "center_back": 0.04,
                "unknown": 0.12,
            },
        }
    )
    player_role_antizero_floors: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "shots_on_target": {
                "forward": 0.25,
                "winger": 0.25,
                "attacking_mid": 0.25,
            },
            "second_half_shots_on_target": {
                "forward": 0.18,
                "winger": 0.18,
                "attacking_mid": 0.18,
            },
        }
    )
    player_ga_assist_rate: float = 0.70
    player_ga_penalty_conversion: float = 0.78
    player_ga_set_piece_assist_lambda: float = 0.08
    player_ga_min_profile_weight_for_focal: float = 0.55
    player_ga_goal_share_bounds: dict[str, list[float]] = field(
        default_factory=lambda: {
            "focal": [0.28, 0.32],
            "attacker": [0.10, 0.30],
            "other": [0.02, 0.18],
        }
    )
    player_ga_assist_share_bounds: dict[str, list[float]] = field(
        default_factory=lambda: {
            "focal": [0.22, 0.32],
            "attacker": [0.08, 0.26],
            "other": [0.02, 0.18],
        }
    )
    player_prior_effective_matches: float = 8.0
    player_thin_profile_upside_weight: float = 0.50
    player_team_context_exponent: float = 0.65
    player_team_context_bounds: list[float] = field(
        default_factory=lambda: [0.70, 1.45]
    )
    manager_pre_regime_weight: float = 0.35
    manager_regime_start_dates: dict[str, str] = field(default_factory=dict)
    global_cards_per_match: float = 4.2
    card_dispersion: float = 8.0
    card_foul_elasticity: float = 0.65
    penalty_base_probability: float = 0.23
    red_card_base_probability: float = 0.11
    red_card_prior_matches: float = 80.0
    penalty_box_pressure_elasticity: float = 0.35
    red_card_foul_elasticity: float = 0.55
    default_player_start_probability: float = 0.55
    default_player_sub_probability: float = 0.30
    default_start_minutes: float = 75.0
    default_sub_minutes: float = 22.0
    player_other_share_prior: float = 3.0
    game_state_chasing_multiplier: dict[str, float] = field(
        default_factory=lambda: {
            "corners": 1.18,
            "offsides": 1.08,
            "shots_on_target": 1.12,
            "fouls": 0.97,
        }
    )
    game_state_leading_multiplier: dict[str, float] = field(
        default_factory=lambda: {
            "corners": 0.88,
            "offsides": 0.95,
            "shots_on_target": 0.92,
            "fouls": 1.03,
            "cards": 1.08,
        }
    )
    use_tournament_incentives: bool = True
    tournament_secure_probability: float = 0.78
    tournament_coast_lead_threshold: int = 2
    tournament_blowout_lead_threshold: int = 3
    tournament_coast_leading_multiplier: dict[str, float] = field(
        default_factory=lambda: {
            "goals": 0.62,
            "corners": 0.58,
            "offsides": 0.72,
            "shots_on_target": 0.60,
            "fouls": 0.78,
        }
    )
    tournament_blowout_leading_multiplier: dict[str, float] = field(
        default_factory=lambda: {
            "goals": 0.46,
            "corners": 0.42,
            "offsides": 0.60,
            "shots_on_target": 0.45,
            "fouls": 0.68,
        }
    )
    structured_possession_tactical_styles: list[str] = field(
        default_factory=lambda: [
            "structured_possession",
            "positional_possession",
            "control",
        ]
    )
    structured_possession_coast_leading_multiplier: dict[str, float] = field(
        default_factory=lambda: {
            "goals": 0.74,
            "corners": 0.90,
            "offsides": 0.86,
            "shots_on_target": 0.88,
            "fouls": 0.78,
        }
    )
    structured_possession_blowout_leading_multiplier: dict[str, float] = field(
        default_factory=lambda: {
            "goals": 0.58,
            "corners": 0.78,
            "offsides": 0.74,
            "shots_on_target": 0.76,
            "fouls": 0.68,
        }
    )
    tournament_trailing_multiplier: dict[str, float] = field(
        default_factory=lambda: {
            "goals": 1.08,
            "corners": 1.22,
            "offsides": 1.12,
            "shots_on_target": 1.18,
            "fouls": 1.04,
        }
    )
    tournament_damage_limitation_multiplier: dict[str, float] = field(
        default_factory=lambda: {
            "goals": 0.84,
            "corners": 0.92,
            "offsides": 0.88,
            "shots_on_target": 0.88,
            "fouls": 0.96,
        }
    )
    backtest_prop_weights: dict[str, float] = field(default_factory=dict)
    first_half_fallback_share: dict[str, float] = field(
        default_factory=lambda: {
            "corners": 0.46,
            "fouls": 0.49,
            "cards": 0.44,
            "offsides": 0.47,
            "shots_on_target": 0.45,
        }
    )

    @classmethod
    def load(cls, path: str | Path | None) -> "Settings":
        if path is None:
            return cls()
        with Path(path).open(encoding="utf-8") as handle:
            values: dict[str, Any] = json.load(handle)
        allowed = {item.name for item in fields(cls)}
        values = {key: value for key, value in values.items() if key in allowed}
        return cls(**values)

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    def resolve(self, root: str | Path) -> "Settings":
        root_path = Path(root)
        values = asdict(self)
        for key in (
            "database_path",
            "raw_cache_dir",
            "artifact_path",
            "market_csv_path",
            "market_definition_path",
            "club_dominance_csv_path",
            "competition_strength_csv_path",
            "team_registry_path",
        ):
            candidate = Path(values[key])
            if not candidate.is_absolute():
                values[key] = str(root_path / candidate)
        return Settings(**values)
