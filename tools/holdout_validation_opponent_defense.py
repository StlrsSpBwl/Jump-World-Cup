#!/usr/bin/env python3
"""Held-out validation for opponent-defense-adjusted count thresholds.

`closed_form.team_count_rate` estimates "team has K+ shots-on-target/corners/
cards" purely from that team's own historical average, with zero conditioning
on the opponent. That's fine on average but blind to a specific matchup: a
strong attacking team facing a side that suppresses shots (a deep block, a
disciplined defense) will not hit its flat historical rate. The France-Paraguay
card lost -47 RBP on "France 7+ SOT" (priced 78% off France's own history; the
match was a low-shot, single-penalty-decided 1-0) -- exactly this gap, and
there's no DK market for team-total SOT/corners/cards to fall back on, so the
model has to get the mean right on its own.

This script tests whether blending a team's own attack rate with the specific
opponent's *allowed* rate (mean of the stat posted against them, from the
`opponent` column already in `team_match_stats`) lowers held-out Brier vs the
flat team-only rate, using a standard attack*defense/league-average blend
(same shape as a Poisson expected-goals model). Train/test split is by match,
like `tools/holdout_validation.py`, so there's no leakage.

Usage:
    python tools/holdout_validation_opponent_defense.py --stat shots_on_target
    python tools/holdout_validation_opponent_defense.py --stat corners
    python tools/holdout_validation_opponent_defense.py --stat cards
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from worldcup_props.closed_form import CountRate, count_threshold  # noqa: E402

THRESHOLDS = {
    "shots_on_target": [4, 5, 6, 7, 8],
    "corners": [4, 5, 6, 7, 8],
    "cards": [2, 3, 4, 5],
}
MIN_TRAIN_GAMES = 5


def load_rows(db: str, stat: str) -> list[tuple[object, str, str, float]]:
    """[(match_id, team, opponent, value), ...] for rows with a non-null stat."""
    con = sqlite3.connect(db)
    rows = con.execute(
        f"SELECT match_id, team, opponent, {stat} FROM team_match_stats WHERE {stat} IS NOT NULL"
    ).fetchall()
    return [(mid, team, opp, float(v)) for mid, team, opp, v in rows]


def validate(db: str, stat: str, seeds: int = 8, weight: float = 1.0) -> dict:
    rows = load_rows(db, stat)
    match_ids = sorted({r[0] for r in rows}, key=str)
    thresholds = THRESHOLDS[stat]

    naive_briers, adj_briers, deltas = [], [], []
    pairs_used = []

    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(match_ids))
        train_ids = {match_ids[i] for i in order[: len(order) // 2].tolist()}
        test_ids = {match_ids[i] for i in order[len(order) // 2 :].tolist()}

        train_rows = [r for r in rows if r[0] in train_ids]
        test_rows = [r for r in rows if r[0] in test_ids]

        attack_vals: dict[str, list[float]] = defaultdict(list)
        allowed_vals: dict[str, list[float]] = defaultdict(list)
        for _, team, opp, v in train_rows:
            attack_vals[team].append(v)
            allowed_vals[opp].append(v)
        league_avg = float(np.mean([v for *_, v in train_rows]))

        attack_mean = {t: float(np.mean(v)) for t, v in attack_vals.items() if len(v) >= MIN_TRAIN_GAMES}
        attack_var = {t: float(np.var(v, ddof=1)) for t, v in attack_vals.items() if len(v) >= MIN_TRAIN_GAMES}
        allowed_mean = {t: float(np.mean(v)) for t, v in allowed_vals.items() if len(v) >= MIN_TRAIN_GAMES}

        seed_naive, seed_adj, n_pairs = [], [], 0
        for _, team, opp, v in test_rows:
            if team not in attack_mean or opp not in allowed_mean:
                continue
            naive_mean = attack_mean[team]
            full_adj_mean = attack_mean[team] * allowed_mean[opp] / league_avg
            adj_mean = (1.0 - weight) * naive_mean + weight * full_adj_mean
            var = max(attack_var[team], naive_mean, adj_mean)  # keep NB/Poisson well-posed
            naive_rate = CountRate(mean=naive_mean, var=var)
            adj_rate = CountRate(mean=adj_mean, var=var)
            n_pairs += 1
            for k in thresholds:
                outcome = 1.0 if v >= k else 0.0
                p_naive = count_threshold(naive_rate, k)
                p_adj = count_threshold(adj_rate, k)
                seed_naive.append((p_naive - outcome) ** 2)
                seed_adj.append((p_adj - outcome) ** 2)

        naive_briers.append(np.mean(seed_naive))
        adj_briers.append(np.mean(seed_adj))
        deltas.append(np.mean(seed_adj) - np.mean(seed_naive))
        pairs_used.append(n_pairs)

    return {
        "stat": stat,
        "thresholds": thresholds,
        "seeds": seeds,
        "weight": weight,
        "mean_pairs_per_seed": round(float(np.mean(pairs_used)), 1),
        "naive_brier": round(float(np.mean(naive_briers)), 4),
        "adjusted_brier": round(float(np.mean(adj_briers)), 4),
        "mean_delta": round(float(np.mean(deltas)), 4),
        "delta_std": round(float(np.std(deltas)), 4),
        "per_seed_deltas": [round(float(d), 4) for d in deltas],
        "validated": bool(np.mean(deltas) < 0),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="data/worldcup_props.sqlite")
    ap.add_argument("--stat", default="shots_on_target", choices=list(THRESHOLDS))
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--weight", type=float, default=1.0,
                     help="blend weight toward the opponent-adjusted mean (1.0 = full adjustment)")
    args = ap.parse_args()
    r = validate(args.db, args.stat, args.seeds, args.weight)
    print(f"\nHold-out: opponent-defense-adjusted (weight={r['weight']}) vs flat team rate for "
          f"'{r['stat']}' thresholds {r['thresholds']}")
    print(f"  pairs/seed (both teams >= {MIN_TRAIN_GAMES} train games): {r['mean_pairs_per_seed']}")
    print(f"  flat team-rate Brier      : {r['naive_brier']}")
    print(f"  opponent-adjusted Brier   : {r['adjusted_brier']}")
    print(f"  mean delta (adj-flat)     : {r['mean_delta']:+}  (std {r['delta_std']}, {r['seeds']} seeds)")
    print(f"  per-seed deltas           : {r['per_seed_deltas']}")
    print(f"  => {'VALIDATED (lowers held-out Brier)' if r['validated'] else 'NOT validated (no robust edge)'}\n")


if __name__ == "__main__":
    main()
