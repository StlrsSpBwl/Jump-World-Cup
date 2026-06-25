#!/usr/bin/env python3
"""Held-out validation for supremacy-conditioned contest-agent corrections.

The standard backtest cannot gate corrections that key on "who is the favorite",
because the historical match data carries no pre-match supremacy signal (no odds,
no ELOs) -- every match resolves to an even matchup, so the correction never
fires there. This script is the correct gate instead: it derives a supremacy
signal directly from the data (a team's average stat differential), fits the
empirical favorite-dominance rate on a TRAIN split, and measures whether keying
on supremacy lowers Brier on a held-out TEST split versus the flat ~0.50 baseline
the simulator produces on these props.

A negative mean delta (corrected - flat) across seeds means the correction's
direction generalizes out-of-sample and earns its place.

Usage:
    python tools/holdout_validation.py --stat shots_on_target --period second_half
    python tools/holdout_validation.py --stat corners --period full
"""

from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict

import numpy as np

PERIODS = ("full", "first_half", "second_half")


def _team_value(total: float | None, first_half: float | None, period: str) -> float | None:
    if total is None:
        return None
    if period == "full":
        return float(total)
    if first_half is None:
        return None
    if period == "first_half":
        return float(first_half)
    return float(total) - float(first_half)  # second_half


def _supremacy_bucket(gap: float) -> int:
    gap = abs(gap)
    return 0 if gap < 0.5 else 1 if gap < 1.5 else 2 if gap < 3 else 3


def load_pairs(db: str, stat: str, period: str) -> list[tuple[str, str, float, float]]:
    """Return (home_team, away_team, home_period_value, away_period_value) per match."""
    con = sqlite3.connect(db)
    rows = con.execute(
        f"SELECT match_id, team, {stat} AS total, first_half_{stat} AS fh "
        "FROM team_match_stats"
    ).fetchall()
    by_match: dict[object, list] = defaultdict(list)
    for match_id, team, total, fh in rows:
        by_match[match_id].append((team, _team_value(total, fh, period)))
    pairs = []
    for members in by_match.values():
        if len(members) != 2:
            continue
        (ta, va), (tb, vb) = members
        if va is None or vb is None:
            continue
        pairs.append((ta, tb, va, vb))
    return pairs


def validate(db: str, stat: str, period: str, seeds: int = 8) -> dict:
    # supremacy rating uses FULL-match differential (overall team strength)
    strength_pairs = load_pairs(db, stat, "full")
    period_pairs = load_pairs(db, stat, period)
    deltas, flats, corrs = [], [], []
    bucket_rates_acc: dict[int, list[float]] = defaultdict(list)
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(period_pairs))
        cut = len(idx) // 2
        train_ids = set(idx[:cut].tolist())

        # team strength ratings from TRAIN only
        diffs: dict[str, list[float]] = defaultdict(list)
        for i, (ta, tb, va, vb) in enumerate(strength_pairs):
            if i not in train_ids:
                continue
            diffs[ta].append(va - vb)
            diffs[tb].append(vb - va)
        rating = {t: float(np.mean(v)) for t, v in diffs.items() if len(v) >= 2}

        # TRAIN empirical favorite-dominance rate by supremacy bucket
        train_hits: dict[int, list[float]] = defaultdict(list)
        for i, (ta, tb, va, vb) in enumerate(period_pairs):
            if i not in train_ids or ta not in rating or tb not in rating:
                continue
            sup = rating[ta] - rating[tb]
            strong, weak = (va, vb) if sup >= 0 else (vb, va)
            y = 1.0 if strong > weak else 0.5 if strong == weak else 0.0
            train_hits[_supremacy_bucket(sup)].append(y)
        brate = {b: float(np.mean(v)) for b, v in train_hits.items() if v}
        for b, r in brate.items():
            bucket_rates_acc[b].append(r)

        # TEST Brier: flat 0.50 vs supremacy-bucket predictor
        bf, bc = [], []
        for i, (ta, tb, va, vb) in enumerate(period_pairs):
            if i in train_ids or ta not in rating or tb not in rating:
                continue
            sup = rating[ta] - rating[tb]
            strong, weak = (va, vb) if sup >= 0 else (vb, va)
            y = 1.0 if strong > weak else 0.5 if strong == weak else 0.0
            p = brate.get(_supremacy_bucket(sup), 0.5)
            bf.append((0.5 - y) ** 2)
            bc.append((p - y) ** 2)
        flats.append(np.mean(bf))
        corrs.append(np.mean(bc))
        deltas.append(np.mean(bc) - np.mean(bf))

    return {
        "stat": stat,
        "period": period,
        "seeds": seeds,
        "bucket_rates": {b: round(float(np.mean(v)), 3) for b, v in sorted(bucket_rates_acc.items())},
        "flat_brier": round(float(np.mean(flats)), 4),
        "corrected_brier": round(float(np.mean(corrs)), 4),
        "mean_delta": round(float(np.mean(deltas)), 4),
        "delta_std": round(float(np.std(deltas)), 4),
        "validated": bool(np.mean(deltas) < 0),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="data/worldcup_props.sqlite")
    ap.add_argument("--stat", default="shots_on_target")
    ap.add_argument("--period", default="second_half", choices=PERIODS)
    ap.add_argument("--seeds", type=int, default=8)
    args = ap.parse_args()
    r = validate(args.db, args.stat, args.period, args.seeds)
    labels = {0: "~even", 1: "small", 2: "clear", 3: "strong"}
    print(f"\nHold-out validation: P(favorite has more {args.stat} in {args.period})")
    print(f"  supremacy bucket rates (train): "
          + ", ".join(f"{labels[b]}={v}" for b, v in r["bucket_rates"].items()))
    print(f"  flat 0.50 baseline Brier : {r['flat_brier']}")
    print(f"  supremacy-corrected Brier: {r['corrected_brier']}")
    print(f"  mean delta (corr-flat)   : {r['mean_delta']:+}  (std {r['delta_std']}, {args.seeds} seeds)")
    print(f"  => {'VALIDATED (correction lowers held-out Brier)' if r['validated'] else 'NOT validated'}\n")


if __name__ == "__main__":
    main()
