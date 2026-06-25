#!/usr/bin/env python3
"""Held-out validation for supremacy-conditioned contest-agent corrections.

The standard backtest cannot gate corrections that key on "who is the favorite",
because the historical match data carries no pre-match supremacy signal (no odds,
no ELOs) -- every match resolves to an even matchup, so the correction never
fires there. This script is the correct gate instead: it derives a supremacy
signal directly from the data, fits the empirical favorite-dominance rate on a
TRAIN split, and measures whether keying on supremacy lowers Brier on a held-out
TEST split versus the flat ~0.50 baseline the simulator produces on these props.

The supremacy signal and the outcome stat are SEPARATE: supremacy must come from
a strength proxy (shots-on-target differential), while the outcome can be any
stat. This matters for fouls/cards -- a foul differential is NOT a strength
signal (weak teams foul more), so testing "underdog fouls more" requires
`--supremacy-stat shots_on_target --stat fouls`.

A negative mean delta (corrected - flat) across seeds means the correction's
direction generalizes out-of-sample and earns its place.

Usage:
    python tools/holdout_validation.py --stat shots_on_target --period second_half
    python tools/holdout_validation.py --stat corners --period full
    python tools/holdout_validation.py --stat fouls --supremacy-stat shots_on_target
"""

from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict

import numpy as np

PERIODS = ("full", "first_half", "second_half")


def _team_value(total, first_half, period):
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


def load_values(db: str, stat: str, period: str) -> dict:
    """match_id -> {team: period_value}, for matches with both teams present."""
    con = sqlite3.connect(db)
    rows = con.execute(
        f"SELECT match_id, team, {stat} AS total, first_half_{stat} AS fh "
        "FROM team_match_stats"
    ).fetchall()
    by_match: dict[object, dict] = defaultdict(dict)
    for match_id, team, total, fh in rows:
        v = _team_value(total, fh, period)
        if v is not None:
            by_match[match_id][team] = v
    return {mid: d for mid, d in by_match.items() if len(d) == 2}


def validate(db: str, stat: str, period: str, seeds: int = 8, supremacy_stat: str | None = None) -> dict:
    supremacy_stat = supremacy_stat or stat
    sup = load_values(db, supremacy_stat, "full")  # strength signal (full match)
    out = load_values(db, stat, period)             # outcome
    common = sorted(set(sup) & set(out), key=str)
    deltas, flats, corrs = [], [], []
    bucket_acc: dict[int, list[float]] = defaultdict(list)

    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(common))
        train_ids = {common[i] for i in order[: len(order) // 2].tolist()}

        # team strength ratings from the SUPREMACY stat, TRAIN only
        diffs: dict[str, list[float]] = defaultdict(list)
        for mid in train_ids:
            (ta, va), (tb, vb) = sup[mid].items()
            diffs[ta].append(va - vb)
            diffs[tb].append(vb - va)
        rating = {t: float(np.mean(v)) for t, v in diffs.items() if len(v) >= 2}

        def fav_outcome(mid):
            """y = 1 if favorite has more of the outcome stat, plus the bucket."""
            (ta, _), (tb, _) = sup[mid].items()
            if ta not in rating or tb not in rating:
                return None
            spr = rating[ta] - rating[tb]
            fav, und = (ta, tb) if spr >= 0 else (tb, ta)
            of, ou = out[mid].get(fav), out[mid].get(und)
            if of is None or ou is None:
                return None
            y = 1.0 if of > ou else 0.5 if of == ou else 0.0
            return _supremacy_bucket(spr), y

        # TRAIN empirical favorite-dominance rate by bucket
        train_hits: dict[int, list[float]] = defaultdict(list)
        for mid in train_ids:
            r = fav_outcome(mid)
            if r:
                train_hits[r[0]].append(r[1])
        brate = {b: float(np.mean(v)) for b, v in train_hits.items() if v}
        for b, rt in brate.items():
            bucket_acc[b].append(rt)

        # TEST Brier: flat 0.50 vs supremacy-bucket predictor
        bf, bc = [], []
        for mid in common:
            if mid in train_ids:
                continue
            r = fav_outcome(mid)
            if not r:
                continue
            b, y = r
            p = brate.get(b, 0.5)
            bf.append((0.5 - y) ** 2)
            bc.append((p - y) ** 2)
        flats.append(np.mean(bf))
        corrs.append(np.mean(bc))
        deltas.append(np.mean(bc) - np.mean(bf))

    return {
        "stat": stat,
        "period": period,
        "supremacy_stat": supremacy_stat,
        "seeds": seeds,
        "bucket_rates": {b: round(float(np.mean(v)), 3) for b, v in sorted(bucket_acc.items())},
        "flat_brier": round(float(np.mean(flats)), 4),
        "corrected_brier": round(float(np.mean(corrs)), 4),
        "mean_delta": round(float(np.mean(deltas)), 4),
        "delta_std": round(float(np.std(deltas)), 4),
        "validated": bool(np.mean(deltas) < 0),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="data/worldcup_props.sqlite")
    ap.add_argument("--stat", default="shots_on_target", help="outcome stat being corrected")
    ap.add_argument("--period", default="second_half", choices=PERIODS)
    ap.add_argument("--supremacy-stat", default=None,
                    help="strength signal (default: same as --stat; use shots_on_target for fouls/cards)")
    ap.add_argument("--seeds", type=int, default=8)
    args = ap.parse_args()
    r = validate(args.db, args.stat, args.period, args.seeds, args.supremacy_stat)
    labels = {0: "~even", 1: "small", 2: "clear", 3: "strong"}
    print(f"\nHold-out: P(favorite has more {r['stat']} in {r['period']}) "
          f"[supremacy = {r['supremacy_stat']} diff]")
    print("  favorite-dominance rate by supremacy (train): "
          + ", ".join(f"{labels[b]}={v}" for b, v in r["bucket_rates"].items()))
    print(f"  flat 0.50 baseline Brier : {r['flat_brier']}")
    print(f"  supremacy-corrected Brier: {r['corrected_brier']}")
    print(f"  mean delta (corr-flat)   : {r['mean_delta']:+}  (std {r['delta_std']}, {args.seeds} seeds)")
    print(f"  => {'VALIDATED (lowers held-out Brier)' if r['validated'] else 'NOT validated (no robust edge)'}\n")


if __name__ == "__main__":
    main()
