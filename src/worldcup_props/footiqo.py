"""Current-tournament data engine (Footiqo).

Replaces the stale 2024 DB as the source for structural props. Provides three
things, deliberately weighted by how much each can be trusted:

1. **Tournament base-rate distributions** (n≈85 games) for "X+ total shots /
   corners / cards" props — empirical P(total >= k). This is solid; use it.
2. **Shrunk, opponent-agnostic team rates** for "team X+ SOT / more corners".
   A team has only 3-4 games, so the raw average is noise — we shrink it toward
   the tournament-wide single-team mean (empirical Bayes) so a fluke game can't
   dominate. Trust grows with games played.
3. **Match lambdas from the local odds file** (H/D/A + O/U de-vigged and
   calibrated) — no web search needed.

The design rule (per the project's hard-won lessons): the base rates are banked,
the team rates are a *shrunk prior* not a bet, and everything is meant to be
cross-checked, not blindly overridden. See `validate_against` for the flagger.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .closed_form import CountRate, goal_props

# contest/broadcast name -> the name as stored in the Footiqo files
NAME_ALIASES = {
    "united states": "USA", "us": "USA",
    "dr congo": "D.R. Congo", "drc": "D.R. Congo", "congo dr": "D.R. Congo",
    "bosnia and herzegovina": "Bosnia & Herzegovina",
    "czechia": "Czech Republic",
    "cote d'ivoire": "Ivory Coast", "côte d'ivoire": "Ivory Coast",
    "south korea": "South Korea", "korea republic": "South Korea",
}

# stat key -> (home column, away column) in the source CSVs
_ATTACK = {"shots": ("HTSFT", "ATSFT"), "shots_on_target": ("HSONFT", "ASONFT")}
_CORNERS = {"corners": ("HCFT", "ACFT"), "yellows": ("HYCFT", "AYCFT")}


def _norm(team: str) -> str:
    return NAME_ALIASES.get(team.strip().lower(), team.strip())


def _read(path: Path) -> list[dict]:
    raw = "".join(ln for ln in path.read_text(encoding="utf-8-sig").splitlines(keepends=True) if ln.strip())
    return list(csv.DictReader(io.StringIO(raw)))


def _int(v) -> int | None:
    v = (v or "").strip()
    return int(v) if v.lstrip("-").isdigit() else None


@dataclass
class Match:
    home: str
    away: str
    stats: dict = field(default_factory=dict)  # (side, stat) -> int
    odds: dict = field(default_factory=dict)


class Footiqo:
    def __init__(self, folder: str | Path):
        folder = Path(folder)
        attack = {r["id"]: r for r in _read(folder / "Database - Attack  Poss - World Cup - CS.csv")}
        corners = {r["id"]: r for r in _read(folder / "Database - Corners  Cards - World Cup - CS.csv")}
        odds = {r["id"]: r for r in _read(folder / "Database - Odds - World Cup - CS.csv")}
        reds = {(_norm(r["homeTeam"]), _norm(r["awayTeam"])): r
                for r in _read(folder / "worldcup_red_cards_2026.csv")}
        self.matches: list[Match] = []
        for mid, ar in attack.items():
            h, a = _norm(ar["homeTeam"]), _norm(ar["awayTeam"])
            m = Match(home=h, away=a)
            for stat, (hc, ac) in _ATTACK.items():
                m.stats[("H", stat)] = _int(ar.get(hc)); m.stats[("A", stat)] = _int(ar.get(ac))
            cr = corners.get(mid, {})
            for stat, (hc, ac) in _CORNERS.items():
                m.stats[("H", stat)] = _int(cr.get(hc)); m.stats[("A", stat)] = _int(cr.get(ac))
            rr = reds.get((h, a), {})
            hred, ared = _int(rr.get("home_red_cards")) or 0, _int(rr.get("away_red_cards")) or 0
            # cards = yellows + reds
            for side, red in (("H", hred), ("A", ared)):
                y = m.stats.get((side, "yellows"))
                m.stats[(side, "cards")] = (y + red) if y is not None else None
            od = odds.get(mid, {})
            m.odds = {k: float(od[k]) for k in ("H", "D", "A", "O25", "U25") if od.get(k)}
            self.matches.append(m)

    # ------------------------------------------------------------------ base rates
    def _totals(self, stat: str, exclude: tuple | None = None) -> np.ndarray:
        vals = []
        for m in self.matches:
            if exclude and {m.home, m.away} == set(exclude):
                continue
            h, a = m.stats.get(("H", stat)), m.stats.get(("A", stat))
            if h is not None and a is not None:
                vals.append(h + a)
        return np.array(vals)

    def total_rate(self, stat: str, k: int, exclude: tuple | None = None) -> float | None:
        """Empirical P(total `stat` across both teams >= k) over all games."""
        vals = self._totals(stat, exclude)
        return float((vals >= k).mean()) if len(vals) else None

    # ------------------------------------------------------------------ team rates
    def _team_vals(self, team: str, stat: str, exclude: tuple | None = None) -> list[int]:
        team = _norm(team)
        out = []
        for m in self.matches:
            if exclude and {m.home, m.away} == set(exclude):
                continue
            if m.home == team and m.stats.get(("H", stat)) is not None:
                out.append(m.stats[("H", stat)])
            elif m.away == team and m.stats.get(("A", stat)) is not None:
                out.append(m.stats[("A", stat)])
        return out

    def _single_team_mean(self, stat: str) -> float:
        vals = [m.stats[(s, stat)] for m in self.matches for s in ("H", "A")
                if m.stats.get((s, stat)) is not None]
        return float(np.mean(vals)) if vals else 0.0

    def team_rate(self, team: str, stat: str, k_prior: float = 4.0,
                  exclude: tuple | None = None) -> CountRate | None:
        """Shrunk team rate: (n·team_mean + k·tournament_mean)/(n+k), empirical Bayes.

        Small n → pulled toward the tournament single-team mean (a fluke game can't
        dominate); more games → trusts the team. Variance keeps the pooled dispersion.
        """
        vals = self._team_vals(team, stat, exclude)
        if not vals:
            return None
        n = len(vals)
        team_mean = float(np.mean(vals))
        prior = self._single_team_mean(stat)
        shrunk = (n * team_mean + k_prior * prior) / (n + k_prior)
        # dispersion from the pooled single-team distribution
        pool = [m.stats[(s, stat)] for m in self.matches for s in ("H", "A")
                if m.stats.get((s, stat)) is not None]
        disp = (np.var(pool) / max(np.mean(pool), 1e-6)) if pool else 1.0
        return CountRate(mean=shrunk, var=max(shrunk, shrunk * disp))

    # ------------------------------------------------------------------ odds -> lambda
    def match_lambdas(self, home: str, away: str) -> tuple[float, float] | None:
        """De-vig the local odds row and calibrate (lambda_home, lambda_away)."""
        home, away = _norm(home), _norm(away)
        row = next((m for m in self.matches if m.home == home and m.away == away), None)
        if row is None or not {"H", "D", "A", "O25"} <= set(row.odds):
            return None
        o = row.odds
        inv = {k: 1.0 / o[k] for k in ("H", "D", "A")}
        s = sum(inv.values())
        hw, dr, aw = inv["H"] / s, inv["D"] / s, inv["A"] / s
        io_, iu = 1.0 / o["O25"], 1.0 / o.get("U25", 1.0 / (1 - 1 / o["O25"]))
        over = io_ / (io_ + iu)
        best = None
        for lb in np.arange(0.5, 2.8, 0.02):
            for lj in np.arange(0.3, 2.2, 0.02):
                g = goal_props(lb, lj)
                err = (g["home_win"] - hw) ** 2 + (g["draw"] - dr) ** 2 + (g["away_win"] - aw) ** 2 + (g["over_2_5"] - over) ** 2
                if best is None or err < best[0]:
                    best = (err, lb, lj)
        return float(best[1]), float(best[2])
