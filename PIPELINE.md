# Forecasting a contest card — RUNBOOK

Read this first before forecasting any card. "The pipeline" = the code in
`src/worldcup_props/` **plus** the rules below. The code without the rules gives
wrong numbers (stale DB, undercounted stars, SOT mispriced vs deep blocks). The
rules are the load-bearing half — they were learned the hard way (see §TRAPS).

---

## THE SEQUENCE (every card)

1. **Lineups** — note (a) which questioned players start vs are **benched**
   (benched star = `lineup` sub-discount), (b) any injuries/rotation.
2. **Market odds** — web-search the match. De-vig moneyline → outcome probs;
   de-vig O/U → total. Grab **anytime-scorer lines** for questioned players and
   the **advance** line. This is the anchor; never hand over model-only outcome/λ.
3. **Calibrate λ** — grid-search `goal_props(λh, λa)` to fit
   (home_win, draw, away_win, over2.5). ~5 lines; see any recent card script.
4. **Run the engine** — `forecast_card(home, away, λh, λa, questions,
   footiqo=Footiqo("Footiqo data"), lineup_status=..., roles=...)`.
   It routes every question; structural rows pull Footiqo (current tournament),
   goal rows come from λ.
5. **Override (only these):**
   - **Scorer / "score-or-assist" for a star → market anytime line.** The role
     prior undercounts stars badly (see TRAPS: star undercount).
   - **Advance → market advance line** (model over-rates the underdog's ET share).
   - **SOT rows → style-check** (see SOT CEILING below).
6. **Apply THE GATE** (below). Submit.

---

## THE GATE (the single most important rule)

Every number is the **model / market / crowd** value **unless** you can tag the
deviation `data`, `market`, or `lineup`. If you can't tag it, you don't move it.

- `market` — a sportsbook line contradicts the model prior (scorer, advance, win).
- `data`   — a current-tournament number the model lacks (Footiqo, team_styles).
- `lineup` — a confirmed team-sheet fact the crowd is mispricing (benched star).

**No narrative fades.** "Cagey game," "they won't finish," "old legs," "physical
knockout," "world champs will pile up" — these are NOT tags. The crowd already
prices them. Every narrative deviation this project made lost points; a flat 50%
once beat a confident-wrong card by ~70 RBP. **When ungrounded, sit near the
crowd/center — do not deviate.**

---

## SOT CEILING (the fix that stops the −47s)

SOT depends on the **opponent's defensive block**, not just the favorite's λ or
star quality. Look the opponent up in **`Footiqo data/team_styles.csv`** (column
`shots_against` = how deep they sit):

- **Deep block** (high `shots_against`: Paraguay 18.5, Norway 14.8, Senegal 11.5)
  → **suppress the favorite's SOT.** Anchor to the opponent's *SOT-conceded*
  rate, NOT the favorite's SOT-for. A bus turns shots into blocks, not SOT.
  (France vs Paraguay: correct ≈ 0.25, not 0.78.)
- **Open / leaky** (low-mid, or a side that concedes SOT: Norway, Brazil) →
  **let SOT scale up** with dominance. (France vs Sweden: correct high, ~0.65.)
- **Suffocator** (low `shots_against`: Spain 4.8, Argentina 6.0) → **fade the
  *opponent's* SOT props** (their opponents get almost nothing on target).

**MAGNITUDE CAUTION on the suffocator anchor (Spain–Belgium, 2026-07-10):**
the direction is real, but don't let a thin `allowed` sample push the anchor to
an extreme. First pass on this card weighted 80% toward Spain's SOT-allowed
rate (0.75/game, **n=4**) for Belgium's team-SOT and De Bruyne's SOT props,
landing at 22% and 27%. Checking the codebase's own attack-defense
multiplicative formula (the same shape as `opponent_adjusted_team_count_rate`)
on the same inputs gives an even MORE extreme 6.6% — a small `allowed` sample
plugged into that kind of formula amplifies rather than corrects, the same
mechanism behind the Argentina corners/SOT disasters. The suffocator signal
itself checked out as real here (Spain suppressed Uruguay and Austria, not
just weak sides, to ≤1 SOT each) — but the opponent GENERATING the SOT
(Belgium) is also a genuinely prolific attacking side (22.2 shots/game raw),
and n=4 is still thin. **Rule: apply the suffocator direction, but cap the
weight on the opponent's allowed-rate at roughly 60-65%, not 80%+ — moderate
the magnitude even when the qualitative read is confirmed, and never trust the
pure multiplicative attack-defense formula on a Footiqo-scale (<8 game)
sample, it's validated for DB-scale (1000+ match) samples only.**

Corners/total-shots scale with dominance **always** (a bus concedes corners).
Only **SOT** needs the block adjustment. Never inflate a favorite's SOT above the
opponent's SOT-conceded rate.

**IMPORTANT CORRECTION (2026-07-07): the contest crowd is NOT known at
submission time — predictions lock before the crowd number is revealed. Any
rule phrased as "default to the crowd" is not executable and must not be
used.** The two Argentina blowout-card SOT/corners losses (Egypt −120.80
combined, Cape Verde −178.98 on SOT alone) triggered an overcorrection: a
"never go below crowd" hard rule that (a) can't actually be followed pre-lock
and (b) turned out, on checking the FULL settled-log history rather than just
these two recent cards, to not even be well-supported — Argentina and Uruguay
have each missed their own historical 6+ SOT prop before. The real, usable fix
has to come from data available before lock: `Model vs the Crowd vs
Outcome.xlsx` (405-row settled log) itself, the same way the offsides fix
worked — a base rate for the PROP TYPE in general, not a peek at any specific
match's crowd.

**Base rates pulled from the settled log for "X or more" threshold props
(re-derive with the query below before trusting a new number on this
category):**
- **SOT team-thresholds ≥5 (n=8):** actual hit rate **62.5%**, vs a typical
  crowd/model estimate around 55-56%. Real counter-examples exist in the same
  sample (Argentina 6+ SOT and Uruguay 6+ SOT both missed) — this is a mild
  upward tilt, not a rule that always fires.
- **Corners "X or more" (n=11, thresholds 3-9):** actual hit rate **63.6%**,
  vs a typical crowd/model estimate around 44-46% — a bigger, more consistent
  gap than SOT.
- Query: filter the log for `'shot' in q and 'target' in q` or `'corner' in q`
  AND a regex `(\d+)\s*or more` threshold match; compare mean(outcome) to
  mean(crowd)/mean(you) for the relevant threshold band. Re-run this before
  every card with a "X+ SOT/corners" question — it's cheap and it's the actual
  usable prior, unlike guessing off 3-4 current-tournament games.

**Tested and rejected as a fix: reweighting the thin Footiqo sample alone
(shrink vs. no-shrink, opponent-adjust vs. not) does not solve this.** On the
Egypt card, dropping only the opponent-adjustment step recovers +13.26 of the
−120.80 lost (still a −107.54 disaster); using the fully raw, unshrunk 3-game
Argentina rate helps SOT (+26.17) but makes corners *worse* (−24.08 — one
fluke 1-corner game in Argentina's own 3-game log drags the raw average down),
netting to only +2.09 overall. **A 3-4 game Footiqo sample is too thin to
support a confident SOT/corners threshold call in either direction, no matter
how it's reweighted — the fix is an outside base rate, not better math on the
same thin data.**

**Practical rule:** for any single-team or match-total "X or more" SOT/corners
prop, first check for a direct sportsbook prop market on that exact stat
(distinct from moneyline/O-U/anytime-scorer — a real pre-lock anchor). If none
exists, pull the settled-log base rate for that threshold band (above) and
blend it with the Footiqo/DB number rather than trusting either alone — shade
a low Footiqo-derived estimate up toward the historical base rate, but don't
swing to an extreme given the real counter-examples. This is a moderate,
quantified tilt, not an absolute law.

**PLAYER-LEVEL SOT (e.g. "[Player] 1+ SOT") — separate rule, confirmed on the
Colombia–Switzerland card.** Q4 Embolo 1+ SOT: submitted 65% using a generic
"forward" role-prior (1.35 SOT/90, no player-specific data), crowd was 56%,
actual NO, −15.59. His REAL per-game log this tournament (findable via web
search, e.g. dimers.com projections or match reports) was 1.0 SOT/game across
4 games (Algeria 2sh/1SOT, Qatar 4sh/1SOT, Canada 2sh/2SOT, Bosnia 1sh/0SOT) —
implying ~0.63, and a direct site projection implied ~0.55, both much closer
to the crowd than the role-prior's 65%. **This is the same avoidable mistake
as Álvarez (see [[project-vision-agent-pipeline]]/settled-card-lessons): a
generic positional role-prior stood in for a specific named player's own
real record. Rule: before submitting ANY named-player SOT/shots/goals prop,
search for that player's own per-game shot/SOT log this tournament first —
only fall back to a role prior if no real log is findable.** This is
distinct from the team-level SOT finding above and was genuinely avoidable
pre-lock (unlike the team-level Switzerland 4+ SOT miss on the same card,
where their own raw 4-game log [5,4,7,7] actually supported an even HIGHER
number than submitted and it still missed — that one looks like real
matchup variance, not a process failure; don't conflate the two).

**INTERNAL CONSISTENCY between a team's own count prop and its players' count
props in the SAME match — confirmed miss on France–Morocco (2026-07-09).** Q6
Morocco 3+ SOT was correctly read bearish (submitted 46% vs crowd 58%, correct
NO, +32.10 — the "cagey, suppressed" match read was right). But Q2 Hakimi 1+
SOT (53% vs crowd 44%, NO, −8.97) and Q4 Díaz 1+ SOT (55% vs crowd 46%, NO,
−11.35) were computed independently off each player's own historical per-game
log, with no adjustment for the same match-level suppression already
identified for their own team. If a team's SOT output is read as suppressed
for THIS match, that isn't independent of whether its individual players clear
their own SOT bar — it's close to mechanically linked (fewer team shots on
target means fewer players individually reaching 1+). **Rule: when a
team-level SOT/shots/corners prop for a match comes out meaningfully below
that team's own historical rate (a "this match is suppressed" read), shade
that same team's individual player SOT props down too, don't compute them in
isolation from a generic per-game log as if the match-level signal doesn't
apply to them.**

**Corners base-rate under-application — France 5+ corners (−13.95, submitted
55% vs crowd 66%, actual YES).** The settled-log tilt for corner thresholds
(actual hit rate 63.6% historically, n=11 — see above) was already known
before this card, but the raw Footiqo estimate (52.4%) only got nudged a few
points to 55% instead of being weighted substantially toward the established
base rate. **When a real settled-log calibration finding exists and materially
diverges from a raw Footiqo/DB number, weight toward it properly — a token
nudge undersells a finding that's already been derived and validated.**

---

## STRUCTURAL PROPS → FOOTIQO, NOT THE DB

The SQLite DB ends **July 2024** — it roughly **doubles card props** and
**suppresses favorite volume**. Use `Footiqo data/` (current tournament) for
shots, SOT, corners, cards. The `footiqo=` arg to `forecast_card` handles this.

- **Cards** — Poisson on the tournament rate (**~2.4 yellows/game**, banked from
  `worldcup_yellow_cards_2026.csv`). Bump modestly for knockout intensity.
  **FETCH THE REFEREE APPOINTMENT before pricing any cards prop — this is a
  real gap, confirmed costly on Spain-Belgium (2026-07-10): submitted 21% for
  4+ cards (crowd 38%, actual YES) → −44.20, the worst row on that card.** The
  referee for that match, Michael Oliver, had issued 13 yellows across 3
  tournament games (the most of any referee) with zero reds/pens — a
  same-tournament, pre-lock, web-searchable fact that would have pushed this
  toward the crowd's number instead of well below it. `card.py`/`closed_form.py`
  never reference referee data at all, even though `Question.referee` and the
  DB's `referees` table already exist for this purpose (from the older
  `forecast.py` simulator path) — the current card pipeline just doesn't wire
  it in. Web-search "[Match] referee appointment [tournament]" and the ref's
  name + "cards this tournament" before finalizing any card-count prop; blend
  their personal rate with the team/tournament baseline the same way a
  sportsbook prop line would override a generic estimate.
  **ALSO CHECK FIXTURE-SPECIFIC INTENSITY, not just the referee stat — confirmed
  gap, England-Argentina SF (2026-07-15).** Q9 "both teams receive a card":
  submitted 40% (crowd 66%, actual YES) −42.38, the worst row on that card.
  The pre-match research had *already surfaced* real, reported context — FBI
  security assessment calling it the "highest risk" game of the tournament,
  press explicitly framing it as "football's angriest rivalry" — and it was
  read, then ignored in favor of the referee's plain statistical card rate.
  **This is not narrative-chasing (THE GATE's "cagey game," "won't finish"
  fades are ungrounded vibes about how a game will play out) — a security
  assessment and a historically hostile fixture are REPORTED FACTS about this
  specific pairing, same evidentiary tier as a referee's card history or a
  lineup change. When research turns up a real fact like this, it has to
  actually move the number, not just get mentioned and dropped.**
- **Corners / shots** — anchor to shot **volume** (a high-shot team draws
  corners even if its raw corner average is low). Opponent-adjust.
- **"X+ total shots"** — pull both teams' actual per-game shot logs, **strip
  minnow-blowout outliers**, opponent-weight. Never a stale-DB average.

## CALIBRATED BASE RATES (exotic props)

From the scraped logs, shrunk vs the crowd (don't bank tiny samples):
- **penalty OR red = 0.30** (was eyeballed 0.40; 13 pens + 9 reds / 82 games,
  shrunk toward crowd 0.38). **red only = 0.11.** **penalty only ≈ 0.25.**
- **goal before 1st hydration break = 0.36** (tournament base; do NOT discount
  it down for "low-tempo" — that lost).
- **offsides — no Footiqo column, but `Model vs the Crowd vs Outcome.xlsx` has 37
  settled rows of "Will [Team] be caught offside 2+ times?"** Actual base rate
  0.459, crowd averaged 0.474 (crowd is well-calibrated here — no exploitable
  bias either direction). Back out a per-team Poisson rate from that base rate
  (solve `1 - poisson.cdf(1, lambda) = 0.459` → lambda ≈ 1.55/team) and convolve
  both teams for match-total offside questions instead of guessing off style
  narrative: P(4+ total) ≈ 0.376, P(3+) ≈ 0.60, P(5+) ≈ 0.20. **Do NOT nudge off
  the crowd/this derived rate for "open/high-press" style reasons — that cost
  −13.82 RBP on the USA-Belgium card** (submitted 0.55 vs crowd 0.46, actual NO).
  (Scrape `worldcup_offsides_2026.csv` to get a real current-tournament number
  and stop leaning on the settled-log proxy.)

  **RESOLVED (France–Spain SF, 2026-07-14): the tension above is confirmed, not
  contradicted.** Submitted 38% for 4+ total offsides (only a token nudge above
  the base rate); the match had already produced 4 offsides in the FIRST HALF
  ALONE, locking the full-match total at ≥4 (a cumulative count only rises) —
  confirmed loss, −22 per the settled card. This is the mirror case of
  USA-Belgium (which nudged UP off vague "both teams open" narrative and lost
  −13.82). **The two results together refine the rule rather than conflict:
  "don't nudge for style" holds for a generic/vague style label on one
  so-so team, but a genuinely extreme matchup (two of the tournament's most
  intense high-line, heavy-press sides, at the semifinal stage) is a real,
  strong signal that deserves a much bigger adjustment than a token nudge —
  not "never adjust," but "match the size of the adjustment to the size of the
  signal." A single "DOMINANT/OPEN" team_styles label is weak evidence; BOTH
  teams being elite, extreme pressing sides in a single match is much
  stronger evidence, and should move the number substantially, not by a point
  or two.**

- **Exotic props with ZERO data source (e.g. "on-field VAR review") — default
  close to the crowd, don't deviate on pure reasoning.** This principle stands
  on its own logic (no data source means no basis for an independently
  "reasoned" number that drifts from crowd — the same discipline already
  applied to offsides before the settled-log fix was found) but is NOT backed
  by the France–Spain SF example that originally prompted it: the settled
  card showed Q4 VAR review as 30% (crowd 42%) → actual YES, −21.58, but the
  organizer's settlement was WRONG — the referee never actually went to the
  pitchside monitor in the first half; that was a scoring error, not a real
  event. **Retracted as evidence — do not cite this specific number again.**
  The "snap to crowd on zero-data props" guidance is kept as a sound
  default, but currently rests on general principle, not a validated example.
  When a settled result looks surprising for a rare/exotic event, it's worth
  a sanity check against what's actually reported to have happened before
  banking a "confirmed miss" lesson from it.

---

## FETCH, DON'T GUESS

Every big miss came from reasoning off a stale model instead of fetching info
that was available:
- Shot counts, card rates, penalties → Footiqo / web (per-game logs exist).
- **Tactical style** (open vs deep block) → knockout previews say it plainly, or
  derive from `team_styles.csv`. You have web access — use it before every card.
- When a scraped file gives a shocking number, **confirm it against a real source
  in 30 seconds** (FotMob/FBref) before banking it.

---

## TRAPS (each one cost real RBP)

- **SOT vs deep block** → overcounted (France-Paraguay, −47). Fix: SOT CEILING.
- **Star SOT/scorer undercount** — Bellingham, Mbappé, Yamal 2+ SOT all hit
  ~0.50 while the role prior said ~0.42. **Stars → market line, not role prior.**
- **Correlated narrative fades** (Spain card) — one team-view smeared across 5
  scoring rows = 5× the downside when wrong. Express a view **once**, at most.
- **Oversized deviations detonate** — RBP is quadratic. A 25-pt miss costs ~6× a
  10-pt miss. Keep structural deviations ≤ ~10 pts of the crowd unless rock-solid.
- **Stale-DB card props** ran 2× too high; **stale-DB favorite SOT** ran too low.
- **Averages lie without opponent context** — Ecuador's "14.7 shots" was one
  27-shot minnow game; Australia's "concedes 15.7" was group games, not a
  knockout bus. Opponent-weight everything.
- **Live in-game counts ≠ forecast** — don't re-derive a pre-game number from the
  70th-minute scoreboard; fix the *prior* for next time instead.

---

## DATA SOURCES

`Footiqo data/`: `Database - Attack Poss` (shots/SOT/off-target/possession),
`Database - Corners Cards`, `Database - Odds`, `worldcup_yellow_cards_2026.csv`,
`worldcup_red_cards_2026.csv`, `worldcup_penalty_awards_2026.csv` (event log,
incomplete — cross-check FotMob), `team_styles.csv` (style + shots_against).
Market odds: web-search per match. `Model vs the Crowd vs Outcome.xlsx` = 405
settled rows = the calibration set for base rates and crowd deltas.
