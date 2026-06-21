# World Cup Prop Probability Forecaster

## RBP Lab Performance Dashboard

This repository also includes a local Streamlit GUI for evaluating the
generative simulator and a Claude baseline against settled crowd probabilities.
It starts with sample data and writes dashboard records to `rbp.db`, separate
from the model-training database.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

On macOS, you can instead double-click **Open RBP Lab.command** in Finder. It
starts the local server and opens `http://localhost:8501` automatically. Keep
the Terminal window open while using RBP Lab; close it or press `Control-C` to
stop the app.

Use **Add / Edit Match** to upload model and Claude PDFs plus a settlement CSV,
or enter every row manually. The settlement CSV accepts case-insensitive
variants of:

```csv
question_text,crowd_prob,outcome,weight
Over 2.5 total goals,0.55,1,1.0
```

A combined CSV may also include `p_model`, `p_claude`, and `category`. Outcomes
accept `1/0`, `yes/no`, or `void`. Probabilities are stored on a `0–1` scale;
percent strings are normalized during import.

Dashboard assumptions:

- Question text is similar enough across sources for normalized and fuzzy
  matching. Every inferred or unmatched row remains visible in the editor.
- Void questions remain in SQLite but do not contribute to any metric.
- Competition-specific RBP normalization is not known. The sidebar exposes
  weighted sum/mean aggregation and a reversible score-sign convention.
- SQLite stores the canonical convention where positive means better than the
  comparison forecast. Flipping the sidebar sign changes the view, not history.

The chart toolbar camera exports PNG files. Dedicated PNG download buttons are
also shown when the local Plotly image engine is available. The dashboard
exports filtered question records and category/probability-bin crowd
calibration as CSV.

### Direct Result Folder Import

The **Import Result Folders** page can read settled Sports Predict PDFs directly
from:

```text
~/Desktop/Prediction results/Model
~/Desktop/Prediction results/Claude
```

Override the parent folder with `PREDICTION_RESULTS_DIR`. The importer pairs
filename variants such as `SPA_CV`/`SPN_CVD`, runs macOS Vision OCR locally,
and caches OCR results under `data/ocr_cache`. No PDF content is uploaded.
Complete cards supply model probability, Claude probability, crowd probability,
and outcome, so they can be sent directly to the dashboard without a settlement
CSV.

The match trend uses the official RBP printed in the page-one match summary
(for example, `United States vs Paraguay = +28.58`). It does not reconstruct
that match total from OCR question rows. Matches without an official page-one
score, including seed/demo records, are excluded from the official trend. The
trend is ordered by the settled match date extracted from the PDF.

Some browser-generated PDFs clip a card when it crosses a printed page break.
Those incomplete cards are explicitly listed in the preview and excluded; the
importer does not infer missing probabilities.

### Fixture Submission Reminders

The reliable reminder channels are independent of Streamlit:

1. Open **Fixtures & Reminders**, enter or import fixtures, then download the
   combined `.ics`. Import it into Apple Calendar, Google Calendar, or Outlook.
   Each kickoff event includes a native alarm 30 minutes before kickoff by
   default.
2. Schedule `reminder_runner.py` once per minute. It reads `rbp.db`, ignores
   submitted/skipped fixtures, records `reminded_at`, and auto-marks passed
   pending fixtures as missed.

The calendar is generated directly without an extra ICS dependency. Kickoffs
are stored as timezone-aware UTC timestamps; manual and CSV values without a
timezone are interpreted in `LOCAL_TZ` (`America/New_York` by default).

Run one reminder check manually:

```bash
source .venv/bin/activate
python reminder_runner.py --once
python reminder_runner.py --lead 30 --channel stdout
```

Desktop delivery uses `plyer`. If native notification delivery is unavailable,
the runner writes a conspicuous reminder to stdout so cron or scheduler logs
still capture it.

Example cron entry:

```cron
* * * * * cd "/absolute/path/to/Jump-World-Cup" && "/absolute/path/to/Jump-World-Cup/.venv/bin/python" reminder_runner.py --once >> reminder.log 2>&1
```

On Windows Task Scheduler, create a basic task with a one-minute repeating
trigger. Set **Program/script** to the virtual environment's `python.exe`, set
**Add arguments** to `reminder_runner.py --once`, and set **Start in** to the
repository directory.

Reminder configuration is available through environment variables:

```bash
export REMINDER_LEAD_MINUTES=30
export UPCOMING_WINDOW_HOURS=24
export LOCAL_TZ=America/New_York
export REMINDER_CHANNEL=desktop  # desktop, email, both, or stdout
export EMAIL_REMINDERS_ENABLED=false
```

Email is off by default and credentials are never stored in the repository.
To opt in, set `EMAIL_REMINDERS_ENABLED=true` and provide `SMTP_HOST`,
`SMTP_PORT`, `SMTP_USER`, `SMTP_PASS`, and `REMINDER_EMAIL_TO`.

Fixture assumptions:

- CSV imports require `match_label,kickoff`; optional columns are `tz`,
  `competition_stage`, `submission_status`, and `notes`.
- Fixture labels are upserted case-insensitively to avoid accidental duplicates.
- The standalone runner and imported calendar are the reliable channels. The
  in-app banner only runs while Streamlit is open.
- A reminder fires once in its current lead-time window. The re-nag rule is
  isolated in `rbp_lab/reminders.py`.

A Python 3.11+ project for calibrated probabilities on international football
match-stat props:

- `P(team A has more fouls/corners/offsides than team B)`
- `P(team A has more corners than team B at halftime)`
- `P(team A records at least k fouls/corners/offsides)`
- player shots, shots on target, goals, assists, and goal-or-assist props
- team cards plus penalty/red-card match events

The optimization target is Brier score, not classification accuracy. The
pipeline therefore exposes tie mass, applies hierarchical shrinkage, supports
market blending, and tunes a final calibration shrinkage on temporal holdouts.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp config.example.json config.json
worldcup-props --config config.json init-db
```

Install the optional soccerdata adapter with:

```bash
pip install -e ".[soccerdata]"
```

All paths in the config are resolved relative to the current working
directory. The default raw cache is `data/raw`; HTTP providers never refetch a
cached URL/parameter combination.

## Data Refresh

### Normalized CSV

`data/matches_template.csv` is the canonical import contract. One row is one
match and contains both teams' full-match and optional first-half statistics.

```bash
worldcup-props --config config.json ingest-csv path/to/matches.csv
```

Imports are idempotent on `(source, source_match_id)`.

### StatsHub (Primary)

StatsHub exposes the needed statistics through the read-only JSON feeds used
by its public team-stat pages. The importer collects:

- full-time fouls, corners, offsides, shots on target, and possession
- first-half fouls, corners, offsides, and shots on target
- World Cup referee matches, fouls per game, and cards per game

The checked-in `data/statshub_teams.csv` is populated with USA, Paraguay,
Mexico, and South Africa as a working example:

```bash
worldcup-props --config config.json ingest-statshub
worldcup-props --config config.json validate \
  --output artifacts/validation.json
```

Add the remaining national teams using their StatsHub team IDs and
confederations. The current World Cup fixture feed can discover listed teams:

```bash
worldcup-props --config config.json statshub-discover \
  --output data/statshub_discovered_teams.csv
```

Review the discovered file, fill its confederation column, then merge it into
`data/statshub_teams.csv`. Discovery reflects fixtures currently returned by
StatsHub and should not be assumed to be a complete tournament roster.

Each refresh requests up to 500 matches per team. Responses are cached under
`data/raw/statshub`, duplicate events are merged by event ID, and repeat
imports are idempotent. These are public site endpoints rather than a
documented API contract, so run tests and validation before retraining.

### FBref (Optional Fallback)

There are two supported routes:

1. Use soccerdata to export its raw schedule, misc, and possession tables.
2. Generate a manifest from an FBref competition schedule page, then ingest
   every linked match report with the built-in polite, cached parser.

```bash
worldcup-props --config config.json fbref-export \
  --league INT-World-Cup --season 2022

worldcup-props --config config.json build-fbref-manifest \
  "https://fbref.com/en/comps/1/2022/schedule/2022-World-Cup-Scores-and-Fixtures" \
  data/fbref_manifest.csv \
  --competition "FIFA World Cup" \
  --competition-type world_cup

worldcup-props --config config.json ingest-fbref data/fbref_manifest.csv
```

`data/fbref_manifest_template.csv` is intentionally only a header schema;
`build-fbref-manifest` attempts to populate it. The generated rows contain the
match-report URL, FBref match ID, date, teams, competition tags, and neutral
flag. Confederations, Elo ratings, and referee names can be enriched before
ingestion when those inputs are available.

The manifest parser handles full-match fouls, corners, offsides, and possession
when those rows exist in the report. First-half fields remain null unless they
are supplied by a normalized CSV or another provider. The model then uses its
empirical pooled first-half share.

Respect FBref's terms and robots policy. The client identifies itself, waits
3.5 seconds between uncached requests, and caches every raw response.

### API-Football

Set `API_FOOTBALL_KEY`. `APIFootballProvider` provides cached fixture and
fixture-statistics calls for a custom refresh script. Normalize provider
responses to `MatchRow` or the CSV contract before ingestion; this keeps
provider-specific IDs and naming decisions outside the statistical model.

### Referees

Use `data/referees_template.csv` for Transfermarkt/WhoScored-derived or manual
referee aggregates:

```bash
worldcup-props --config config.json ingest-referees data/referees.csv
```

Observed match-level referee totals and manual aggregates are both shrunk
toward the global foul mean using `referee_prior_matches`.

### Validation

```bash
worldcup-props --config config.json validate --output artifacts/validation.json
```

Validation checks:

- matches without exactly two team rows
- missing core statistics
- duplicate date/home/away combinations
- negative or implausibly large values
- first-half values greater than full-match values
- per-team coverage counts for each stat and first-half corners

Issues are also persisted in `data_quality_issues`.

## Model

### Fouls: Total Then Split

The simulator draws:

1. Match total `T` from a negative binomial.
2. Home foul share from a beta distribution.
3. Home fouls from `Binomial(T, share)` and away fouls as the remainder.

The total log-rate includes the shrunken referee effect, competition context,
and absolute Elo gap. If no referee is supplied, each simulation integrates
over the empirical referee distribution.

The split logit includes:

- each team's sample-size-shrunk historical foul share
- projected possession deficit
- pressing-proxy difference, when available

The beta concentration captures match-to-match split overdispersion.

### Cards, Rare Events, and Players

Cards, penalty/red-card events, and player events are generated inside the
same six-segment simulation as goals, fouls, corners, offsides, and shots on
target:

- team cards depend on the simulated fouls, team card history, game state, and
  the selected referee's cards-per-match rate
- penalty and red-card hazards depend on simulated foul/box pressure,
  supremacy, and referee tendency
- each team's simulated shots on target and goals are allocated across
  available players plus an explicit `__other__` bucket, so allocations equal
  team totals in every draw

Player profiles are club-derived per-90 inputs. Load them with:

```bash
worldcup-props --config config.json ingest-player-profiles \
  data/player_profiles.csv
```

Use `data/player_profiles_template.csv` as the schema. Lineup uncertainty is a
separate input:

```bash
worldcup-props --config config.json ingest-lineups data/lineups.csv
```

`data/lineups_template.csv` accepts probabilistic pre-lineup entries or
confirmed starter/bench/out statuses. Confirmed lineups collapse the minutes
mixture; `predict-player --explain` reports the probability change from the
configured pre-lineup baseline.

### Territory Stats and Game State

Corners, offsides, and shots on target use negative-binomial rates combining:

- the team's shrunken attacking rate
- the opponent's shrunken concession rate
- an Elo/history possession projection
- a nonlinear possession/supremacy split learned from paired team data

If `data/club_dominance.csv` is present, training uses it for the dominance
regression; otherwise it estimates from the international database with
regularized league-average priors. The optional club file should contain
paired team rows with `match_id`, `team`, `is_home`, both Elo columns,
`possession`, and the three territory stats.

Every simulation first draws a six-segment goal path. Trailing teams receive
configurable chasing multipliers and leaders receive configurable leading
multipliers, so halftime and second-half questions use the realized game state
rather than a fixed full-match scaling.

### Shared Match Flow

The v2 candidate simulator can draw one latent match-flow value per Monte Carlo match.
The favorite and underdog receive opposite, mean-preserving multipliers for
goals, corners, shots on target, and offsides. This makes dominant-side counts
positively correlated rather than independent margins. `shared_normal` is the
retained default; set it to `none` for the v1 ablation baseline.

Market-calibrated goal totals also propagate into territory-stat volume using
configurable elasticities, so lower scoring expectations reduce simulated
shots and corners coherently.

Player props use role-aware base-rate floors for thin profiles. Confirmed
attacking starters shrink toward a positional prior instead of toward zero,
and `predict-player --explain` reports the role base rate, expected-minutes
fraction, team-context multiplier, profile data weight, and any anti-zero
floor applied.

Team rates shrink first toward confederation rates and then indirectly toward
the global pool. Recency weights use a configurable half-life, and friendlies
receive a configurable downweight.

### Halftime

Team-specific first-half rates are used where available. Sparse or absent
coverage shrinks toward the empirical pool-wide first-half share of the
full-match rate.

## Train and Backtest

Fit an uncalibrated model:

```bash
worldcup-props --config config.json train
```

Run the fixed temporal holdouts:

```bash
worldcup-props --config config.json backtest \
  --output artifacts/backtest.json
```

The holdout windows are:

- 2022 World Cup: November 20 through December 18, 2022
- Euro 2024: June 14 through July 14, 2024
- Copa America 2024: June 20 through July 14, 2024

For every tournament, training data is strictly earlier than its opening date.
The report includes raw and calibrated Brier scores, calibration tables,
reliability/resolution/uncertainty decomposition, and comparison with:

- a 50/50 baseline adjusted for tie mass on strict comparison props
- a confederation/global average-rate negative-binomial baseline

It emits a warning whenever the calibrated model loses to either baseline.
Optional `backtest_prop_weights` in the config applies contest-style weights
to aggregate Brier scores and calibration fitting.

Fit the final model and install the Brier-optimal per-prop shrinkage parameters:

```bash
worldcup-props --config config.json train --calibrate \
  --backtest-report artifacts/backtest.json \
  --backtest-simulations 10000
```

Increase backtest simulations for the final contest run if compute permits.
Prediction defaults to exactly 100,000 simulations.

Per-prop recalibration compares linear shrinkage with a Platt map using
leave-one-tournament-out scoring, then stores the selected map fitted on all
historical holdouts. Reliability tables are reported before and after.

Run cumulative feature ablations in build order:

```bash
worldcup-props --config config.json ablation \
  --simulations 5000 --output artifacts/ablation.json
```

Each stage reports per-prop Brier deltas and a `retained` decision.

## Predict

```bash
worldcup-props --config config.json predict \
  --home USA --away Paraguay \
  --stat offsides --type threshold --k 2

worldcup-props --config config.json predict \
  --home "South Africa" --away Mexico \
  --stat fouls --type more_than --referee "Name"

worldcup-props --config config.json predict \
  --home USA --away Paraguay \
  --stat corners --type halftime_more_than --explain

worldcup-props --config config.json predict \
  --home USA --away Paraguay \
  --stat cards --type more_than --referee "Name"

worldcup-props --config config.json predict-player \
  --home Canada --away Bosnia \
  --team Canada --player "Jonathan David" \
  --event goals --k 1 --explain

worldcup-props --config config.json predict-player \
  --home USA --away Paraguay \
  --team Paraguay --player "Julio Enciso" \
  --event second_half_shots_on_target --k 1 --explain

worldcup-props --config config.json predict-event \
  --home USA --away Paraguay --event penalty_or_red --explain
```

JSON output contains:

- final blended probability
- market-calibrated simulator, explicit Elo-fallback model-only, and raw probabilities
- market probability, when available
- `P(home > away)`, `P(tie)`, and `P(home < away)`
- 80% simulation interval for the count or count difference
- recency-weighted effective sample sizes for both teams
- fitted goal lambdas, market fit residuals, rate, territory, referee,
  calibration, and simulation metadata

`--explain` adds an ordered audit containing the base rate, raw simulator,
latent-flow summary, market fusion, recalibration, ESS gate, optional field
bias, crowd-anchor risk control, and final probability.

### Coverage Safeguards

Coverage safeguards are enabled by default. They prevent sparse or unknown
teams from producing confident count-stat forecasts:

- team count shares and rates are capped to configurable plausible ranges
- market-driven count-volume propagation is capped
- if either team is below the configured effective-match threshold, the
  simulated probability is pulled toward its calibrated base rate
- a definition-matched match market contributes limited pseudo-coverage, but
  cannot make missing historical count data look fully observed
- missing player profiles remain a hard error rather than triggering an
  invented player estimate

The raw and guarded probabilities, effective-match counts, market coverage
credit, cap activation, and warnings are available in `--explain`.

### Tie Definition

`tie_handling` is deliberately configurable:

- `strict`: event is only `A > B` (default)
- `half`: event probability includes half the tie mass
- `home`: event probability includes all tie mass
- `away`: ties do not count for the home event

Confirm the contest's exact wording before submitting. The three-way masses
are always reported separately, regardless of this setting.

## Batch Mode

Start with `data/questions_template.csv`:

```bash
worldcup-props --config config.json batch \
  data/questions.csv artifacts/submission.csv
```

The output retains every input column and appends probabilities, tie mass,
interval bounds, and effective sample sizes.

## Match Markets and Goal Calibration

Set `THE_ODDS_API_KEY`, then fetch supported World Cup markets:

```bash
worldcup-props --config config.json ingest-odds-api \
  --sport soccer_fifa_world_cup
```

After starting lineups are announced, update the lineup CSV and explicitly
refresh prices:

```bash
worldcup-props --config config.json ingest-lineups data/lineups.csv
worldcup-props --config config.json refresh-lineup-markets \
  --sport soccer_fifa_world_cup
```

This is the supported formation/lineup adjustment: refreshed market goal
lambdas flow back through the latent match state and all count processes.
Pair-synergy embeddings remain an ablation-gated, disabled stub.
Unlike routine refreshes, `refresh-lineup-markets` bypasses the local odds
cache before replacing it with the post-lineup response.

The manual fallback accepts prices from any bookmaker or prediction market:

```bash
worldcup-props --config config.json ingest-markets data/markets.csv
```

Required columns are:

```text
match,market,selection,decimal_odds,book,timestamp
```

Optional `line` and `definition` columns are supported. Raw implied and
de-vigged probabilities are persisted. Two-way markets default to Shin,
while 1X2 uses proportional normalization. Aggregation uses Pinnacle when
present and otherwise the median fair probability across books.

Before each simulation, 1X2, totals, and Asian-handicap probabilities are fit
to a Dixon-Coles score matrix by weighted least squares. The resulting
`lambda_home` and `lambda_away` drive the score path and all state-dependent
count processes. Output includes every target residual and clearly reports
`model_only` when no usable market is available.

Use `--no-market` to force the Elo fallback. Direct question markets are
blended in logit space only when `data/market_definitions.json` explicitly
marks settlement definitions as matching. Liquid markets default to 70%
market weight; thin player or penalty markets default to 50%.

## Results Log

Enter one result or ingest `data/results_template.csv`:

```bash
worldcup-props --config config.json log-result \
  --match "South Korea vs Czechia" --question "South Korea win" \
  --question-type match_winner --submitted 0.43 --crowd 0.47 \
  --outcome 1 --market-blended 0.44

worldcup-props --config config.json ingest-results data/results.csv
worldcup-props --config config.json results-report \
  --output artifacts/results_report.json
```

The report gives cumulative and per-question-type Brier scores for submissions,
the crowd, and retroactive market-blended forecasts, plus Brier decomposition
and crowd bias. Tournament backtests also compare model-only and
market-calibrated lambdas wherever historical market snapshots are available.

The field-bias layer is implemented but disabled by default. Enable it only
after enough crowd outcomes exist; adjustments are capped by
`field_bias_max_deviation`. Manager-regime weighting is configured through
`manager_regime_start_dates`; regime dates after a historical cutoff are
ignored to prevent leakage.

The crowd-anchor layer is enabled by default as contest risk control, not as a
football model input. Fixated question types such as penalty/red-card OR,
compound BTTS+3, 4+ cards, and 2+ offsides are kept inside empirically observed
crowd bands; loose buckets such as match winner, corners, second-half SOT
comparison, and player second-half SOT pass through. If settled `forecast_results`
show the crowd forecast for a type moving over the tournament, the anchor shifts
slightly and reports that drift in `--explain`.

## Tests

```bash
pytest
```

Tests cover de-vigging, Brier decomposition, shrinkage behavior, SQLite
validation, artifact round-tripping, probability partitioning, referee
conditioning, and an end-to-end threshold forecast on synthetic data.
