# Project Brief — Market-Anchored Probabilistic Forecasting Engine

*Self-contained summary for resume / portfolio use. Paste into any chat tool to iterate.*

## One-line
A Python system that forecasts soccer match-event ("prop") probabilities by anchoring a Monte Carlo simulator to de-vigged betting-market odds, with a held-out-validated correction layer; competed live in a crowd-forecasting contest scored on relative Brier skill (RBP) and consistently beat the crowd.

## What it does (technical)
- **Monte Carlo match simulator** — Dixon-Coles goal model + negative-binomial count props (shots-on-target, corners, cards, fouls, offsides); 60k+ simulated paths per match yield per-prop probabilities.
- **Market calibration** — fits goal intensities (λ_home, λ_away) to de-vigged 1X2 / totals / BTTS lines via weighted least squares; de-vigging uses Shin and proportional methods. Adding BTTS as a fit target pins the home/away split that 1X2+totals leave under-determined.
- **Odds-ingestion pipeline** — DraftKings HAR parser + manual/web CSV → SQLite calibration store.
- **Contest-agent correction layer** — post-model adjustments for systematic miscalibration, each gated by held-out validation.
- **Validation harness** — train/test, multi-seed held-out Brier for supremacy-conditioned corrections.

## Results (honest, quantified)
- **Beat the contest crowd** on relative-Brier (RBP) scoring across the tournament — e.g., a representative slate scored **+220 RBP over 59 props (~75% beat the crowd)**.
- **Favorite-2nd-half-SOT correction**: fit on 2,408 historical matches, **held-out Brier −0.023 (8-seed)**; recovered ~+5 RBP on a settled loss.
- **Corner-inflation correction**: held-out Brier **−0.021**.
- **Refuted** an "underdog fouls more" heuristic via the same held-out test (delta **+0.0005**, no edge) and kept it out of the model — discipline over intuition.
- **Edge decomposition**: ~39% from a lineup-information advantage, ~61% from model calibration.

## Methodology highlights (the part that signals rigor)
- **Outcome-grounded, held-out validation** — corrections fit to actual outcomes, validated on a held-out split, not tuned to crowd numbers.
- **Caught my own tooling limit**: the standard backtest couldn't gate a supremacy-conditioned correction (historical data had no odds/ELOs → no favorite signal), so I built a dedicated held-out validator instead — a concrete in-sample vs out-of-sample lesson.
- **Separated calibration alpha from information alpha** and reasoned explicitly about overfitting and edge decay.
- **No hardcoded magic constants** — all correction magnitudes measured from data.

## Honest scope (say this in interviews — accuracy reads as maturity)
- It's a **hybrid**: market-anchored simulator + validated correction layer, with **manual lineup/odds inputs**. Not a fully autonomous agent.
- Final submission is manual (the contest platform blocks automation).

## Skills / keywords
Probabilistic modeling · Monte Carlo simulation · Bayesian/MLE calibration · backtesting & held-out validation · market microstructure (de-vigging, Shin) · Python · SQLite · data pipelines · model evaluation (Brier / calibration).

## Resume bullets (pick 3–4)
- Built a market-anchored Monte Carlo forecaster: fit Dixon-Coles goal intensities to de-vigged 1X2/totals/BTTS lines, simulated 60k+ paths, derived per-prop probabilities.
- Engineered an odds-ingestion pipeline (DraftKings HAR parser, manual/web → SQLite) with Shin/proportional de-vigging.
- Diagnosed systematic miscalibration from settled results and encoded a supremacy-conditioned correction fit to 2,408 matches, lowering held-out Brier 0.023.
- Found the standard backtest couldn't validate the correction (no historical odds/ELOs) and built a reusable train/test, multi-seed held-out harness — keeping fixes outcome-grounded, not overfit.
- Beat the contest crowd on relative-Brier scoring, with an explicit decomposition separating model-calibration alpha from lineup-information advantage.

## Interview talking points
- Lead with the **rigor, not the win**: "I caught that my own backtest couldn't validate a correction, so I built the right held-out test" beats any score.
- Be ready to whiteboard: **de-vigging, Dixon-Coles, why Brier, train/test leakage.**
- Own the **overfitting discussion** (calibration vs information edge, how the edge could decay) — it signals you know the limits of your own work.
- Don't call it an "autonomous AI agent." It's a hybrid forecasting system — say so.
