# Findings

What each iteration was supposed to prove, what it actually showed, and what I
found when I audited my own results.

This document exists because the honest output of this project is not a return
figure — it is a set of diagnoses. Several numbers I previously reported were
wrong. They are corrected here rather than removed.

---

## 1. The headline correction: a reported Sharpe I no longer stand behind

The `intraday_ml_research` iteration was documented with these figures:

| Metric | Reported |
|---|---|
| ROC-AUC | 0.5407 |
| Sharpe ratio | 1.28 |
| Hit rate | 64.5% |
| Max drawdown | < 2% |

**These numbers are mutually inconsistent, and I should have caught it at the
time.** An ROC-AUC of 0.5407 describes a classifier that is barely better than a
coin flip. A 1.28 Sharpe cannot be extracted from a signal that weak. The AUC is
computed directly from model predictions and is the trustworthy figure; the Sharpe
passes through a backtester, and the backtester is where the errors were.

Three defects, each independently sufficient to inflate the result:

**The trading threshold was fitted on the test set.** `main.py` sets
`threshold = probs.quantile(0.90)` where `probs` are the model's predictions *on
the test period*. The decision rule is therefore chosen with knowledge of the data
it is evaluated against.

**Exits were computed against the wrong frame.** `backtester.py` filters to signal
rows first, then applies `trades.groupby('symbol')['close'].shift(-horizon)`. That
shift operates on the filtered rows, so "the close 24 bars later" is really "the
close at this symbol's 24th subsequent *signal*" — which can be weeks away.

**Barriers were applied without path dependency.** P&L is
`clip(horizon_return, -SL, +PT)`. A trade that traded through its stop and
recovered is scored as a winner. The labels use ATR-scaled barriers while the
backtest uses fixed 3.0% / 1.2% ones, so the model was also being evaluated
against a different target than it was trained on.

The equity curve in [`results/intraday_ml_research/`](../results/intraday_ml_research/)
is preserved unedited. It climbs to roughly 2,600 cumulative percentage points in
eleven months on a near-straight line. That shape is the signature of these bugs,
and it is more useful as a diagnostic exhibit than it ever was as a result.

---

## 2. Transaction costs decided every rule-based backtest

Decomposing all three trade ledgers into gross P&L and costs:

| Iteration | Gross | Costs | Net | Median position | Cost/position |
|---|---:|---:|---:|---:|---:|
| VWAP reversion (v3) | +₹7,003 | ₹3,717 | +₹3,286 | ₹24,537 | 0.36% |
| Gap momentum (v10) | +₹4,178 | ₹10,060 | −₹5,882 | ₹10,410 | 0.59% |
| Opening range breakout | −₹6,626 | ₹90,300 | −₹96,927 | ₹9,456 | 0.62% |

Gap momentum found gross-profitable trades and still lost money. The ORB
strategy's gross loss was ₹6,626 — friction accounts for **93%** of the ₹96,927
it actually lost. Its gross win rate was 46.1% against a net win rate of 19.8%.

The cause is the interaction between a flat ₹20-per-leg brokerage and risk-based
position sizing. Sizing positions at 1.5% account risk against an ORB-width stop
produced positions around ₹9,500, where ₹40 of round-trip brokerage is 0.42%
before slippage. The strategy needed to clear ~0.6% per trade just to break even.

**What I would do differently:** enforce a minimum position value derived from the
cost model, not a minimum share count (`MIN_QTY = 3` does not solve this), and
make cost-per-trade a first-class constraint in strategy selection rather than a
deduction applied at the end.

---

## 3. Both profit targets were unreachable

| Iteration | Target hit | Stop hit | Timed out at EOD |
|---|---:|---:|---:|
| VWAP reversion (v3) | 20 | 4 | 18 |
| Gap momentum (v10) | **0** | 60 | 106 |
| Opening range breakout | 124 | 290 | 1,078 |

Gap momentum did not reach its target once in 166 trades. ORB timed out in 72% of
its 1,492 trades, hitting target in 8%. A target that is never reached is not a
target — it is a slow stop-loss with extra steps. Both strategies were effectively
running an exit policy nobody designed.

This is only visible in the exit-reason distribution, which is why the trade
ledgers are published rather than summarised.

---

## 4. Validation defects found during audit

Listed by severity. All are present in the code as committed; none of the
performance figures in this repository should be read as reproducible until they
are addressed.

**Labels and holding periods cross the overnight boundary.** `labeler.py` groups
only by `symbol`, never by session date. A 24-bar horizon starting at 15:25
resolves against the next morning's prices. For a project whose entire premise is
intraday, this quietly imports overnight gap moves that no live intraday system
could capture.

**The LOB holdout is not a holdout.** In `lob_microstructure_ml/main.py`,
`TimeSeriesSplit` runs across the entire dataframe, and the "test set" is then
taken as `df_labeled.iloc[split_idx:]`. Because `apply_triple_barrier_elite_v8`
sorts by `['symbol', 'datetime']`, that slice is the last *symbols alphabetically*
across their full history — not the last 20% in time. The evaluation set was both
trained on and misconstructed.

**Order-book ask sizes are the wrong quantity.** `lob_processing.py` maps
`L{i}-SellNo` to `L{i}-AskSize`. `SellNo` is the number of orders resting at a
level, not their size. Every ask-side depth, queue-imbalance, and OFI feature is
computed against a different variable than intended — which undermines the
microstructure work specifically.

**Order-flow imbalance crosses symbol boundaries.** `get_ofi_delta` is called on
whole chunks rather than per symbol, so differences are taken across the boundary
between one symbol's last row and the next symbol's first. Rolling features also
reset at file-chunk edges, and a `(symbol, 5-minute)` bucket spanning two chunks
emits duplicate rows.

**The ORB backtest is not deterministic.** `engine.py` calls `random.shuffle` on
the symbol list, unseeded, to decide which signals fill the position limit. The
same code on the same data returns different results per run.

**The opening range is 20 minutes, not 15.** `_get_orb` computes
`orb_end = first_bar + 15min` and then filters `datetime <= orb_end`, which admits
a fourth 5-minute bar. Every stop derived from that range is wider than designed.

**Concurrent positions are uncapped.** `MAX_POSITIONS` limits new signals per
timestamp but no open-position ledger is maintained, so with a 24-bar hold the
actual simultaneous exposure exceeds the configured limit.

**`PURGE_BARS` is configured and never used.** `config.py` defines it with the
comment *"Must equal HORIZON_BARS to prevent label leakage."* Nothing reads it.
The train/test split is unpurged and labels straddle the boundary.

**Calibration is not time-series safe.** `CalibratedClassifierCV(base_model, cv=3)`
refits the estimator across folds of the calibration window rather than calibrating
the already-trained model (`cv='prefit'`). The LOB pipeline gets this right; the
XGBoost one does not.

**The leakage sanity check proves nothing.** `sanity_check.py` shuffles labels and
asserts the backtest is unchanged. It always will be — the backtester never reads
labels for P&L. The test passes by construction while appearing to verify
something. It also labels with `triple_barrier_labeling` while training uses
`atr_barrier_labeling`, so it checks a different target than the model learned.

**`analyze_backtest.py` cannot run.** It calls `Backtester.sweep_thresholds`,
which does not exist, and passes `pt=`/`sl=` to `atr_barrier_labeling`, whose
parameters are `atr_multiplier_pt`/`atr_multiplier_sl`.

**Early-pipeline outputs carry their own flags.** `threshold_sweep_v2.csv`
contains a single row with a profit factor of exactly 1.0 — a placeholder, not a
measurement. Timestamps in `filtered_signals_v2.csv` (e.g. `03:45:00`) are UTC
while the rest of the pipeline assumes IST, an unlabelled 5.5-hour offset.

---

## 5. What the feature importances actually said

The v1 LightGBM ranking in
[`results/early_ml_pipeline/feature_importance.csv`](../results/early_ml_pipeline/feature_importance.csv)
is dominated by `day_progress` — time elapsed in the session — by a wide margin
over any price or volume feature. The model's strongest signal was *what time it
is*, which is intraday seasonality rather than tradeable edge.

The corresponding threshold sweep is equally direct: at a 0.40 threshold,
precision is 0.109 against recall of 0.729. The classifier was firing on 31% of
all bars to capture 73% of positive cases, at roughly the base rate.

Neither of these is a good result. Both are useful ones, and neither was visible
while the outputs sat ungitignored on a laptop.

---

## 6. Corrected order of work

If these results are to become reproducible, in dependency order:

1. Seed the RNG in the backtest engine — nothing else is measurable until runs repeat.
2. Add session-boundary grouping to labels and holding periods.
3. Fix the exit-price shift to operate on the bar series, not the signal subset.
4. Rebuild the LOB split chronologically and exclude the holdout from CV.
5. Verify `SellNo` against the raw feed and rebuild the ask-side features.
6. Wire up `PURGE_BARS`; select thresholds on validation data only.
7. Add a portfolio simulator that tracks concurrent positions, capital, and path-dependent exits.
8. Re-run against a frozen dataset and publish whatever comes out, including if it is flat or negative.
