# Intraday ML research outputs (projects/intraday_ml_research)

> The reported metrics for this iteration — 1.28 Sharpe, 64.5% hit rate — **do not
> survive audit.** The reported ROC-AUC of 0.5407 does, and it describes a signal
> barely better than chance. See
> [`docs/findings.md`](../../docs/findings.md#1-the-headline-correction-a-reported-sharpe-i-no-longer-stand-behind).

## What is here

| File | What it is |
|---|---|
| `equity_curve.png` | Backtest equity curve at the 0.90-quantile threshold |
| `equity_curve_t33/38/55/65/70.png` | The same backtest at five fixed probability thresholds — a sensitivity study |
| `feature_importance.png` | Top-15 XGBoost feature importances |
| `paper_trading_dry.log` | Live dry-run log, 2 and 12 April 2026 |

## Reading the equity curves

`equity_curve.png` rises to roughly 2,600 cumulative percentage points over eleven
months on a nearly straight line. Real intraday equity curves do not look like
this. The shape is produced by three defects — a threshold fitted on the test set,
exit prices shifted against the wrong frame, and barriers applied without path
dependency — all documented in the findings.

They are preserved unedited because the curve is a useful exhibit of what
leakage looks like when it is plotted.

The five threshold variants remain genuinely informative as a *relative*
comparison: they show how signal count and curve shape respond to the decision
threshold, independent of the absolute level being inflated.

## The paper-trading log

`paper_trading_dry.log` records the dry-run execution layer operating on live
market data: signal generation with a Nifty regime filter, ATR-based stop exits,
60-minute maximum holds, and API timeout handling. Sample:

```
EXIT ADANIPORTS | Reason: ATR Stop Loss | Entry: 1336.60 | Exit: 1342.80 | PnL: -0.46%
EXIT APOLLOHOSP | Reason: Max hold 60min | Entry: 7191.00 | Exit: 7181.00 | PnL: +0.14%
```

No live orders were placed. The surviving bridge code is an incomplete fragment,
preserved and labelled in
[`archives/kite_bridge_experiment/`](../../archives/kite_bridge_experiment/).
