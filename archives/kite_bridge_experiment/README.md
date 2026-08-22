# Kite bridge — abandoned experiment

**Status: incomplete. Preserved as history, not as working code.**

`kite_bridge.py` is what survived of the live signal bridge for the
`intraday_ml_research` pipeline. It is a single orphaned method —
`get_ranked_signals(self, ...)` with no enclosing class — so it cannot be
imported or run in this state.

It is kept here because the experiment was real even though the file is not
complete. The dry-run execution layer it belonged to did operate on live market
data: see
[`results/intraday_ml_research/paper_trading_dry.log`](../../results/intraday_ml_research/paper_trading_dry.log),
which logs signal generation, ATR stop exits, 60-minute maximum holds, and Nifty
regime filtering across 2 and 12 April 2026. No live orders were ever placed.

The intended design, per the original project notes: dry-run mode, signal
deduplication against open positions, time-of-day gating through the 11:30–13:30
lunch window, and automatic horizon exits at 30 minutes to match backtest logic.
What remains implements only the ranking and threshold step.

**Do not run this.** It is here so the record of what was attempted is complete.
