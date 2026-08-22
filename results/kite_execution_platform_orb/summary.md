# Opening range breakout (projects/kite_execution_platform)

**1,492 trades · 887 trading days · Mar 2021 – Feb 2026 · net −₹96,927 (−96.9%)**

| | |
|---|---|
| Gross P&L / costs / net | −₹6,626 / ₹90,300 / **−₹96,927** |
| Win rate (gross / net) | 46.1% / 19.8% |
| Profit factor | 0.24 |
| Max drawdown | −₹98,011 |
| Avg win / avg loss | +₹101 / −₹106 |
| Exits | 124 target · 290 stop · 1,078 EOD |
| Direction split | 974 long / 518 short |
| Median position value | ₹9,456 (cost 0.62% of position) |

**The largest sample and the worst result — published because of both.** Nearly
five years and 1,492 trades makes this the only iteration with enough data to say
anything, and what it says is that the strategy as implemented destroys capital.

The decomposition is what makes it useful. Gross P&L was −₹6,626 — close to
break-even. Costs of ₹90,300 account for **93% of the realised loss.** The gross
win rate of 46.1% collapses to 19.8% net: roughly a quarter of all trades were
profitable before friction and unprofitable after it.

72% of trades ended in an end-of-day timeout, target hit in 8%.

**Two known code defects affect these numbers directly** and are documented in
[`docs/findings.md`](../../docs/findings.md#4-validation-defects-found-during-audit):
signal selection uses an unseeded `random.shuffle`, so this ledger is one draw
rather than a reproducible figure; and the opening range spans 20 minutes rather
than the configured 15, widening every stop. Treat this as a diagnosis, not a
measurement.

`trades.csv` is the full ledger. `equity_curve.csv` is daily cumulative P&L.
