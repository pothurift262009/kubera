# Gap momentum + VWAP (archives/rule_based_v10)

**166 trades · 97 trading days · Mar 2021 – Jan 2022 · net −₹5,882 (−5.9%)**

| | |
|---|---|
| Gross P&L / costs / net | +₹4,178 / ₹10,060 / **−₹5,882** |
| Win rate (gross / net) | 51.2% / 38.0% |
| Profit factor | 0.62 |
| Max drawdown | −₹5,780 |
| Avg win / avg loss | +₹154 / −₹152 |
| Exits | **0 target** · 60 stop · 106 EOD |
| Median position value | ₹10,410 (cost 0.59% of position) |

**The instructive result.** This strategy was gross profitable — +₹4,178 before
costs — and still lost ₹5,882, because ₹10,060 of friction was larger than the
edge. It is the clearest evidence in the repository that the constraint was
execution economics, not signal quality.

Second finding: **it never hit its profit target once in 166 trades.** Every trade
resolved as a stop or an end-of-day timeout. A target that unreachable means the
exit policy in effect was not the exit policy designed.

`trades.csv` is the full ledger. `equity_curve.csv` is daily cumulative P&L.
