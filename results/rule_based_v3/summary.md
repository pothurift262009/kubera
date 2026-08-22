# VWAP mean reversion (archives/rule_based_v3)

**42 trades · 36 trading days · Apr 2023 – Mar 2026 · net +₹3,286 (+3.3%)**

| | |
|---|---|
| Gross P&L / costs / net | +₹7,003 / ₹3,717 / **+₹3,286** |
| Win rate (gross / net) | 78.6% / 66.7% |
| Profit factor | 1.76 |
| Max drawdown | −₹742 |
| Avg win / avg loss | +₹272 / −₹309 |
| Exits | 20 target · 4 stop · 18 EOD |
| Median position value | ₹24,537 (cost 0.36% of position) |

**This is the only profitable iteration and the least conclusive one.** 42 trades
across 36 days in a three-year window is far too small a sample to separate edge
from variance, and the confidence interval on a 66.7% win rate at n=42 spans most
of the useful range.

What it does show: it took positions 2.5× larger than the later strategies, so
fixed brokerage consumed 0.36% of each position instead of 0.62%. On the cost
decomposition that difference is most of the gap between this result and the
others — see [`docs/findings.md`](../../docs/findings.md#2-transaction-costs-decided-every-rule-based-backtest).

`trades.csv` is the full ledger. `equity_curve.csv` is daily cumulative P&L.
