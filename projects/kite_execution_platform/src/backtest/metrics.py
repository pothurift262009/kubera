import numpy as np
import pandas as pd


def sharpe(returns: pd.Series, risk_free: float = 0.065) -> float:
    if returns.std() == 0:
        return 0.0
    daily_rf = risk_free / 252
    excess   = returns - daily_rf
    return round(float(excess.mean() / excess.std() * np.sqrt(252)), 2)


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd   = (equity - peak) / peak * 100
    return round(float(dd.min()), 2)


def compute(trades_df: pd.DataFrame, equity_curve: pd.DataFrame, initial_capital: float) -> dict:
    if trades_df.empty:
        return {}

    daily_pnl     = trades_df.groupby("date")["pnl"].sum()
    daily_returns = daily_pnl / initial_capital

    wins   = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] < 0]

    total_win  = wins["pnl"].sum()
    total_loss = abs(losses["pnl"].sum())
    pf         = round(total_win / total_loss, 2) if total_loss else float("inf")

    sl_count     = len(trades_df[trades_df["exit_reason"] == "SL"])
    target_count = len(trades_df[trades_df["exit_reason"] == "TARGET"])
    eod_count    = len(trades_df[trades_df["exit_reason"] == "EOD"])
    n            = len(trades_df)

    return {
        "Sharpe Ratio":      sharpe(daily_returns),
        "Max Drawdown":      f"{max_drawdown(equity_curve['capital'])}%",
        "Win Rate":          f"{len(wins)/n*100:.1f}%",
        "Profit Factor":     pf,
        "Avg Trade P&L":     f"₹{trades_df['pnl'].mean():,.0f}",
        "Avg Win":           f"₹{wins['pnl'].mean():,.0f}"   if len(wins)   else "—",
        "Avg Loss":          f"₹{losses['pnl'].mean():,.0f}" if len(losses) else "—",
        "Best Trade":        f"₹{trades_df['pnl'].max():,.0f}",
        "Worst Trade":       f"₹{trades_df['pnl'].min():,.0f}",
        "Total Trades":      n,
        "Avg Trades / Day":  round(n / trades_df["date"].nunique(), 1),
        "SL Hit %":          f"{sl_count/n*100:.1f}%",
        "Target Hit %":      f"{target_count/n*100:.1f}%",
        "EOD Exit %":        f"{eod_count/n*100:.1f}%",
    }
