import numpy as np
import pandas as pd


def sharpe(returns: pd.Series, risk_free: float = 0.065) -> float:
    if returns.std() == 0:
        return 0.0
    excess = returns - risk_free / 252
    return round(float(excess.mean() / excess.std() * np.sqrt(252)), 2)


def compute(trades_df: pd.DataFrame, equity_curve: pd.DataFrame, initial_capital: float) -> dict:
    if trades_df.empty:
        return {}

    daily_pnl     = trades_df.groupby("date")["pnl"].sum()
    daily_returns = daily_pnl / initial_capital

    wins   = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] < 0]
    n      = len(trades_df)

    pf = round(wins["pnl"].sum() / abs(losses["pnl"].sum()), 2) if len(losses) else float("inf")

    ec = equity_curve.copy()
    ec["peak"]     = ec["capital"].cummax()
    ec["drawdown"] = (ec["capital"] - ec["peak"]) / ec["peak"] * 100

    sl_count     = len(trades_df[trades_df["exit_reason"] == "SL"])
    target_count = len(trades_df[trades_df["exit_reason"] == "TARGET"])
    eod_count    = len(trades_df[trades_df["exit_reason"] == "EOD"])

    return {
        "Sharpe Ratio":      sharpe(daily_returns),
        "Max Drawdown":      f"{round(ec['drawdown'].min(), 2)}%",
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
