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
    wins          = trades_df[trades_df["pnl"] > 0]
    losses        = trades_df[trades_df["pnl"] < 0]
    n             = len(trades_df)
    pf            = round(wins["pnl"].sum() / abs(losses["pnl"].sum()), 2) if len(losses) else float("inf")

    ec = equity_curve.copy()
    ec["peak"]     = ec["capital"].cummax()
    ec["drawdown"] = (ec["capital"] - ec["peak"]) / ec["peak"] * 100

    exit_counts = trades_df["exit_reason"].value_counts()

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
        "Trading Days":      trades_df["date"].nunique(),
        "Avg Trades / Day":  round(n / trades_df["date"].nunique(), 1),
        "SL Hit %":          f"{exit_counts.get('SL',0)/n*100:.1f}%",
        "Target Hit %":      f"{exit_counts.get('TARGET',0)/n*100:.1f}%",
        "Time Exit %":       f"{exit_counts.get('TIME',0)/n*100:.1f}%",
        "EOD Exit %":        f"{exit_counts.get('EOD',0)/n*100:.1f}%",
    }
