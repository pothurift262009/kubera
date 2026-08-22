import pandas as pd  # type: ignore
import numpy as np   # type: ignore
import logging

logger = logging.getLogger(__name__)

class Backtester:
    """
    Expert-level Intraday Backtesting Engine.
    - Simulates trades based on model signals and management rules.
    - Models brokerage, slippage, and transaction costs realistically.
    """
    def __init__(self, config):
        self.config = config
        self.confidence_threshold = config["execution"]["confidence_threshold"]
        self.top_n = config["execution"]["top_n_per_step"]
        self.max_pos = config["execution"]["max_positions"]

        # Costs
        self.brokerage = config["backtest"]["brokerage_pct"]
        self.slippage = config["backtest"]["slippage_pct"]
        self.sl_pct = config["backtest"]["stop_loss_pct"]
        self.tp_pct = config["backtest"]["take_profit_pct"]

    def run_backtest(self, test_df, y_prob):
        """Simulates trading and calculates performance metrics."""
        logger.info("Executing trading simulation engine...")

        bt_df = test_df.copy()
        bt_df["y_prob"] = y_prob

        # Filter signals based on confidence threshold
        bt_df["is_signal"] = (bt_df["y_prob"] > self.confidence_threshold).astype(int)

        # Strategy: Top-N selection per timestep
        # We only rank where is_signal == 1
        signals = bt_df[bt_df["is_signal"] == 1].copy()

        if len(signals) == 0:
            logger.warning("ZERO TRADES in backtest. Model not strong enough for current threshold.")
            return None

        # Rank signals at each date to pick Top-N
        signals["rank"] = signals.groupby("date")["y_prob"].rank(ascending=False, method="first")
        trades = signals[signals["rank"] <= self.top_n].copy()

        # Simulation Logic - vectorized
        # Actual return minus costs
        trades["raw_ret"] = trades["future_ret"]
        total_cost = (self.brokerage + self.slippage) * 2 # In and out

        # Apply SL/TP boundaries (Simplified for 5m candles)
        # Assuming we exit at future_ret OR SL/TP
        trades["actual_ret"] = trades["raw_ret"].clip(lower=-self.sl_pct, upper=self.tp_pct)
        trades["pnl"] = trades["actual_ret"] - total_cost

        # Performance analysis
        trade_count = len(trades)
        total_pnl = trades["pnl"].sum()
        win_rate = (trades["pnl"] > 0).mean() * 100
        avg_trade = trades["pnl"].mean() * 100

        # Drawdown calculation
        eq = trades["pnl"].cumsum()
        roll_max = eq.expanding().max()
        dd = roll_max - eq
        max_dd = dd.max()

        # Sharpe Calculation (Dynamic annualization derived from actual bar frequency)
        sharpe = 0
        if trades["pnl"].std() > 0:
            n_unique_days = max(1, len(bt_df["date"].dt.date.unique()))
            bars_per_year = (len(bt_df) / n_unique_days) * 252
            sharpe = (trades["pnl"].mean() / trades["pnl"].std()) * np.sqrt(bars_per_year)

        logger.info(f"--- 🚀 STRATEGY PERFORMANCE ---")
        logger.info(f"Total Transactions: {trade_count:,}")
        logger.info(f"Win Rate (Strategy): {win_rate:.2f}%")
        logger.info(f"Cumulative Alpha: {total_pnl:.4f}")
        logger.info(f"Average Return/Trade: {avg_trade:.4f}%")
        logger.info(f"Max Portfolio Drawdown: {max_dd:.4f}")
        logger.info(f"Estimated Sharpe Ratio: {sharpe:.2f}")
        logger.info(f"-------------------------------")

        return {
            "pnl": total_pnl,
            "win_rate": win_rate,
            "avg_trade": avg_trade,
            "trade_count": trade_count,
            "max_dd": max_dd,
            "sharpe_ratio": sharpe
        }
