import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import config

logger = logging.getLogger(__name__)


class Backtester:
    def __init__(self, df, model, feature_cols, target_col='label'):
        self.df = df.copy()
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.results = None

        print("\nScoring dataset for backtesting...")
        self.df['prob'] = model.predict_proba(self.df[self.feature_cols])[:, 1]

    def run_backtest(self, threshold=None, pt=None, sl=None,
                     transaction_cost=None, slippage=None,
                     max_positions=None, horizon=None, plot=True):

        if pt is None:
            pt = config.PT_PCT
        if sl is None:
            sl = config.SL_PCT
        if transaction_cost is None:
            transaction_cost = config.TRANSACTION_COST_PCT
        if slippage is None:
            slippage = config.SLIPPAGE_PCT
        if max_positions is None:
            max_positions = config.MAX_POSITIONS
        if horizon is None:
            horizon = config.HORIZON_BARS

        df = self.df.copy()

        # ✅ FIXED: threshold + rank together
        df = df.sort_values(["date", "prob"], ascending=[True, False])
        df["rank"] = df.groupby("date").cumcount() + 1
        df["signal"] = (
            (df["prob"] >= threshold) &
            (df["rank"] <= max_positions)
        ).astype(int)

        trades = df[df["signal"] == 1].copy()

        if trades.empty:
            return self._empty_results()

        trades['prev_trade'] = trades.groupby('symbol')['date'].shift(1)
        trades['gap'] = (
            trades['date'] - trades['prev_trade']
        ).dt.total_seconds() / 60

        min_gap = horizon * 5
        trades = trades[
            (trades['gap'].isna()) | (trades['gap'] >= min_gap)
        ]

        if trades.empty:
            return self._empty_results()

        trades['future_close'] = trades.groupby('symbol')['close'].shift(-horizon)
        trades = trades.dropna()

        trades['ret'] = (trades['future_close'] - trades['close']) / trades['close'] * 100

        trades['pnl'] = np.clip(trades['ret'], -sl, pt) - transaction_cost - slippage

        pnl_series = trades.groupby('date')['pnl'].mean()

        total_trades = len(trades)
        hit_rate = (trades['pnl'] > 0).mean()

        cum_pnl = pnl_series.cumsum()
        max_dd = (cum_pnl - cum_pnl.cummax()).min()
        total_return = cum_pnl.iloc[-1]

        sharpe = (
            pnl_series.mean() / pnl_series.std() * np.sqrt(252 * 75)
            if pnl_series.std() > 0 else 0
        )

        gross_profit = trades[trades['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades[trades['pnl'] <= 0]['pnl'].sum())
        pf = gross_profit / gross_loss if gross_loss > 0 else 0

        self.results = {
            'Total Trades': total_trades,
            'Hit Rate (%)': round(hit_rate * 100, 2),
            'Total Return (%)': round(total_return, 2),
            'Max Drawdown (%)': round(max_dd, 2),
            'Sharpe': round(sharpe, 2),
            'Profit Factor': round(pf, 2),
        }

        if plot:
            self._plot(cum_pnl)

        self._print()
        return self.results

    def _plot(self, pnl):
        plt.figure(figsize=(10, 5))
        pnl.plot()
        plt.title("Equity Curve")
        plt.grid()
        plt.savefig("equity_curve.png")

    def _print(self):
        print("\n--- Backtest Results ---")
        for k, v in self.results.items():
            print(f"{k}: {v}")

    def _empty_results(self):
        return {k: 0 for k in [
            'Total Trades', 'Hit Rate (%)',
            'Total Return (%)', 'Max Drawdown (%)',
            'Sharpe', 'Profit Factor'
        ]}