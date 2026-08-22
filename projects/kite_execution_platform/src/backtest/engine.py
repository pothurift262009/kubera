"""
Backtest Engine
---------------
Iterates over every trading day in the dataset.
For each day:
  1. Ask strategy for signals across all symbols.
  2. Simulate each trade candle-by-candle (SL / TARGET / EOD exit).
  3. Deduct realistic costs (brokerage + slippage).
  4. Log trades to SQLite.
"""
from datetime import time
import pandas as pd
from src.strategy.orb import ORBStrategy, Signal
from src.utils import db
from src.utils.logger import setup_logger

logger = setup_logger("BacktestEngine")


class BacktestEngine:
    def __init__(self, config: dict, stock_data: dict):
        self.config       = config
        self.stock_data   = stock_data
        self.strategy     = ORBStrategy(config)
        self.brokerage    = config["costs"]["brokerage"]       # ₹20 per leg
        self.slippage_pct = config["costs"]["slippage_pct"]
        self.max_pos      = config["strategy"]["max_positions"]
        self.initial_cap  = config["capital"]["total"]

    # ------------------------------------------------------------------ #

    def run(self, start_date=None, end_date=None, progress_cb=None) -> dict:
        db.clear_backtest_trades()

        dates = self._all_dates()
        if start_date: dates = [d for d in dates if d >= start_date]
        if end_date:   dates = [d for d in dates if d <= end_date]

        capital        = self.initial_cap
        equity_curve   = [{"date": str(dates[0]), "capital": capital}]
        all_trades     = []

        for i, date in enumerate(dates):
            if progress_cb:
                progress_cb(i / len(dates))

            day_trades, day_pnl = self._run_day(date, capital)
            all_trades.extend(day_trades)
            capital += day_pnl
            equity_curve.append({"date": str(date), "capital": capital})

        if progress_cb:
            progress_cb(1.0)

        logger.info(f"Backtest done. Trades={len(all_trades)}  Capital=₹{capital:,.0f}")
        return self._summarise(all_trades, equity_curve, capital)

    # ------------------------------------------------------------------ #

    def _run_day(self, date, capital: float):
        import random
        signals     = []
        cap_per_pos = capital / self.max_pos

        symbols = list(self.stock_data.keys())
        random.shuffle(symbols)

        for symbol in symbols:
            if len(signals) >= self.max_pos:
                break
            df   = self.stock_data[symbol]
            sigs = self.strategy.generate_signals(df, date, cap_per_pos)
            if sigs:
                signals.append(sigs[0])

        trades, pnl = [], 0.0
        for sig in signals:
            t = self._simulate(sig, date)
            if t:
                trades.append(t)
                pnl += t["pnl"]
                db.log_trade(t)

        return trades, pnl

    def _simulate(self, sig: Signal, date) -> dict | None:
        df     = self.stock_data[sig.symbol]
        day_df = df[df["datetime"].dt.date == date]

        entry_ts = pd.Timestamp(sig.candle_time)
        post     = day_df[day_df["datetime"] > entry_ts]

        exit_price  = sig.entry_price
        exit_reason = "EOD"
        exit_time   = str(day_df["datetime"].iloc[-1]) if not day_df.empty else sig.candle_time

        for _, c in post.iterrows():
            # Hard EOD square-off at 15:15
            if c["datetime"].time() >= time(15, 15):
                exit_price  = c["open"]
                exit_reason = "EOD"
                exit_time   = str(c["datetime"])
                break

            if sig.direction == "LONG":
                if c["low"]  <= sig.sl:
                    exit_price, exit_reason, exit_time = sig.sl,     "SL",     str(c["datetime"]); break
                if c["high"] >= sig.target:
                    exit_price, exit_reason, exit_time = sig.target, "TARGET", str(c["datetime"]); break
            else:
                if c["high"] >= sig.sl:
                    exit_price, exit_reason, exit_time = sig.sl,     "SL",     str(c["datetime"]); break
                if c["low"]  <= sig.target:
                    exit_price, exit_reason, exit_time = sig.target, "TARGET", str(c["datetime"]); break

        # --- P&L calc ---
        if sig.direction == "LONG":
            gross = (exit_price - sig.entry_price) * sig.quantity
        else:
            gross = (sig.entry_price - exit_price) * sig.quantity

        entry_val = sig.entry_price * sig.quantity
        exit_val  = exit_price      * sig.quantity
        costs     = self.brokerage * 2 + (entry_val + exit_val) * self.slippage_pct
        net_pnl   = gross - costs

        return {
            "date":        str(date),
            "symbol":      sig.symbol,
            "direction":   sig.direction,
            "entry_price": round(sig.entry_price, 2),
            "exit_price":  round(exit_price,      2),
            "quantity":    sig.quantity,
            "sl":          round(sig.sl,           2),
            "target":      round(sig.target,       2),
            "pnl":         round(net_pnl,          2),
            "pnl_pct":     round(net_pnl / entry_val * 100, 3) if entry_val else 0,
            "exit_reason": exit_reason,
            "entry_time":  sig.candle_time,
            "exit_time":   exit_time,
            "mode":        "backtest",
        }

    # ------------------------------------------------------------------ #

    def _all_dates(self) -> list:
        s = set()
        for df in self.stock_data.values():
            s.update(df["datetime"].dt.date.unique())
        return sorted(s)

    def _summarise(self, trades: list, equity_curve: list, final_cap: float) -> dict:
        if not trades:
            return {"error": "No trades generated. Check data path and column names."}

        df = pd.DataFrame(trades)
        ec = pd.DataFrame(equity_curve)

        wins       = df[df["pnl"] > 0]
        losses     = df[df["pnl"] < 0]
        total_win  = wins["pnl"].sum()
        total_loss = abs(losses["pnl"].sum())
        pf         = round(total_win / total_loss, 2) if total_loss else float("inf")

        ec["peak"]     = ec["capital"].cummax()
        ec["drawdown"] = (ec["capital"] - ec["peak"]) / ec["peak"] * 100

        return {
            "total_trades":     len(df),
            "win_rate":         round(len(wins) / len(df) * 100, 1),
            "total_pnl":        round(df["pnl"].sum(), 2),
            "final_capital":    round(final_cap, 2),
            "return_pct":       round((final_cap - self.initial_cap) / self.initial_cap * 100, 2),
            "avg_win":          round(wins["pnl"].mean(),   2) if len(wins)   else 0,
            "avg_loss":         round(losses["pnl"].mean(), 2) if len(losses) else 0,
            "profit_factor":    pf,
            "max_drawdown_pct": round(ec["drawdown"].min(), 2),
            "trades_df":        df,
            "equity_curve":     ec,
        }
