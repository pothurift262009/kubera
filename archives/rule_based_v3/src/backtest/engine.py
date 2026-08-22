"""
Backtest Engine — with verbose logging + date range filtering
"""
from datetime import time as dtime, date as date_type
import pandas as pd
import random
from src.strategy.vwap_reversion import VWAPReversionStrategy, Signal
from src.utils import db
from src.utils.logger import setup_logger

logger = setup_logger("Backtest")


class BacktestEngine:
    def __init__(self, config: dict, stock_data: dict):
        self.config       = config
        self.stock_data   = stock_data
        self.strategy     = VWAPReversionStrategy(config)
        self.brokerage    = config["costs"]["brokerage"]
        self.slippage_pct = config["costs"]["slippage_pct"]
        self.max_pos      = config["strategy"]["max_positions"]
        self.initial_cap  = config["capital"]["total"]

    # ------------------------------------------------------------------ #

    def run(self, start_date=None, end_date=None, progress_cb=None, log_cb=None) -> dict:
        db.clear_backtest_trades()

        def log(msg):
            logger.info(msg)
            if log_cb:
                log_cb(msg)

        all_dates = self._all_dates()

        if start_date:
            all_dates = [d for d in all_dates if d >= start_date]
        if end_date:
            all_dates = [d for d in all_dates if d <= end_date]

        if not all_dates:
            return {"error": "No trading dates found in selected range."}

        log(f"Backtest period : {all_dates[0]} → {all_dates[-1]}")
        log(f"Trading days    : {len(all_dates)}")
        log(f"Symbols loaded  : {len(self.stock_data)}")
        log(f"Capital         : ₹{self.initial_cap:,.0f}")
        log(f"Max positions   : {self.max_pos}/day")
        log(f"Risk/trade      : {self.config['strategy']['risk_pct']*100:.1f}%")
        log("─" * 50)

        capital      = self.initial_cap
        equity_curve = [{"date": str(all_dates[0]), "capital": capital}]
        all_trades   = []
        month_pnl    = {}

        for i, date in enumerate(all_dates):
            if progress_cb:
                progress_cb(i / len(all_dates))

            day_trades, day_pnl = self._run_day(date, capital, log)
            all_trades.extend(day_trades)
            capital += day_pnl
            equity_curve.append({"date": str(date), "capital": capital})

            # Monthly summary log
            month_key = str(date)[:7]
            month_pnl[month_key] = month_pnl.get(month_key, 0) + day_pnl
            if date == all_dates[-1] or str(all_dates[i+1] if i+1 < len(all_dates) else date)[:7] != month_key:
                log(f"Month {month_key} | P&L: ₹{month_pnl[month_key]:+,.0f} | Capital: ₹{capital:,.0f}")

        if progress_cb:
            progress_cb(1.0)

        log("─" * 50)
        log(f"DONE | Trades: {len(all_trades)} | Final Capital: ₹{capital:,.0f}")

        return self._summarise(all_trades, equity_curve, capital)

    # ------------------------------------------------------------------ #

    def _run_day(self, date, capital: float, log) -> tuple:
        signals     = []
        cap_per_pos = capital / self.max_pos
        symbols     = list(self.stock_data.keys())
        random.shuffle(symbols)

        for symbol in symbols:
            if len(signals) >= self.max_pos:
                break
            sigs = self.strategy.generate_signals(self.stock_data[symbol], date, cap_per_pos)
            if sigs:
                signals.append(sigs[0])

        if not signals:
            return [], 0.0

        log(f"[{date}] {len(signals)} signal(s) found")

        trades, pnl = [], 0.0
        for sig in signals:
            t = self._simulate(sig, date)
            if t:
                emoji = "✅" if t["pnl"] > 0 else "❌"
                log(f"  {emoji} {sig.symbol} {sig.direction} | "
                    f"entry={sig.entry_price} exit={t['exit_price']} "
                    f"qty={sig.quantity} | P&L=₹{t['pnl']:+.0f} [{t['exit_reason']}]")
                trades.append(t)
                pnl += t["pnl"]
                db.log_trade(t)

        return trades, pnl

    # ------------------------------------------------------------------ #

    def _simulate(self, sig: Signal, date) -> dict | None:
        df     = self.stock_data[sig.symbol]
        day_df = df[df["datetime"].dt.date == date].copy()
        day_df = self.strategy._add_vwap(day_df)

        entry_ts = pd.Timestamp(sig.candle_time)
        post     = day_df[day_df["datetime"] > entry_ts]

        exit_price  = sig.entry_price
        exit_reason = "EOD"
        exit_time   = str(day_df["datetime"].iloc[-1]) if not day_df.empty else sig.candle_time
        position    = {"direction": sig.direction, "sl": sig.sl,
                       "entry_price": sig.entry_price, "target": sig.target}

        for _, c in post.iterrows():
            if c["datetime"].time() >= dtime(15, 15):
                exit_price, exit_reason, exit_time = c["open"], "EOD", str(c["datetime"])
                break
            vwap   = c.get("vwap", sig.target)
            candle = {"high": c["high"], "low": c["low"], "close": c["close"]}
            result = self.strategy.check_exit(position, candle, vwap)
            if result == "SL":
                exit_price, exit_reason, exit_time = sig.sl, "SL", str(c["datetime"])
                break
            elif result == "TARGET":
                exit_price, exit_reason, exit_time = vwap, "TARGET", str(c["datetime"])
                break

        gross     = ((exit_price - sig.entry_price) if sig.direction == "LONG"
                     else (sig.entry_price - exit_price)) * sig.quantity
        entry_val = sig.entry_price * sig.quantity
        exit_val  = exit_price * sig.quantity
        costs     = self.brokerage * 2 + (entry_val + exit_val) * self.slippage_pct
        net_pnl   = gross - costs

        return {
            "date": str(date), "symbol": sig.symbol,
            "direction": sig.direction,
            "entry_price": round(sig.entry_price, 2),
            "exit_price": round(exit_price, 2),
            "quantity": sig.quantity,
            "sl": round(sig.sl, 2), "target": round(sig.target, 2),
            "pnl": round(net_pnl, 2),
            "pnl_pct": round(net_pnl / entry_val * 100, 3) if entry_val else 0,
            "exit_reason": exit_reason,
            "entry_time": sig.candle_time, "exit_time": exit_time,
            "mode": "backtest",
        }

    def _all_dates(self) -> list:
        s = set()
        for df in self.stock_data.values():
            s.update(df["datetime"].dt.date.unique())
        return sorted(s)

    def _summarise(self, trades: list, equity_curve: list, final_cap: float) -> dict:
        if not trades:
            return {"error": "No trades generated. Try lowering vwap_deviation_pct in config.yaml"}

        df = pd.DataFrame(trades)
        ec = pd.DataFrame(equity_curve)
        wins   = df[df["pnl"] > 0]
        losses = df[df["pnl"] < 0]
        pf     = round(wins["pnl"].sum() / abs(losses["pnl"].sum()), 2) if len(losses) else float("inf")

        ec["peak"]     = ec["capital"].cummax()
        ec["drawdown"] = (ec["capital"] - ec["peak"]) / ec["peak"] * 100

        return {
            "total_trades":     len(df),
            "win_rate":         round(len(wins) / len(df) * 100, 1),
            "total_pnl":        round(df["pnl"].sum(), 2),
            "final_capital":    round(final_cap, 2),
            "return_pct":       round((final_cap - self.initial_cap) / self.initial_cap * 100, 2),
            "avg_win":          round(wins["pnl"].mean(), 2)   if len(wins)   else 0,
            "avg_loss":         round(losses["pnl"].mean(), 2) if len(losses) else 0,
            "profit_factor":    pf,
            "max_drawdown_pct": round(ec["drawdown"].min(), 2),
            "trades_df":        df,
            "equity_curve":     ec,
        }
