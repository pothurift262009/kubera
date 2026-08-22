"""
PaperTrader — simulates live trading without placing real orders.
Identical logic to AutoTrader but all order calls are replaced with logs.
Tracks virtual P&L, positions, and logs everything to DB with mode='paper'.
"""
import threading
from datetime import datetime, time, date
import pandas as pd
from src.data.feed import LiveFeed
from src.strategy.vwap_reversion import VWAPReversionStrategy
from src.execution.broker import KiteBroker
from src.utils import db
from src.utils.logger import setup_logger

logger = setup_logger("PaperTrader")

SQUARE_OFF  = time(15, 15)


class PaperTrader:
    def __init__(self, config: dict, symbols: list):
        self.config  = config
        self.symbols = symbols
        self.broker  = KiteBroker(config)   # used only for LTP / instrument tokens
        self.strat   = VWAPReversionStrategy(config)
        self.max_pos = config["strategy"]["max_positions"]
        self.capital = config["capital"]["total"]

        self._positions:    dict = {}
        self._signals_fired: set = set()
        self._candle_buf:   dict = {}
        self._running            = False
        self._squared_off        = False
        self._daily_pnl          = 0.0
        self._trade_count        = 0

    # ------------------------------------------------------------------ #
    #  Start / Stop
    # ------------------------------------------------------------------ #

    def start(self):
        logger.info("=" * 55)
        logger.info("  PAPER TRADER — no real orders will be placed")
        logger.info(f"  Symbols    : {len(self.symbols)}")
        logger.info(f"  Capital    : ₹{self.capital:,.0f} (virtual)")
        logger.info(f"  Max pos    : {self.max_pos}")
        logger.info(f"  Risk/trade : {self.config['strategy']['risk_pct']*100:.1f}%")
        logger.info("=" * 55)

        token_map = self.broker.get_instrument_tokens(self.symbols)
        if not token_map:
            logger.error("No instrument tokens found. Check symbol names and access token.")
            return

        inv_map = {v: k for k, v in token_map.items()}

        self._feed = LiveFeed(
            self.config["kite"]["api_key"],
            self.config["kite"]["access_token"],
        )
        self._feed.add_candle_callback(self._on_candle)
        self._feed.subscribe(inv_map)
        self._feed.start()
        self._running = True

        threading.Thread(target=self._eod_monitor, daemon=True).start()
        logger.info("PaperTrader live ✅  Waiting for candles...")

    def stop(self):
        self._running = False
        self._feed.stop()
        logger.info(f"PaperTrader stopped. Day P&L=₹{self._daily_pnl:+.0f}  Trades={self._trade_count}")

    # ------------------------------------------------------------------ #
    #  Candle handler
    # ------------------------------------------------------------------ #

    def _on_candle(self, symbol: str, candle: dict):
        if not self._running:
            return

        now = candle["timestamp"].time() if hasattr(candle.get("timestamp", ""), "time") \
              else datetime.now().time()

        logger.info(f"[CANDLE] {symbol} {candle.get('timestamp','')} | "
                    f"O={candle['open']} H={candle['high']} "
                    f"L={candle['low']} C={candle['close']} V={candle['volume']}")

        # ── Square-off ────────────────────────────────────────────────
        if now >= SQUARE_OFF and not self._squared_off:
            logger.info("15:15 — closing all paper positions at market")
            self._square_off_all(candle["close"])
            self._squared_off = True
            return

        if self._squared_off:
            return

        # ── Exit check ────────────────────────────────────────────────
        if symbol in self._positions:
            pos  = self._positions[symbol]
            vwap = self._get_live_vwap(symbol)
            exit_reason = self.strat.check_exit(pos, candle, vwap)

            if exit_reason:
                logger.info(f"[PAPER EXIT] {symbol} | reason={exit_reason} | price={candle['close']}")
                self._close_position(symbol, candle["close"], exit_reason)
            else:
                logger.info(f"[PAPER HOLD] {symbol} {pos['direction']} | "
                            f"entry={pos['entry_price']}  SL={pos['sl']}  "
                            f"target={pos['target']}  VWAP={vwap:.2f}")
            return

        # ── Signal check ──────────────────────────────────────────────
        if symbol in self._signals_fired:
            return

        if len(self._positions) >= self.max_pos:
            return

        self._candle_buf.setdefault(symbol, []).append(candle)
        if len(self._candle_buf[symbol]) < 5:
            return

        df      = self._buf_to_df(symbol)
        today   = date.today()
        cap_per = self.capital / self.max_pos
        signals = self.strat.generate_signals(df, today, cap_per)

        if signals:
            sig = signals[0]
            logger.info(f"[PAPER SIGNAL] {sig.direction} {symbol} | "
                        f"entry={sig.entry_price}  SL={sig.sl}  target={sig.target}  "
                        f"VWAP={sig.vwap}  RSI={sig.rsi}  dev={sig.deviation}%  qty={sig.quantity}")
            logger.info(f"  ⚠️  PAPER MODE — no real order placed")
            self._enter_position(sig)

    # ------------------------------------------------------------------ #
    #  Virtual position management
    # ------------------------------------------------------------------ #

    def _enter_position(self, sig):
        symbol = sig.symbol
        self._positions[symbol] = {
            "direction":   sig.direction,
            "entry_price": sig.entry_price,
            "sl":          sig.sl,
            "target":      sig.target,
            "quantity":    sig.quantity,
            "entry_time":  str(datetime.now()),
            "vwap":        sig.vwap,
        }
        self._signals_fired.add(symbol)
        logger.info(f"[PAPER ENTRY] {sig.direction} {symbol} | "
                    f"qty={sig.quantity}  entry={sig.entry_price}  "
                    f"SL={sig.sl}  target={sig.target}")

    def _close_position(self, symbol: str, price: float, reason: str):
        pos = self._positions.pop(symbol, None)
        if not pos:
            return

        if pos["direction"] == "LONG":
            gross = (price - pos["entry_price"]) * pos["quantity"]
        else:
            gross = (pos["entry_price"] - price) * pos["quantity"]

        costs   = 20 * 2 + (pos["entry_price"] + price) * pos["quantity"] * 0.001
        net_pnl = gross - costs

        self._daily_pnl  += net_pnl
        self.capital     += net_pnl
        self._trade_count += 1

        emoji = "✅" if net_pnl > 0 else "❌"
        logger.info(f"[PAPER CLOSE] {emoji} {symbol} {pos['direction']} | "
                    f"entry={pos['entry_price']}  exit={price}  "
                    f"qty={pos['quantity']}  P&L=₹{net_pnl:+.0f}  [{reason}]")
        logger.info(f"  Day P&L=₹{self._daily_pnl:+.0f}  "
                    f"Virtual Capital=₹{self.capital:,.0f}  "
                    f"Trades today={self._trade_count}")

        db.log_trade({
            "date":        str(date.today()),
            "symbol":      symbol,
            "direction":   pos["direction"],
            "entry_price": pos["entry_price"],
            "exit_price":  round(price, 2),
            "quantity":    pos["quantity"],
            "sl":          pos["sl"],
            "target":      pos["target"],
            "pnl":         round(net_pnl, 2),
            "pnl_pct":     round(net_pnl / (pos["entry_price"] * pos["quantity"]) * 100, 3),
            "exit_reason": reason,
            "entry_time":  pos["entry_time"],
            "exit_time":   str(datetime.now()),
            "mode":        "paper",
        })

    def _square_off_all(self, price: float):
        for symbol in list(self._positions.keys()):
            self._close_position(symbol, price, "EOD")
        logger.info(f"[PAPER EOD] All closed. Day P&L=₹{self._daily_pnl:+.0f}")

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def _get_live_vwap(self, symbol: str) -> float:
        buf = self._candle_buf.get(symbol, [])
        if not buf:
            return 0.0
        df  = self._buf_to_df(symbol)
        typ = (df["high"] + df["low"] + df["close"]) / 3
        return float((typ * df["volume"]).sum() / df["volume"].sum())

    def _buf_to_df(self, symbol: str) -> pd.DataFrame:
        rows = self._candle_buf[symbol]
        df   = pd.DataFrame(rows)
        df["symbol"]   = symbol
        df["datetime"] = pd.to_datetime(df["timestamp"])
        return df

    def _eod_monitor(self):
        import time as tmod
        while self._running:
            if datetime.now().time() >= SQUARE_OFF and not self._squared_off:
                logger.info("[EOD MONITOR] Triggering paper EOD square-off")
                ltp = {}
                try:
                    ltp = self.broker.get_ltp(list(self._positions.keys()))
                except Exception:
                    pass
                for sym in list(self._positions.keys()):
                    price = ltp.get(sym, self._positions[sym]["entry_price"])
                    self._close_position(sym, price, "EOD")
                self._squared_off = True
            tmod.sleep(30)

    def get_status(self) -> dict:
        return {
            "running":       self._running,
            "positions":     self._positions,
            "signals_fired": list(self._signals_fired),
            "daily_pnl":     self._daily_pnl,
            "capital":       self.capital,
            "trade_count":   self._trade_count,
        }
