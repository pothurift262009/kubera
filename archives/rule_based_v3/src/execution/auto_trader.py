"""
AutoTrader — fully automatic live trading engine
-------------------------------------------------
- Connects to Kite WebSocket
- Builds 5-min candles in real time
- Fires VWAP Reversion signals automatically
- Places market entry + SL bracket orders via Kite API
- Squares off all positions at 15:15
- Logs every event verbosely
"""
import threading
from datetime import datetime, time, date
import pandas as pd
from src.data.feed import LiveFeed
from src.strategy.vwap_reversion import VWAPReversionStrategy
from src.execution.broker import KiteBroker
from src.utils import db
from src.utils.logger import setup_logger

logger = setup_logger("AutoTrader")

SQUARE_OFF = time(15, 15)
MARKET_OPEN = time(9, 15)


class AutoTrader:
    def __init__(self, config: dict, symbols: list):
        self.config  = config
        self.symbols = symbols
        self.broker  = KiteBroker(config)
        self.strat   = VWAPReversionStrategy(config)
        self.max_pos = config["strategy"]["max_positions"]
        self.capital = config["capital"]["total"]

        # State
        self._positions: dict  = {}   # {symbol: position_dict}
        self._signals_fired    = set()
        self._candle_buf: dict = {}   # {symbol: [candles]}
        self._sl_orders: dict  = {}   # {symbol: order_id}
        self._running          = False
        self._squared_off      = False
        self._daily_pnl        = 0.0

    # ------------------------------------------------------------------ #
    #  Start / Stop
    # ------------------------------------------------------------------ #

    def start(self):
        logger.info("=" * 55)
        logger.info("  AutoTrader STARTING")
        logger.info(f"  Symbols     : {len(self.symbols)}")
        logger.info(f"  Capital     : ₹{self.capital:,.0f}")
        logger.info(f"  Max pos     : {self.max_pos}")
        logger.info(f"  Risk/trade  : {self.config['strategy']['risk_pct']*100:.1f}%")
        logger.info(f"  Square-off  : {SQUARE_OFF}")
        logger.info("=" * 55)

        token_map = self.broker.get_instrument_tokens(self.symbols)
        if not token_map:
            logger.error("No instrument tokens found. Check symbol names.")
            return

        inv_map = {v: k for k, v in token_map.items()}
        logger.info(f"Subscribed tokens: {list(token_map.keys())}")

        self._feed = LiveFeed(
            self.config["kite"]["api_key"],
            self.config["kite"]["access_token"],
        )
        self._feed.add_candle_callback(self._on_candle)
        self._feed.subscribe(inv_map)
        self._feed.start()
        self._running = True

        # Background thread to monitor square-off time
        threading.Thread(target=self._eod_monitor, daemon=True).start()
        logger.info("AutoTrader live ✅  Waiting for market candles...")

    def stop(self):
        self._running = False
        self._feed.stop()
        logger.info("AutoTrader stopped.")

    # ------------------------------------------------------------------ #
    #  Candle handler (called for every completed 5-min candle)
    # ------------------------------------------------------------------ #

    def _on_candle(self, symbol: str, candle: dict):
        if not self._running:
            return

        now = candle["timestamp"].time() if hasattr(candle.get("timestamp", ""), "time") else datetime.now().time()

        logger.info(f"[CANDLE] {symbol} {candle.get('timestamp','')} | "
                    f"O={candle['open']} H={candle['high']} "
                    f"L={candle['low']} C={candle['close']} V={candle['volume']}")

        # ── Square-off check ──────────────────────────────────────────
        if now >= SQUARE_OFF and not self._squared_off:
            logger.info("15:15 reached — squaring off all positions")
            self._square_off_all(candle["close"])
            self._squared_off = True
            return

        if self._squared_off:
            return

        # ── Exit check for open positions ─────────────────────────────
        if symbol in self._positions:
            pos  = self._positions[symbol]
            vwap = self._get_live_vwap(symbol)
            exit_reason = self.strat.check_exit(pos, candle, vwap)

            if exit_reason:
                logger.info(f"[EXIT] {symbol} | reason={exit_reason} | price={candle['close']}")
                self._close_position(symbol, candle["close"], exit_reason)
            else:
                logger.info(f"[HOLD] {symbol} {pos['direction']} | "
                            f"entry={pos['entry_price']} SL={pos['sl']} target={pos['target']} VWAP={vwap:.2f}")
            return

        # ── Signal check ──────────────────────────────────────────────
        if symbol in self._signals_fired:
            return

        if len(self._positions) >= self.max_pos:
            return

        # Accumulate candles for VWAP/RSI calculation
        self._candle_buf.setdefault(symbol, []).append(candle)
        if len(self._candle_buf[symbol]) < 5:
            return

        df      = self._buf_to_df(symbol)
        today   = date.today()
        cap_per = self.capital / self.max_pos
        signals = self.strat.generate_signals(df, today, cap_per)

        if signals:
            sig = signals[0]
            logger.info(f"[SIGNAL] {sig.direction} {symbol} | "
                        f"entry={sig.entry_price} SL={sig.sl} target={sig.target} "
                        f"VWAP={sig.vwap} RSI={sig.rsi} dev={sig.deviation}% qty={sig.quantity}")
            self._enter_position(sig)

    # ------------------------------------------------------------------ #
    #  Position management
    # ------------------------------------------------------------------ #

    def _enter_position(self, sig):
        symbol = sig.symbol
        try:
            if sig.direction == "LONG":
                oid = self.broker.buy(symbol, sig.quantity)
            else:
                oid = self.broker.sell(symbol, sig.quantity)

            logger.info(f"[ORDER] Entry placed | {sig.direction} {symbol} qty={sig.quantity} | order_id={oid}")

            # Place SL order
            sl_oid = self.broker.place_sl_order(symbol, sig.direction, sig.quantity, sig.sl)
            self._sl_orders[symbol] = sl_oid
            logger.info(f"[ORDER] SL placed | {symbol} sl={sig.sl} | order_id={sl_oid}")

            self._positions[symbol] = {
                "direction":   sig.direction,
                "entry_price": sig.entry_price,
                "sl":          sig.sl,
                "target":      sig.target,
                "quantity":    sig.quantity,
                "entry_time":  str(datetime.now()),
            }
            self._signals_fired.add(symbol)

        except Exception as e:
            logger.error(f"[ERROR] Entry failed for {symbol}: {e}")

    def _close_position(self, symbol: str, price: float, reason: str):
        pos = self._positions.pop(symbol, None)
        if not pos:
            return

        try:
            # Cancel pending SL order
            if symbol in self._sl_orders:
                try:
                    self.broker.cancel_order(self._sl_orders.pop(symbol))
                except Exception:
                    pass

            if pos["direction"] == "LONG":
                self.broker.sell(symbol, pos["quantity"])
                pnl = (price - pos["entry_price"]) * pos["quantity"]
            else:
                self.broker.buy(symbol, pos["quantity"])
                pnl = (pos["entry_price"] - price) * pos["quantity"]

            costs = 20 * 2 + (pos["entry_price"] + price) * pos["quantity"] * 0.001
            net   = pnl - costs
            self._daily_pnl += net
            self.capital    += net

            emoji = "✅" if net > 0 else "❌"
            logger.info(f"[CLOSED] {emoji} {symbol} {pos['direction']} | "
                        f"entry={pos['entry_price']} exit={price} qty={pos['quantity']} "
                        f"P&L=₹{net:+.0f} [{reason}] | Day P&L=₹{self._daily_pnl:+.0f}")

            db.log_trade({
                "date":        str(date.today()),
                "symbol":      symbol,
                "direction":   pos["direction"],
                "entry_price": pos["entry_price"],
                "exit_price":  price,
                "quantity":    pos["quantity"],
                "sl":          pos["sl"],
                "target":      pos["target"],
                "pnl":         round(net, 2),
                "pnl_pct":     round(net / (pos["entry_price"] * pos["quantity"]) * 100, 3),
                "exit_reason": reason,
                "entry_time":  pos["entry_time"],
                "exit_time":   str(datetime.now()),
                "mode":        "live",
            })

        except Exception as e:
            logger.error(f"[ERROR] Close failed for {symbol}: {e}")

    def _square_off_all(self, price: float):
        for symbol in list(self._positions.keys()):
            logger.info(f"[EOD] Squaring off {symbol}")
            self._close_position(symbol, price, "EOD")
        logger.info(f"[EOD] All positions closed. Day P&L=₹{self._daily_pnl:+.0f}")

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
        """Background thread: force square-off at 15:15 even if no new candle arrives."""
        import time as tmod
        while self._running:
            if datetime.now().time() >= SQUARE_OFF and not self._squared_off:
                logger.info("[EOD MONITOR] Triggering EOD square-off from background thread")
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

    # ------------------------------------------------------------------ #
    #  Status (used by dashboard)
    # ------------------------------------------------------------------ #

    def get_status(self) -> dict:
        return {
            "running":      self._running,
            "positions":    self._positions,
            "signals_fired": list(self._signals_fired),
            "daily_pnl":    self._daily_pnl,
            "capital":      self.capital,
        }
