"""
LiveTrader
----------
Wires together LiveFeed → ORBStrategy → RiskManager → KiteBroker.
Runs in a background thread; call start() / stop().
"""
import threading
from datetime import datetime, time
from src.data.feed import LiveFeed
from src.strategy.orb import ORBStrategy
from src.execution.broker import KiteBroker
from src.execution.risk import RiskManager
from src.utils import db
from src.utils.logger import setup_logger
import pandas as pd

logger = setup_logger("LiveTrader")

SQUARE_OFF_TIME = time(15, 15)


class LiveTrader:
    def __init__(self, config: dict, symbols: list):
        self.config  = config
        self.symbols = symbols
        self.broker  = KiteBroker(config)
        self.risk    = RiskManager(config)
        self.strat   = ORBStrategy(config)

        # Per-symbol intraday candle accumulator
        self._candle_buf: dict = {}     # {symbol: [candles today]}
        self._orb_done:   set  = set()  # symbols whose ORB window has closed
        self._signals_fired: set = set()

        self._running = False
        self._thread  = None

    # ------------------------------------------------------------------ #

    def start(self):
        token_map = self.broker.get_instrument_tokens(self.symbols)
        inv_map   = {v: k for k, v in token_map.items()}  # {token: symbol}

        self._feed = LiveFeed(
            self.config["kite"]["api_key"],
            self.config["kite"]["access_token"],
        )
        self._feed.add_candle_callback(self._on_candle)
        self._feed.subscribe(inv_map)
        self._feed.start()
        self._running = True
        logger.info(f"LiveTrader started for {len(self.symbols)} symbols")

    def stop(self):
        self._running = False
        self._feed.stop()
        logger.info("LiveTrader stopped")

    # ------------------------------------------------------------------ #

    def _on_candle(self, symbol: str, candle: dict):
        if not self._running:
            return

        now = candle["timestamp"].time() if hasattr(candle.get("timestamp", ""), "time") else datetime.now().time()

        # Square off at 15:15
        if now >= SQUARE_OFF_TIME:
            if symbol in self.risk.open_positions:
                self._exit(symbol, candle["close"], "EOD")
            return

        # Accumulate candles for ORB logic
        self._candle_buf.setdefault(symbol, []).append(candle)

        # Check exit for open positions
        if symbol in self.risk.open_positions:
            exit_reason = self.strat.check_exit(self.risk.open_positions[symbol], candle)
            if exit_reason:
                self._exit(symbol, candle["close"], exit_reason)
            return

        # Only fire one signal per symbol per day
        if symbol in self._signals_fired:
            return

        # Build a minimal DataFrame from accumulated candles to pass to strategy
        if len(self._candle_buf[symbol]) < 4:   # need at least some candles
            return

        df = self._buf_to_df(symbol)
        today = datetime.today().date()
        signals = self.strat.generate_signals(df, today, self.risk.capital / self.risk.max_positions)

        if signals and self.risk.can_trade():
            sig = signals[0]
            logger.info(f"Signal: {sig.direction} {symbol} @ {sig.entry_price}")
            try:
                if sig.direction == "LONG":
                    self.broker.buy(symbol, sig.quantity)
                else:
                    self.broker.sell(symbol, sig.quantity)

                # Place bracket SL order
                self.broker.place_sl_order(symbol, sig.direction, sig.quantity, sig.sl)

                self.risk.add_position(symbol, {
                    "direction":   sig.direction,
                    "entry_price": sig.entry_price,
                    "sl":          sig.sl,
                    "target":      sig.target,
                    "quantity":    sig.quantity,
                })
                self._signals_fired.add(symbol)
            except Exception as e:
                logger.error(f"Order placement failed for {symbol}: {e}")

    def _exit(self, symbol: str, price: float, reason: str):
        pos = self.risk.open_positions.get(symbol)
        if not pos:
            return
        try:
            if pos["direction"] == "LONG":
                self.broker.sell(symbol, pos["quantity"])
            else:
                self.broker.buy(symbol, pos["quantity"])
            pnl = self.risk.close_position(symbol, price, reason)
            db.log_trade({
                "date":        str(datetime.today().date()),
                "symbol":      symbol,
                "direction":   pos["direction"],
                "entry_price": pos["entry_price"],
                "exit_price":  price,
                "quantity":    pos["quantity"],
                "sl":          pos["sl"],
                "target":      pos["target"],
                "pnl":         round(pnl, 2),
                "pnl_pct":     round(pnl / (pos["entry_price"] * pos["quantity"]) * 100, 3),
                "exit_reason": reason,
                "entry_time":  str(datetime.now()),
                "exit_time":   str(datetime.now()),
                "mode":        "live",
            })
        except Exception as e:
            logger.error(f"Exit failed for {symbol}: {e}")

    def _buf_to_df(self, symbol: str) -> pd.DataFrame:
        rows = self._candle_buf[symbol]
        df = pd.DataFrame(rows)
        df["symbol"]   = symbol
        df["datetime"] = pd.to_datetime(df["timestamp"])
        return df
