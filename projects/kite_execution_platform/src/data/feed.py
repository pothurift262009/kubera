"""
Live 5-min candle builder on top of Kite WebSocket ticks.
Usage:
    feed = LiveFeed(api_key, access_token)
    feed.add_candle_callback(my_callback)   # fn(symbol, candle_dict)
    feed.subscribe(token_map)               # {instrument_token: symbol}
    feed.start()
"""
from collections import defaultdict
from kiteconnect import KiteTicker
from src.utils.logger import setup_logger

logger = setup_logger("LiveFeed")


class LiveFeed:
    def __init__(self, api_key: str, access_token: str):
        self.kws = KiteTicker(api_key, access_token)
        self.token_map: dict = {}          # {token: symbol}
        self.candles: dict = {}            # {token: current candle dict}
        self._callbacks: list = []

        self.kws.on_ticks   = self._on_ticks
        self.kws.on_connect = lambda ws, r: logger.info("WebSocket connected")
        self.kws.on_close   = lambda ws, c, r: logger.warning(f"WebSocket closed: {c}")
        self.kws.on_error   = lambda ws, c, r: logger.error(f"WebSocket error: {c} {r}")

    # ------------------------------------------------------------------ #
    def add_candle_callback(self, fn):
        """Register callback(symbol: str, candle: dict) for closed 5-min candles."""
        self._callbacks.append(fn)

    def subscribe(self, token_map: dict):
        """token_map = {instrument_token (int): symbol (str)}"""
        self.token_map = token_map
        tokens = list(token_map.keys())
        self.kws.subscribe(tokens)
        self.kws.set_mode(self.kws.MODE_FULL, tokens)
        logger.info(f"Subscribed to {len(tokens)} instruments")

    def start(self):
        self.kws.connect(threaded=True)

    def stop(self):
        self.kws.close()

    # ------------------------------------------------------------------ #
    def _on_ticks(self, ws, ticks):
        for tick in ticks:
            token = tick["instrument_token"]
            price  = tick["last_price"]
            volume = tick.get("volume_traded", tick.get("volume", 0))
            ts     = tick["timestamp"]

            # 5-min bucket key
            bucket = (ts.minute // 5) * 5
            key = f"{ts.date()}_{ts.hour:02d}_{bucket:02d}"

            if token not in self.candles or self.candles[token]["key"] != key:
                # Close the previous candle
                if token in self.candles:
                    self._emit(token, self.candles[token])
                # Open a new candle
                self.candles[token] = {
                    "key": key, "open": price, "high": price,
                    "low": price, "close": price,
                    "volume": volume, "timestamp": ts,
                }
            else:
                c = self.candles[token]
                c["high"]   = max(c["high"], price)
                c["low"]    = min(c["low"],  price)
                c["close"]  = price
                c["volume"] += volume

    def _emit(self, token: int, candle: dict):
        symbol = self.token_map.get(token, str(token))
        for fn in self._callbacks:
            try:
                fn(symbol, candle)
            except Exception as e:
                logger.error(f"Candle callback error: {e}")
