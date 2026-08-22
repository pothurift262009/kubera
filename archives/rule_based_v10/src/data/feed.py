from kiteconnect import KiteTicker
from src.utils.logger import setup_logger
import time

logger = setup_logger("LiveFeed")


class LiveFeed:
    def __init__(self, api_key: str, access_token: str):
        self.kws        = KiteTicker(api_key, access_token)
        self.token_map: dict  = {}
        self.candles:   dict  = {}
        self._callbacks: list = []
        self._tokens:   list  = []

        self.kws.on_close = lambda ws, c, r: logger.warning(f"WebSocket closed: {c}")
        self.kws.on_error = lambda ws, c, r: logger.error(f"WebSocket error: {c} {r}")
        self.kws.on_ticks = self._on_ticks

    def add_candle_callback(self, fn):
        self._callbacks.append(fn)

    def subscribe(self, token_map: dict):
        """Store token map — actual subscription happens after connect."""
        self.token_map = token_map
        self._tokens   = list(token_map.keys())

    def start(self):
        tokens = self._tokens

        def on_connect(ws, response):
            logger.info(f"WebSocket connected ✅ — subscribing {len(tokens)} instruments")
            ws.subscribe(tokens)
            ws.set_mode(ws.MODE_FULL, tokens)

        self.kws.on_connect = on_connect
        self.kws.connect(threaded=True)
        logger.info("WebSocket connecting...")

    def stop(self):
        self.kws.close()

    def _on_ticks(self, ws, ticks):
        for tick in ticks:
            token  = tick["instrument_token"]
            price  = tick["last_price"]
            volume = tick.get("volume_traded", tick.get("volume", 0))
            ts     = tick["timestamp"]

            bucket = (ts.minute // 5) * 5
            key    = f"{ts.date()}_{ts.hour:02d}_{bucket:02d}"

            if token not in self.candles or self.candles[token]["key"] != key:
                if token in self.candles:
                    self._emit(token, self.candles[token])
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
                logger.error(f"Candle callback error [{symbol}]: {e}")
