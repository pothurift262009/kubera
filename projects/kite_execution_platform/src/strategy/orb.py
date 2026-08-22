"""
Opening Range Breakout (ORB) Strategy
--------------------------------------
Logic:
  1. Compute the high/low of the first N minutes after market open (ORB range).
  2. After the ORB window closes, watch for a candle close ABOVE orb_high
     or BELOW orb_low WITH a volume spike (> avg * multiplier).
  3. Entry = breakout candle close price.
  4. SL    = opposite side of the ORB range.
  5. Target = entry ± (risk * RR ratio).
  6. Max 1 signal per stock per day (first valid breakout wins).
"""
from dataclasses import dataclass
from typing import Optional
import pandas as pd
from src.utils.logger import setup_logger

logger = setup_logger("ORBStrategy")


@dataclass
class Signal:
    symbol:      str
    direction:   str        # 'LONG' | 'SHORT'
    entry_price: float
    sl:          float
    target:      float
    quantity:    int
    candle_time: str
    orb_high:    float
    orb_low:     float


class ORBStrategy:
    def __init__(self, config: dict):
        cfg = config["strategy"]
        self.orb_minutes   = cfg["orb_minutes"]
        self.rr_ratio      = cfg["rr_ratio"]
        self.vol_mult      = cfg["volume_multiplier"]
        self.risk_pct      = cfg["risk_pct"]
        self.max_positions = cfg["max_positions"]
        self.capital       = config["capital"]["total"]

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    # Minimum shares to make trade worth brokerage cost
    MIN_QTY = 3

    def generate_signals(self, df: pd.DataFrame, date, capital_available: float) -> list[Signal]:
        """Return up to 1 Signal for the given (symbol, date)."""
        orb = self._get_orb(df, date)
        if orb is None:
            return []

        orb_high, orb_low, avg_vol = orb

        # ORB range must be 0.4%–2.5% of price (tight enough to be meaningful)
        mid     = (orb_high + orb_low) / 2
        rng_pct = (orb_high - orb_low) / mid
        if not (0.004 <= rng_pct <= 0.025):
            return []

        day_df   = df[df["datetime"].dt.date == date]
        mkt_open = day_df["datetime"].min()
        orb_end  = mkt_open + pd.Timedelta(minutes=self.orb_minutes)
        post_orb = day_df[day_df["datetime"] > orb_end]

        # Trend filter: day open vs ORB midpoint
        day_open = day_df.iloc[0]["open"]

        # Only take first valid breakout
        for _, row in post_orb.iterrows():
            # Skip candles within 30 min of EOD
            if row["datetime"].time() >= __import__("datetime").time(12, 00):
                break

            # Volume must spike above average
            if row["volume"] < avg_vol * self.vol_mult:
                continue

            # Breakout candle body must be decisive (close near high/low)
            candle_range = row["high"] - row["low"]
            if candle_range == 0:
                continue

            if row["close"] > orb_high:
                # Trend filter: only LONG if day opened above midpoint (uptrend day)
                if day_open < mid * 0.998:
                    continue
                # Body must close in upper 40% of candle
                if (row["close"] - row["low"]) / candle_range < 0.6:
                    continue
                entry = row["close"]
                sl    = orb_low
                risk  = entry - sl
                if risk <= 0:
                    continue
                target = entry + risk * self.rr_ratio
                qty    = self._size(entry, sl, capital_available)
                if qty < self.MIN_QTY:
                    continue
                return [Signal(
                    symbol      = df["symbol"].iloc[0],
                    direction   = "LONG",
                    entry_price = entry,
                    sl          = sl,
                    target      = target,
                    quantity    = qty,
                    candle_time = str(row["datetime"]),
                    orb_high    = orb_high,
                    orb_low     = orb_low,
                )]

            elif row["close"] < orb_low:
                # Trend filter: only SHORT if day opened below midpoint (downtrend day)
                if day_open > mid * 1.002:
                    continue
                # Body must close in lower 40% of candle
                if (row["high"] - row["close"]) / candle_range < 0.6:
                    continue
                entry = row["close"]
                sl    = orb_high
                risk  = sl - entry
                if risk <= 0:
                    continue
                target = entry - risk * self.rr_ratio
                qty    = self._size(entry, sl, capital_available)
                if qty < self.MIN_QTY:
                    continue
                return [Signal(
                    symbol      = df["symbol"].iloc[0],
                    direction   = "SHORT",
                    entry_price = entry,
                    sl          = sl,
                    target      = target,
                    quantity    = qty,
                    candle_time = str(row["datetime"]),
                    orb_high    = orb_high,
                    orb_low     = orb_low,
                )]
        return []

    # ------------------------------------------------------------------ #
    #  Live mode: check a single new candle against an open position
    # ------------------------------------------------------------------ #

    def check_exit(self, position: dict, candle: dict) -> Optional[str]:
        """
        Returns 'SL' | 'TARGET' | None.
        position keys: direction, entry_price, sl, target
        candle keys: high, low, close
        """
        if position["direction"] == "LONG":
            if candle["low"]  <= position["sl"]:     return "SL"
            if candle["high"] >= position["target"]: return "TARGET"
        else:
            if candle["high"] >= position["sl"]:     return "SL"
            if candle["low"]  <= position["target"]: return "TARGET"
        return None

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def _get_orb(self, df: pd.DataFrame, date) -> Optional[tuple]:
        day = df[df["datetime"].dt.date == date]
        if day.empty:
            return None
        mkt_open = day["datetime"].min()
        orb_end  = mkt_open + pd.Timedelta(minutes=self.orb_minutes)
        orb_df   = day[day["datetime"] <= orb_end]
        if len(orb_df) < 2:
            return None
        return orb_df["high"].max(), orb_df["low"].min(), orb_df["volume"].mean()

    def _size(self, entry: float, sl: float, capital: float) -> int:
        """Risk-based position sizing capped by capital/max_positions."""
        risk_amt  = capital * self.risk_pct
        risk_unit = abs(entry - sl)
        if risk_unit == 0:
            return 0
        qty_by_risk    = int(risk_amt / risk_unit)
        qty_by_capital = int((capital / self.max_positions) / entry)
        return min(qty_by_risk, qty_by_capital)
