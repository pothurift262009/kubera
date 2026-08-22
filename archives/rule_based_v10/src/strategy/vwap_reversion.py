"""
VWAP Mean Reversion Strategy — v4 HIGH PROFIT
Improvements over v2:
  1. Extended target  : VWAP + 0.3% buffer (let winners run past VWAP)
  2. Partial exit     : 50% of position exits at VWAP, rest runs to extended target
  3. Re-entry allowed : after a profitable exit, allow new signal same day same stock
  4. Wider watchlist  : 40 stocks instead of 30
"""
from dataclasses import dataclass, field
from typing import Optional
from datetime import time
import numpy as np
import pandas as pd
from src.utils.logger import setup_logger

logger = setup_logger("VWAPStrategy")

ENTRY_START = time(9, 45)
ENTRY_END   = time(13, 0)

HIGH_BETA_STOCKS = {
    # Large cap momentum
    "ADANIENT", "ADANIPORTS", "BAJFINANCE", "BAJAJFINSV",
    "AXISBANK", "ICICIBANK", "SBIN", "HDFCBANK", "KOTAKBANK",
    "TATAMOTORS", "TATASTEEL", "JSWSTEEL", "HINDALCO",
    "WIPRO", "TECHM", "INFY", "TCS", "HCLTECH",
    "TITAN", "M&M", "MARUTI", "HEROMOTOCO",
    "SUNPHARMA", "DRREDDY", "CIPLA",
    "RELIANCE", "LT", "ULTRACEMCO",
    "DIVISLAB", "APOLLOHOSP",
    # Additional high-beta additions
    "INDUSINDBK", "FEDERALBNK", "BANDHANBNK",
    "GRASIM", "ASIANPAINT", "NESTLEIND",
    "BPCL", "IOC", "COALINDIA",
    "TATACONSUM", "BRITANNIA",
}


@dataclass
class Signal:
    symbol:      str
    direction:   str
    entry_price: float
    sl:          float
    target:      float          # extended target (VWAP + buffer)
    vwap_target: float          # VWAP level — partial exit here
    quantity:    int
    half_qty:    int            # qty to exit at VWAP (partial)
    candle_time: str
    vwap:        float
    rsi:         float
    deviation:   float


class VWAPReversionStrategy:
    def __init__(self, config: dict):
        cfg = config["strategy"]
        self.min_deviation   = cfg.get("vwap_deviation_pct", 1.5) / 100
        self.rsi_period      = cfg.get("rsi_period", 14)
        self.rsi_oversold    = cfg.get("rsi_oversold", 32)
        self.rsi_overbought  = cfg.get("rsi_overbought", 68)
        self.sl_ratio        = cfg.get("sl_ratio", 1.0)
        self.vol_mult        = cfg.get("volume_multiplier", 2.0)
        self.max_positions   = cfg["max_positions"]
        self.risk_pct        = cfg["risk_pct"]
        self.capital         = config["capital"]["total"]
        self.min_qty         = cfg.get("min_qty", 3)
        self.target_buffer   = cfg.get("target_buffer_pct", 0.3) / 100  # 0.3% past VWAP
        self.partial_exit    = cfg.get("partial_exit", True)

    # ------------------------------------------------------------------ #

    def generate_signals(self, df: pd.DataFrame, date, capital_available: float,
                         reentry: bool = False) -> list[Signal]:
        """
        reentry=True: already traded this stock today but allow new signal
                      (only if previous trade was profitable)
        """
        symbol = df["symbol"].iloc[0]
        if symbol not in HIGH_BETA_STOCKS:
            return []

        day_df = df[df["datetime"].dt.date == date].copy()
        if len(day_df) < self.rsi_period + 5:
            return []

        day_df  = self._add_vwap(day_df)
        day_df  = self._add_rsi(day_df)
        avg_vol = day_df["volume"].mean()

        for _, row in day_df.iterrows():
            ts = row["datetime"]
            if not (ENTRY_START <= ts.time() <= ENTRY_END):
                continue

            vwap   = row["vwap"]
            close  = row["close"]
            rsi    = row["rsi"]
            volume = row["volume"]

            if pd.isna(rsi) or pd.isna(vwap) or vwap == 0:
                continue

            deviation = (close - vwap) / vwap

            if volume < avg_vol * self.vol_mult:
                continue

            body = abs(row["close"] - row["open"])
            if body / close < 0.0015:
                continue

            # ── LONG ──────────────────────────────────────────────────
            if (deviation <= -self.min_deviation
                    and rsi < self.rsi_oversold
                    and row["close"] > row["open"]):

                entry       = close
                vwap_target = round(vwap, 2)
                # Extended target: VWAP + buffer
                ext_target  = round(vwap * (1 + self.target_buffer), 2)
                risk        = abs(deviation) * entry * self.sl_ratio
                sl          = round(entry - risk, 2)
                if sl >= entry or risk <= 0:
                    continue

                qty = self._size(entry, sl, capital_available)
                if qty < self.min_qty:
                    continue

                half_qty = max(1, qty // 2) if self.partial_exit else 0

                logger.info(f"  SIGNAL LONG  {symbol} | entry={entry} vwap={vwap_target} "
                            f"target={ext_target} rsi={round(rsi,1)} dev={round(deviation*100,2)}% "
                            f"qty={qty} half={half_qty} {'[RE-ENTRY]' if reentry else ''}")

                return [Signal(symbol=symbol, direction="LONG",
                               entry_price=entry, sl=sl,
                               target=ext_target, vwap_target=vwap_target,
                               quantity=qty, half_qty=half_qty,
                               candle_time=str(ts),
                               vwap=vwap_target, rsi=round(rsi,1),
                               deviation=round(deviation*100,2))]

            # ── SHORT ─────────────────────────────────────────────────
            elif (deviation >= self.min_deviation
                    and rsi > self.rsi_overbought
                    and row["close"] < row["open"]):

                entry       = close
                vwap_target = round(vwap, 2)
                ext_target  = round(vwap * (1 - self.target_buffer), 2)
                risk        = abs(deviation) * entry * self.sl_ratio
                sl          = round(entry + risk, 2)
                if sl <= entry or risk <= 0:
                    continue

                qty = self._size(entry, sl, capital_available)
                if qty < self.min_qty:
                    continue

                half_qty = max(1, qty // 2) if self.partial_exit else 0

                logger.info(f"  SIGNAL SHORT {symbol} | entry={entry} vwap={vwap_target} "
                            f"target={ext_target} rsi={round(rsi,1)} dev={round(deviation*100,2)}% "
                            f"qty={qty} half={half_qty} {'[RE-ENTRY]' if reentry else ''}")

                return [Signal(symbol=symbol, direction="SHORT",
                               entry_price=entry, sl=sl,
                               target=ext_target, vwap_target=vwap_target,
                               quantity=qty, half_qty=half_qty,
                               candle_time=str(ts),
                               vwap=vwap_target, rsi=round(rsi,1),
                               deviation=round(deviation*100,2))]

        return []

    def check_exit(self, position: dict, candle: dict, vwap: float) -> Optional[str]:
        """
        Returns: 'SL' | 'PARTIAL' | 'TARGET' | None
        PARTIAL fires when price hits VWAP (exit half qty, move SL to breakeven)
        TARGET fires when price hits extended target
        """
        direction = position["direction"]
        sl        = position.get("current_sl", position["sl"])
        target    = position["target"]
        vwap_tgt  = position.get("vwap_target", vwap)
        partial   = position.get("partial_done", False)

        if direction == "LONG":
            if candle["low"] <= sl:
                return "SL"
            # Partial exit at VWAP (if not done yet)
            if not partial and candle["high"] >= vwap_tgt:
                return "PARTIAL"
            # Full exit at extended target
            if candle["high"] >= target:
                return "TARGET"
        else:
            if candle["high"] >= sl:
                return "SL"
            if not partial and candle["low"] <= vwap_tgt:
                return "PARTIAL"
            if candle["low"] <= target:
                return "TARGET"

        return None

    # ------------------------------------------------------------------ #

    @staticmethod
    def _add_vwap(df: pd.DataFrame) -> pd.DataFrame:
        typical    = (df["high"] + df["low"] + df["close"]) / 3
        cum_tp_vol = (typical * df["volume"]).cumsum()
        cum_vol    = df["volume"].cumsum()
        df["vwap"] = cum_tp_vol / cum_vol.replace(0, np.nan)
        return df

    def _add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        delta = df["close"].diff()
        gain  = delta.clip(lower=0)
        loss  = (-delta).clip(lower=0)
        avg_g = gain.ewm(com=self.rsi_period - 1, min_periods=self.rsi_period).mean()
        avg_l = loss.ewm(com=self.rsi_period - 1, min_periods=self.rsi_period).mean()
        rs    = avg_g / avg_l.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))
        return df

    def _size(self, entry: float, sl: float, capital: float) -> int:
        risk_amt    = capital * self.risk_pct
        risk_unit   = abs(entry - sl)
        if risk_unit == 0:
            return 0
        qty_risk    = int(risk_amt / risk_unit)
        qty_capital = int((capital / self.max_positions) / entry)
        return min(qty_risk, qty_capital)
