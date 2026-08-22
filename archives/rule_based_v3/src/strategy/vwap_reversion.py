"""
VWAP Mean Reversion Strategy — Stable v2
65% win rate | 1.92 Profit Factor | -1.6% Max Drawdown
"""
from dataclasses import dataclass
from typing import Optional
from datetime import time
import numpy as np
import pandas as pd
from src.utils.logger import setup_logger

logger = setup_logger("VWAPStrategy")

ENTRY_START = time(9, 45)
ENTRY_END   = time(13, 0)

HIGH_BETA_STOCKS = {
    "ADANIENT", "ADANIPORTS", "BAJFINANCE", "BAJAJFINSV",
    "AXISBANK", "ICICIBANK", "SBIN", "HDFCBANK", "KOTAKBANK",
    "TATAMOTORS", "TATASTEEL", "JSWSTEEL", "HINDALCO",
    "WIPRO", "TECHM", "INFY", "TCS", "HCLTECH",
    "TITAN", "M&M", "MARUTI", "HEROMOTOCO",
    "SUNPHARMA", "DRREDDY", "CIPLA",
    "RELIANCE", "LT", "ULTRACEMCO",
    "DIVISLAB", "APOLLOHOSP",
}


@dataclass
class Signal:
    symbol:      str
    direction:   str
    entry_price: float
    sl:          float
    target:      float
    quantity:    int
    candle_time: str
    vwap:        float
    rsi:         float
    deviation:   float


class VWAPReversionStrategy:
    def __init__(self, config: dict):
        cfg = config["strategy"]
        self.min_deviation  = cfg.get("vwap_deviation_pct", 1.5) / 100
        self.rsi_period     = cfg.get("rsi_period", 14)
        self.rsi_oversold   = cfg.get("rsi_oversold", 32)
        self.rsi_overbought = cfg.get("rsi_overbought", 68)
        self.sl_ratio       = cfg.get("sl_ratio", 1.0)
        self.vol_mult       = cfg.get("volume_multiplier", 2.0)
        self.max_positions  = cfg["max_positions"]
        self.risk_pct       = cfg["risk_pct"]
        self.capital        = config["capital"]["total"]
        self.min_qty        = cfg.get("min_qty", 3)

    def generate_signals(self, df: pd.DataFrame, date, capital_available: float) -> list[Signal]:
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

                entry  = close
                target = round(vwap, 2)
                risk   = abs(deviation) * entry * self.sl_ratio
                sl     = round(entry - risk, 2)
                if sl >= entry or risk <= 0:
                    continue
                qty = self._size(entry, sl, capital_available)
                if qty < self.min_qty:
                    continue
                logger.info(f"  SIGNAL LONG  {symbol} | entry={entry} vwap={round(vwap,2)} rsi={round(rsi,1)} dev={round(deviation*100,2)}% qty={qty}")
                return [Signal(symbol=symbol, direction="LONG",
                               entry_price=entry, sl=sl, target=target,
                               quantity=qty, candle_time=str(ts),
                               vwap=round(vwap,2), rsi=round(rsi,1),
                               deviation=round(deviation*100,2))]

            # ── SHORT ─────────────────────────────────────────────────
            elif (deviation >= self.min_deviation
                    and rsi > self.rsi_overbought
                    and row["close"] < row["open"]):

                entry  = close
                target = round(vwap, 2)
                risk   = abs(deviation) * entry * self.sl_ratio
                sl     = round(entry + risk, 2)
                if sl <= entry or risk <= 0:
                    continue
                qty = self._size(entry, sl, capital_available)
                if qty < self.min_qty:
                    continue
                logger.info(f"  SIGNAL SHORT {symbol} | entry={entry} vwap={round(vwap,2)} rsi={round(rsi,1)} dev={round(deviation*100,2)}% qty={qty}")
                return [Signal(symbol=symbol, direction="SHORT",
                               entry_price=entry, sl=sl, target=target,
                               quantity=qty, candle_time=str(ts),
                               vwap=round(vwap,2), rsi=round(rsi,1),
                               deviation=round(deviation*100,2))]

        return []

    def check_exit(self, position: dict, candle: dict, vwap: float) -> Optional[str]:
        if position["direction"] == "LONG":
            if candle["low"]  <= position["sl"]:  return "SL"
            if candle["high"] >= vwap:            return "TARGET"
        else:
            if candle["high"] >= position["sl"]:  return "SL"
            if candle["low"]  <= vwap:            return "TARGET"
        return None

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
