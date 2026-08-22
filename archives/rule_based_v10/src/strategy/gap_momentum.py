"""
Gap & Momentum Strategy — v10 (Institutional Grade)
-----------------------------------------------------
Improvements over v9:
  1. Gap >= 1% OR gap > 0.5x ATR (whichever is larger)
  2. First 15-min volume > previous day avg 15-min volume (real conviction)
  3. VWAP filter: LONG only if price above VWAP, SHORT only below
  4. Retest confirmation: wait for pullback after breakout (reduces fake breakouts)
  5. SL = 75% of opening range (not full range)
  6. Partial exit: 50% at 1R, trail rest with VWAP
  7. Time exit: VWAP cross OR 11:15 hard cutoff
  8. Risk 1% per trade (conservative, more trades)
"""
from dataclasses import dataclass, field
from typing import Optional
from datetime import time
import pandas as pd
import numpy as np
from src.utils.logger import setup_logger

logger = setup_logger("GapMomentum")

MARKET_OPEN  = time(9, 15)
ORB_END      = time(9, 30)
ENTRY_CUTOFF = time(10, 15)   # only trade early momentum
EXIT_CUTOFF  = time(11, 15)   # hard exit — momentum fades


@dataclass
class Signal:
    symbol:       str
    direction:    str
    entry_price:  float
    sl:           float
    target_1r:    float    # 50% exit here
    target_trail: float    # trail rest with VWAP
    quantity:     int
    half_qty:     int
    candle_time:  str
    gap_pct:      float
    orb_high:     float
    orb_low:      float
    vwap:         float
    atr:          float


class GapMomentumStrategy:
    def __init__(self, config: dict):
        cfg = config["strategy"]
        self.min_gap_pct    = cfg.get("min_gap_pct", 1.0) / 100
        self.atr_gap_mult   = cfg.get("atr_gap_mult", 0.5)
        self.rr_ratio       = cfg.get("rr_ratio", 2.0)
        self.sl_range_pct   = cfg.get("sl_range_pct", 0.75)   # 75% of ORB range
        self.max_positions  = cfg["max_positions"]
        self.risk_pct       = cfg["risk_pct"]
        self.capital        = config["capital"]["total"]
        self.min_qty        = cfg.get("min_qty", 3)
        self.retest_buffer  = cfg.get("retest_buffer_pct", 0.1) / 100

    # ------------------------------------------------------------------ #

    def generate_signals(self, df: pd.DataFrame, date, capital_available: float) -> list[Signal]:
        symbol = df["symbol"].iloc[0]

        prev_close = self._prev_close(df, date)
        if prev_close is None:
            return []

        # ATR from last 14 days
        atr = self._atr(df, date)

        day_df = df[df["datetime"].dt.date == date].copy()
        if len(day_df) < 6:
            return []

        open_price = day_df.iloc[0]["open"]
        gap_pct    = (open_price - prev_close) / prev_close

        # ── Gap filter: 1% OR 0.5x ATR ────────────────────────────────
        min_gap = max(self.min_gap_pct, self.atr_gap_mult * atr / prev_close)
        if abs(gap_pct) < min_gap:
            return []

        # ── Opening range ──────────────────────────────────────────────
        orb_df  = day_df[day_df["datetime"].dt.time <= ORB_END]
        if len(orb_df) < 2:
            return []

        orb_high = orb_df["high"].max()
        orb_low  = orb_df["low"].min()
        orb_range = orb_high - orb_low
        if orb_range <= 0:
            return []

        # ── Volume filter: 15-min vol > prev day 15-min avg ───────────
        orb_vol     = orb_df["volume"].sum()
        prev_15_vol = self._prev_15min_vol(df, date)
        if prev_15_vol and orb_vol < prev_15_vol:
            logger.debug(f"  {symbol} volume fail: {orb_vol:.0f} < {prev_15_vol:.0f}")
            return []

        # ── VWAP for context ───────────────────────────────────────────
        day_df  = self._add_vwap(day_df)
        post_orb = day_df[
            (day_df["datetime"].dt.time > ORB_END) &
            (day_df["datetime"].dt.time <= ENTRY_CUTOFF)
        ]

        for idx, row in post_orb.iterrows():
            vwap = row.get("vwap", open_price)

            # ── LONG setup ────────────────────────────────────────────
            if gap_pct >= min_gap:
                # VWAP filter: price must be above VWAP
                if row["close"] < vwap:
                    continue

                # Breakout above ORB high
                if row["close"] > orb_high:
                    # Wait for retest: next candle pulls back toward ORB high
                    retest_ok = self._check_retest(
                        post_orb, idx, "LONG", orb_high, self.retest_buffer)
                    if not retest_ok:
                        continue

                    entry = orb_high * (1 + self.retest_buffer)  # just above retest level
                    sl    = orb_high - orb_range * self.sl_range_pct
                    risk  = entry - sl
                    if risk <= 0 or risk / entry > 0.03:
                        continue

                    target_1r    = round(entry + risk, 2)
                    target_trail = round(entry + risk * self.rr_ratio, 2)
                    qty          = self._size(entry, sl, capital_available)
                    if qty < self.min_qty:
                        continue
                    half_qty = max(1, qty // 2)

                    logger.info(
                        f"  GAP LONG  {symbol} | gap={gap_pct*100:.2f}% atr={atr:.2f} "
                        f"orb={orb_low:.2f}-{orb_high:.2f} entry={entry:.2f} "
                        f"sl={sl:.2f} 1R={target_1r} 2R={target_trail} qty={qty}"
                    )
                    return [Signal(symbol=symbol, direction="LONG",
                                   entry_price=round(entry,2), sl=round(sl,2),
                                   target_1r=target_1r, target_trail=target_trail,
                                   quantity=qty, half_qty=half_qty,
                                   candle_time=str(row["datetime"]),
                                   gap_pct=round(gap_pct*100,2),
                                   orb_high=round(orb_high,2), orb_low=round(orb_low,2),
                                   vwap=round(vwap,2), atr=round(atr,2))]

            # ── SHORT setup ───────────────────────────────────────────
            elif gap_pct <= -min_gap:
                if row["close"] > vwap:
                    continue

                if row["close"] < orb_low:
                    retest_ok = self._check_retest(
                        post_orb, idx, "SHORT", orb_low, self.retest_buffer)
                    if not retest_ok:
                        continue

                    entry = orb_low * (1 - self.retest_buffer)
                    sl    = orb_low + orb_range * self.sl_range_pct
                    risk  = sl - entry
                    if risk <= 0 or risk / entry > 0.03:
                        continue

                    target_1r    = round(entry - risk, 2)
                    target_trail = round(entry - risk * self.rr_ratio, 2)
                    qty          = self._size(entry, sl, capital_available)
                    if qty < self.min_qty:
                        continue
                    half_qty = max(1, qty // 2)

                    logger.info(
                        f"  GAP SHORT {symbol} | gap={gap_pct*100:.2f}% atr={atr:.2f} "
                        f"entry={entry:.2f} sl={sl:.2f} 1R={target_1r} qty={qty}"
                    )
                    return [Signal(symbol=symbol, direction="SHORT",
                                   entry_price=round(entry,2), sl=round(sl,2),
                                   target_1r=target_1r, target_trail=target_trail,
                                   quantity=qty, half_qty=half_qty,
                                   candle_time=str(row["datetime"]),
                                   gap_pct=round(gap_pct*100,2),
                                   orb_high=round(orb_high,2), orb_low=round(orb_low,2),
                                   vwap=round(vwap,2), atr=round(atr,2))]

        return []

    # ------------------------------------------------------------------ #

    def check_exit(self, position: dict, candle: dict, vwap: float) -> Optional[str]:
        """
        Returns: 'SL' | 'PARTIAL' | 'TARGET' | 'VWAP_CROSS' | None
        """
        direction    = position["direction"]
        sl           = position["sl"]
        target_1r    = position["target_1r"]
        target_trail = position["target_trail"]
        partial_done = position.get("partial_done", False)

        if direction == "LONG":
            if candle["low"]  <= sl:               return "SL"
            if not partial_done and candle["high"] >= target_1r: return "PARTIAL"
            if partial_done and candle["high"] >= target_trail:  return "TARGET"
            # VWAP cross exit: price falls back below VWAP after partial
            if partial_done and candle["close"] < vwap:          return "VWAP_CROSS"
        else:
            if candle["high"] >= sl:               return "SL"
            if not partial_done and candle["low"] <= target_1r:  return "PARTIAL"
            if partial_done and candle["low"] <= target_trail:   return "TARGET"
            if partial_done and candle["close"] > vwap:          return "VWAP_CROSS"

        return None

    # ------------------------------------------------------------------ #

    def _check_retest(self, post_orb: pd.DataFrame, current_idx,
                      direction: str, level: float, buffer: float) -> bool:
        """
        After breakout, check if next 1-2 candles pull back to level (retest).
        Returns True if retest found (confirming the breakout).
        """
        rows = post_orb[post_orb.index > current_idx].head(3)
        for _, r in rows.iterrows():
            if direction == "LONG":
                # Pulls back to within buffer% of ORB high, then holds above
                if r["low"] <= level * (1 + buffer) and r["close"] >= level:
                    return True
            else:
                if r["high"] >= level * (1 - buffer) and r["close"] <= level:
                    return True
        # No retest found within 3 candles — still allow if strong momentum
        # (candle that broke out had strong body)
        curr = post_orb.loc[current_idx]
        body = abs(curr["close"] - curr["open"])
        if body / curr["close"] > 0.003:   # strong candle → skip retest req
            return True
        return False

    @staticmethod
    def _add_vwap(df: pd.DataFrame) -> pd.DataFrame:
        typical    = (df["high"] + df["low"] + df["close"]) / 3
        cum_tp_vol = (typical * df["volume"]).cumsum()
        cum_vol    = df["volume"].cumsum()
        df = df.copy()
        df["vwap"] = cum_tp_vol / cum_vol.replace(0, np.nan)
        return df

    def _prev_close(self, df: pd.DataFrame, date) -> Optional[float]:
        prev = df[df["datetime"].dt.date < date]
        if prev.empty:
            return None
        last_day = prev["datetime"].dt.date.max()
        return float(prev[prev["datetime"].dt.date == last_day]["close"].iloc[-1])

    def _prev_15min_vol(self, df: pd.DataFrame, date) -> Optional[float]:
        """Average 15-min opening volume from last 5 trading days."""
        prev = df[df["datetime"].dt.date < date]
        if prev.empty:
            return None
        days = sorted(prev["datetime"].dt.date.unique())[-5:]
        vols = []
        for d in days:
            day_df = prev[prev["datetime"].dt.date == d]
            orb_vol = day_df[day_df["datetime"].dt.time <= ORB_END]["volume"].sum()
            if orb_vol > 0:
                vols.append(orb_vol)
        return float(np.mean(vols)) if vols else None

    def _atr(self, df: pd.DataFrame, date, period: int = 14) -> float:
        prev = df[df["datetime"].dt.date < date]
        if len(prev) < period * 5:
            return 0.0
        daily = prev.groupby(prev["datetime"].dt.date).agg(
            high=("high","max"), low=("low","min"), close=("close","last")
        ).reset_index()
        daily["prev_close"] = daily["close"].shift(1)
        daily["tr"] = daily.apply(
            lambda r: max(r["high"]-r["low"],
                          abs(r["high"]-r["prev_close"]),
                          abs(r["low"]-r["prev_close"]))
            if not pd.isna(r["prev_close"]) else r["high"]-r["low"], axis=1)
        return float(daily["tr"].tail(period).mean())

    def _size(self, entry: float, sl: float, capital: float) -> int:
        risk_amt    = capital * self.risk_pct
        risk_unit   = abs(entry - sl)
        if risk_unit == 0:
            return 0
        qty_risk    = int(risk_amt / risk_unit)
        qty_capital = int((capital / self.max_positions) / entry)
        return min(qty_risk, qty_capital)
