"""
═══════════════════════════════════════════════════════════════
🌊 REGIME DETECTION — Market State Classification
═══════════════════════════════════════════════════════════════
Classifies market state along two axes:

1. VOLATILITY REGIME
   - Low vol (calm)     : current_vol < 0.8 × avg_vol
   - Normal vol         : 0.8–1.5 × avg_vol
   - High vol (storm)   : current_vol > 1.5 × avg_vol

2. TREND REGIME
   - Trending up        : price > MA20 AND MA5 > MA10
   - Trending down      : price < MA20 AND MA5 < MA10
   - Sideways/choppy    : otherwise

WHY: Markets behave fundamentally differently in each regime.
A signal that works in trending+calm markets fails in
choppy+volatile markets. Regime awareness prevents bleed.
═══════════════════════════════════════════════════════════════
"""

import pandas as pd  # type: ignore
import numpy as np   # type: ignore
import logging
import time

logger = logging.getLogger(__name__)


class RegimeDetector:
    """
    Detects market regime for each stock at each timestamp.
    All features are lagged to prevent data leakage.
    """

    def __init__(self, config: dict):
        self.config = config
        self.regime_cfg = config.get("regime", {})
        self.vol_lookback = self.regime_cfg.get("volatility_lookback", 20)
        self.trend_lookback = self.regime_cfg.get("trend_lookback", 20)
        self.vol_expansion = self.regime_cfg.get("vol_expansion_threshold", 1.5)

    def detect_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds regime classification columns to the DataFrame.
        All regime features use PAST data only (no leakage).
        """
        logger.info("🌊 REGIME DETECTION:")
        t0 = time.time()

        df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
        g = df.groupby("symbol", observed=True)

        # ── Ensure base features exist ──────────────────────
        if "ret_1" not in df.columns:
            df["ret_1"] = g["close"].pct_change(1)

        # ═══════════════════════════════════════════════════
        # 1. VOLATILITY REGIME (per-stock)
        # ═══════════════════════════════════════════════════
        logger.info("  [1] Volatility regime...")

        # Current realized vol (fast window)
        df["_vol_fast"] = g["ret_1"].transform(
            lambda x: x.rolling(5, min_periods=3).std()
        )
        # Average realized vol (slow window)
        df["_vol_slow"] = g["ret_1"].transform(
            lambda x: x.rolling(self.vol_lookback, min_periods=10).std()
        )

        # Volatility ratio
        df["vol_ratio"] = np.where(
            df["_vol_slow"] > 0,
            df["_vol_fast"] / df["_vol_slow"],
            1.0
        )

        # Classify: 0=low, 1=normal, 2=high
        df["vol_regime"] = np.where(
            df["vol_ratio"] > self.vol_expansion, 2,
            np.where(df["vol_ratio"] < 0.8, 0, 1)
        ).astype(np.int8)

        # Vol expanding or contracting (derivative)
        df["vol_expanding"] = g["vol_ratio"].transform(
            lambda x: (x - x.shift(3))
        )

        # ═══════════════════════════════════════════════════
        # 2. TREND REGIME (per-stock)
        # ═══════════════════════════════════════════════════
        logger.info("  [2] Trend regime...")

        # Moving averages for trend detection
        df["_ma_5"] = g["close"].transform(
            lambda x: x.rolling(5, min_periods=5).mean()
        )
        df["_ma_10"] = g["close"].transform(
            lambda x: x.rolling(10, min_periods=10).mean()
        )
        df["_ma_20"] = g["close"].transform(
            lambda x: x.rolling(self.trend_lookback, min_periods=10).mean()
        )

        # Trend classification
        # Trending up: price > MA20 AND MA5 > MA10
        up_trend = (df["close"] > df["_ma_20"]) & (df["_ma_5"] > df["_ma_10"])
        # Trending down: price < MA20 AND MA5 < MA10
        down_trend = (df["close"] < df["_ma_20"]) & (df["_ma_5"] < df["_ma_10"])

        # 0=down, 1=sideways, 2=up
        df["trend_regime"] = np.where(
            up_trend, 2,
            np.where(down_trend, 0, 1)
        ).astype(np.int8)

        # Trend strength (distance from MA20, normalized)
        df["trend_strength"] = (
            (df["close"] - df["_ma_20"]) / df["_ma_20"]
        ).astype(np.float32)

        # MA alignment score: +1 each for MA5>MA10, MA10>MA20, price>MA5
        ma_align = (
            (df["_ma_5"] > df["_ma_10"]).astype(np.int8) +
            (df["_ma_10"] > df["_ma_20"]).astype(np.int8) +
            (df["close"] > df["_ma_5"]).astype(np.int8)
        )
        df["ma_alignment"] = ma_align.astype(np.int8)

        # ═══════════════════════════════════════════════════
        # 3. MARKET-WIDE REGIME (cross-sectional)
        # ═══════════════════════════════════════════════════
        logger.info("  [3] Market-wide regime...")

        # Market-wide volatility regime (shifted by 1 for leakage prevention)
        mkt_vol_at_t = df.groupby("date")["vol_ratio"].transform("mean")
        df["mkt_vol_regime"] = (
            mkt_vol_at_t.groupby(df["symbol"], observed=True).shift(1)
        ).astype(np.float32)

        # Market breadth: % of stocks in uptrend (shifted)
        mkt_breadth_at_t = df.groupby("date")["trend_regime"].transform(
            lambda x: np.mean(x == 2)
        )
        df["mkt_breadth"] = (
            mkt_breadth_at_t.groupby(df["symbol"], observed=True).shift(1)
        ).astype(np.float32)

        # ═══════════════════════════════════════════════════
        # 4. COMPOSITE REGIME SCORE
        # ═══════════════════════════════════════════════════
        # Higher = more favorable for long signals
        # Favorable: uptrend + low/normal vol + good breadth
        df["regime_score"] = (
            (df["trend_regime"] / 2.0) * 0.4 +         # Trend component
            (1.0 - df["vol_regime"] / 2.0) * 0.3 +     # Vol component (inverted)
            df["mkt_breadth"].fillna(0.5) * 0.3          # Breadth component
        ).astype(np.float32)

        # Cleanup intermediate columns
        df = df.drop(columns=["_vol_fast", "_vol_slow", "_ma_5", "_ma_10", "_ma_20"],
                      errors="ignore")

        elapsed = time.time() - t0
        logger.info(f"✅ REGIME DETECTION COMPLETE | {elapsed:.1f}s")

        # Log regime distribution
        for regime_col in ["vol_regime", "trend_regime"]:
            dist = df[regime_col].value_counts(normalize=True).sort_index()
            logger.info(f"  {regime_col}: {dict(dist.round(3))}")

        return df
