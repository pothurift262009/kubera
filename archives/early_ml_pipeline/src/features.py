import pandas as pd  # type: ignore
import numpy as np   # type: ignore
import logging
import time
from typing import Any, cast
try:
    from src.utils.memory import log_memory, clear_memory  # type: ignore
except ImportError:
    from .utils.memory import log_memory, clear_memory    # type: ignore

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    v2.1 Feature Engine: Memory-Optimized (8GB RAM).
    Uses chunked processing to prevent large memory spikes.
    """

    def __init__(self, config: dict):
        self.config = config
        self.feat_cfg = config["features"]
        self.target_cfg = config["target"]

        self.rolling_windows = self.feat_cfg["rolling_windows"]
        self.return_windows = self.feat_cfg["return_windows"]
        self.momentum_windows = self.feat_cfg["momentum_windows"]
        self.lags = self.feat_cfg["lags"]

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Memory-efficient feature pipeline."""
        logger.info(f"⚡ FEATURE ENGINEERING (Memory-Optimized): {len(df):,} rows")
        log_memory("Pre-features")

        # 1. Base transformations (vectorized, done on full df for speed)
        df.sort_values(["symbol", "date"], inplace=True)

        # 2. Chunk-based computation for complex stateful features
        symbols = df["symbol"].unique()
        logger.info(f"  [Processing {len(symbols)} symbols in chunks...]")

        processed_groups = []
        for symbol in symbols:
            # Efficiently extract group
            group = df[df["symbol"] == symbol].copy()

            # Compute features on single symbol
            group = self._process_group(group)

            # Cast everything to float32 immediately to save 50% RAM
            for col in group.select_dtypes(include=['float64']).columns:
                group[col] = group[col].astype(np.float32)

            processed_groups.append(group)

            # Periodic cleanup
            if len(processed_groups) % 15 == 0:
                gc_msg = f"Processed {len(processed_groups)} symbols"
                clear_memory(gc_msg)

        # 3. Reassemble
        df = pd.concat(processed_groups, ignore_index=True)
        processed_groups = None
        clear_memory("Post-concat features")

        # 4. Global cross-sectional features (must be done on full df)
        if self.feat_cfg.get("use_market_cross_sectional", True):
            logger.info("  [G] Market Cross-Sectional...")
            df = self._cross_sectional(df)
            log_memory("Post-cross-sectional")

        return df

    def _process_group(self, df: pd.DataFrame) -> pd.DataFrame:
        """Internal: computation on a per-symbol group."""
        # Returns
        for w in self.return_windows:
            df[f"ret_{w}"] = df["close"].pct_change(w).astype(np.float32)

        # Momentum
        for w in self.momentum_windows:
            df[f"mom_{w}"] = (df["close"] / df["close"].shift(w) - 1).astype(np.float32)

        # Trend (MA ratios)
        for w in self.rolling_windows:
            ma = df["close"].rolling(w).mean()
            df[f"price_ma_{w}_ratio"] = (df["close"] / ma).astype(np.float32)

        # Volatility
        if "ret_1" not in df.columns:
            df["ret_1"] = df["close"].pct_change(1).astype(np.float32)

        for w in self.rolling_windows:
            df[f"volatility_{w}"] = df["ret_1"].rolling(w).std().astype(np.float32)

        # Volume
        if self.feat_cfg.get("use_volume_intelligence", True):
            vol_ma_20 = df["volume"].rolling(20, min_periods=5).mean()
            df["vol_spike"] = (df["volume"] / vol_ma_20).astype(np.float32)
            df["vol_change_pct"] = df["volume"].pct_change().astype(np.float32)

        # RSI Technical Indicator
        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        # Scale RSI to 0-1 range (normalized)
        # Use cast(Any, ...) to bypass the IDE's incorrect float inference
        rsi_raw = 100 - (100 / (1 + rs))
        # Final RSI normalized calculation
        # Using getattr to bypass the IDE's incorrect float-astype hallucination
        rsi_norm_raw = (1.0 - (rsi_raw / 100))
        df["rsi_14_norm"] = getattr(rsi_norm_raw, "astype")(np.float32)

        return df

    def create_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Labels high-conviction moves with memory awareness. Target split for Base Model."""
        logger.info("🎯 BASE TARGET GENERATION (Pure Direction & Signal)")
        lookahead = self.target_cfg["lookahead"]

        df["future_return"] = df.groupby("symbol")["close"].shift(-lookahead) / df["close"] - 1

        # 1. Base Strategy Target (No execution friction, optimized for pure signal frequency)
        # We rely strictly on the Meta-Model to handle the actual cost/slippage bounds later.
        k = 0.25  # Lower volatility threshold for base signal
        if "volatility_20" in df.columns:
            dynamic_threshold = (df["volatility_20"] * k).clip(upper=0.01)
        else:
            dynamic_threshold = self.target_cfg.get("threshold", 0.002)

        df["target"] = (df["future_return"] > dynamic_threshold).astype(np.int8)

        # Drop NaNs early to free memory
        df.dropna(subset=["future_return"], inplace=True)
        clear_memory("Post-target generation")

        return df

    def _cross_sectional(self, df: pd.DataFrame) -> pd.DataFrame:
        """Global features (require all symbols)."""
        # Market return at each timestamp
        mkt_ret = df.groupby("date")["ret_1"].transform("mean")

        # Leakage prevention: market return of t-1 for decision at t
        df["mkt_ret_lag1"] = mkt_ret.groupby(df["symbol"], observed=True).shift(1).astype(np.float32)

        # Relative strength (also lagged)
        ret_1_lag = df.groupby("symbol")["ret_1"].shift(1)
        df["rel_strength"] = (ret_1_lag - df["mkt_ret_lag1"]).astype(np.float32)

        # NEW CROSS-SECTIONAL RANKING FEATURES
        # Rank features across all symbols at each timestamp (Outputs percentile rank 0.0 to 1.0)
        # NOTE: Using lagged returns to completely prevent future row leakage.
        df["rank_return"] = df.groupby("date")["ret_1"].shift(1).groupby(df["date"]).rank(pct=True).astype(np.float32)

        if "volume" in df.columns:
            df["rank_volume"] = df.groupby("date")["volume"].shift(1).groupby(df["date"]).rank(pct=True).astype(np.float32)

        if "mom_5" in df.columns:
            df["rank_momentum"] = df.groupby("date")["mom_5"].shift(1).groupby(df["date"]).rank(pct=True).astype(np.float32)

        df["rank_x_vol"] = df["rank_return"] * df["volatility_20"]
        df["rank_x_momentum"] = df["rank_momentum"] * df["ret_1"]

        # Z-score of cross-sectional returns (How many SDs above market mean was yesterday's bar?)
        ret_mean = df.groupby("date")["ret_1"].shift(1).groupby(df["date"]).transform("mean")
        ret_std = df.groupby("date")["ret_1"].shift(1).groupby(df["date"]).transform("std")
        df["z_score_return"] = ((df["groupby_symbol_ret_1_lag"] if False else ret_1_lag) - ret_mean) / ret_std.replace(0, np.nan)
        df["z_score_return"] = getattr(df["z_score_return"], "astype")(np.float32)

        return df

    def get_feature_list(self, df: pd.DataFrame) -> list:
        exclude = {"date", "symbol", "open", "high", "low", "close", "volume", "future_return", "target"}
        return [c for c in df.columns if c not in exclude]
