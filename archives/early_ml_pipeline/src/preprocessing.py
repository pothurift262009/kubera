import pandas as pd
import numpy as np
import logging
import time
from src.utils.memory import log_memory, clear_memory

logger = logging.getLogger(__name__)


class Preprocessor:
    """
    Memory-efficient cleaner for NIFTY50 5-min OHLCV data.
    Optimized for 8GB RAM with in-place operations.
    """

    def __init__(self, config: dict):
        self.config = config

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full preprocessing pipeline with memory profiling."""
        logger.info("🏗️ PREPROCESSING (Memory-Optimized)")
        log_memory("Pre-process")

        # ── 1. Datetime & Casting ───────────────────────────
        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            df["date"] = pd.to_datetime(df["date"])

        # Early cast to float32 to halve memory vs float64
        float_cols = ["open", "high", "low", "close"]
        df[float_cols] = df[float_cols].astype(np.float32)
        df["volume"] = df["volume"].astype(np.int32)

        # ── 2. Deduplication ──────────────────────────────────
        df.sort_values(["symbol", "date"], inplace=True)
        df.drop_duplicates(subset=["symbol", "date"], inplace=True)
        log_memory("Post-dedup")

        # ── 3. Market Hours (09:15 – 15:25) ────────────────────
        market_start = pd.Timestamp("09:15").time()
        market_end = pd.Timestamp("15:25").time()

        # Efficient masking (avoiding large temporary objects)
        time_vals = df["date"].dt.time
        mask = (time_vals >= market_start) & (time_vals <= market_end)
        df = df[mask].reset_index(drop=True)
        clear_memory("Post market-hour filter")

        # ── 4. Missing Value Imputation ──────────────────────
        # Use transform in-place as much as possible
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            df[col] = df.groupby("symbol", observed=True)[col].transform(lambda x: x.ffill().bfill())

        df.dropna(subset=float_cols + ["volume"], inplace=True)
        log_memory("Post-imputation")

        # ── 5. Integrity Checks (In-Place) ──────────────────
        # Avoid creating full boolean dataframes
        for col in float_cols:
            bad_mask = df[col] <= 0
            if bad_mask.any():
                logger.warning(f"  ⚠️ Removing {bad_mask.sum()} non-positive rows in {col}")
                df = df[~bad_mask].reset_index(drop=True)

        clear_memory("Final preprocessing cleanup")
        return df
