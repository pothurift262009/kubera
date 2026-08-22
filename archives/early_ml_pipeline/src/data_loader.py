"""
═══════════════════════════════════════════════════════════════
📂 DATA LOADER — High-Performance OHLCV Ingestion
═══════════════════════════════════════════════════════════════
- Optimized dtypes for 4M+ row datasets
- Parquet caching for fast iteration
- Memory-efficient categorical encoding
"""

import pandas as pd
import os
import logging
import time
from typing import List, Optional
from src.utils.memory import log_memory, clear_memory

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Production-grade data loader optimized for 8GB RAM.
    Uses int32/float32 and selective loading for Parquet.
    """

    def __init__(self, config: dict):
        self.config = config
        self.raw_path = config["data"]["raw"]
        self.processed_path = config["data"]["processed"]

    def load_raw_csv(self) -> pd.DataFrame:
        """Loads raw CSV with optimized dtypes."""
        if not os.path.exists(self.raw_path):
            logger.error(f"❌ Raw data file missing: {self.raw_path}")
            raise FileNotFoundError(f"Missing {self.raw_path}")

        logger.info(f"📂 Loading raw CSV: {self.raw_path}")
        log_memory("Pre-load CSV")

        dtypes = {
            "symbol": "category",
            "open": "float32",
            "high": "float32",
            "low": "float32",
            "close": "float32",
            "volume": "int32", # int32 is enough for NIFTY 5-min bars (up to 2B)
        }

        try:
            df = pd.read_csv(
                self.raw_path,
                dtype=dtypes,
                parse_dates=["date"],
                low_memory=True, # Critical for RAM efficiency
            )

            # Ensure date is represented efficiently
            df["date"] = pd.to_datetime(df["date"])

            log_memory("Post-load CSV")
            return df
        except Exception as e:
            logger.error(f"❌ CSV load failed: {e}")
            raise

    def save_processed(self, df: pd.DataFrame) -> None:
        """Saves processed DataFrame to Parquet for fast reloading."""
        logger.info(f"💾 Saving processed Parquet → {self.processed_path}")
        os.makedirs(os.path.dirname(self.processed_path), exist_ok=True)
        # Using snappy compression for speed/disk balance
        df.to_parquet(self.processed_path, index=False, engine="pyarrow", compression="snappy")
        logger.info("✅ Parquet save complete.")
        clear_memory("Post-save Parquet")

    def load_processed(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Loads cached Parquet with optional column selection."""
        if not os.path.exists(self.processed_path):
            logger.info("No cached data found.")
            return None

        logger.info(f"⚡ Loading cached Parquet (Selective cols: {columns is not None})")
        log_memory("Pre-load Parquet")

        try:
            df = pd.read_parquet(
                self.processed_path,
                columns=columns, # Only load what is needed
                engine="pyarrow"
            )

            # Downcast to float32 if any float64 snuck in
            float_cols = df.select_dtypes(include=['float64']).columns
            if not float_cols.empty:
                df[float_cols] = df[float_cols].astype('float32')

            log_memory("Post-load Parquet")
            return df
        except Exception as e:
            logger.error(f"❌ Parquet load failed: {e}")
            return None
