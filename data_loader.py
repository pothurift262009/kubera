import pandas as pd
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_ohlcv(path: str) -> pd.DataFrame:
    """
    Load KB_OP.csv.gz (OHLCV + basic order book data) with performance optimizations.
    """
    logger.info(f"Loading OHLCV data from {path}...")
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"OHLCV file not found: {path}")
    
    # Use fast-path string slicing for datetime parsing
    df = pd.read_csv(path)
    
    # Rename columns early
    rename_cols = {
        '#RIC': 'symbol', 'Date-Time': 'datetime',
        'Last': 'close', 'Open': 'open',
        'High': 'high', 'Low': 'low', 'Volume': 'volume'
    }
    df = df.rename(columns=rename_cols)
    
    # Fast datetime parsing
    df['datetime'] = pd.to_datetime(df['datetime'].str[:19], format="%Y-%m-%dT%H:%M:%S")
    df['datetime'] = df['datetime'].dt.tz_localize('Asia/Kolkata')
    
    # Filter NSE trading hours: 09:00 - 15:30
    df = df.set_index('datetime').between_time('09:00', '15:30').reset_index()
    
    # Drop rows without close prices
    df = df.dropna(subset=['close'])
    
    # Convert to efficient types
    df['symbol'] = df['symbol'].astype('category')
    
    # Sort for merge_asof
    df = df.sort_values(['symbol', 'datetime']).reset_index(drop=True)
    
    logger.info(f"Successfully loaded {len(df):,} rows of OHLCV data.")
    return df

def merge_with_lob_asof(df_ohlcv, df_lob):
    """
    High-fidelity time-aware merging using merge_asof.
    FIXED BUG 4: Ensuring matching tz-aware datetimes in IST.
    FIXED: Global sorting by 'datetime' for merge_asof requirements.
    """
    logger.info("Performing Elite merge_asof between OHLCV and LOB...")
    
    # Ensure both have matching tz-aware datetime in IST
    for df in [df_ohlcv, df_lob]:
        if df['datetime'].dt.tz is None:
            df['datetime'] = df['datetime'].dt.tz_localize('Asia/Kolkata')
        else:
            df['datetime'] = df['datetime'].dt.tz_convert('Asia/Kolkata')
    
    # CRITICAL: merge_asof requires 'on' key to be globally sorted
    df_ohlcv = df_ohlcv.sort_values('datetime')
    df_lob   = df_lob.sort_values('datetime')
    
    # Align types
    df_ohlcv['symbol'] = df_ohlcv['symbol'].astype(str)
    df_lob['symbol']   = df_lob['symbol'].astype(str)
    
    return pd.merge_asof(
        df_ohlcv, df_lob,
        on='datetime', by='symbol',
        direction='backward'
    )

if __name__ == "__main__":
    pass
