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

def merge_with_lob_asof(df_ohlcv: pd.DataFrame, df_lob: pd.DataFrame) -> pd.DataFrame:
    """
    High-fidelity time-aware merging using merge_asof.
    Ensures LOB data aligned with OHLCV bar is strictly backward-looking.
    """
    logger.info("Performing merge_asof between OHLCV and LOB data...")
    
    # Ensure both are sorted by datetime for merge_asof
    df_ohlcv = df_ohlcv.sort_values('datetime')
    df_lob = df_lob.sort_values('datetime')
    
    # Align types
    df_ohlcv['symbol'] = df_ohlcv['symbol'].astype(str)
    df_lob['symbol'] = df_lob['symbol'].astype(str)
    
    # Merge_asof: for each row in df_ohlcv, match the nearest row in df_lob 
    # that has a timestamp <= ohlcv_timestamp.
    df_merged = pd.merge_asof(
        df_ohlcv, 
        df_lob, 
        on='datetime', 
        by='symbol', 
        direction='backward'
    )
    
    logger.info(f"Merge complete. Final shape: {df_merged.shape}")
    return df_merged

if __name__ == "__main__":
    pass
