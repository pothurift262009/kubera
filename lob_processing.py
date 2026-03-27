import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_ofi_delta(p, s):
    """
    Robust OFI calculation using np.where to avoid alignment/boolean indexing issues.
    """
    p_diff = p.diff().fillna(0)
    s_diff = s.diff().fillna(0)
    delta = np.where(p_diff > 0, s.values,
            np.where(p_diff == 0, s_diff.values,
            -s.shift(1).fillna(0).values))
    return pd.Series(delta, index=p.index)

def compute_lob_features_elite(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elite LOB features: Microprice, Robust OFI, Slope, Rolling Book States.
    """
    # 1. Prices and Spreads
    df['mid_price'] = (df['L1-AskPrice'] + df['L1-BidPrice']) / 2
    bid_sz = df['L1-BidSize'].fillna(1e-10)
    ask_sz = df['L1-AskSize'].fillna(1e-10)
    df['microprice'] = (df['L1-BidPrice'] * ask_sz + df['L1-AskPrice'] * bid_sz) / (bid_sz + ask_sz)
    
    df['spread'] = df['L1-AskPrice'] - df['L1-BidPrice']
    df['spread_pct'] = df['spread'] / (df['mid_price'] + 1e-10)
    
    # 2. Book Imbalances (OBI) L1-L5
    for i in range(1, 6):
        b = df[f'L{i}-BidSize'].fillna(0)
        a = df[f'L{i}-AskSize'].fillna(0)
        df[f'obi_l{i}'] = (b - a) / (b + a).replace(0, np.nan)
        
    # 3. Robust Order Flow Imbalance (OFI)
    bid_ofi = get_ofi_delta(df['L1-BidPrice'], df['L1-BidSize'])
    ask_ofi = get_ofi_delta(df['L1-AskPrice'], df['L1-AskSize'])
    df['ofi'] = bid_ofi.values - ask_ofi.values
    
    # 4. Total Book Slope and Intensity
    df['total_bid_size'] = sum(df[f'L{i}-BidSize'].fillna(0) for i in range(1, 6))
    df['total_ask_size'] = sum(df[f'L{i}-AskSize'].fillna(0) for i in range(1, 6))
    df['book_depth_imbalance'] = (df['total_bid_size'] - df['total_ask_size']) / (df['total_bid_size'] + df['total_ask_size'] + 1e-10)
    
    # 5. Rolling stats of book states
    g = df.groupby('symbol')
    df['obi_l1_m5'] = g['obi_l1'].transform(lambda x: x.rolling(5).mean())
    df['spread_std5'] = g['spread_pct'].transform(lambda x: x.rolling(5).std())
    
    return df

def process_lob_elite(input_path: str, output_path: str, chunksize: int = 2_000_000, max_chunks: int = None):
    """
    Elite LOB processor with incremental writing.
    Stays NAIVE for better Parquet compatibility in pandas 3.0.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"LOB file not found: {input_path}")
        
    cols = ['#RIC', 'Date-Time']
    for i in range(1, 6):
        cols += [f'L{i}-BidPrice', f'L{i}-BidSize', f'L{i}-AskPrice', f'L{i}-SellNo']
            
    rename_dict = {'#RIC': 'symbol', 'Date-Time': 'datetime'}
    for i in range(1, 6):
        rename_dict[f'L{i}-SellNo'] = f'L{i}-AskSize'
        
    writer = None
    logger.info(f"Starting Elite LOB processing from {input_path}...")
    
    count = 0
    for chunk in pd.read_csv(input_path, compression='gzip', usecols=cols, chunksize=chunksize):
        if max_chunks and count >= max_chunks:
            logger.info(f"Reached max_chunks limit: {max_chunks}")
            break
        
        chunk = chunk.rename(columns=rename_dict)
        # BUG FIX: Stay NAIVE for Parquet compatibility
        chunk['datetime'] = pd.to_datetime(chunk['datetime'].str[:19], format="%Y-%m-%dT%H:%M:%S")
        
        # Elite Features
        chunk = compute_lob_features_elite(chunk)
        
        # Resample
        resampled = chunk.set_index('datetime').groupby(['symbol', pd.Grouper(freq='5min')]).last().reset_index()
        
        # Select key features
        keep_cols = ['symbol', 'datetime', 'mid_price', 'microprice', 'spread_pct',
                     'obi_l1', 'obi_l2', 'obi_l1_m5', 'spread_std5', 'ofi', 
                     'book_depth_imbalance']
        resampled = resampled[keep_cols].dropna(subset=['datetime'])
        
        # Write
        table = pa.Table.from_pandas(resampled)
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema, compression='snappy')
        writer.write_table(table)
        count += 1
        logger.info(f"Processed chunk {count}")
        
    if writer:
        writer.close()
    logger.info("LOB Processing Complete.")

if __name__ == "__main__":
    pass
