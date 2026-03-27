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
    State-of-the-art Vectorized OFI calculation.
    Logic: 
    if P(t) > P(t-1): size(t)
    if P(t) == P(t-1): delta_size(t)
    if P(t) < P(t-1): -size(t-1)
    """
    p_diff = p.diff().fillna(0)
    s_diff = s.diff().fillna(0)
    delta = np.where(p_diff > 0, s.values,
            np.where(p_diff == 0, s_diff.values,
            -s.shift(1).fillna(0).values))
    return pd.Series(delta, index=p.index)

def compute_lob_features_elite_v2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Advanced Microstructure Alpha: OFI Momentum, Microprice Drift, and Depth Dynamics.
    """
    # 1. Base Price & Spreads
    df['mid_price'] = (df['L1-AskPrice'] + df['L1-BidPrice']) / 2
    bid_sz, ask_sz = df['L1-BidSize'].fillna(1e-10), df['L1-AskSize'].fillna(1e-10)
    df['microprice'] = (df['L1-BidPrice'] * ask_sz + df['L1-AskPrice'] * bid_sz) / (bid_sz + ask_sz)
    df['micro_mid_diff'] = (df['microprice'] - df['mid_price']) / (df['mid_price'] + 1e-10)
    
    df['spread'] = df['L1-AskPrice'] - df['L1-BidPrice']
    df['spread_pct'] = df['spread'] / (df['mid_price'] + 1e-10)
    
    # 2. Advanced OFI (Order Flow Imbalance)
    bid_ofi = get_ofi_delta(df['L1-BidPrice'], df['L1-BidSize'])
    ask_ofi = get_ofi_delta(df['L1-AskPrice'], df['L1-AskSize'])
    df['ofi'] = bid_ofi.values - ask_ofi.values
    
    # OFI Momentum: rate of change of imbalance
    g = df.groupby('symbol')
    df['ofi_mom'] = g['ofi'].transform(lambda x: x.diff())
    
    # 3. Queue Imbalance (OBI) L1-L5
    for i in range(1, 6):
        b, a = df[f'L{i}-BidSize'].fillna(0), df[f'L{i}-AskSize'].fillna(0)
        df[f'obi_l{i}'] = (b - a) / (b + a + 1e-10)
    
    # 4. Book Depth & Slope
    df['total_depth'] = sum(df[f'L{i}-BidSize'] + df[f'L{i}-AskSize'] for i in range(1, 6))
    df['depth_imbalance'] = (sum(df[f'L{i}-BidSize'] for i in range(1, 6)) - sum(df[f'L{i}-AskSize'] for i in range(1, 6))) / (df['total_depth'] + 1e-10)
    
    # Depth Acceleration (Alpha: hidden buying/selling activity)
    df['depth_acc'] = g['depth_imbalance'].transform(lambda x: x.diff())
    
    # 5. Rolling microstructure states (Vectorized)
    df['ofi_ema_5'] = g['ofi'].transform(lambda x: x.ewm(span=5, adjust=False).mean())
    df['spread_std_5'] = g['spread_pct'].transform(lambda x: x.rolling(5).std())
    
    return df

def process_lob_elite_v2(input_path: str, output_path: str, chunksize: int = 1_000_000, max_chunks: int = None):
    """
    Memory-efficient LOB processing pipeline for multi-year NSE data.
    """
    if not os.path.exists(input_path): raise FileNotFoundError(input_path)
        
    cols = ['#RIC', 'Date-Time']
    for i in range(1, 6):
        cols += [f'L{i}-BidPrice', f'L{i}-BidSize', f'L{i}-AskPrice', f'L{i}-SellNo']
            
    rename_dict = {'#RIC': 'symbol', 'Date-Time': 'datetime'}
    for i in range(1, 6): rename_dict[f'L{i}-SellNo'] = f'L{i}-AskSize'
        
    writer = None
    logger.info(f"Staring Enhanced LOB Processing (V2) from {input_path}...")
    
    count = 0
    for chunk in pd.read_csv(input_path, compression='gzip', usecols=cols, chunksize=chunksize):
        if max_chunks and count >= max_chunks: break
        
        chunk = chunk.rename(columns=rename_dict)
        chunk['datetime'] = pd.to_datetime(chunk['datetime'].str[:19], format="%Y-%m-%dT%H:%M:%S")
        
        # Apply Enhanced Alpha Features
        chunk = compute_lob_features_elite_v2(chunk)
        
        # Chronological Resampling
        resampled = chunk.set_index('datetime').groupby(['symbol', pd.Grouper(freq='5min')]).last().reset_index()
        
        # Pruning redundant columns for high-fidelity alignment
        keep_cols = [
            'symbol', 'datetime', 'microprice', 'micro_mid_diff', 'spread_pct', 
            'ofi', 'ofi_mom', 'ofi_ema_5', 'obi_l1', 'obi_l2', 'depth_imbalance', 
            'depth_acc', 'spread_std_5'
        ]
        resampled = resampled[keep_cols].dropna(subset=['datetime'])
        
        # Parquet I/O
        table = pa.Table.from_pandas(resampled)
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema, compression='snappy')
        writer.write_table(table)
        count += 1
        if count % 10 == 0: logger.info(f"Processed {count} chunks...")
        
    if writer: writer.close()
    logger.info("Enhanced LOB Processing Complete.")

if __name__ == "__main__":
    pass
