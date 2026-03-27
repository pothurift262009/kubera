import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def apply_triple_barrier_elite(df: pd.DataFrame, config: dict = None) -> pd.DataFrame:
    """
    Elite Triple Barrier Labeling with configurable parameters.
    Labels: 0 = Short, 1 = Flat, 2 = Long.
    """
    if config is None:
        config = {'tp_mult': 1.5, 'sl_mult': 1.0, 'max_bars': 6}
        
    tp_mult = config.get('tp_mult', 1.5)
    sl_mult = config.get('sl_mult', 1.0)
    max_bars = config.get('max_bars', 6)
    
    logger.info(f"Applying Elite TBM (tp={tp_mult}, sl={sl_mult}, horizon={max_bars})...")
    
    df = df.sort_values(['symbol', 'datetime']).reset_index(drop=True)
    
    # Bug Fix: Use .bfill() for pandas 3.0 compatibility
    tr_vol = df['atr_14'].bfill()
    
    tp_long = df['close'] + tp_mult * tr_vol
    sl_long = df['close'] - sl_mult * tr_vol
    tp_short = df['close'] - tp_mult * tr_vol
    sl_short = df['close'] + sl_mult * tr_vol
    
    hit_tp_l = np.full(len(df), 999)
    hit_sl_l = np.full(len(df), 999)
    hit_tp_s = np.full(len(df), 999)
    hit_sl_s = np.full(len(df), 999)
    
    for i in range(1, max_bars + 1):
        # We group by symbol to avoid crossing stocks during shift
        h_fut = df.groupby('symbol')['high'].shift(-i).values
        l_fut = df.groupby('symbol')['low'].shift(-i).values
        
        # Long
        m_tp_l = (hit_tp_l == 999) & (h_fut >= tp_long.values)
        hit_tp_l[m_tp_l] = i
        m_sl_l = (hit_sl_l == 999) & (l_fut <= sl_long.values)
        hit_sl_l[m_sl_l] = i
        
        # Short
        m_tp_s = (hit_tp_s == 999) & (l_fut <= tp_short.values)
        hit_tp_s[m_tp_s] = i
        m_sl_s = (hit_sl_s == 999) & (h_fut >= sl_short.values)
        hit_sl_s[m_sl_s] = i
        
    df['label'] = 1
    long_win = (hit_tp_l <= max_bars) & (hit_tp_l < hit_sl_l)
    short_win = (hit_tp_s <= max_bars) & (hit_tp_s < hit_sl_s)
    
    df.loc[long_win, 'label'] = 2
    df.loc[short_win, 'label'] = 0
    
    # Filter edge tails
    last_idx = df.groupby('symbol').tail(max_bars).index
    df.loc[last_idx, 'label'] = 1
    
    logger.info(f"Target distribution: {df['label'].value_counts(normalize=True).to_dict()}")
    return df
