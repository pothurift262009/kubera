import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def apply_triple_barrier_elite_v2(df: pd.DataFrame, config: dict = None) -> pd.DataFrame:
    """
    Elite Triple Barrier Method (TBM) with Strict Decision Information.
    Uses volatility at decision time (T-1) to set barriers for future windows.
    Labels: 0 = Short, 1 = Flat, 2 = Long.
    """
    if config is None:
        config = {'tp_mult': 2.0, 'sl_mult': 1.0, 'max_bars': 6}
        
    tp_mult, sl_mult, max_bars = config['tp_mult'], config['sl_mult'], config['max_bars']
    
    logger.info(f"Applying Triple Barrier (V2) with tp={tp_mult}, sl={sl_mult}, horizon={max_bars}...")
    
    df = df.sort_values(['symbol', 'datetime']).reset_index(drop=True)
    
    # Critical: Use Decision-Time Volatility (T-1) for Barrier Setting
    # NOTE: df['atr_14'] is already shifted(1) in feature_engineering.py
    tr_vol = df['atr_14'].bfill()
    
    tp_long = df['close'] + tp_mult * tr_vol
    sl_long = df['close'] - sl_mult * tr_vol
    tp_short = df['close'] - tp_mult * tr_vol
    sl_short = df['close'] + sl_mult * tr_vol
    
    hit_tp_l, hit_sl_l = [np.full(len(df), 999) for _ in range(2)]
    hit_tp_s, hit_sl_s = [np.full(len(df), 999) for _ in range(2)]
    
    for i in range(1, max_bars + 1):
        # Peak/Trough within future horizon
        h_fut = df.groupby('symbol')['high'].shift(-i).values
        l_fut = df.groupby('symbol')['low'].shift(-i).values
        
        # Long Barriers
        m_tp_l = (hit_tp_l == 999) & (h_fut >= tp_long.values)
        hit_tp_l[m_tp_l] = i
        m_sl_l = (hit_sl_l == 999) & (l_fut <= sl_long.values)
        hit_sl_l[m_sl_l] = i
        
        # Short Barriers
        m_tp_s = (hit_tp_s == 999) & (l_fut <= tp_short.values)
        hit_tp_s[m_tp_s] = i
        m_sl_s = (hit_sl_s == 999) & (h_fut >= sl_short.values)
        hit_sl_s[m_sl_s] = i
        
    df['label'] = 1
    l_win = (hit_tp_l <= max_bars) & (hit_tp_l < hit_sl_l)
    s_win = (hit_tp_s <= max_bars) & (hit_tp_s < hit_sl_s)
    
    df.loc[l_win, 'label'] = 2
    df.loc[s_win, 'label'] = 0
    
    # Filter edge tails (not enough bars to confirm label)
    df.loc[df.groupby('symbol').tail(max_bars).index, 'label'] = 1
    
    logger.info(f"Label Stats: {df['label'].value_counts(normalize=True).to_dict()}")
    return df
