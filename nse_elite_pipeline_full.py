import pandas as pd
import numpy as np
import logging
import os
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
import lightgbm as lgb
import catboost as cb
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import TimeSeriesSplit

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler("elite_pipeline_full.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Elite_NSE_Pipeline")

# ==========================================
# 1. DATA LOADING & SYNCHRONIZATION
# ==========================================

def load_ohlcv(path: str) -> pd.DataFrame:
    """Load KB_OP.csv.gz with optimized datetime parsing and IST localization."""
    logger.info(f"Loading OHLCV data from {path}...")
    if not os.path.exists(path):
        raise FileNotFoundError(f"OHLCV file not found: {path}")
    
    df = pd.read_csv(path)
    rename_cols = {
        '#RIC': 'symbol', 'Date-Time': 'datetime',
        'Last': 'close', 'Open': 'open',
        'High': 'high', 'Low': 'low', 'Volume': 'volume'
    }
    df = df.rename(columns=rename_cols)
    df['datetime'] = pd.to_datetime(df['datetime'].str[:19], format="%Y-%m-%dT%H:%M:%S")
    df['datetime'] = df['datetime'].dt.tz_localize('Asia/Kolkata')
    df = df.set_index('datetime').between_time('09:00', '15:30').reset_index()
    df = df.dropna(subset=['close'])
    df['symbol'] = df['symbol'].astype('category')
    df = df.sort_values(['symbol', 'datetime']).reset_index(drop=True)
    logger.info(f"Successfully loaded {len(df):,} rows.")
    return df

def merge_with_lob_asof(df_ohlcv, df_lob):
    """Time-aware synchronization using merge_asof with timezone alignment."""
    for df in [df_ohlcv, df_lob]:
        if df['datetime'].dt.tz is None:
            df['datetime'] = df['datetime'].dt.tz_localize('Asia/Kolkata')
        else:
            df['datetime'] = df['datetime'].dt.tz_convert('Asia/Kolkata')
    
    df_ohlcv = df_ohlcv.sort_values('datetime')
    df_lob   = df_lob.sort_values('datetime')
    df_ohlcv['symbol'] = df_ohlcv['symbol'].astype(str)
    df_lob['symbol']   = df_lob['symbol'].astype(str)
    
    logger.info("Synchronizing datasets via merge_asof...")
    return pd.merge_asof(df_ohlcv, df_lob, on='datetime', by='symbol', direction='backward')

# ==========================================
# 2. LOB MICROSTRUCTURE PROCESSING
# ==========================================

def get_ofi_delta(p, s):
    """Vectorized Order Flow Imbalance logic."""
    p_diff = p.diff().fillna(0)
    s_diff = s.diff().fillna(0)
    delta = np.where(p_diff > 0, s.values,
            np.where(p_diff == 0, s_diff.values,
            -s.shift(1).fillna(0).values))
    return pd.Series(delta, index=p.index)

def compute_lob_features_elite(df: pd.DataFrame) -> pd.DataFrame:
    """Microprice, Robust OFI, Slope, and Rolling Book States."""
    df['mid_price'] = (df['L1-AskPrice'] + df['L1-BidPrice']) / 2
    bid_sz = df['L1-BidSize'].fillna(1e-10)
    ask_sz = df['L1-AskSize'].fillna(1e-10)
    df['microprice'] = (df['L1-BidPrice'] * ask_sz + df['L1-AskPrice'] * bid_sz) / (bid_sz + ask_sz)
    df['spread'] = df['L1-AskPrice'] - df['L1-BidPrice']
    df['spread_pct'] = df['spread'] / (df['mid_price'] + 1e-10)
    
    for i in range(1, 6):
        b = df[f'L{i}-BidSize'].fillna(0)
        a = df[f'L{i}-AskSize'].fillna(0)
        df[f'obi_l{i}'] = (b - a) / (b + a).replace(0, np.nan)
        
    bid_ofi = get_ofi_delta(df['L1-BidPrice'], df['L1-BidSize'])
    ask_ofi = get_ofi_delta(df['L1-AskPrice'], df['L1-AskSize'])
    df['ofi'] = bid_ofi.values - ask_ofi.values
    
    df['total_bid_size'] = sum(df[f'L{i}-BidSize'].fillna(0) for i in range(1, 6))
    df['total_ask_size'] = sum(df[f'L{i}-AskSize'].fillna(0) for i in range(1, 6))
    df['book_depth_imbalance'] = (df['total_bid_size'] - df['total_ask_size']) / (df['total_bid_size'] + df['total_ask_size'] + 1e-10)
    
    g = df.groupby('symbol')
    df['obi_l1_m5'] = g['obi_l1'].transform(lambda x: x.rolling(5).mean())
    df['spread_std5'] = g['spread_pct'].transform(lambda x: x.rolling(5).std())
    return df

def process_lob_elite(input_path: str, output_path: str, chunksize: int = 2_000_000, max_chunks: int = None):
    """Incremental LOB processing to Parquet."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"LOB file not found: {input_path}")
        
    cols = ['#RIC', 'Date-Time']
    for i in range(1, 6):
        cols += [f'L{i}-BidPrice', f'L{i}-BidSize', f'L{i}-AskPrice', f'L{i}-SellNo']
            
    rename_dict = {'#RIC': 'symbol', 'Date-Time': 'datetime'}
    for i in range(1, 6):
        rename_dict[f'L{i}-SellNo'] = f'L{i}-AskSize'
        
    writer = None
    logger.info(f"Processing Elite LOB from {input_path}...")
    
    count = 0
    for chunk in pd.read_csv(input_path, compression='gzip', usecols=cols, chunksize=chunksize):
        if max_chunks and count >= max_chunks: break
        chunk = chunk.rename(columns=rename_dict)
        chunk['datetime'] = pd.to_datetime(chunk['datetime'].str[:19], format="%Y-%m-%dT%H:%M:%S")
        chunk = compute_lob_features_elite(chunk)
        resampled = chunk.set_index('datetime').groupby(['symbol', pd.Grouper(freq='5min')]).last().reset_index()
        
        keep_cols = ['symbol', 'datetime', 'mid_price', 'microprice', 'spread_pct',
                     'obi_l1', 'obi_l2', 'obi_l1_m5', 'spread_std5', 'ofi', 'book_depth_imbalance']
        resampled = resampled[keep_cols].dropna(subset=['datetime'])
        
        table = pa.Table.from_pandas(resampled)
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema, compression='snappy')
        writer.write_table(table)
        count += 1
        if count % 5 == 0: logger.info(f"Processed {count} chunks...")
        
    if writer: writer.close()
    logger.info("LOB Processing Complete.")

# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================

def add_technical_indicators_elite(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized indicators with transform safety."""
    df = df.sort_values(['symbol', 'datetime']).reset_index(drop=True)
    g = df.groupby('symbol')

    def rsi(s, p):
        d = s.diff(); g = d.clip(lower=0); l = -d.clip(upper=0)
        return 100 - 100 / (1 + g.ewm(com=p-1, adjust=False).mean() / (l.ewm(com=p-1, adjust=False).mean() + 1e-10))

    for p in [7, 14, 21]:
        df[f'rsi_{p}'] = g['close'].transform(lambda x: rsi(x, p))
    
    df['macd'] = g['close'].transform(lambda x: x.ewm(span=12).mean() - x.ewm(span=26).mean())
    df['macd_signal'] = g['macd'].transform(lambda x: x.ewm(span=9).mean())
    
    pc = g['close'].shift(1)
    tr = pd.concat([df['high']-df['low'], (df['high']-pc).abs(), (df['low']-pc).abs()], axis=1).max(axis=1)
    df['atr_14'] = g.apply(lambda x: tr.loc[x.index].ewm(alpha=1/14, adjust=False).mean(), include_groups=False).squeeze().reset_index(level=0, drop=True)
    
    up = df['high'].diff(); dw = -df['low'].diff()
    up.loc[df['symbol'] != df['symbol'].shift(1)] = 0; dw.loc[df['symbol'] != df['symbol'].shift(1)] = 0
    df['pdm'] = np.where((up > dw) & (up > 0), up, 0)
    df['mdm'] = np.where((dw > up) & (dw > 0), dw, 0)
    
    pdi = 100 * g['pdm'].transform(lambda x: pd.Series(x).ewm(alpha=1/14, adjust=False).mean()) / (df['atr_14'] + 1e-10)
    mdi = 100 * g['mdm'].transform(lambda x: pd.Series(x).ewm(alpha=1/14, adjust=False).mean()) / (df['atr_14'] + 1e-10)
    df['adx_14'] = g.apply(lambda x: (100 * (pdi.loc[x.index] - mdi.loc[x.index]).abs() / (pdi.loc[x.index] + mdi.loc[x.index] + 1e-10)).ewm(alpha=1/14, adjust=False).mean(), include_groups=False).squeeze().reset_index(level=0, drop=True)
    
    sma20 = g['close'].transform(lambda x: x.rolling(20).mean())
    std20 = g['close'].transform(lambda x: x.rolling(20).std())
    df['bb_width'] = (4 * std20) / (sma20 + 1e-10)
    return df.drop(columns=['pdm', 'mdm'], errors='ignore')

def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    df['is_trending'] = (df['adx_14'] > 25).astype(int)
    df['is_ranging'] = (df['adx_14'] < 20).astype(int)
    g = df.groupby('symbol')
    r_max = g['high'].transform(lambda x: x.rolling(30).max())
    r_min = g['low'].transform(lambda x: x.rolling(30).min())
    s_std = g['close'].transform(lambda x: x.rolling(30).std())
    df['hurst_proxy'] = (r_max - r_min) / (s_std + 1e-10)
    return df

def run_feature_pipeline_elite(df: pd.DataFrame) -> pd.DataFrame:
    df = df[['symbol', 'datetime', 'open', 'high', 'low', 'close', 'volume']].copy()
    df = add_technical_indicators_elite(df)
    df = add_regime_features(df)
    
    g = df.groupby('symbol')
    for lag in [1, 3, 6, 12, 24]:
        df[f'return_{lag}b'] = g['close'].transform(lambda x: x.pct_change(lag))
    
    vol_sma20 = g['volume'].transform(lambda x: x.rolling(20).mean())
    df['vol_ratio'] = df['volume'] / (vol_sma20 + 1e-10)
    df['rsi_vol_interaction'] = df['rsi_14'] * df['vol_ratio']
    
    exclude = {'symbol', 'datetime', 'label'}
    raw_cols = {'open', 'high', 'low', 'close', 'volume'}
    shift_cols = [c for c in df.columns if c not in exclude and c not in raw_cols]
    df[shift_cols] = df.groupby('symbol')[shift_cols].shift(1)
    
    t = df['datetime'].dt.hour * 60 + df['datetime'].dt.minute
    df['time_sin'] = np.sin(2 * np.pi * t / (24 * 60))
    df['time_cos'] = np.cos(2 * np.pi * t / (24 * 60))
    return df.dropna(subset=['rsi_14', 'adx_14', 'bb_width'])

# ==========================================
# 4. LABELING & MODELING
# ==========================================

def apply_triple_barrier_elite(df: pd.DataFrame, config: dict = None) -> pd.DataFrame:
    if config is None: config = {'tp_mult': 1.5, 'sl_mult': 1.0, 'max_bars': 6}
    tp_mult, sl_mult, max_bars = config['tp_mult'], config['sl_mult'], config['max_bars']
    
    df = df.sort_values(['symbol', 'datetime']).reset_index(drop=True)
    tr_vol = df['atr_14'].bfill()
    tp_l, sl_l = df['close'] + tp_mult * tr_vol, df['close'] - sl_mult * tr_vol
    tp_s, sl_s = df['close'] - tp_mult * tr_vol, df['close'] + sl_mult * tr_vol
    
    h_l, s_l, h_s, s_s = [np.full(len(df), 999) for _ in range(4)]
    for i in range(1, max_bars + 1):
        hf, lf = df.groupby('symbol')['high'].shift(-i).values, df.groupby('symbol')['low'].shift(-i).values
        m_tp_l = (h_l == 999) & (hf >= tp_l.values); h_l[m_tp_l] = i
        m_sl_l = (s_l == 999) & (lf <= sl_l.values); s_l[m_sl_l] = i
        m_tp_s = (h_s == 999) & (lf <= tp_s.values); h_s[m_tp_s] = i
        m_sl_s = (s_s == 999) & (hf >= sl_s.values); s_s[m_sl_s] = i
        
    df['label'] = 1
    df.loc[(h_l <= max_bars) & (h_l < s_l), 'label'] = 2
    df.loc[(h_s <= max_bars) & (h_s < s_s), 'label'] = 0
    df.loc[df.groupby('symbol').tail(max_bars).index, 'label'] = 1
    return df

def train_elite_ensemble(df: pd.DataFrame, feature_cols: list, target_col: str = 'label', n_splits: int = 3):
    df = df.sort_values('datetime').reset_index(drop=True)
    valid_cols = [c for c in feature_cols if df[c].isna().mean() < 0.5]
    X, y = df[valid_cols].fillna(0), df[target_col].fillna(1).astype(int)
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    best_lgb, best_cb = None, None
    for fold, (t_idx, v_idx) in enumerate(tscv.split(X)):
        Xt, Xv, yt, yv = X.iloc[t_idx], X.iloc[v_idx], y.iloc[t_idx], y.iloc[v_idx]
        lgb_m = lgb.LGBMClassifier(n_estimators=500, is_unbalance=True, verbose=-1).fit(Xt, yt, eval_set=[(Xv, yv)], callbacks=[lgb.early_stopping(30, verbose=False)])
        cb_m = cb.CatBoostClassifier(iterations=500, auto_class_weights='Balanced', verbose=0).fit(Xt, yt, eval_set=(Xv, yv), early_stopping_rounds=30)
        best_lgb, best_cb = lgb_m, cb_m
        logger.info(f"Fold {fold+1} complete.")
        
    imp = pd.DataFrame({'feature': valid_cols, 'importance': best_lgb.feature_importances_})
    return best_lgb, best_cb, imp, valid_cols

# ==========================================
# 5. BACKTESTING
# ==========================================

def run_backtest_elite(df, preds, probs, cost=0.0005, slippage=0.0002, prob_threshold=0.55):
    bt_df = df.copy()
    bt_df['pred'], bt_df['pred_prob'] = preds, np.max(probs, axis=1)
    pos_map = {0: -1, 1: 0, 2: 1}
    bt_df['target_pos'] = bt_df['pred'].map(pos_map)
    bt_df.loc[bt_df['pred_prob'] < prob_threshold, 'target_pos'] = 0
    bt_df['actual_pos'] = bt_df.groupby('symbol')['target_pos'].shift(1).fillna(0)
    bt_df['net_ret'] = (bt_df['actual_pos'] * bt_df.groupby('symbol')['close'].pct_change()) - (bt_df.groupby('symbol')['actual_pos'].diff().abs().fillna(0) * (cost+slippage))
    
    p_rets = bt_df.groupby('datetime')['net_ret'].mean()
    cum_rets = (1 + p_rets).cumprod()
    max_dd = ((cum_rets / cum_rets.cummax()) - 1).min()
    sharpe = np.sqrt(75*252) * p_rets.mean() / (p_rets.std() + 1e-10)
    
    logger.info(f"Backtest: Total Return {cum_rets.iloc[-1]-1:.2%}, Max DD {max_dd:.2%}, Sharpe {sharpe:.2f}")
    metrics = {'Total Return': cum_rets.iloc[-1]-1, 'Max DD': max_dd, 'Sharpe': sharpe}
    return metrics

# ==========================================
# EXECUTION
# ==========================================

def main():
    PATHS = {'ohlcv': '/Users/Pothuri/Downloads/kubera/KB_OP.csv.gz', 'lob': '/Users/Pothuri/Downloads/kubera/kblobop.csv.gz', 'parquet': '/Users/Pothuri/Downloads/kubera/lob_5m_elite.parquet'}
    
    if not os.path.exists(PATHS['parquet']): 
        process_lob_elite(PATHS['lob'], PATHS['parquet'])
    
    df_ohlcv = load_ohlcv(PATHS['ohlcv'])
    df_feat = run_feature_pipeline_elite(df_ohlcv)
    df_lob = pd.read_parquet(PATHS['parquet'])
    df_lob['datetime'] = pd.to_datetime(df_lob['datetime'])
    df_lob['datetime'] = df_lob['datetime'].dt.tz_localize('Asia/Kolkata') if df_lob['datetime'].dt.tz is None else df_lob['datetime'].dt.tz_convert('Asia/Kolkata')
    
    df_comb = merge_with_lob_asof(df_feat, df_lob)
    df_lab = apply_triple_barrier_elite(df_comb)
    
    f_cols = [c for c in df_lab.columns if c not in {'symbol', 'datetime', 'label'}]
    best_l, best_c, imp, f_cols = train_elite_ensemble(df_lab, f_cols)
    
    test_df = df_lab.iloc[int(0.8*len(df_lab)):]
    probs = (best_l.predict_proba(test_df[f_cols]) + best_c.predict_proba(test_df[f_cols])) / 2
    preds = np.argmax(probs, axis=1)
    
    logger.info("\n" + classification_report(test_df['label'], preds))
    run_backtest_elite(test_df, preds, probs)

if __name__ == "__main__":
    main()
