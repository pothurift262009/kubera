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

# Configure LOGGING (V2 Prod-Spec)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler("elite_pipeline_v2_f.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Elite_NSE_Pipeline_V2")

# ==========================================
# 1. DATA LOADING & PRODUCTION MERGE
# ==========================================

def load_ohlcv(path: str) -> pd.DataFrame:
    """Load KB_OP.csv.gz with optimized datetimes for NSE sessions."""
    if not os.path.exists(path): raise FileNotFoundError(path)
    df = pd.read_csv(path)
    rename_cols = {'#RIC': 'symbol', 'Date-Time': 'datetime', 'Last': 'close', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Volume': 'volume'}
    df = df.rename(columns=rename_cols)
    df['datetime'] = pd.to_datetime(df['datetime'].str[:19], format="%Y-%m-%dT%H:%M:%S")
    df['datetime'] = df['datetime'].dt.tz_localize('Asia/Kolkata')
    df = df.set_index('datetime').between_time('09:00', '15:30').reset_index()
    df = df.dropna(subset=['close'])
    df['symbol'] = df['symbol'].astype('category')
    return df.sort_values(['symbol', 'datetime']).reset_index(drop=True)

def merge_with_lob_asof(df_ohlcv, df_lob):
    """High-fidelity synchronization for multi-instrument trading."""
    for d in [df_ohlcv, df_lob]:
        if d['datetime'].dt.tz is None: d['datetime'] = d['datetime'].dt.tz_localize('Asia/Kolkata')
        else: d['datetime'] = d['datetime'].dt.tz_convert('Asia/Kolkata')
    
    df_ohlcv, df_lob = df_ohlcv.sort_values('datetime'), df_lob.sort_values('datetime')
    df_ohlcv['symbol'], df_lob['symbol'] = df_ohlcv['symbol'].astype(str), df_lob['symbol'].astype(str)
    
    return pd.merge_asof(df_ohlcv, df_lob, on='datetime', by='symbol', direction='backward')

# ==========================================
# 2. MICROSTRUCTURE ALPHA (LOB V2)
# ==========================================

def get_ofi_delta(p, s):
    p_diff, s_diff = p.diff().fillna(0), s.diff().fillna(0)
    delta = np.where(p_diff > 0, s.values, np.where(p_diff == 0, s_diff.values, -s.shift(1).fillna(0).values))
    return pd.Series(delta, index=p.index)

def compute_lob_features_elite_v2(df: pd.DataFrame) -> pd.DataFrame:
    """Microprice, OFI Momentum, and Book Imbalance Acceleration."""
    df['mid_price'] = (df['L1-AskPrice'] + df['L1-BidPrice']) / 2
    b_sz, a_sz = df['L1-BidSize'].fillna(1e-10), df['L1-AskSize'].fillna(1e-10)
    df['microprice'] = (df['L1-BidPrice'] * a_sz + df['L1-AskPrice'] * b_sz) / (b_sz + a_sz)
    df['spread_pct'] = (df['L1-AskPrice'] - df['L1-BidPrice']) / (df['mid_price'] + 1e-10)
    df['m_m_diff'] = (df['microprice'] - df['mid_price']) / (df['mid_price'] + 1e-10)
    
    df['ofi'] = get_ofi_delta(df['L1-BidPrice'], df['L1-BidSize']) - get_ofi_delta(df['L1-AskPrice'], df['L1-AskSize'])
    g = df.groupby('symbol')
    df['ofi_mom'] = g['ofi'].transform(lambda x: x.diff())
    
    for i in range(1, 4):
        b, a = df[f'L{i}-BidSize'].fillna(0), df[f'L{i}-AskSize'].fillna(0)
        df[f'obi_l{i}'] = (b - a) / (b + a + 1e-10)
        
    df['depth_imb'] = (sum(df[f'L{i}-BidSize'] for i in range(1, 4)) - sum(df[f'L{i}-AskSize'] for i in range(1, 4))) / (sum(df[f'L{i}-BidSize'] + df[f'L{i}-AskSize'] for i in range(1, 4)) + 1e-10)
    df['depth_acc'] = g['depth_imb'].transform(lambda x: x.diff())
    return df

def process_lob_v2(in_p, out_p, chunks=500_000, max_c=None):
    cols = ['#RIC', 'Date-Time']
    for i in range(1, 6): cols += [f'L{i}-BidPrice', f'L{i}-BidSize', f'L{i}-AskPrice', f'L{i}-SellNo']
    r_dict = {'#RIC': 'symbol', 'Date-Time': 'datetime'}
    for i in range(1, 6): r_dict[f'L{i}-SellNo'] = f'L{i}-AskSize'
    
    writer, count = None, 0
    for chunk in pd.read_csv(in_p, compression='gzip', usecols=cols, chunksize=chunks):
        if max_c and count >= max_c: break
        chunk = chunk.rename(columns=r_dict)
        chunk['datetime'] = pd.to_datetime(chunk['datetime'].str[:19], format="%Y-%m-%dT%H:%M:%S")
        chunk = compute_lob_features_elite_v2(chunk)
        res = chunk.set_index('datetime').groupby(['symbol', pd.Grouper(freq='5min')]).last().reset_index()
        k_cols = ['symbol', 'datetime', 'microprice', 'm_m_diff', 'spread_pct', 'ofi', 'ofi_mom', 'obi_l1', 'depth_imb', 'depth_acc']
        table = pa.Table.from_pandas(res[k_cols].dropna(subset=['datetime']))
        if writer is None: writer = pq.ParquetWriter(out_p, table.schema, compression='snappy')
        writer.write_table(table)
        count += 1
        if count % 10 == 0: logger.info(f"Chunk {count} complete...")
    if writer: writer.close()

# ==========================================
# 3. FEATURE ENGINEERING V2
# ==========================================

def run_features_v2(df: pd.DataFrame) -> pd.DataFrame:
    df = df[['symbol', 'datetime', 'open', 'high', 'low', 'close', 'volume']].copy()
    g = df.groupby('symbol')
    
    # Technical Alpha
    ema12, ema26 = g['close'].transform(lambda x: x.ewm(span=12).mean()), g['close'].transform(lambda x: x.ewm(span=26).mean())
    df['macd_r'] = (ema12 - ema26) / (ema26 + 1e-10)
    df['rsi_14'] = g['close'].transform(lambda x: 100 - (100 / (1 + x.diff().clip(lower=0).ewm(span=14).mean() / ((-x.diff().clip(upper=0)).ewm(span=14).mean() + 1e-10))))
    
    tr = pd.concat([df['high']-df['low'], (df['high']-df['close'].shift(1)).abs(), (df['low']-df['close'].shift(1)).abs()], axis=1).max(axis=1)
    df['atr_14'] = g.apply(lambda x: tr.loc[x.index].ewm(alpha=1/14, adjust=False).mean(), include_groups=False).squeeze().reset_index(level=0, drop=True)
    df['atr_pct'] = df['atr_14'] / (df['close'] + 1e-10)
    
    # Regime Features
    r_max, r_min, s_std = g['high'].transform(lambda x: x.rolling(30).max()), g['low'].transform(lambda x: x.rolling(30).min()), g['close'].transform(lambda x: x.rolling(30).std())
    df['hurst'] = (r_max - r_min) / (s_std + 1e-10)
    
    for l in [1, 3, 6]: df[f'ret_{l}b'] = g['close'].transform(lambda x: x.pct_change(l))
    df['mom_accel'] = df['ret_1b'] - g['ret_1b'].shift(1)
    
    # Strict Lagging
    pred_cols = [c for c in df.columns if c not in {'symbol', 'datetime'}]
    df[pred_cols] = df.groupby('symbol')[pred_cols].shift(1)
    
    t = df['datetime'].dt.hour * 60 + df['datetime'].dt.minute
    df['t_sin'], df['t_cos'] = np.sin(2*np.pi*t/(24*60)), np.cos(2*np.pi*t/(24*60))
    return df.dropna(subset=['rsi_14', 'atr_pct'])

# ==========================================
# 4. TRIPLE BARRIER LABELING
# ==========================================

def apply_tbm_v2(df: pd.DataFrame, config={'tp_m': 2.0, 'sl_m': 1.0, 'h': 6}):
    df, tp_m, sl_m, h = df.sort_values(['symbol', 'datetime']).reset_index(drop=True), config['tp_m'], config['sl_m'], config['h']
    v = df['atr_14'].bfill()
    tpl, sll = df['close'] + tp_m*v, df['close'] - sl_m*v
    tps, sls = df['close'] - tp_m*v, df['close'] + sl_m*v
    
    hl, sl, hs, ss = [np.full(len(df), 999) for _ in range(4)]
    for i in range(1, h + 1):
        hf, lf = df.groupby('symbol')['high'].shift(-i).values, df.groupby('symbol')['low'].shift(-i).values
        m_tp_l = (hl == 999) & (hf >= tpl.values); hl[m_tp_l] = i
        m_sl_l = (sl == 999) & (lf <= sll.values); sl[m_sl_l] = i
        m_tp_s = (hs == 999) & (lf <= tps.values); hs[m_tp_s] = i
        m_sl_s = (ss == 999) & (hf >= sls.values); ss[m_sl_s] = i
        
    df['label'] = 1
    df.loc[(hl <= h) & (hl < sl), 'label'] = 2
    df.loc[(hs <= h) & (hs < ss), 'label'] = 0
    return df

# ==========================================
# 5. MODELING & BACKTESTING V2
# ==========================================

def train_ensemble_v2(df, f_cols, n_splits=3):
    df = df.sort_values('datetime').reset_index(drop=True)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    tr_idx, _ = next(tscv.split(df))
    lgb_pre = lgb.LGBMClassifier(n_estimators=100, verbose=-1).fit(df.loc[tr_idx, f_cols].fillna(0), df.loc[tr_idx, 'label'])
    sel_cols = pd.Series(lgb_pre.feature_importances_, index=f_cols).sort_values(ascending=False).head(20).index.tolist()
    
    best_l, best_c = None, None
    for fold, (t_idx, v_idx) in enumerate(tscv.split(df)):
        Xt, Xv, yt, yv = df.loc[t_idx, sel_cols].fillna(0), df.loc[v_idx, sel_cols].fillna(0), df.loc[t_idx, 'label'], df.loc[v_idx, 'label']
        best_l = lgb.LGBMClassifier(n_estimators=500, class_weight='balanced', verbose=-1).fit(Xt, yt, eval_set=[(Xv, yv)], callbacks=[lgb.early_stopping(30, verbose=False)])
        best_c = cb.CatBoostClassifier(iterations=500, auto_class_weights='Balanced', verbose=0).fit(Xt, yt, eval_set=(Xv, yv), early_stopping_rounds=30)
    return best_l, best_c, sel_cols

def run_backtest_v2(df, probs, cost=0.0005, slippage=0.0002, threshold=0.65):
    bt = df.copy()
    bt['s_p'], bt['l_p'] = probs[:, 0], probs[:, 2]
    bt['target'] = 0
    bt.loc[bt['l_p'] > threshold, 'target'] = 1
    bt.loc[bt['s_p'] > threshold, 'target'] = -1
    
    # Vol/Spread Filters
    bt['target'] *= (bt['spread_pct'] < 0.0015).astype(int) * (bt['atr_pct'] > bt['atr_pct'].quantile(0.2)).astype(int)
    bt['pos'] = bt.groupby('symbol')['target'].transform(lambda x: x.replace(0, np.nan).ffill().fillna(0)).shift(1).fillna(0)
    
    bt['r_net'] = (bt['pos'] * bt.groupby('symbol')['close'].pct_change()) - (bt.groupby('symbol')['pos'].diff().abs() * (cost+slippage))
    
    rets = bt.groupby('datetime')['r_net'].mean()
    cum = (1 + rets).cumprod()
    sharpe = np.sqrt(75*252) * rets.mean() / (rets.std() + 1e-10)
    logger.info(f"V2 STATS: Sharpe {sharpe:.2f}, Return {cum.iloc[-1]-1:.2%}, Trades {len(bt[bt.groupby('symbol')['pos'].diff().abs()>0])}")
    return sharpe

def main():
    PATHS = {'ohlcv': '/Users/Pothuri/Downloads/kubera/KB_OP.csv.gz', 'lob': '/Users/Pothuri/Downloads/kubera/kblobop.csv.gz', 'v2_p': '/Users/Pothuri/Downloads/kubera/lob_v2_elite.parquet'}
    if not os.path.exists(PATHS['v2_p']): process_lob_v2(PATHS['lob'], PATHS['v2_p'])
    df_o = load_ohlcv(PATHS['ohlcv'])
    df_f = run_features_v2(df_o)
    df_l = pd.read_parquet(PATHS['v2_p'])
    df_l['datetime'] = pd.to_datetime(df_l['datetime']).dt.tz_localize('Asia/Kolkata')
    df_c = merge_with_lob_asof(df_f, df_l)
    df_lab = apply_tbm_v2(df_c)
    
    f_cols = [c for c in df_lab.columns if c not in {'symbol', 'datetime', 'label'}]
    m_l, m_c, sel_f = train_ensemble_v2(df_lab, f_cols)
    
    test_df = df_lab.iloc[int(0.8*len(df_lab)):]
    probs = (m_l.predict_proba(test_df[sel_f]) + m_c.predict_proba(test_df[sel_f])) / 2
    run_backtest_v2(test_df, probs)

if __name__ == "__main__":
    main()
