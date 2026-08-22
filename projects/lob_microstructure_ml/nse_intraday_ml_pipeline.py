"""
NSE Intraday ML Pipeline — Full Pipeline
==========================================
Steps:
  1. Install dependencies
  2. Load & preprocess KB_OP.csv.gz (OHLCV + order book)
  3. Load kblobop.csv.gz in chunks, compute LOB features → lob_5m.parquet
  4. Manual feature engineering (RSI, MACD, EMA, ATR, BB, VWAP, OBV, CCI …)
  5. Merge LOB features (auto-skipped if lob_5m.parquet not yet ready)
  6. Triple barrier labeling  →  nifty10_final.parquet
  7. LightGBM + CatBoost ensemble, evaluation, model save

Run each step independently or as a full pipeline.
"""

import os
import subprocess
import sys

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Install dependencies
# ─────────────────────────────────────────────────────────────────────────────
def step1_install():
    print("=" * 60)
    print("STEP 1: Installing dependencies...")
    print("=" * 60)
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "pandas", "numpy", "pandas_ta", "lightgbm", "catboost",
        "scikit-learn", "pyarrow", "tqdm"
    ])
    print("  ✓ Dependencies installed.\n")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Load KB_OP.csv.gz
# ─────────────────────────────────────────────────────────────────────────────
def step2_load_ohlcv(path='/Users/Pothuri/Downloads/kubera/KB_OP.csv.gz'):
    import pandas as pd
    import warnings
    warnings.filterwarnings('ignore')

    print("=" * 60)
    print("STEP 2: Loading KB_OP.csv.gz...")
    print("=" * 60)

    df = pd.read_csv(path)
    df = df.rename(columns={
        '#RIC': 'symbol', 'Date-Time': 'datetime',
        'Last': 'close', 'Open': 'open',
        'High': 'high', 'Low': 'low', 'Volume': 'volume'
    })

    # Fast datetime parse (avoid slow inference path)
    df['datetime'] = (
        pd.to_datetime(df['datetime'].str[:19], format="%Y-%m-%dT%H:%M:%S")
          .dt.tz_localize('Asia/Kolkata')
    )

    df = df.set_index('datetime').between_time('09:00', '15:30').reset_index()
    df = df.dropna(subset=['close'])
    df = df.sort_values(['symbol', 'datetime']).reset_index(drop=True)

    print(f"  ✓ Loaded {len(df):,} rows, {df['symbol'].nunique()} symbols.\n")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Process kblobop.csv.gz → lob_5m.parquet
# ─────────────────────────────────────────────────────────────────────────────
def step3_process_lob(
    path='/Users/Pothuri/Downloads/kubera/kblobop.csv.gz',
    out='/Users/Pothuri/Downloads/kubera/lob_5m.parquet',
    chunksize=2_000_000
):
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    import warnings
    warnings.filterwarnings('ignore')

    print("=" * 60)
    print("STEP 3: Processing kblobop.csv.gz in chunks...")
    print("=" * 60)

    columns_to_load = ['#RIC', 'Date-Time']
    for i in range(1, 6):
        columns_to_load += [f'L{i}-BidPrice', f'L{i}-BidSize',
                            f'L{i}-AskPrice', f'L{i}-SellNo']  # SellNo = AskSize

    rename_dict = {'#RIC': 'symbol', 'Date-Time': 'datetime'}
    for i in range(1, 6):
        rename_dict[f'L{i}-SellNo'] = f'L{i}-AskSize'

    resampled_chunks = []

    for chunk in tqdm(pd.read_csv(path, compression='gzip',
                                  usecols=columns_to_load, chunksize=chunksize)):
        chunk = chunk.rename(columns=rename_dict)
        chunk['datetime'] = (
            pd.to_datetime(chunk['datetime'].str[:19], format="%Y-%m-%dT%H:%M:%S")
              .dt.tz_localize('Asia/Kolkata')
        )

        tot_bid = sum(chunk[f'L{i}-BidSize'].fillna(0) for i in range(1, 6))
        tot_ask = sum(chunk[f'L{i}-AskSize'].fillna(0) for i in range(1, 6))

        chunk['lob_obi']        = (tot_bid - tot_ask) / (tot_bid + tot_ask).replace(0, np.nan)
        chunk['mid_price']      = (chunk['L1-AskPrice'] + chunk['L1-BidPrice']) / 2
        chunk['lob_spread']     = chunk['L1-AskPrice'] - chunk['L1-BidPrice']
        chunk['lob_spread_pct'] = chunk['lob_spread'] / chunk['mid_price'].replace(0, np.nan)

        for i in range(1, 6):
            b = chunk[f'L{i}-BidSize'].fillna(0)
            a = chunk[f'L{i}-AskSize'].fillna(0)
            chunk[f'lob_obi_l{i}'] = (b - a) / (b + a).replace(0, np.nan)

        chunk['lob_bid_queue'] = chunk['L1-BidSize'].fillna(0) / tot_bid.replace(0, np.nan)
        chunk['lob_ask_queue'] = chunk['L1-AskSize'].fillna(0) / tot_ask.replace(0, np.nan)

        chunk = chunk.dropna(subset=['datetime'])
        chunk.set_index('datetime', inplace=True)
        res = chunk.groupby(['symbol', pd.Grouper(freq='5min')]).last().reset_index()
        resampled_chunks.append(res)

    print("  Concatenating and final-resampling...")
    df_lob = pd.concat(resampled_chunks)
    df_lob.set_index('datetime', inplace=True)
    df_lob = df_lob.groupby(['symbol', pd.Grouper(freq='5min')]).last().reset_index()

    df_lob.to_parquet(out)
    print(f"  ✓ Saved LOB features → {out}  ({len(df_lob):,} rows)\n")
    return df_lob


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Manual feature engineering on OHLCV
# ─────────────────────────────────────────────────────────────────────────────
def step4_feature_engineering(df):
    import pandas as pd
    import numpy as np
    import warnings
    warnings.filterwarnings('ignore')

    print("=" * 60)
    print("STEP 4: Feature engineering...")
    print("=" * 60)

    df = df.sort_values(['symbol', 'datetime']).reset_index(drop=True)

    # ── helpers ──────────────────────────────────────────────
    def rsi(s, p):
        d = s.diff(); g = d.clip(lower=0); l = -d.clip(upper=0)
        return 100 - 100 / (1 + g.ewm(com=p-1, adjust=False).mean()
                                 / l.ewm(com=p-1, adjust=False).mean())

    def _atr(hi, lo, cl, p=14):
        tr = pd.concat([hi-lo,
                        (hi - cl.shift()).abs(),
                        (lo - cl.shift()).abs()], axis=1).max(axis=1)
        return tr.ewm(alpha=1/p, adjust=False).mean()

    def _adx(hi, lo, cl, p=14):
        pdm = hi.diff().clip(lower=0)
        ndm = (-lo.diff()).clip(lower=0)
        tr_val = _atr(hi, lo, cl, p)
        pdi = 100 * pdm.ewm(alpha=1/p, adjust=False).mean() / tr_val
        ndi = 100 * ndm.ewm(alpha=1/p, adjust=False).mean() / tr_val
        dx  = 100 * (pdi - ndi).abs() / (pdi + ndi + 1e-8)
        return dx.ewm(alpha=1/p, adjust=False).mean()

    g = df.groupby('symbol')

    # RSI
    print("  Computing RSI...")
    df['rsi_14'] = g['close'].transform(lambda x: rsi(x, 14))
    df['rsi_7']  = g['close'].transform(lambda x: rsi(x, 7))

    # MACD
    print("  Computing MACD...")
    ema_f = g['close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    ema_s = g['close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
    df['macd']        = ema_f - ema_s
    df['macd_signal'] = g['macd'].transform(lambda x: x.ewm(span=9, adjust=False).mean())

    # Stochastic
    print("  Computing Stochastic...")
    h14 = g['high'].transform(lambda x: x.rolling(14).max())
    l14 = g['low'].transform(lambda x: x.rolling(14).min())
    df['stoch_k'] = 100 * (df['close'] - l14) / (h14 - l14 + 1e-8)
    df['stoch_d'] = g['stoch_k'].transform(lambda x: x.rolling(3).mean())

    # EMA + crossovers
    print("  Computing EMAs...")
    df['ema_9']  = g['close'].transform(lambda x: x.ewm(span=9,  adjust=False).mean())
    df['ema_21'] = g['close'].transform(lambda x: x.ewm(span=21, adjust=False).mean())
    df['ema_50'] = g['close'].transform(lambda x: x.ewm(span=50, adjust=False).mean())
    df['ema_9_21_cross']  = df['ema_9'] - df['ema_21']
    df['ema_21_50_cross'] = df['ema_21'] - df['ema_50']

    # ATR & ADX
    print("  Computing ATR / ADX...")
    df['atr_14'] = (df.groupby('symbol', group_keys=False)
                      .apply(lambda x: _atr(x['high'], x['low'], x['close']))
                      .reset_index(level=0, drop=True))
    df['adx_14'] = (df.groupby('symbol', group_keys=False)
                      .apply(lambda x: _adx(x['high'], x['low'], x['close']))
                      .reset_index(level=0, drop=True))

    # Bollinger Bands
    print("  Computing Bollinger Bands...")
    sma20 = g['close'].transform(lambda x: x.rolling(20).mean())
    std20 = g['close'].transform(lambda x: x.rolling(20).std())
    ub = sma20 + 2*std20; lb = sma20 - 2*std20
    df['bb_width']    = (ub - lb) / sma20
    df['bb_position'] = (df['close'] - lb) / (ub - lb + 1e-8)

    # VWAP deviation (intraday, resets per day per symbol)
    print("  Computing VWAP...")
    df['_date'] = df['datetime'].dt.date
    df['_tp']   = (df['high'] + df['low'] + df['close']) / 3
    df['vwap']  = (df.groupby(['symbol', '_date'])
                     .apply(lambda x: (x['_tp']*x['volume']).cumsum()
                                       / x['volume'].cumsum())
                     .reset_index(level=[0,1], drop=True))
    df['vwap_dev'] = (df['close'] - df['vwap']) / df['vwap']

    # Volume ratio
    df['volume_ratio'] = (df['volume']
                          / (g['volume'].transform(lambda x: x.rolling(20).mean()) + 1e-8))

    # OBV
    print("  Computing OBV...")
    df['obv'] = (df.groupby('symbol', group_keys=False)
                   .apply(lambda x: (x['volume'] * np.sign(x['close'].diff())).cumsum())
                   .reset_index(level=0, drop=True))

    # CCI, ROC, MOM
    print("  Computing CCI / ROC / MOM...")
    sma_tp  = g['_tp'].transform(lambda x: x.rolling(20).mean())
    mad_tp  = g['_tp'].transform(lambda x: x.rolling(20).apply(
                                    lambda s: np.abs(s - s.mean()).mean()))
    df['cci']    = (df['_tp'] - sma_tp) / (0.015 * mad_tp + 1e-8)
    df['roc_10'] = g['close'].transform(lambda x: x.pct_change(10))
    df['mom_10'] = g['close'].transform(lambda x: x - x.shift(10))

    # ADX already computed above

    # Time features
    print("  Computing time features...")
    t = df['datetime'].dt.hour * 60 + df['datetime'].dt.minute
    df['time_sin']    = np.sin(2 * np.pi * t / (24*60))
    df['time_cos']    = np.cos(2 * np.pi * t / (24*60))
    df['dayofweek']   = df['datetime'].dt.dayofweek
    df['is_first_30'] = ((df['datetime'].dt.hour == 9) &
                         (df['datetime'].dt.minute <= 45)).astype(int)
    df['is_last_30']  = (df['datetime'].dt.hour == 15).astype(int)

    # Gap %
    df['_prev_close'] = g['close'].shift(1)
    df['gap_pct'] = np.where(
        df['_date'] != df.groupby('symbol')['_date'].shift(1),
        (df['open'] - df['_prev_close']) / (df['_prev_close'] + 1e-8),
        0.0
    )

    # Return lags: 1, 3, 6, 12 bars
    print("  Computing return lags...")
    for lag in [1, 3, 6, 12]:
        df[f'return_lag_{lag}'] = g['close'].transform(lambda x: x.pct_change(lag))

    # Drop temporary columns
    df = df.drop(columns=['_date', '_tp', '_prev_close'], errors='ignore')

    print(f"  ✓ Feature engineering done. Shape: {df.shape}\n")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Merge LOB features (if lob_5m.parquet is ready)
# ─────────────────────────────────────────────────────────────────────────────
def step5_merge_lob(
    df,
    lob_path='/Users/Pothuri/Downloads/kubera/lob_5m.parquet'
):
    import pandas as pd
    print("=" * 60)
    print("STEP 5: Merging LOB features...")
    print("=" * 60)

    if not os.path.exists(lob_path):
        print("  ⚠  lob_5m.parquet not found — skipping LOB merge.\n")
        return df

    df_lob = pd.read_parquet(lob_path)
    df_lob['datetime'] = pd.to_datetime(df_lob['datetime'])
    df['datetime']     = pd.to_datetime(df['datetime'])
    df_lob['symbol']   = df_lob['symbol'].astype(str)
    df['symbol']       = df['symbol'].astype(str)

    df_lob = df_lob.sort_values(['symbol', 'datetime'])
    df     = df.sort_values(['symbol', 'datetime'])

    df = pd.merge(df, df_lob, on=['symbol', 'datetime'], how='left')

    lob_cols = [c for c in df_lob.columns if c not in ['symbol', 'datetime']]
    df[lob_cols] = df.groupby('symbol')[lob_cols].ffill()

    print(f"  ✓ LOB merged. Shape: {df.shape}\n")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Triple Barrier Labeling
# ─────────────────────────────────────────────────────────────────────────────
def step6_label(
    df,
    tp_mult=1.5, sl_mult=1.0, max_bars=6,
    out='/Users/Pothuri/Downloads/kubera/nifty10_final.parquet'
):
    import pandas as pd
    import numpy as np
    print("=" * 60)
    print("STEP 6: Triple barrier labeling...")
    print("=" * 60)

    df = df.sort_values(['symbol', 'datetime']).reset_index(drop=True)
    df['atr_14'] = df['atr_14'].fillna(method='bfill')

    tp_long  = df['close'] + tp_mult * df['atr_14']
    sl_long  = df['close'] - sl_mult * df['atr_14']
    tp_short = df['close'] - tp_mult * df['atr_14']
    sl_short = df['close'] + sl_mult * df['atr_14']

    hit_tp_l = pd.Series(999, index=df.index)
    hit_sl_l = pd.Series(999, index=df.index)
    hit_tp_s = pd.Series(999, index=df.index)
    hit_sl_s = pd.Series(999, index=df.index)

    for i in range(1, max_bars + 1):
        hf = df['high'].shift(-i)
        lf = df['low'].shift(-i)
        hit_tp_l[(hf >= tp_long)  & (hit_tp_l == 999)] = i
        hit_sl_l[(lf <= sl_long)  & (hit_sl_l == 999)] = i
        hit_tp_s[(lf <= tp_short) & (hit_tp_s == 999)] = i
        hit_sl_s[(hf >= sl_short) & (hit_sl_s == 999)] = i

    long_win  = (hit_tp_l <= max_bars) & (hit_tp_l < hit_sl_l)
    short_win = (hit_tp_s <= max_bars) & (hit_tp_s < hit_sl_s)

    df['label'] = 1                          # default: Flat
    df.loc[short_win, 'label'] = 0           # Short
    df.loc[long_win,  'label'] = 2           # Long

    # Nullify last max_bars rows per symbol (no valid lookahead)
    df.loc[df.groupby('symbol').tail(max_bars).index, 'label'] = 1

    df = df.dropna()
    df.to_parquet(out)

    print(f"  ✓ Saved labeled dataset → {out}  ({len(df):,} rows)")
    print("  Label distribution:")
    print(df['label'].value_counts(normalize=True).round(3).to_string())
    print()
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — LightGBM + CatBoost Ensemble
# ─────────────────────────────────────────────────────────────────────────────
def step7_train(df, model_dir='/Users/Pothuri/Downloads/kubera/models'):
    import pandas as pd
    import numpy as np
    from sklearn.metrics import accuracy_score, classification_report
    import lightgbm as lgb
    import catboost as cb

    print("=" * 60)
    print("STEP 7: Training LightGBM + CatBoost ensemble...")
    print("=" * 60)

    EXCLUDE = {'symbol', 'datetime', 'label', 'vwap', 'obv'}
    features = [c for c in df.columns
                if c not in EXCLUDE and df[c].dtype != object]

    df_m = df.dropna(subset=features + ['label']).sort_values('datetime').reset_index(drop=True)
    split_idx = int(len(df_m) * 0.8)
    train_df  = df_m.iloc[:split_idx].copy()
    test_df   = df_m.iloc[split_idx:].copy()
    print(f"  Train: {len(train_df):,}  |  Test: {len(test_df):,}")

    # Upsample Long class to match Short
    short_n = (train_df['label'] == 0).sum()
    long_n  = (train_df['label'] == 2).sum()
    if 0 < long_n < short_n:
        extras = (train_df[train_df['label'] == 2]
                  .sample(n=short_n - long_n, replace=True, random_state=42))
        train_df = (pd.concat([train_df, extras])
                      .sort_values('datetime').reset_index(drop=True))
        print(f"  Upsampled Long class by {short_n - long_n} rows.")

    X_train, y_train = train_df[features], train_df['label']
    X_test,  y_test  = test_df[features],  test_df['label']

    # LightGBM
    print("  Training LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=63,
        is_unbalance=True, random_state=42, n_jobs=-1)
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(30, verbose=False),
                   lgb.log_evaluation(period=-1)])

    # CatBoost
    print("  Training CatBoost...")
    cb_model = cb.CatBoostClassifier(
        iterations=300, learning_rate=0.05, depth=6,
        verbose=0, random_state=42, thread_count=-1)
    cb_model.fit(X_train, y_train,
                 eval_set=(X_test, y_test),
                 early_stopping_rounds=30)

    # Ensemble prediction
    preds = np.argmax(
        (lgb_model.predict_proba(X_test) + cb_model.predict_proba(X_test)) / 2,
        axis=1)

    print(f"\n  Accuracy: {accuracy_score(y_test, preds):.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, preds,
          target_names=['Short(0)', 'Flat(1)', 'Long(2)']))

    # Feature importance
    imp_df = (pd.DataFrame({'feature': features,
                             'importance': lgb_model.feature_importances_})
                .sort_values('importance', ascending=False))
    print("  Top 15 Feature Importances (LightGBM):")
    print(imp_df.head(15).to_string(index=False))

    # Save
    os.makedirs(model_dir, exist_ok=True)
    lgb_model.booster_.save_model(os.path.join(model_dir, 'lgb_model.txt'))
    cb_model.save_model(os.path.join(model_dir, 'cb_model.cbm'))
    print(f"\n  ✓ Models saved to {model_dir}/\n")

    return lgb_model, cb_model, imp_df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — Run the full pipeline
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # step1_install()           # Uncomment to install packages first

    df = step2_load_ohlcv()
    df = step4_feature_engineering(df)
    df = step5_merge_lob(df)   # auto-skips if lob_5m.parquet not ready
    df = step6_label(df)
    step7_train(df)

    # To also run Step 3 (LOB processing):
    # step3_process_lob()
