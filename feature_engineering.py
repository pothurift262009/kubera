import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def add_technical_indicators_elite(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Computing elite technical indicators...")
    df = df.sort_values(['symbol', 'datetime']).reset_index(drop=True)
    g = df.groupby('symbol')

    # RSI (7, 14, 21)
    def rsi(s, p):
        d = s.diff(); g = d.clip(lower=0); l = -d.clip(upper=0)
        return 100 - 100 / (1 + g.ewm(com=p-1, adjust=False).mean() / (l.ewm(com=p-1, adjust=False).mean() + 1e-10))

    for p in [7, 14, 21]:
        df[f'rsi_{p}'] = g['close'].transform(lambda x: rsi(x, p))
    
    # MACD
    ema12 = g['close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    ema26 = g['close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
    df['macd'] = ema12 - ema26
    df['macd_signal'] = g['macd'].transform(lambda x: x.ewm(span=9, adjust=False).mean())
    
    # ATR & ADX
    pc = g['close'].shift(1)
    tr = pd.concat([df['high']-df['low'], (df['high']-pc).abs(), (df['low']-pc).abs()], axis=1).max(axis=1)
    df['atr_14'] = g.apply(lambda x: tr.loc[x.index].ewm(alpha=1/14, adjust=False).mean(), include_groups=False).squeeze().reset_index(level=0, drop=True)
    
    # ADX logic
    up = df['high'].diff(); dw = -df['low'].diff()
    up.loc[df['symbol'] != df['symbol'].shift(1)] = 0; dw.loc[df['symbol'] != df['symbol'].shift(1)] = 0
    pdm = np.where((up > dw) & (up > 0), up, 0); mdm = np.where((dw > up) & (dw > 0), dw, 0)
    
    pdi = 100 * g.apply(lambda x: pd.Series(pdm[x.index]).ewm(alpha=1/14, adjust=False).mean(), include_groups=False).squeeze().reset_index(level=0, drop=True) / (df['atr_14'] + 1e-10)
    mdi = 100 * g.apply(lambda x: pd.Series(mdm[x.index]).ewm(alpha=1/14, adjust=False).mean(), include_groups=False).squeeze().reset_index(level=0, drop=True) / (df['atr_14'] + 1e-10)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-10)
    df['adx_14'] = g.apply(lambda x: dx.loc[x.index].ewm(alpha=1/14, adjust=False).mean(), include_groups=False).squeeze().reset_index(level=0, drop=True)
    
    # Bollinger Bands
    sma20 = g['close'].transform(lambda x: x.rolling(20).mean())
    std20 = g['close'].transform(lambda x: x.rolling(20).std())
    df['bb_width'] = (4 * std20) / (sma20 + 1e-10)
    df['bb_pos'] = (df['close'] - (sma20 - 2 * std20)) / (4 * std20 + 1e-10)
    
    return df

def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Computing regime detection features...")
    # Trend vs Mean-Reversion based on ADX & Volatility
    df['is_trending'] = (df['adx_14'] > 25).astype(int)
    df['is_ranging'] = (df['adx_14'] < 20).astype(int)
    
    # Hurst Exponent Proxy (Rolling high-low range / rolling std)
    g = df.groupby('symbol')
    r_max = g['high'].transform(lambda x: x.rolling(30).max())
    r_min = g['low'].transform(lambda x: x.rolling(30).min())
    s_std = g['close'].transform(lambda x: x.rolling(30).std())
    df['hurst_proxy'] = (r_max - r_min) / (s_std + 1e-10)
    
    return df

def add_advanced_momentum_elite(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Computing multi-horizon momentum and interactions...")
    g = df.groupby('symbol')
    
    # Returns & Momentum
    for lag in [1, 3, 6, 12, 24]:
        df[f'return_{lag}b'] = g['close'].transform(lambda x: x.pct_change(lag))
        df[f'mom_{lag}b'] = g['close'].transform(lambda x: x - x.shift(lag))

    # Volatility regimes
    df['vol_regime'] = g['return_1b'].transform(lambda x: x.rolling(20).std())
    
    # Interactions
    vol_sma20 = g['volume'].transform(lambda x: x.rolling(20).mean())
    df['vol_ratio'] = df['volume'] / (vol_sma20 + 1e-10)
    df['rsi_vol_interaction'] = df['rsi_14'] * df['vol_ratio']
    
    return df

def run_feature_pipeline_elite(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Base columns
    df = df[['symbol', 'datetime', 'open', 'high', 'low', 'close', 'volume']].copy()
    
    # 2. Indicators
    df = add_technical_indicators_elite(df)
    df = add_regime_features(df)
    df = add_advanced_momentum_elite(df)
    
    # 3. LEAKAGE PREVENTION: Shift all features group-wise
    logger.info("Elite leakage check: Applying group-wise shift(1)...")
    exclude = {'symbol', 'datetime', 'label'}
    feature_cols = [c for c in df.columns if c not in exclude]
    
    df[feature_cols] = df.groupby('symbol')[feature_cols].shift(1)
    
    # Time features (non-leaky, but we use the bar's open time)
    t = df['datetime'].dt.hour * 60 + df['datetime'].dt.minute
    df['time_sin'] = np.sin(2 * np.pi * t / (24 * 60))
    df['time_cos'] = np.cos(2 * np.pi * t / (24 * 60))
    
    return df.dropna(subset=['rsi_14', 'adx_14', 'bb_width'])
