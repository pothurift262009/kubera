import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def add_technical_indicators_elite_v2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elite Production indicators: RSI, MACD, ATR-normalized Returns.
    """
    logger.info("Computing elite technical indicators v2...")
    df = df.sort_values(['symbol', 'datetime']).reset_index(drop=True)
    g = df.groupby('symbol')

    # RSI (14, 21)
    def rsi(s, p):
        d = s.diff(); g = d.clip(lower=0); l = -d.clip(upper=0)
        return 100 - 100 / (1 + g.ewm(com=p-1, adjust=False).mean() / (l.ewm(com=p-1, adjust=False).mean() + 1e-10))

    for p in [14, 21]:
        df[f'rsi_{p}'] = g['close'].transform(lambda x: rsi(x, p))
    
    # MACD Ratio (better normalization)
    ema12 = g['close'].transform(lambda x: x.ewm(span=12).mean())
    ema26 = g['close'].transform(lambda x: x.ewm(span=26).mean())
    df['macd_ratio'] = (ema12 - ema26) / (ema26 + 1e-10)
    
    # ATR & ADX (Normalized)
    pc = g['close'].shift(1)
    tr = pd.concat([df['high']-df['low'], (df['high']-pc).abs(), (df['low']-pc).abs()], axis=1).max(axis=1)
    df['tr'] = tr
    df['atr_14'] = g['tr'].transform(lambda x: x.ewm(alpha=1/14, adjust=False).mean())
    df['atr_pct'] = df['atr_14'] / (df['close'] + 1e-10)
    
    # ADX logic (Proprietary optimization)
    up = df['high'].diff(); dw = -df['low'].diff()
    up.loc[df['symbol'] != df['symbol'].shift(1)] = 0; dw.loc[df['symbol'] != df['symbol'].shift(1)] = 0
    df['pdm'] = np.where((up > dw) & (up > 0), up, 0); df['mdm'] = np.where((dw > up) & (dw > 0), dw, 0)
    
    tr14 = g['tr'].transform(lambda x: x.rolling(14).sum() + 1e-10)
    pdi = 100 * g['pdm'].transform(lambda x: x.rolling(14).sum()) / tr14
    mdi = 100 * g['mdm'].transform(lambda x: x.rolling(14).sum()) / tr14
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-10)
    df['adx_14'] = g.apply(lambda x: dx.loc[x.index].ewm(span=14).mean(), include_groups=False).squeeze().reset_index(level=0, drop=True)
    
    # Bollinger Bands Position (Z-score approach)
    sma20 = g['close'].transform(lambda x: x.rolling(20).mean())
    std20 = g['close'].transform(lambda x: x.rolling(20).std())
    df['bb_zscore'] = (df['close'] - sma20) / (std20 + 1e-10)
    
    return df

def add_regime_features_v2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Volatility & Trend Regimes: Hurst, ATR-Norm, Vol-Ratio.
    """
    logger.info("Computing regime detection features v2...")
    df['trend_strength'] = df['adx_14'].clip(0, 100)
    
    g = df.groupby('symbol')
    r_max = g['high'].transform(lambda x: x.rolling(30).max())
    r_min = g['low'].transform(lambda x: x.rolling(30).min())
    s_std = g['close'].transform(lambda x: x.rolling(30).std())
    df['hurst_proxy'] = (r_max - r_min) / (s_std + 1e-10)
    
    # Volatility Regime
    df['vol_6'] = g['close'].transform(lambda x: x.pct_change().rolling(6).std())
    df['vol_24'] = g['close'].transform(lambda x: x.pct_change().rolling(24).std())
    df['vol_ratio'] = df['vol_6'] / (df['vol_24'] + 1e-10)
    
    # Interaction: High Vol * RSI Deviation
    df['rsi_vol_interaction'] = (df['rsi_14'] - 50) * df['vol_ratio']
    
    return df

def add_advanced_momentum_v2(df: pd.DataFrame) -> pd.DataFrame:
    """Multi-horizon momentum and acceleration."""
    logger.info("Computing short-term alpha momentum...")
    g = df.groupby('symbol')
    
    for lag in [1, 3, 6]:
        df[f'ret_{lag}b'] = g['close'].transform(lambda x: x.pct_change(lag))
    
    # Momentum Acceleration
    df['mom_accel'] = df['ret_1b'] - g['ret_1b'].shift(1)
    
    # VWAP Deviation
    df['pv'] = df['close'] * df['volume']
    df['cum_pv'] = g['pv'].transform(lambda x: x.cumsum())
    df['cum_v']  = g['volume'].transform(lambda x: x.cumsum())
    df['vwap'] = df['cum_pv'] / (df['cum_v'] + 1e-10)
    df['vwap_dev'] = (df['close'] - df['vwap']) / (df['vwap'] + 1e-10)
    
    return df

def run_feature_pipeline_elite_v2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Orchestrate full elite feature pipeline with strict leakage prevention.
    """
    df = df[['symbol', 'datetime', 'open', 'high', 'low', 'close', 'volume']].copy()
    
    # 1. Base Indicators
    df = add_technical_indicators_elite_v2(df)
    df = add_regime_features_v2(df)
    df = add_advanced_momentum_v2(df)
    
    # 2. Strict Lagging: Apply group-wise shift(1) to ALL features
    # NOTE: Labels will be computed on raw, non-shifted prices later.
    logger.info("Elite leakage enforcement: Shifting all predictive features...")
    predict_cols = [c for c in df.columns if c not in {'symbol', 'datetime'}]
    df[predict_cols] = df.groupby('symbol')[predict_cols].shift(1)
    
    # 3. Cyclical Time Features
    t = df['datetime'].dt.hour * 60 + df['datetime'].dt.minute
    df['time_sin'] = np.sin(2 * np.pi * t / (24 * 60))
    df['time_cos'] = np.cos(2 * np.pi * t / (24 * 60))
    
    return df.dropna(subset=['rsi_14', 'adx_14', 'vwap_dev', 'bb_zscore']).drop(columns=['tr', 'pdm', 'mdm', 'pv', 'cum_pv', 'cum_v'], errors='ignore')
