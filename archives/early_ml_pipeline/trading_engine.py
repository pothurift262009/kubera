import pandas as pd  # type: ignore
import pandas_ta as ta  # type: ignore
import numpy as np  # type: ignore

def prepare_features(df):
    """
    Core function to generate features from OHLCV data.
    Input: DataFrame with [date, symbol, open, high, low, close, volume]
    Output: DataFrame with advanced features, ready for model prediction.
    """
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"])

    # Basic Returns
    df["ret_1"] = df.groupby("symbol")["close"].pct_change()

    # Market-wide return (Beta)
    market_avg = df.groupby("date")["ret_1"].mean().rename("mkt_ret_1")
    df = df.merge(market_avg, on="date", how="left")
    df["rel_ret_1"] = df["ret_1"] - df["mkt_ret_1"]

    def apply_ta(group):
        group = group.copy()
        # Technical Indicators - RSI, MACD, BBands
        group.ta.rsi(length=14, append=True)
        group.ta.macd(fast=12, slow=26, signal=9, append=True)
        group.ta.bbands(length=20, std=2, append=True)
        group.ta.atr(length=14, append=True)

        # Normalized ATR (Volatility)
        group["NATR"] = group["ATR_14"] / group["close"] * 100

        # Volume Spike
        group["vol_ma_20"] = group["volume"].rolling(20).mean()
        group["vol_spike"] = group["volume"] / group["vol_ma_20"]

        # Multi-period Returns
        for lag in [3, 5, 10]:
            group[f"ret_{lag}"] = group["close"].pct_change(lag)

        # Distance from EMA 20
        group["ema_20"] = ta.ema(group["close"], length=20)
        group["dist_ema_20"] = (group["close"] / group["ema_20"]) - 1

        # Intraday progress
        minutes = group["date"].dt.hour * 60 + group["date"].dt.minute
        group["day_progress"] = (minutes - 555) / (930 - 555)

    df = df.groupby("symbol", group_keys=False).apply(apply_ta)

    # Feature list excluding raw data and targets
    exclude = ["date", "symbol", "open", "high", "low", "close", "volume", "future_ret_3", "target"]
    features = [c for c in df.columns if c not in exclude]

    # Shift all valid features forward by 1 candle to prevent look-ahead leakage
    df[features] = df.groupby("symbol")[features].shift(1)
    df = df.dropna(subset=features)

    return df, features

def get_target(df, lookahead=3):
    """
    Calculates the target variable for training.
    """
    COST = 0.0006
    MARGIN = 0.0005

    df = df.copy()
    df["future_ret_3"] = df.groupby("symbol")["close"].shift(-lookahead) / df["close"] - 1
    df["target"] = (df["future_ret_3"] > (COST + MARGIN)).astype(int)
    return df
