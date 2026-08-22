import pandas as pd
import pandas_ta as ta
import numpy as np
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    def __init__(self, data):
        self.df = data

    def generate_features(self):
        print("Generating technical indicators...")
        processed_groups = []
        for symbol, group in self.df.groupby('symbol'):
            group = self._calc_symbol_features(group)
            processed_groups.append(group)

        self.df = pd.concat(processed_groups).reset_index(drop=True)
        self.df = self.df.dropna()
        return self.df

    def _calc_symbol_features(self, df):
        df = df.copy()

        # 1. Momentum
        df['rsi'] = ta.rsi(df['close'], length=14)
        df['roc_5'] = ta.roc(df['close'], length=5)
        df['roc_20'] = ta.roc(df['close'], length=20)

        # 2. EMA Gaps (20, 50, 100)
        ema_20 = ta.ema(df['close'], length=20)
        ema_50 = ta.ema(df['close'], length=50)
        ema_100 = ta.ema(df['close'], length=100)
        df['ema_gap_20'] = (df['close'] - ema_20) / ema_20
        df['ema_gap_50'] = (df['close'] - ema_50) / ema_50
        df['ema_gap_100'] = (df['close'] - ema_100) / ema_100

        # 3. Bollinger Bands
        bbands = ta.bbands(df['close'], length=20, std=2)
        bbu_col = [c for c in bbands.columns if c.startswith('BBU_')][0]
        bbl_col = [c for c in bbands.columns if c.startswith('BBL_')][0]
        bbm_col = [c for c in bbands.columns if c.startswith('BBM_')][0]
        df['bb_width'] = (bbands[bbu_col] - bbands[bbl_col]) / bbands[bbm_col]
        df['bb_position'] = (df['close'] - bbands[bbl_col]) / (bbands[bbu_col] - bbands[bbl_col])

        # 4. ATR (normalized by close)
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14) / df['close']

        # 5. Volume
        df['vol_sma_14'] = df['volume'] / ta.sma(df['volume'], length=14)
        df['mfi'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=14)
        df['vol_acceleration'] = df['volume'] / df['volume'].rolling(3).mean()

        # 6. Multi-timeframe returns (5min bars)
        df['return_1bar'] = df['close'].pct_change(1)
        df['return_3bar'] = df['close'].pct_change(3)
        df['return_6bar'] = df['close'].pct_change(6)
        df['return_12bar'] = df['close'].pct_change(12)
        df['return_24bar'] = df['close'].pct_change(24)
        df['volatility_6bar'] = df['return_1bar'].rolling(6).std()

        # 7. Momentum alignment across timeframes
        df['momentum_aligned'] = (
            (df['return_3bar'] > 0).astype(int) +
            (df['return_6bar'] > 0).astype(int) +
            (df['return_12bar'] > 0).astype(int)
        )

        # 8. Trend strength
        adx = ta.adx(df['high'], df['low'], df['close'], length=14)
        df['adx'] = adx['ADX_14']
        df['dmp'] = adx['DMP_14']
        df['dmn'] = adx['DMN_14']
        df['di_diff'] = df['dmp'] - df['dmn']
        df['regime'] = (df['adx'] > 20).astype(int)

        # 9. MACD
        macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
        df['macd_hist'] = macd['MACDh_12_26_9']
        df['macd_slope'] = df['macd_hist'] - df['macd_hist'].shift(1)

        # 10. Stochastic RSI
        stochrsi = ta.stochrsi(df['close'], length=14)
        df['stochrsi_k'] = stochrsi['STOCHRSIk_14_14_3_3']
        df['stochrsi_d'] = stochrsi['STOCHRSId_14_14_3_3']

        # 11. VWAP (daily eset per symbol)
        vol_sum = df['volume'].groupby(df['date'].dt.date).cumsum()
        typical_price = (df['high'] + df['low'] + df['close']) / 3.0
        df['vwap'] = (typical_price * df['volume']).groupby(
            df['date'].dt.date).cumsum() / vol_sum
        df['vwap_dev'] = (df['close'] - df['vwap']) / df['vwap'] * 100

        # 12. Opening range (first 3 bars = 15 min) — SAFE version
        df['day'] = df['date'].dt.date

        def _opening_range_high(x, n=3):
            first_n = x.values[:min(n, len(x))]
            or_val = first_n.max() if len(first_n) > 0 else np.nan
            if len(x) <= n:
                return pd.Series(first_n, index=x.index)
            result = list(first_n) + [or_val] * (len(x) - n)
            return pd.Series(result, index=x.index)

        def _opening_range_low(x, n=3):
            first_n = x.values[:min(n, len(x))]
            or_val = first_n.min() if len(first_n) > 0 else np.nan
            if len(x) <= n:
                return pd.Series(first_n, index=x.index)
            result = list(first_n) + [or_val] * (len(x) - n)
            return pd.Series(result, index=x.index)

        df['or_high'] = df.groupby('day')['high'].transform(_opening_range_high)
        df['or_low'] = df.groupby('day')['low'].transform(_opening_range_low)

        or_range = (df['or_high'] - df['or_low']).replace(0, np.nan)
        df['or_position'] = (df['close'] - df['or_low']) / or_range
        df['or_breakout'] = np.where(
            df['close'] > df['or_high'], 1,
            np.where(df['close'] < df['or_low'], -1, 0)
        )

        # 13. Time
        df['hour'] = df['date'].dt.hour
        df['minute'] = df['date'].dt.minute
        df['minutes_from_open'] = (
            (df['hour'] - 9) * 60 + (df['minute'] - 15)
        ).clip(lower=0)

        return df

    def rank_features(self):
        print("Ranking features cross-sectionally...")
        self.df['market_return'] = self.df.groupby('date')['return_1bar'].transform('mean')
        self.df['relative_return'] = self.df['return_1bar'] - self.df['market_return']


        rank_cols = [
            'rsi', 'roc_5', 'roc_20',
            'ema_gap_20', 'ema_gap_50', 'ema_gap_100',
            'bb_width', 'bb_position', 'atr',
            'vol_sma_14', 'mfi', 'vol_acceleration',
            'return_3bar', 'return_6bar', 'return_12bar', 'return_24bar',
            'adx', 'di_diff', 'macd_hist', 'macd_slope',
            'stochrsi_k', 'stochrsi_d', 'vwap_dev',
            'or_position', 'momentum_aligned'
        ]
        for col in rank_cols:
            self.df[f'rank_{col}'] = self.df.groupby('date')[col].rank(pct=True)

        return self.df

    @staticmethod
    def get_feature_list():
        return [
            # Momentum
            'rsi', 'roc_5', 'roc_20',
            # EMA gaps
            'ema_gap_20', 'ema_gap_50', 'ema_gap_100',
            # Bollinger Bands
            'bb_width', 'bb_position',
            # Volatility
            'atr', 'volatility_6bar',
            # Volume
            'vol_sma_14', 'mfi', 'vol_acceleration',
            # Multi-timeframe returns
            'return_1bar', 'return_3bar', 'return_6bar',
            'return_12bar', 'return_24bar',
            # Momentum alignment
            'momentum_aligned',
            # Trend
            'adx', 'dmp', 'dmn', 'di_diff', 'regime',
            # MACD
            'macd_hist', 'macd_slope',
            # Stochastic RSI
            'stochrsi_k', 'stochrsi_d',
            # VWAP
            'vwap_dev',
            # Opening range
            'or_position', 'or_breakout',
            # Time
            'minutes_from_open',
            # Cross-sectional
            'relative_return',

            'rank_rsi', 'rank_roc_5', 'rank_roc_20',
            'rank_ema_gap_20', 'rank_ema_gap_50', 'rank_ema_gap_100',
            'rank_bb_width', 'rank_bb_position', 'rank_atr',
            'rank_vol_sma_14', 'rank_mfi', 'rank_vol_acceleration',
            'rank_return_3bar', 'rank_return_6bar',
            'rank_return_12bar', 'rank_return_24bar',
            'rank_adx', 'rank_di_diff',
            'rank_macd_hist', 'rank_macd_slope',
            'rank_stochrsi_k', 'rank_stochrsi_d',
            'rank_vwap_dev', 'rank_or_position',
            'rank_momentum_aligned'
        ]
