import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class Labeler:
    def __init__(self, data):
        self.df = data

    def triple_barrier_labeling(self, pt=0.6, sl=0.3, horizon=12):
        """
        Implements Triple Barrier Method.
        pt: Profit Take (%)
        sl: Stop Loss (%)
        horizon: Max bars to hold the trade
        """
        print(f"Applying Triple Barrier Labeling (PT: {pt}%, SL: {sl}%, Horizon: {horizon} bars)...")

        self.df['label'] = 0
        processed_groups = []
        for symbol, group in self.df.groupby('symbol'):
            group = group.copy()
            close = group['close'].values
            n = len(close)
            if n <= horizon:
                group['label'] = 0
                processed_groups.append(group)
                continue

            labels = np.zeros(n)

            # Vectorized sliding window
            windows = np.lib.stride_tricks.sliding_window_view(close, horizon + 1)
            entry_prices = windows[:, 0:1]
            future_path = windows[:, 1:]

            pct_returns = (future_path - entry_prices) / entry_prices * 100.0

            tp_hits = pct_returns >= pt
            sl_hits = pct_returns <= -sl

            has_tp = np.any(tp_hits, axis=1)
            has_sl = np.any(sl_hits, axis=1)

            first_tp = np.where(has_tp, np.argmax(tp_hits, axis=1), horizon + 1)
            first_sl = np.where(has_sl, np.argmax(sl_hits, axis=1), horizon + 1)

            labels[:len(windows)] = np.where(
                first_tp < first_sl, 1,
                np.where(first_sl < first_tp, -1, 0)
            )

            group['label'] = labels
            processed_groups.append(group)

        self.df = pd.concat(processed_groups).reset_index(drop=True)

        dist = self.df['label'].value_counts(normalize=True)
        print(f"Label distribution:\n{dist}")
        return self.df

    def atr_barrier_labeling(self, atr_multiplier_pt=1.5, atr_multiplier_sl=1.0, horizon=30):
        """
        PT = atr_multiplier_pt × ATR, SL = atr_multiplier_sl × ATR.
        Expects 'atr' column (ATR / Close).
        """
        print(f"Applying ATR-based Triple Barrier (PT: {atr_multiplier_pt}x ATR, "
              f"SL: {atr_multiplier_sl}x ATR, Horizon: {horizon} bars)...")

        self.df['label'] = 0
        processed_groups = []
        for symbol, group in self.df.groupby('symbol'):
            group = group.copy()
            close = group['close'].values
            atr_vals = group['atr'].values
            n = len(close)
            if n <= horizon:
                group['label'] = 0
                processed_groups.append(group)
                continue

            labels = np.zeros(n)

            windows = np.lib.stride_tricks.sliding_window_view(close, horizon + 1)
            entry_prices = windows[:, 0:1]
            future_path = windows[:, 1:]

            pct_returns = (future_path - entry_prices) / entry_prices * 100.0

            bar_pt = atr_vals[:len(windows), np.newaxis] * 100.0 * atr_multiplier_pt
            bar_sl = atr_vals[:len(windows), np.newaxis] * 100.0 * atr_multiplier_sl
            group['pt_dynamic'] = atr_vals * 100.0 * atr_multiplier_pt
            group['sl_dynamic'] = atr_vals * 100.0 * atr_multiplier_sl

            tp_hits = pct_returns >= bar_pt
            sl_hits = pct_returns <= -bar_sl

            has_tp = np.any(tp_hits, axis=1)
            has_sl = np.any(sl_hits, axis=1)

            first_tp = np.where(has_tp, np.argmax(tp_hits, axis=1), horizon + 1)
            first_sl = np.where(has_sl, np.argmax(sl_hits, axis=1), horizon + 1)

            labels[:len(windows)] = np.where(
                first_tp < first_sl, 1,
                np.where(first_sl < first_tp, -1, 0)
            )

            group['label'] = labels
            processed_groups.append(group)

        self.df = pd.concat(processed_groups).reset_index(drop=True)
        return self.df

    def simple_labeling(self, horizon=30, threshold=0.1):
        """Simple binary labeling: 1 if return > threshold after horizon bars, else 0."""
        print(f"Applying Simple Binary Labeling (Horizon: {horizon} bars, Threshold: {threshold}%)...")
        self.df['future_return'] = (
            self.df.groupby('symbol')['close'].shift(-horizon) / self.df['close'] - 1
        )
        self.df['label'] = (self.df['future_return'] * 100 > threshold).astype(int)
        self.df = self.df.dropna()
        return self.df
