import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def load_data(self):
        logger.info(f"Loading data from {self.file_path}...")
        print(f"Loading data from {self.file_path}...")
        self.df = pd.read_parquet(self.file_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values(['symbol', 'date']).reset_index(drop=True)
        print(f"Loaded shape: {self.df.shape}")
        return self.df

    def resample_to_5min(self):
        print("Resampling to 5-minute bars...")
        self.df = self.df.set_index('date')
        self.df = (
            self.df.groupby('symbol')
            .resample('5min')
            .agg({'open': 'first', 'high': 'max', 'low': 'min',
                  'close': 'last', 'volume': 'sum'})
            .dropna()
            .reset_index()
        )
        self.df = self.df.sort_values(['symbol', 'date']).reset_index(drop=True)
        self.optimize_memory()
        print(f"Resampled shape: {self.df.shape}")
        return self.df

    def optimize_memory(self):
        float_cols = self.df.select_dtypes(include=['float64']).columns
        self.df[float_cols] = self.df[float_cols].astype('float32')
        int_cols = self.df.select_dtypes(include=['int64']).columns
        self.df[int_cols] = self.df[int_cols].astype('int32')
        mem_mb = self.df.memory_usage(deep=True).sum() / 1024**2
        print(f"Memory: {mem_mb:.2f} MB")

    def get_symbols(self):
        return self.df['symbol'].unique()
