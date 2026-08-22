import pandas as pd
from src.utils.logger import setup_logger

logger = setup_logger("DataLoader")


class DataLoader:
    """
    Loads 5-min OHLCV CSV and returns a dict of {symbol: DataFrame}.

    Expected CSV columns (case-insensitive):
        symbol, datetime (or date + time), open, high, low, close, volume
    """

    def __init__(self, csv_path: str):
        self.csv_path = csv_path

    def load(self) -> dict:
        logger.info(f"Loading {self.csv_path}")
        df = pd.read_csv(self.csv_path)
        df.columns = [c.lower().strip() for c in df.columns]

        # --- Normalise datetime column ---
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
        elif "date" in df.columns and "time" in df.columns:
            df["datetime"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str))
        elif "date" in df.columns:
            df["datetime"] = pd.to_datetime(df["date"])
        else:
            raise ValueError("CSV must have a 'datetime' or 'date' column")

        # --- Rename common column variants ---
        rename = {"open": "open", "high": "high", "low": "low",
                  "close": "close", "volume": "volume", "vol": "volume"}
        df.rename(columns=rename, inplace=True)

        required = {"symbol", "open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")

        df = df.sort_values(["symbol", "datetime"]).reset_index(drop=True)

        stock_data: dict = {}
        for symbol, grp in df.groupby("symbol"):
            stock_data[symbol] = grp.reset_index(drop=True)

        logger.info(f"Loaded {len(stock_data)} symbols | {len(df):,} rows")
        return stock_data

    @staticmethod
    def get_trading_dates(stock_data: dict) -> list:
        first = list(stock_data.values())[0]
        return sorted(first["datetime"].dt.date.unique())
