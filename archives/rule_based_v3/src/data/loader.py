import pandas as pd
from src.utils.logger import setup_logger

logger = setup_logger("DataLoader")


class DataLoader:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path

    def load(self) -> dict:
        logger.info(f"Reading {self.csv_path} ...")
        df = pd.read_csv(self.csv_path)
        df.columns = [c.lower().strip() for c in df.columns]

        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], utc=False)
            if hasattr(df["datetime"].dtype, "tz") and df["datetime"].dt.tz is not None:
                df["datetime"] = df["datetime"].dt.tz_localize(None)
        elif "date" in df.columns and "time" in df.columns:
            df["datetime"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str))
        elif "date" in df.columns:
            df["datetime"] = pd.to_datetime(df["date"])
        else:
            raise ValueError("CSV must have a 'datetime' or 'date' column")

        df.rename(columns={"vol": "volume", "ticker": "symbol"}, inplace=True)

        required = {"symbol", "open", "high", "low", "close", "volume"}
        missing  = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")

        df = df.sort_values(["symbol", "datetime"]).reset_index(drop=True)

        stock_data = {}
        for symbol, grp in df.groupby("symbol"):
            stock_data[symbol] = grp.reset_index(drop=True)

        dates = df["datetime"].dt.date
        logger.info(
            f"Loaded {len(stock_data)} symbols | {len(df):,} rows | "
            f"Date range: {dates.min()} → {dates.max()}"
        )
        return stock_data
