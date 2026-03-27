"""
NSE Full Universe Data Pipeline
================================
Fetches 5-min OHLCV for ALL NSE-listed stocks (~1,800–2,000 symbols).

Strategy:
  1. Pull the live NSE equity symbol list via the NSE website CSV endpoint
  2. Filter for actively traded stocks (min market cap / avg volume)
  3. Download in parallel batches with rate-limit handling
  4. Save per-stock Parquet files under data/raw/<SYMBOL>.parquet
  5. Build a merged feature-ready dataset

Requirements:
    pip install yfinance pandas requests tqdm pyarrow fastparquet
"""

import os
import time
import warnings
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from io import StringIO

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
RAW_DIR       = Path("data/raw")          # per-stock Parquet files
MERGED_DIR    = Path("data/merged")       # combined datasets
RAW_DIR.mkdir(parents=True, exist_ok=True)
MERGED_DIR.mkdir(parents=True, exist_ok=True)

INTERVAL       = "5m"         # 5-min bars
PERIOD_DAYS    = 59           # yfinance max for 5-min (60 day cap)
SESSION_START  = "09:15"
SESSION_END    = "15:25"
MAX_WORKERS    = 8            # parallel threads (keep ≤10 for yfinance rate limits)
BATCH_PAUSE    = 2.0          # seconds pause between batches of MAX_WORKERS
MIN_AVG_VOLUME = 500_000      # filter: avg daily volume > 500K shares
RETRY_LIMIT    = 3            # retries on download failure


# ─────────────────────────────────────────────────────────────
# STEP 1 — FETCH COMPLETE NSE SYMBOL LIST
# ─────────────────────────────────────────────────────────────

NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}

def fetch_nse_symbol_list() -> pd.DataFrame:
    """
    Downloads the official NSE equity symbol master CSV.
    Returns DataFrame with columns: SYMBOL, NAME, SERIES, ISIN
    Falls back to a bundled Nifty 500 list if NSE blocks the request.
    """
    # NSE publishes a daily refreshed CSV of all listed equities
    url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
    print("Fetching NSE equity master list...")
    try:
        session = requests.Session()
        # warm up cookies
        session.get("https://www.nseindia.com", headers=NSE_HEADERS, timeout=10)
        resp = session.get(url, headers=NSE_HEADERS, timeout=20)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        df.columns = [c.strip().upper().replace(" ", "_") for c in df.columns]
        # Keep only regular equity series (EQ), exclude SME/BE/BZ
        df = df[df["SERIES"] == "EQ"].copy()
        df = df[["SYMBOL", "NAME_OF_COMPANY", "ISIN_NUMBER"]].rename(
            columns={"NAME_OF_COMPANY": "NAME", "ISIN_NUMBER": "ISIN"}
        )
        print(f"  ✓ {len(df):,} NSE EQ symbols loaded from master list")
        return df
    except Exception as e:
        print(f"  ⚠ NSE master fetch failed ({e}). Falling back to Nifty 500 list.")
        return _nifty500_fallback()


def _nifty500_fallback() -> pd.DataFrame:
    """Nifty 500 from NSE index constituents CSV — always publicly accessible."""
    url = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
    try:
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=NSE_HEADERS, timeout=10)
        resp = session.get(url, headers=NSE_HEADERS, timeout=20)
        df = pd.read_csv(StringIO(resp.text))
        df.columns = [c.strip().upper() for c in df.columns]
        df = df[["SYMBOL", "COMPANY_NAME", "ISIN"]].rename(columns={"COMPANY_NAME": "NAME"})
        print(f"  ✓ {len(df)} Nifty 500 symbols loaded (fallback)")
        return df
    except Exception as e:
        print(f"  ✗ Fallback also failed ({e}). Using hardcoded Nifty 50.")
        return pd.DataFrame({"SYMBOL": _NIFTY50_HARDCODED, "NAME": _NIFTY50_HARDCODED})


_NIFTY50_HARDCODED = [
    "RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK","HINDUNILVR","SBIN",
    "BAJFINANCE","BHARTIARTL","KOTAKBANK","ITC","LT","AXISBANK","ASIANPAINT",
    "MARUTI","HCLTECH","SUNPHARMA","TITAN","WIPRO","ULTRACEMCO","NTPC",
    "POWERGRID","BAJAJFINSV","TECHM","ONGC","JSWSTEEL","TATAMOTORS",
    "ADANIPORTS","NESTLEIND","GRASIM","COALINDIA","HINDALCO","DRREDDY",
    "BRITANNIA","DIVISLAB","CIPLA","EICHERMOT","BPCL","IOC","TATASTEEL",
    "TATACONSUM","APOLLOHOSP","HEROMOTOCO","INDUSINDBK","BAJAJ_AUTO",
    "VEDL","M&M","UPL","HDFCLIFE","SBILIFE"
]


# ─────────────────────────────────────────────────────────────
# STEP 2 — DOWNLOAD ONE SYMBOL
# ─────────────────────────────────────────────────────────────

def download_symbol(symbol: str) -> tuple[str, pd.DataFrame | None, str]:
    """
    Download 5-min OHLCV for a single NSE symbol via yfinance.
    Returns (symbol, dataframe_or_None, status_message)
    """
    ticker = f"{symbol}.NS"
    end    = datetime.now()
    start  = end - timedelta(days=PERIOD_DAYS)

    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                interval=INTERVAL,
                progress=False,
                auto_adjust=True,
                multi_level_index=False,  # yfinance ≥ 0.2.38 returns MultiIndex by default
            )
            if df is None or df.empty:
                return symbol, None, "empty"

            # Flatten MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df.columns = [c.lower() for c in df.columns]
            df.index   = pd.to_datetime(df.index)

            # Localize / convert to IST if timezone-aware
            if df.index.tz is not None:
                df.index = df.index.tz_convert("Asia/Kolkata").tz_localize(None)

            # Filter to NSE session
            df = df.between_time(SESSION_START, SESSION_END)
            df.dropna(subset=["close", "volume"], inplace=True)

            # Volume quality filter
            daily_vol = df["volume"].resample("D").sum().mean()
            if daily_vol < MIN_AVG_VOLUME:
                return symbol, None, f"low_volume({daily_vol:.0f})"

            df["symbol"] = symbol
            return symbol, df, "ok"

        except Exception as e:
            if attempt < RETRY_LIMIT:
                time.sleep(1.5 * attempt)
            else:
                return symbol, None, f"error:{e}"

    return symbol, None, "failed"


# ─────────────────────────────────────────────────────────────
# STEP 3 — BATCH DOWNLOAD ALL SYMBOLS
# ─────────────────────────────────────────────────────────────

def download_all_symbols(symbols: list[str]) -> dict:
    """
    Parallel download with progress bar and batch rate-limiting.
    Saves each valid symbol to data/raw/<SYMBOL>.parquet
    Returns summary stats dict.
    """
    total       = len(symbols)
    successful  = []
    failed      = []
    skipped     = []

    print(f"\nDownloading {total:,} symbols in batches of {MAX_WORKERS}...")
    print(f"Data window: last {PERIOD_DAYS} days at {INTERVAL} intervals\n")

    pbar = tqdm(total=total, unit="sym", ncols=80, colour="green")

    # Process in batches to respect rate limits
    batch_size = MAX_WORKERS * 4   # submit MAX_WORKERS*4 symbols per batch
    for batch_start in range(0, total, batch_size):
        batch = symbols[batch_start: batch_start + batch_size]

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(download_symbol, sym): sym for sym in batch}
            for future in as_completed(futures):
                sym, df, status = future.result()

                if status == "ok" and df is not None:
                    # Save to Parquet (efficient columnar storage)
                    out_path = RAW_DIR / f"{sym}.parquet"
                    df.to_parquet(out_path, index=True, compression="snappy")
                    successful.append(sym)
                elif status.startswith("low_volume") or status == "empty":
                    skipped.append((sym, status))
                else:
                    failed.append((sym, status))

                pbar.update(1)
                pbar.set_postfix(ok=len(successful), skip=len(skipped), fail=len(failed))

        # Pause between batches to avoid yfinance rate limiting
        if batch_start + batch_size < total:
            time.sleep(BATCH_PAUSE)

    pbar.close()

    summary = {
        "total_requested": total,
        "successful":       len(successful),
        "skipped":          len(skipped),
        "failed":           len(failed),
        "symbols":          successful,
    }
    return summary, skipped, failed


# ─────────────────────────────────────────────────────────────
# STEP 4 — MERGE ALL PARQUETS INTO ONE DATASET
# ─────────────────────────────────────────────────────────────

def merge_all_to_dataset(symbols: list[str]) -> pd.DataFrame:
    """
    Reads all saved Parquet files and concatenates into a single
    multi-stock DataFrame with (datetime, symbol) as the index.
    Saves merged file to data/merged/all_stocks_5m.parquet
    """
    print(f"\nMerging {len(symbols)} stock files into dataset...")
    dfs = []
    for sym in tqdm(symbols, unit="sym", ncols=80):
        path = RAW_DIR / f"{sym}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            dfs.append(df)

    if not dfs:
        print("No data to merge.")
        return pd.DataFrame()

    combined = pd.concat(dfs, axis=0)
    combined.sort_index(inplace=True)
    combined.index.name = "datetime"

    out_path = MERGED_DIR / "all_stocks_5m.parquet"
    combined.to_parquet(out_path, index=True, compression="snappy")

    size_mb = out_path.stat().st_size / 1_048_576
    print(f"  ✓ Merged dataset saved → {out_path}")
    print(f"    Shape: {combined.shape[0]:,} rows × {combined.shape[1]} cols")
    print(f"    File size: {size_mb:.1f} MB")
    print(f"    Symbols: {combined['symbol'].nunique()}")
    print(f"    Date range: {combined.index.min()} → {combined.index.max()}")
    return combined


# ─────────────────────────────────────────────────────────────
# STEP 5 — QUICK SANITY CHECK
# ─────────────────────────────────────────────────────────────

def sanity_check(df: pd.DataFrame, n_symbols: int = 5):
    """Print basic stats to verify data quality."""
    print("\n── Sanity check ──────────────────────────────────────")
    sample_syms = df["symbol"].unique()[:n_symbols]
    for sym in sample_syms:
        sub = df[df["symbol"] == sym]
        print(f"  {sym:15s}  rows={len(sub):5d}  "
              f"close_range=[{sub['close'].min():.1f}, {sub['close'].max():.1f}]  "
              f"avg_vol={sub['volume'].mean():,.0f}")
    print(f"\n  Total bars:    {len(df):,}")
    print(f"  Total symbols: {df['symbol'].nunique()}")
    print(f"  Memory usage:  {df.memory_usage(deep=True).sum() / 1_048_576:.1f} MB")
    print("──────────────────────────────────────────────────────\n")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = time.time()

    # 1. Get full symbol list
    symbol_df = fetch_nse_symbol_list()
    all_symbols = symbol_df["SYMBOL"].str.strip().str.upper().tolist()
    print(f"\nTotal symbols to process: {len(all_symbols):,}")

    # Optional: restrict to a subset for testing
    # all_symbols = all_symbols[:50]   # ← uncomment to test with first 50

    # 2. Download all
    summary, skipped, failed = download_all_symbols(all_symbols)

    # 3. Print download report
    print(f"\n{'─'*50}")
    print(f"  Download complete in {(time.time()-t0)/60:.1f} min")
    print(f"  ✓ Successful : {summary['successful']:,}")
    print(f"  ~ Skipped    : {len(skipped):,}  (low volume / empty)")
    print(f"  ✗ Failed     : {len(failed):,}")
    if failed:
        print(f"    Failed symbols: {[s for s,_ in failed[:10]]}{'...' if len(failed)>10 else ''}")
    print(f"{'─'*50}\n")

    # Save failed list for retry
    if failed:
        pd.DataFrame(failed, columns=["symbol", "reason"]).to_csv(
            "data/failed_symbols.csv", index=False
        )
        print("  Failed symbols saved to data/failed_symbols.csv")

    # 4. Merge into single dataset
    merged_df = merge_all_to_dataset(summary["symbols"])

    # 5. Sanity check
    if not merged_df.empty:
        sanity_check(merged_df)

    print("Pipeline complete. Next step: run features.py on merged dataset.")
