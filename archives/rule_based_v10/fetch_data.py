"""
Fetch 1-minute OHLCV data for high-beta Nifty stocks from Kite Historical API
Saves to: data/highbeta_1min_5years.csv

Usage:
  python fetch_data.py

Requirements:
  - Set api_key and access_token in config.yaml first
  - Authenticate via dashboard -> Live Trading tab to get access_token
"""
import yaml
import pandas as pd
import time
import os
from datetime import datetime, timedelta
from kiteconnect import KiteConnect

# ── Config ────────────────────────────────────────────────────────────────
with open("config.yaml") as f:
    config = yaml.safe_load(f)

kite = KiteConnect(api_key=config["kite"]["api_key"])
kite.set_access_token(config["kite"]["access_token"])

# ── Optimized stock list ──────────────────────────────────────────────────
SYMBOLS = [
    # Tier 1: High Beta (1.3+), moves 1.7-2.8% daily
    "TATAMOTORS", "ADANIENT",   "BAJFINANCE",  "TATASTEEL",
    "ADANIPORTS", "BAJAJFINSV", "JSWSTEEL",    "HINDALCO",
    "INDUSINDBK", "AXISBANK",   "SBIN",        "ICICIBANK",
    # Tier 2: Medium Beta (1.0-1.2), high liquidity
    "LT",         "M&M",        "TITAN",       "HDFCBANK",
    "KOTAKBANK",  "TECHM",      "MARUTI",      "INFY",
    "WIPRO",      "RELIANCE",
]

END_DATE    = datetime.today()
START_DATE  = END_DATE - timedelta(days=365 * 5)
CHUNK_DAYS  = 50
INTERVAL    = "minute"
OUTPUT_FILE = "data/highbeta_1min_5years.csv"
RESUME_FILE = "data/.fetch_progress.txt"

os.makedirs("data", exist_ok=True)


def load_progress():
    if os.path.exists(RESUME_FILE):
        with open(RESUME_FILE) as f:
            return set(f.read().strip().split("\n"))
    return set()


def save_progress(symbol):
    with open(RESUME_FILE, "a") as f:
        f.write(symbol + "\n")


def fetch_symbol(symbol, token):
    all_data = []
    current  = START_DATE

    while current < END_DATE:
        chunk_end = min(current + timedelta(days=CHUNK_DAYS), END_DATE)
        for attempt in range(2):
            try:
                candles = kite.historical_data(
                    instrument_token=token,
                    from_date=current,
                    to_date=chunk_end,
                    interval=INTERVAL,
                )
                if candles:
                    df = pd.DataFrame(candles)
                    df["symbol"] = symbol
                    all_data.append(df)
                    print(f"    {current.date()} -> {chunk_end.date()} | {len(candles):,} rows", end="\r")
                break
            except Exception as e:
                if attempt == 0:
                    print(f"\n    Retry {current.date()}: {e}")
                    time.sleep(5)
                else:
                    print(f"\n    Skip  {current.date()}: {e}")

        current = chunk_end + timedelta(days=1)
        time.sleep(0.4)

    if not all_data:
        return pd.DataFrame()

    df = pd.concat(all_data, ignore_index=True)
    df.rename(columns={"date": "datetime"}, inplace=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    if df["datetime"].dt.tz is not None:
        df["datetime"] = df["datetime"].dt.tz_localize(None)
    return df[["symbol", "datetime", "open", "high", "low", "close", "volume"]]


# ── Main ──────────────────────────────────────────────────────────────────
print("=" * 55)
print("  High-Beta 1-Min Data Fetcher")
print(f"  Stocks : {len(SYMBOLS)}")
print(f"  Period : {START_DATE.date()} to {END_DATE.date()}")
print(f"  Output : {OUTPUT_FILE}")
print("=" * 55)

print("\nFetching instrument tokens...")
instruments = kite.instruments("NSE")
token_map   = {i["tradingsymbol"]: i["instrument_token"]
               for i in instruments if i["tradingsymbol"] in SYMBOLS}
print(f"Found {len(token_map)}/{len(SYMBOLS)} symbols")

missing = set(SYMBOLS) - set(token_map.keys())
if missing:
    print(f"Not found: {missing}")

done       = load_progress()
all_frames = []

# Load existing data if resuming mid-run
if done and os.path.exists(OUTPUT_FILE):
    print(f"\nResuming — {len(done)} already done. Loading existing data...")
    all_frames.append(pd.read_csv(OUTPUT_FILE))

for i, symbol in enumerate(SYMBOLS):
    if symbol in done:
        print(f"[{i+1}/{len(SYMBOLS)}] {symbol} skipped (already done)")
        continue
    if symbol not in token_map:
        print(f"[{i+1}/{len(SYMBOLS)}] {symbol} skipped (not found in NSE)")
        continue

    print(f"\n[{i+1}/{len(SYMBOLS)}] {symbol}...")
    t0 = time.time()
    df = fetch_symbol(symbol, token_map[symbol])

    if not df.empty:
        print(f"\n  OK {symbol}: {len(df):,} rows | {time.time()-t0:.0f}s")
        all_frames.append(df)

        # Save after every symbol so progress isn't lost
        out = pd.concat(all_frames, ignore_index=True)
        out = out.drop_duplicates(subset=["symbol","datetime"])
        out = out.sort_values(["symbol","datetime"]).reset_index(drop=True)
        out.to_csv(OUTPUT_FILE, index=False)
        save_progress(symbol)
        print(f"  Saved {len(out):,} total rows")
    else:
        print(f"\n  EMPTY {symbol}: no data")

# ── Summary ───────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
if os.path.exists(OUTPUT_FILE):
    final = pd.read_csv(OUTPUT_FILE)
    print(f"  COMPLETE")
    print(f"  Rows     : {len(final):,}")
    print(f"  Symbols  : {final['symbol'].nunique()}")
    print(f"  Range    : {final['datetime'].min()} to {final['datetime'].max()}")
    print(f"  Size     : {os.path.getsize(OUTPUT_FILE)/1024/1024:.1f} MB")
    print(f"\n  Now update config.yaml:")
    print(f"    csv_path: \"{OUTPUT_FILE}\"")
else:
    print("  No data saved. Check your access_token in config.yaml")
print("=" * 55)

if os.path.exists(RESUME_FILE):
    os.remove(RESUME_FILE)
