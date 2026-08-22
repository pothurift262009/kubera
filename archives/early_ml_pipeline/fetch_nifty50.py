from kiteconnect import KiteConnect  # type: ignore
from datetime import datetime, timedelta
import pandas as pd  # type: ignore
import time

# ===== CONFIG =====
API_KEY = "placeholder"
ACCESS_TOKEN = "placeholder"

kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)

# ===== NIFTY 50 LIST =====
NIFTY50 = [
"RELIANCE","TCS","INFY","HDFCBANK","ICICIBANK","HINDUNILVR",
"ITC","KOTAKBANK","LT","SBIN","BHARTIARTL","ASIANPAINT",
"AXISBANK","MARUTI","SUNPHARMA","TITAN","ULTRACEMCO",
"NESTLEIND","WIPRO","HCLTECH","POWERGRID","NTPC",
"TATAMOTORS","M&M","TECHM","BAJFINANCE","BAJAJFINSV",
"INDUSINDBK","ONGC","ADANIENT","ADANIPORTS","JSWSTEEL",
"TATASTEEL","COALINDIA","HDFCLIFE","SBILIFE","DRREDDY",
"CIPLA","GRASIM","BRITANNIA","HEROMOTOCO","EICHERMOT",
"APOLLOHOSP","DIVISLAB","UPL","BAJAJ-AUTO","SHREECEM",
"LTIM","HINDALCO"
]

# ===== GET INSTRUMENT TOKENS =====
print("Fetching instrument tokens...")
instruments = kite.instruments("NSE")
df_inst = pd.DataFrame(instruments)

token_map = df_inst[df_inst["tradingsymbol"].isin(NIFTY50)].set_index("tradingsymbol")["instrument_token"].to_dict()

print(f"Found {len(token_map)} stocks")

# ===== FUNCTION: FETCH DATA IN CHUNKS =====
def fetch_5y_data(token, symbol):
    all_data = []

    end = datetime.now()
    start_limit = end - timedelta(days=1825)  # ~5 years

    while end > start_limit:
        start = end - timedelta(days=60)

        if start < start_limit:
            start = start_limit

        retries = 3
        while retries > 0:
            try:
                data = kite.historical_data(
                    token,
                    from_date=start,
                    to_date=end,
                    interval="5minute"
                )

                for d in data:
                    d["symbol"] = symbol

                all_data.extend(data)
                print(f"{symbol}: {start.date()} → {end.date()} | rows={len(data)}")  # type: ignore
                break

            except Exception as e:
                retries -= 1  # type: ignore
                if retries == 0:
                    raise RuntimeError(f"Kite API failed for {symbol} ({start.date()} to {end.date()}): {e}")  # type: ignore

                wait_time = 2 ** (3 - retries)
                print(f"Error for {symbol}: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)

        end = start
        time.sleep(0.3)  # avoid rate limit

    return pd.DataFrame(all_data)


# ===== MAIN LOOP =====
all_dataframes = []

for symbol, token in token_map.items():
    print(f"\n===== FETCHING {symbol} =====")

    df = fetch_5y_data(token, symbol)

    if not df.empty:
        all_dataframes.append(df)


# ===== MERGE ALL =====
print("\nMerging all stocks...")

final_df = pd.concat(all_dataframes, ignore_index=True)

# Sort properly
final_df = final_df.sort_values(by=["symbol", "date"])

# Save
final_df.to_csv("nifty50_5min_5years.csv", index=False)

print("\nDONE ✅ File saved: nifty50_5min_5years.csv")
print("Total rows:", len(final_df))