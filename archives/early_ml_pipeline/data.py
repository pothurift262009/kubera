from kiteconnect import KiteConnect
from datetime import datetime
import pandas as pd

API_KEY = "placeholder"
ACCESS_TOKEN = "placeholder"

kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)

# ===== GET INSTRUMENT TOKEN =====
instruments = kite.instruments("NSE")
df_inst = pd.DataFrame(instruments)

token = df_inst[df_inst.tradingsymbol == "INFY"]["instrument_token"].values[0]

# ===== FETCH DATA =====
data = kite.historical_data(
    token,
    from_date=datetime(2024,1,1),
    to_date=datetime(2024,2,1),
    interval="5minute"
)

df = pd.DataFrame(data)

print(df.head())

df.to_csv("infy_data.csv", index=False)


import pandas as pd

df = pd.read_csv("infy_data.csv")

df = df.dropna()
df = df.reset_index(drop=True)

df["future_return"] = df["close"].shift(-3) / df["close"] - 1

# 0.2% move in next 15 mins
df["target"] = (df["future_return"] > 0.002).astype(int)
df["return"] = df["close"].pct_change()
df["range"] = df["high"] - df["low"]

df["ma_5"] = df["close"].rolling(5).mean()
df["ma_10"] = df["close"].rolling(10).mean()

df["volatility"] = df["return"].rolling(10).std()

df["volume_ma"] = df["volume"].rolling(10).mean()
df["volume_spike"] = df["volume"] / df["volume_ma"]
df = df.dropna()
from sklearn.ensemble import RandomForestClassifier

features = ["return","range","ma_5","ma_10","volatility","volume_spike"]

X = df[features]
y = df["target"]

model = RandomForestClassifier(n_estimators=100)

split = int(len(df)*0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model.fit(X_train, y_train)

preds = model.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(y_test, preds))