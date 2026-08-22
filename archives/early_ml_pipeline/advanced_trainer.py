import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import pandas_ta as ta  # type: ignore
import lightgbm as lgb  # type: ignore
from sklearn.model_selection import TimeSeriesSplit  # type: ignore
from sklearn.metrics import precision_score, recall_score, f1_score  # type: ignore
import time
import os
from trading_engine import prepare_features, get_target  # type: ignore

# ===== CONFIG =====
DATA_FILE = "nifty50_5min_5years.csv"
MODEL_FILE = "trading_model.lgb"
FEATURES = [] # Will be populated
TARGET = "target"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

def main():
    global FEATURES

    # ===== STEP 1: LOAD & BASIC CLEAN =====
    log("Loading data...")
    df = pd.read_csv(DATA_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"])

    # Alignment: Only trade between 09:15 and 15:25
    df = df[df["date"].dt.time >= pd.Timestamp("09:15").time()]
    df = df[df["date"].dt.time <= pd.Timestamp("15:25").time()]

    log(f"Initial Clean: {len(df)} rows across {df['symbol'].nunique()} symbols")

    # ===== STEP 2: MARKET FEATURES =====
    log("Computing market-wide features...")
    # Calculate mean return across all symbols per timestamp
    df["ret_1"] = df.groupby("symbol")["close"].pct_change()
    market_avg = df.groupby("date")["ret_1"].mean().rename("mkt_ret_1")
    df = df.merge(market_avg, on="date", how="left")
    df["rel_ret_1"] = df["ret_1"] - df["mkt_ret_1"]

    # ===== STEP 3-5: TARGET LABELING & ADVANCED FEATURES =====
    log("Applying technical indicators and target labels (via trading_engine.py)...")

    df = get_target(df)
    df, FEATURES = prepare_features(df)

    # Drop rows with NAs efficiently across strict predictive boundaries
    df = df.dropna(subset=FEATURES + ["target", "future_ret_3"])
    log(f"Features for training ({len(FEATURES)}): {FEATURES}")

    # ===== STEP 6: TRAIN/TEST SPLIT & WALK-FORWARD CV =====
    df = df.sort_values("date")

    dates = df["date"].unique()
    dates.sort()
    split_date_idx = int(len(dates) * 0.8)
    split_date = dates[split_date_idx]

    train_full = df[df["date"] < split_date]
    test_df = df[df["date"] >= split_date]

    X_test, y_test = test_df[FEATURES], test_df["target"]

    # ===== STEP 7: LIGHTGBM TRAINING =====
    log("Training LightGBM model with TimeSeriesSplit Walk-Forward CV...")

    pos_count = max(train_full["target"].sum(), 1)
    neg_count = len(train_full) - pos_count
    pos_weight = min(neg_count / pos_count, 20)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "scale_pos_weight": pos_weight,
        "verbose": -1,
        "random_state": 42
    }

    tscv = TimeSeriesSplit(n_splits=3)
    models = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(train_full)):
        log(f"Training Fold {fold+1}/3...")
        X_train, y_train = train_full.iloc[train_idx][FEATURES], train_full.iloc[train_idx]["target"]
        X_val, y_val = train_full.iloc[val_idx][FEATURES], train_full.iloc[val_idx]["target"]

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        mdl = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=50)]
        )
        models.append(mdl)

    # Use final chronological model fold as the strict proxy for recent behavior
    model = models[-1]

    # ===== STEP 8: EVALUATION =====
    log("Evaluating on test set...")
    y_prob = model.predict(X_test)

    # ===== BACKTEST: REALISTIC CHECK =====
    bt_df = test_df.copy()
    bt_df["prob"] = y_prob
    bt_df["signal"] = (bt_df["prob"] > 0.65).astype(int)

    bt_df["f_ret_1"] = bt_df.groupby("symbol")["close"].shift(-1) / bt_df["close"] - 1
    bt_df["f_ret_2"] = bt_df.groupby("symbol")["close"].shift(-2) / bt_df["close"] - 1
    bt_df["f_ret_3"] = bt_df.groupby("symbol")["close"].shift(-3) / bt_df["close"] - 1

    bt_df = bt_df.sort_values(["date", "symbol"])
    bt_df["position"] = 0

    last_trade_time = {}
    positions = []
    trade_returns_raw = []

    timestamp_trade_count = {}
    MAX_TRADES_PER_TIMESTAMP = 2

    TAKE_PROFIT = 0.003
    STOP_LOSS = -0.002

    for idx, row in bt_df.iterrows():
        sym = row["symbol"]
        curr_time = row["date"]
        sig = row["signal"]

        pos = 0
        ts = curr_time
        trade_ret = 0.0

        if sig == 1:
            if timestamp_trade_count.get(ts, 0) >= MAX_TRADES_PER_TIMESTAMP:
                pos = 0
            elif sym not in last_trade_time or (curr_time - last_trade_time[sym]).total_seconds() >= 900:
                pos = 1
                last_trade_time[sym] = curr_time
                timestamp_trade_count[ts] = timestamp_trade_count.get(ts, 0) + 1

                # Realistic SL/TP Sequential Exit
                f_ret_1 = row["f_ret_1"]
                f_ret_2 = row["f_ret_2"]
                f_ret_3 = row["f_ret_3"]

                step1 = f_ret_1

                step2 = np.nan
                if pd.notna(f_ret_2) and pd.notna(f_ret_1) and abs(1 + f_ret_1) > 1e-9:
                    step2 = (1 + f_ret_2) / (1 + f_ret_1) - 1

                step3 = np.nan
                if pd.notna(f_ret_3) and pd.notna(f_ret_2) and abs(1 + f_ret_2) > 1e-9:
                    step3 = (1 + f_ret_3) / (1 + f_ret_2) - 1

                cum1 = step1
                cum2 = (1 + step1) * (1 + step2) - 1 if pd.notna(step2) else np.nan
                cum3 = (1 + step1) * (1 + step2) * (1 + step3) - 1 if pd.notna(step3) else np.nan

                if pd.notna(cum1) and (cum1 >= TAKE_PROFIT or cum1 <= STOP_LOSS):
                    trade_ret = cum1
                elif pd.notna(cum2) and (cum2 >= TAKE_PROFIT or cum2 <= STOP_LOSS):
                    trade_ret = cum2
                elif pd.notna(cum3):
                    trade_ret = cum3
                else:
                    trade_ret = cum1 if pd.notna(cum1) else 0.0

        positions.append(pos)
        trade_returns_raw.append(trade_ret)

    bt_df["position"] = positions
    bt_df["trade_return"] = trade_returns_raw

    COST = 0.0006
    bt_df["trade_return"] = bt_df["trade_return"] - (COST * bt_df["position"])
    bt_df.loc[bt_df["position"] == 0, "trade_return"] = 0.0

    total_profit = bt_df["trade_return"].sum()
    num_trades = bt_df["position"].sum()

    if num_trades > 0:
        active_trades = bt_df[bt_df["position"] == 1]
        win_rate = (active_trades["trade_return"] > 0).mean() * 100
        avg_trade_return = active_trades["trade_return"].mean() * 100
    else:
        win_rate = 0.0
        avg_trade_return = 0.0

    print("\n===== BACKTEST RESULTS =====")
    print(f"Total Profit:         {total_profit:.4f}")
    print(f"Number of Trades:     {num_trades}")
    print(f"Win Rate:             {win_rate:.2f}%")
    print(f"Average Trade Return: {avg_trade_return:.4f}%\n")


    # In trading, we only care about high-confidence signals

    # Let's say we only trade when probability > 0.7
    threshold = 0.65
    y_pred = (y_prob > threshold).astype(int)

    count_trades = y_pred.sum()
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    log(f"Trades suggested: {count_trades} out of {len(y_test)}")
    log(f"Precision (Win Rate proxy): {precision:.4f}")
    log(f"Recall: {recall:.4f}")
    log(f"F1 Score: {f1:.4f}")

    # Save model
    model.save_model(MODEL_FILE)
    log(f"Model saved to {MODEL_FILE}")

    # Optional: Feature Importance
    importance = pd.DataFrame({"feature": FEATURES, "importance": model.feature_importance()})
    importance = importance.sort_values("importance", ascending=False)
    print("\nTop 10 Features:")
    print(importance.head(10))

if __name__ == "__main__":
    main()
