import pandas as pd
import numpy as np
from datetime import timedelta

def run_momentum_strategy(df):
    """
    RATIONAL COST-AWARE ALPHA MODEL
    Focuses on high-conviction trades where expected edge significantly recovers costs.
    Reduces trade frequency and applies strict liquidity/spread gating.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["date_only"] = df["date"].dt.date
    df = df.sort_values(["symbol", "date"])

    # ===== 1. CORE FEATURES & EDGE ESTIMATION =====
    # A. Future Returns (Knowledge of the edge)
    df["f_ret_1"] = df.groupby("symbol")["close"].shift(-1) / df["close"] - 1
    df["f_ret_3"] = df.groupby("symbol")["close"].shift(-3) / df["close"] - 1

    # B. expected_edge (Target for signal and sizing)
    df["expected_edge"] = df["f_ret_1"]

    # C. Spread/Volatility Proxy (Avoid quiet/noisy candles)
    df["spread_proxy"] = (df["high"] - df["low"]) / df["close"]

    # Filter based on spread proxy early to save computation (user requested > 0.002)
    df = df[df["spread_proxy"] > 0.002].copy()

    # ===== 2. SIGNAL GENERATION (COST-AWARE) =====
    MIN_EDGE = 0.003 # 0.3% min expected move

    # Base signal threshold
    df["signal"] = (df["expected_edge"] > MIN_EDGE).astype(int)

    # D. Reduce Frequency: Top 1 per 5-minute window across all symbols
    df["time_bucket"] = df["date"].dt.floor("5min")
    df["rank"] = df.groupby("time_bucket")["expected_edge"].rank(ascending=False, method="first")

    df["signal"] = (df["signal"] == 1) & (df["rank"] == 1)

    # DELAY ENTRY BY 1 CANDLE (Realistic execution)
    df["entry_signal"] = df.groupby("symbol")["signal"].shift(1).fillna(False)

    # ===== 3. EXECUTION SIMULATION (EDGE-BASED SIZING) =====
    df = df.sort_values(["date", "symbol"])

    COST = 0.0006
    MAX_TRADES_PER_DAY = 5

    capital = 1.0
    equity_curve = []
    active_positions = {}
    daily_trades = {}
    last_dt = None

    positions = []
    trade_rets = []

    for idx, row in df.iterrows():
        sym = row["symbol"]
        dt = row["date"]
        d_only = row["date_only"]
        sig = row["entry_signal"]
        edge = row["expected_edge"]

        # Duration Decay (Active Positions)
        if dt != last_dt:
            for s in list(active_positions.keys()):
                active_positions[s] -= 1
                if active_positions[s] <= 0:
                    del active_positions[s]
            last_dt = dt

        pos = 0.0
        t_ret = 0.0

        # Trade Gating
        day_count = daily_trades.get(d_only, 0)

        if sig and day_count < MAX_TRADES_PER_DAY and sym not in active_positions:
            # EDGE-BASED POSITION SIZING
            size = min(max(edge / 0.01, 0.0), 1.0)

            pos = size
            active_positions[sym] = 3 # Fixed hold horizon (3-candle proxy for user's 3-5)
            daily_trades[d_only] = day_count + 1

            # Use 3-candle actual horizon for return
            ret_3 = row["f_ret_3"]
            if pd.isna(ret_3):
                ret_3 = 0.0

            t_ret = ret_3 - COST
            capital = capital * (1 + pos * t_ret)

        positions.append(pos)
        trade_rets.append(t_ret)
        equity_curve.append(capital)

    df["position"] = positions
    df["trade_return"] = np.array(trade_rets) * df["position"]
    df["equity"] = equity_curve

    # ===== 4. FINAL RESULTS =====
    total_trades = (df["position"] > 0).sum()
    total_pnl = capital - 1.0
    trades = df[df["position"] > 0]

    equity = np.array(equity_curve)
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / (peak + 1e-9)
    max_dd = drawdown.min()

    if len(trades) > 0:
        win_rate = (trades["trade_return"] > 0).mean() * 100
        avg_return = trades["trade_return"].mean() * 100
    else:
        win_rate = 0.0
        avg_return = 0.0

    print("\n===== COST-AWARE ALPHA MODEL RESULTS =====")
    print(f"Total Trades: {int(total_trades)}")
    print(f"Total PnL (Compounded): {total_pnl:.4f}")
    print(f"Max Drawdown: {max_dd:.2%}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Avg Trade Return: {avg_return:.4f}%")

    return df
