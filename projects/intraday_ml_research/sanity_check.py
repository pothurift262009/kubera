import pandas as pd
import numpy as np
import joblib
from backtester import Backtester
from data_loader import DataProcessor
from feature_engineer import FeatureEngineer
from labeler import Labeler
import config


def sanity_check():
    print("\n--- Starting Sanity Check (Label Shuffling) ---")

    model = joblib.load('best_xgb_model.joblib')
    feature_cols = joblib.load('feature_cols.joblib')

    # Minimal pipeline to get test_df
    dp = DataProcessor(config.DATA_FILE)
    df = dp.load_data()
    df = dp.resample_to_5min()

    fe = FeatureEngineer(df)
    df = fe.generate_features()
    df = fe.rank_features()

    lb = Labeler(df)
    df = lb.triple_barrier_labeling(
        pt=config.PT_PCT, sl=config.SL_PCT, horizon=config.HORIZON_BARS
    )

    # Date-based split (matches training methodology)
    unique_dates = sorted(df['date'].unique())
    split_idx = int(len(unique_dates) * config.TRAIN_RATIO)
    split_date = unique_dates[split_idx]
    test_df = df[df['date'] >= split_date].copy()

    print(f"\nTest set: {len(test_df)} rows, "
          f"{test_df['date'].min()} to {test_df['date'].max()}")

    # ── Case 1: Original labels ──────────────────────────────────
    print("\n[Case 1] Original Labels")
    bt = Backtester(test_df, model, feature_cols)
    res_orig = bt.run_backtest(
        threshold=0.60,
        pt=config.PT_PCT, sl=config.SL_PCT,
        transaction_cost=config.TRANSACTION_COST_PCT,
        horizon=config.HORIZON_BARS,
        plot=False
    )
    print(f"Original Sharpe: {res_orig['Sharpe']}")
    print(f"Original Hit Rate: {res_orig['Hit Rate (%)']:.1f}%")

    # ── Case 2: Shuffled labels ──────────────────────────────────
    # NOTE: With the new backtester that simulates from real prices,
    # shuffling labels should have NO effect on PnL (because the backtester
    # no longer reads labels for PnL). This is the correct behavior.
    # If shuffling DOES change results, something is still wrong.

    print("\n[Case 2] Shuffled Labels (should produce IDENTICAL results)")
    test_df_shuffled = test_df.copy()
    test_df_shuffled['label'] = np.random.permutation(test_df_shuffled['label'].values)

    bt_check = Backtester(test_df_shuffled, model, feature_cols, target_col='label')
    res_shuf = bt_check.run_backtest(
        threshold=0.60,
        pt=config.PT_PCT, sl=config.SL_PCT,
        transaction_cost=config.TRANSACTION_COST_PCT,
        horizon=config.HORIZON_BARS,
        plot=False
    )
    print(f"Shuffled Sharpe: {res_shuf['Sharpe']}")
    print(f"Shuffled Hit Rate: {res_shuf['Hit Rate (%)']:.1f}%")

    # ── Validation ───────────────────────────────────────────────
    if abs(res_orig['Total Return (%)'] - res_shuf['Total Return (%)']) < 0.001:
        print("\n✅ PASS: Shuffling labels has NO effect on backtest PnL.")
        print("   This confirms the backtester uses real prices, not labels.")
    else:
        print("\n❌ FAIL: Shuffling labels changed the backtest results!")
        print("   The backtester may still be reading from the label column.")


if __name__ == "__main__":
    sanity_check()
