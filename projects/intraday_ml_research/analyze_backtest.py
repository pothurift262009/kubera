from data_loader import DataProcessor
from feature_engineer import FeatureEngineer
from labeler import Labeler
from backtester import Backtester
import config
import joblib


def main():
    dp = DataProcessor(config.DATA_FILE)
    df = dp.load_data()
    df = dp.resample_to_5min()

    fe = FeatureEngineer(df)
    df = fe.generate_features()
    df = fe.rank_features()

    lb = Labeler(df)
    df = lb.atr_barrier_labeling(
        pt=config.PT_PCT, sl=config.SL_PCT, horizon=config.HORIZON_BARS
    )

    # Load saved model
    print("Loading optimized XGBoost model...")
    model = joblib.load('best_xgb_model.joblib')
    feature_cols = joblib.load('feature_cols.joblib')

    # Time-based split (must match training split)
    unique_dates = sorted(df['date'].unique())
    split_idx = int(len(unique_dates) * config.TRAIN_RATIO)
    split_date = unique_dates[split_idx]
    test_df = df[df['date'] >= split_date]

    bt = Backtester(test_df, model, feature_cols, target_col='label')

    results_df = bt.sweep_thresholds(
        start=0.2, end=0.6, step=0.05,
        pt=config.PT_PCT, sl=config.SL_PCT,
        tc=config.TRANSACTION_COST_PCT,
        horizon=config.HORIZON_BARS
    )

    bt.run_backtest(
        threshold=0.3,
        pt=config.PT_PCT, sl=config.SL_PCT,
        transaction_cost=config.TRANSACTION_COST_PCT,
        slippage=config.SLIPPAGE_PCT,
        horizon=config.HORIZON_BARS,
        plot=True
    )


if __name__ == "__main__":
    main()
