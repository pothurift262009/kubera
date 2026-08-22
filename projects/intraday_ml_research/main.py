from data_loader import DataProcessor
from feature_engineer import FeatureEngineer
from labeler import Labeler
from model_trainer import ModelTrainer
from backtester import Backtester
import config
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('kubera_train.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    # ── File check ─────────────────────────────────────────────
    if not os.path.exists(config.DATA_FILE):
        print(f"File {config.DATA_FILE} not found.")
        return

    # ── 1. Load & Resample ────────────────────────────────────
    dp = DataProcessor(config.DATA_FILE)
    df = dp.load_data()
    df = dp.resample_to_5min()

    # ── 2. Feature Engineering ────────────────────────────────
    fe = FeatureEngineer(df)
    df = fe.generate_features()
    df = fe.rank_features()

    # ── 3. Labeling (ATR-based) ───────────────────────────────
    lb = Labeler(df)
    df = lb.atr_barrier_labeling(
        atr_multiplier_pt=3.0,
        atr_multiplier_sl=1.5,
        horizon=config.HORIZON_BARS
    )

    # ── 4. Model Training ─────────────────────────────────────
    feature_cols = FeatureEngineer.get_feature_list()
    mt = ModelTrainer(df, feature_cols, target_col='label')
    test_df = mt.train()
    mt.save_model('best_xgb_model.joblib')
    mt.get_feature_importance()


    # ── 6. Backtesting ────────────────────────────────────────
    bt = Backtester(test_df, mt.model, feature_cols, target_col='label')

    # ── Probability diagnostics ───────────────────────────────
    probs = bt.df['prob']
    print(f"\n--- Probability Distribution ---")
    print(f"Min: {probs.min():.4f}")
    print(f"25th: {probs.quantile(0.25):.4f}")
    print(f"50th: {probs.quantile(0.50):.4f}")
    print(f"75th: {probs.quantile(0.75):.4f}")
    print(f"90th: {probs.quantile(0.90):.4f}")
    print(f"95th: {probs.quantile(0.95):.4f}")
    print(f"99th: {probs.quantile(0.99):.4f}")
    print(f"Max: {probs.max():.4f}")
    print(f"> 0.20: {(probs > 0.20).sum()}")
    print(f"> 0.30: {(probs > 0.30).sum()}")
    print(f"> 0.50: {(probs > 0.50).sum()}")

    # ── Auto threshold sweep range ────────────────────────────
    p90 = round(probs.quantile(0.90), 2)
    p99 = round(probs.quantile(0.99), 2)

    sweep_start = max(0.10, p90 - 0.05)
    sweep_end = min(0.95, p99 + 0.10)
    sweep_step = max(round((sweep_end - sweep_start) / 8, 2), 0.02)

    print(f"\nAuto sweep range: {sweep_start} → {sweep_end} (step {sweep_step})")

    # ── Direct threshold (quantile-based) ─────────────────────
    threshold = probs.quantile(0.90)
    print(f"\nUsing threshold: {threshold:.2f}")

    bt.run_backtest(
        threshold=threshold,
        pt=config.PT_PCT,
        sl=config.SL_PCT,
        transaction_cost=config.TRANSACTION_COST_PCT,
        slippage=config.SLIPPAGE_PCT,
        horizon=config.HORIZON_BARS,
        plot=True
    )

    print("\nPipeline complete. Model ready.")



if __name__ == "__main__":
    main()