import logging
import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

# Import Elite V7 modules
from data_loader import load_ohlcv, merge_with_lob_asof
from feature_engineering import run_feature_pipeline_elite_v2
from lob_processing import process_lob_elite_v2
from labeling import apply_triple_barrier_elite_v5
from model import train_elite_ensemble_v2
from backtest import run_backtest_elite_v7

# Configure Production Elite Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("elite_pipeline_v7.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Elite_NSE_Pipeline_V7")

def main():
    # PATHS
    KB_OHLCV_PATH = '/Users/Pothuri/Downloads/kubera/KB_OP.csv.gz'
    KB_LOB_PATH = '/Users/Pothuri/Downloads/kubera/kblobop.csv.gz'
    LOB_V2_PARQUET = '/Users/Pothuri/Downloads/kubera/lob_v2_elite.parquet'
    
    # CONFIG (V7 ELITE STRATEGY)
    # Target: Sharpe > 1.3, PF > 1.2
    LABEL_CONFIG = {'tp_mult': 2.0, 'sl_mult': 1.0, 'max_bars': 6}
    BT_CONFIG = {
        'cost': 0.0002, 
        'slippage': 0.0001, 
        'long_threshold': 0.58,      # Relaxed from 0.60
        'short_threshold': 0.58,     # Relaxed from 0.62
        'conf_gap_threshold': 0.10,  # Relaxed from 0.12
        'strength_percentile': 0.15, # Relaxed from 0.25
        'min_hold_bars': 3,          # Maintain V6 Logic (Persistence)
        'vol_filter_quantile': 0.4,  # Maintain V6 Logic (Regime)
        'spread_filter_threshold': 0.0012 # Maintain V6 Logic (Spread)
    }
    
    # 1. PEFORMANCE CHECK
    if not os.path.exists(LOB_V2_PARQUET):
        logger.info("Initializing Elite LOB (V2) processing...")
        process_lob_elite_v2(KB_LOB_PATH, LOB_V2_PARQUET)
    else:
        logger.info(f"Using existing LOB V2 features from {LOB_V2_PARQUET}")

    # 2. LOAD DATA
    df_ohlcv = load_ohlcv(KB_OHLCV_PATH)
    df_features = run_feature_pipeline_elite_v2(df_ohlcv)
    
    # 3. HIGH-FIDELITY SYNCHRONIZATION
    logger.info("Reading LOB V2 features and synchronizing...")
    df_lob = pd.read_parquet(LOB_V2_PARQUET)
    df_lob['datetime'] = pd.to_datetime(df_lob['datetime'])
    if df_lob['datetime'].dt.tz is None:
        df_lob['datetime'] = df_lob['datetime'].dt.tz_localize('Asia/Kolkata')
    else:
        df_lob['datetime'] = df_lob['datetime'].dt.tz_convert('Asia/Kolkata')
    
    df_combined = merge_with_lob_asof(df_features, df_lob)
    
    # 4. TRIPLE BARRIER LABELING (Asymmetric payload)
    df_labeled = apply_triple_barrier_elite_v5(df_combined, config=LABEL_CONFIG)
    
    # 5. ELITE MODELING (TimeSeriesSplit Walk-Forward CV)
    feature_cols = [c for c in df_labeled.columns if c not in {'symbol', 'datetime', 'label'}]
    best_lgb, best_cb, imp, selected_features = train_elite_ensemble_v2(
        df_labeled, feature_cols, n_splits=3)
    
    # 6. ENSEMBLE EVALUATION (Last 20% Hold-out)
    split_idx = int(0.8 * len(df_labeled))
    df_test = df_labeled.iloc[split_idx:]
    
    # Probability Prediction
    p_l = best_lgb.predict_proba(df_test[selected_features])
    p_c = best_cb.predict_proba(df_test[selected_features])
    probs = (p_l + p_c) / 2
    
    # 7. ELITE BACKTESTING V7 (QUALITY FILTERS)
    bt_df, portfolio_rets, metrics = run_backtest_elite_v7(
        df_test, probs, 
        cost=BT_CONFIG['cost'], 
        slippage=BT_CONFIG['slippage'], 
        long_threshold=BT_CONFIG['long_threshold'],
        short_threshold=BT_CONFIG['short_threshold'],
        conf_gap_threshold=BT_CONFIG['conf_gap_threshold'],
        strength_percentile=BT_CONFIG['strength_percentile'],
        min_hold_bars=BT_CONFIG['min_hold_bars'],
        vol_filter_quantile=BT_CONFIG['vol_filter_quantile'],
        spread_filter_threshold=BT_CONFIG['spread_filter_threshold']
    )
    
    # VERIFICATION
    final_sharpe = metrics['Sharpe']
    final_pf = metrics['Profit Factor']
    num_trades = len(bt_df[bt_df.groupby('symbol')['pos_sign'].diff().fillna(0) != 0])
    
    logger.info(f"V7 ELITE SUMMARY: Sharpe={final_sharpe:.2f}, PF={final_pf:.2f}, N_Trades={num_trades}")
    logger.info("Elite Pipeline V7 Execution Finished Successfully.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Elite Pipeline V7 fatal error: {e}", exc_info=True)
