import logging
import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

# Import Elite V12 High-Conviction Alpha modules
from data_loader import load_ohlcv, merge_with_lob_asof
from feature_engineering import run_feature_pipeline_elite_v2
from lob_processing import process_lob_elite_v2
from labeling import apply_triple_barrier_elite_v8
from model import train_elite_ensemble_v2
from backtest import run_backtest_high_conviction_v12 # Fix: Concentration Transition

# Configure Alpha Production Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("alpha_pipeline_v12.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Alpha_NSE_Pipeline_V12")

def main():
    # PATHS
    KB_OHLCV_PATH = '/Users/Pothuri/Downloads/kubera/KB_OP.csv.gz'
    KB_LOB_PATH = '/Users/Pothuri/Downloads/kubera/kblobop.csv.gz'
    LOB_V2_PARQUET = '/Users/Pothuri/Downloads/kubera/lob_v2_elite.parquet'
    
    # CONFIG (V12 HIGH-CONVICTION TRANSITION)
    LABEL_CONFIG = {'tp_mult': 2.2, 'sl_mult': 1.0, 'max_bars': 6}
    BT_CONFIG = {
        'cost': 0.0002, 
        'slippage': 0.0001, 
        'top_n': 3,                 # Keep only top 3 ranked stocks per timestamp
        'min_hold_bars': 3,          
        'spread_filter_threshold': 0.001 # Fix 4: Tighten microstructure
    }
    
    # 1. DATA PREP
    if not os.path.exists(LOB_V2_PARQUET):
        logger.info("Initializing Alpha LOB (V2) processing...")
        process_lob_elite_v2(KB_LOB_PATH, LOB_V2_PARQUET)
    else:
        logger.info(f"Using existing LOB V2 features from {LOB_V2_PARQUET}")

    df_ohlcv = load_ohlcv(KB_OHLCV_PATH)
    df_features = run_feature_pipeline_elite_v2(df_ohlcv)
    
    logger.info("Reading LOB V2 features and synchronizing...")
    df_lob = pd.read_parquet(LOB_V2_PARQUET)
    df_lob['datetime'] = pd.to_datetime(df_lob['datetime'])
    if df_lob['datetime'].dt.tz is None:
        df_lob['datetime'] = df_lob['datetime'].dt.tz_localize('Asia/Kolkata')
    else:
        df_lob['datetime'] = df_lob['datetime'].dt.tz_convert('Asia/Kolkata')
    
    df_combined = merge_with_lob_asof(df_features, df_lob)
    df_labeled = apply_triple_barrier_elite_v8(df_combined, config=LABEL_CONFIG)
    
    # 2. CALIBRATED TRAINING
    feature_cols = [c for c in df_labeled.columns if c not in {'symbol', 'datetime', 'label'}]
    best_lgb, best_cb, imp, selected_features = train_elite_ensemble_v2(
        df_labeled, feature_cols, n_splits=3)
    
    # 3. HOLD-OUT INFERENCE (20%)
    split_idx = int(0.8 * len(df_labeled))
    df_test = df_labeled.iloc[split_idx:]
    
    p_l = best_lgb.predict_proba(df_test[selected_features])
    p_c = best_cb.predict_proba(df_test[selected_features])
    probs = (p_l + p_c) / 2
    
    # 4. HIGH-CONVICTION BACKTESTING V12 (Pure Alpha Concentration)
    bt_df, portfolio_rets, metrics = run_backtest_high_conviction_v12(
        df_test, probs, 
        cost=BT_CONFIG['cost'], 
        slippage=BT_CONFIG['slippage'], 
        top_n=BT_CONFIG['top_n'],
        min_hold_bars=BT_CONFIG['min_hold_bars'],
        spread_filter_threshold=BT_CONFIG['spread_filter_threshold']
    )
    
    # FINAL METRIC VERIFICATION
    final_sharpe = metrics['Sharpe']
    final_pf = metrics['Profit Factor']
    num_trades = len(bt_df[bt_df['pos_sign'].diff().fillna(0) != 0])
    
    logger.info(f"V12 HIGH-CONVICTION SUMMARY: Sharpe={final_sharpe:.2f}, PF={final_pf:.2f}, N_Trades={num_trades}")
    logger.info("Alpha Pipeline V12 execution finished.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Alpha Pipeline V12 fatal error: {e}", exc_info=True)
