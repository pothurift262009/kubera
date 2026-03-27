import logging
import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

# Import Elite modules
from data_loader import load_ohlcv, merge_with_lob_asof
from feature_engineering import run_feature_pipeline_elite
from lob_processing import process_lob_elite
from labeling import apply_triple_barrier_elite
from model import train_elite_ensemble
from backtest import run_backtest_elite

# Configure Elite logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("elite_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Elite_NSE_Pipeline")

def main():
    # PATHS
    KB_OHLCV_PATH = '/Users/Pothuri/Downloads/kubera/KB_OP.csv.gz'
    KB_LOB_PATH = '/Users/Pothuri/Downloads/kubera/kblobop.csv.gz'
    LOB_5M_PARQUET = '/Users/Pothuri/Downloads/kubera/lob_5m_elite.parquet'
    
    # CONFIG
    LABEL_CONFIG = {'tp_mult': 1.5, 'sl_mult': 1.0, 'max_bars': 6}
    BT_CONFIG = {'cost': 0.0005, 'slippage': 0.0002, 'prob_threshold': 0.55}
    
    # 1. PROCESS LOB (Elite)
    if not os.path.exists(LOB_5M_PARQUET):
        logger.info("Initializing Elite LOB processing...")
        process_lob_elite(KB_LOB_PATH, LOB_5M_PARQUET)
    else:
        logger.info(f"Using existing Elite LOB features from {LOB_5M_PARQUET}")

    # 2. LOAD OHLCV
    df_ohlcv = load_ohlcv(KB_OHLCV_PATH)
    
    # 3. FEATURE ENGINEERING (Elite)
    df_features = run_feature_pipeline_elite(df_ohlcv)
    
    # 4. HIGH-FIDELITY MERGE (merge_asof)
    logger.info("Reading LOB parquet and synchronizing...")
    df_lob = pd.read_parquet(LOB_5M_PARQUET)
    
    # FIXED BUG 5: Robust datetime logic for LOB parquet
    df_lob['datetime'] = pd.to_datetime(df_lob['datetime'])
    if df_lob['datetime'].dt.tz is None:
        df_lob['datetime'] = df_lob['datetime'].dt.tz_localize('Asia/Kolkata')
    else:
        df_lob['datetime'] = df_lob['datetime'].dt.tz_convert('Asia/Kolkata')
    
    df_combined = merge_with_lob_asof(df_features, df_lob)
    
    # 5. ELITE LABELING (Triple Barrier)
    df_labeled = apply_triple_barrier_elite(df_combined, config=LABEL_CONFIG)
    
    # 6. ELITE MODELING (Walk-Forward CV)
    # Filter features: everything except metadata and target
    feature_cols = [c for c in df_labeled.columns if c not in {'symbol', 'datetime', 'label'}]
    
    # FIXED BUG 7: Call signature update
    best_lgb, best_cb, imp, feature_cols = train_elite_ensemble(
        df_labeled, feature_cols, n_splits=3)
    
    # 7. ELITE BACKTESTING
    # Use last 20% as holdout for backtest
    split_idx = int(0.8 * len(df_labeled))
    df_test = df_labeled.iloc[split_idx:]
    
    # Ensemble predictions on test set
    p_l = best_lgb.predict_proba(df_test[feature_cols])
    p_c = best_cb.predict_proba(df_test[feature_cols])
    probs = (p_l + p_c) / 2
    preds = np.argmax(probs, axis=1)
    
    # Final Model Metrics
    logger.info("\n=== HOLD-OUT CLASSIFICATION REPORT ===")
    logger.info("\n" + classification_report(df_test['label'], preds))
    
    bt_df, portfolio_rets, metrics = run_backtest_elite(
        df_test, preds, probs, 
        cost=BT_CONFIG['cost'], 
        slippage=BT_CONFIG['slippage'], 
        prob_threshold=BT_CONFIG['prob_threshold']
    )
    
    logger.info("Elite Pipeline Execution Finished Successfully.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Elite Pipeline encountered a fatal error: {e}", exc_info=True)
