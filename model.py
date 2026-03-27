import pandas as pd
import numpy as np
import logging
import lightgbm as lgb
import catboost as cb
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import TimeSeriesSplit
import os

logger = logging.getLogger(__name__)

def train_elite_ensemble_v2(df: pd.DataFrame, feature_cols: list, target_col: str = 'label', n_splits: int = 3):
    """
    Elite Production Walk-Forward CV: Selective Feature Selection & Balanced Optimization.
    Trains an ensemble of LGBM + CatBoost on each fold using Top-N features only.
    """
    logger.info(f"Starting Elite Walk-Forward CV (V2) with {n_splits} splits...")
    
    # Sort chronologically for valid time-series split
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # 1. Pruning Low-Signal features (Basic Correlation Heat-map equivalent)
    valid_cols = [c for c in feature_cols if df[c].isna().mean() < 0.3]
    X, y = df[valid_cols].fillna(0), df[target_col].fillna(1).astype(int)
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Pre-select features on the first fold for stability
    train_idx, val_idx = next(tscv.split(X))
    X_f1, y_f1 = X.iloc[train_idx], y.iloc[train_idx]
    
    logger.info(f"Performing Initial Feature Selection on {len(X_f1)} samples...")
    lgb_pre = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1).fit(X_f1, y_f1)
    imp = pd.Series(lgb_pre.feature_importances_, index=valid_cols).sort_values(ascending=False)
    
    # Top 25 alpha features only (Reduce noise & overfit)
    selected_cols = imp.head(25).index.tolist()
    logger.info(f"Selected Top 25 Predictive Features: {selected_cols}")
    
    # 2. Main CV loop on selected ALPHA features
    fold_metrics, best_lgb, best_cb = [], None, None
    X_s = X[selected_cols]
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_s)):
        X_train, X_val = X_s.iloc[train_idx], X_s.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        logger.info(f"Fold {fold+1}: Train size={len(X_train)}, Val size={len(X_val)}")
        
        # 1. Balanced LightGBM (Gradient-Based One-Side Sampling)
        lgb_fold = lgb.LGBMClassifier(
            n_estimators=1000, learning_rate=0.02, num_leaves=31, 
            class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1,
            early_stopping_rounds=50
        )
        lgb_fold.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                     callbacks=[lgb.early_stopping(30, verbose=False)])
        
        # 2. Balanced CatBoost (Ordered Boosting)
        cb_fold = cb.CatBoostClassifier(
            iterations=1000, learning_rate=0.02, depth=6,
            auto_class_weights='Balanced', random_state=42, verbose=0,
            early_stopping_rounds=50
        )
        cb_fold.fit(X_train, y_train, eval_set=(X_val, y_val))
        
        # 3. Ensemble Probabilities
        p_l = lgb_fold.predict_proba(X_val); p_c = cb_fold.predict_proba(X_val)
        probs = (p_l + p_c) / 2
        preds = np.argmax(probs, axis=1)
        
        acc = accuracy_score(y_val, preds)
        fold_metrics.append(acc)
        logger.info(f"Fold {fold+1} Acc: {acc:.4f}")
        
        best_lgb, best_cb = lgb_fold, cb_fold
        
    avg_acc = np.mean(fold_metrics)
    logger.info(f"Average End-to-End CV Acc: {avg_acc:.4f}")
    
    # Save production weights
    os.makedirs('models', exist_ok=True)
    best_lgb.booster_.save_model('models/lgb_v2.txt')
    best_cb.save_model('models/cb_v2.cbm')
    
    # Importance Stability
    final_imp = pd.DataFrame({'feature': selected_cols, 'importance': best_lgb.feature_importances_}).sort_values('importance', ascending=False)
    
    return best_lgb, best_cb, final_imp, selected_cols
