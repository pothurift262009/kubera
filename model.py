import pandas as pd
import numpy as np
import logging
import lightgbm as lgb
import catboost as cb
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import TimeSeriesSplit
import os

logger = logging.getLogger(__name__)

def train_elite_ensemble(df: pd.DataFrame, feature_cols: list, target_col: str = 'label', n_splits: int = 3):
    """
    Elite Walk-Forward Cross-Validation (TimeSeriesSplit).
    Trains an ensemble of LGBM + CatBoost on each fold.
    FIXED BUG 7: NaN cleanup and updated return signature.
    """
    logger.info(f"Starting Elite Walk-Forward CV with {n_splits} splits...")
    
    # Sort by datetime to ensure chronological split
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Bug 7: Drop columns with >50% NaN and fill remaining with 0
    valid_cols = [c for c in feature_cols if df[c].isna().mean() < 0.5]
    feature_cols = valid_cols
    X = df[feature_cols].fillna(0)
    y = df[target_col].fillna(1).astype(int)
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    fold_metrics = []
    best_lgb = None
    best_cb = None
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        logger.info(f"Fold {fold+1}: Train size={len(X_train)}, Val size={len(X_val)}")
        
        # 1. LightGBM
        lgb_fold = lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.03, num_leaves=31,
            is_unbalance=True, random_state=42, n_jobs=-1, verbose=-1
        )
        lgb_fold.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                     callbacks=[lgb.early_stopping(30, verbose=False)])
        
        # 2. CatBoost
        cb_fold = cb.CatBoostClassifier(
            iterations=500, learning_rate=0.03, depth=6,
            auto_class_weights='Balanced', random_state=42, verbose=0,
            early_stopping_rounds=30
        )
        cb_fold.fit(X_train, y_train, eval_set=(X_val, y_val))
        
        # 3. Ensemble Eval
        p_l = lgb_fold.predict_proba(X_val)
        p_c = cb_fold.predict_proba(X_val)
        fold_probs = (p_l + p_c) / 2
        fold_preds = np.argmax(fold_probs, axis=1)
        
        acc = accuracy_score(y_val, fold_preds)
        fold_metrics.append(acc)
        logger.info(f"Fold {fold+1} Accuracy: {acc:.4f}")
        
        best_lgb = lgb_fold
        best_cb = cb_fold
        
    avg_acc = np.mean(fold_metrics)
    logger.info(f"Average CV Accuracy: {avg_acc:.4f}")
    
    # Feature Importance (Final Fold)
    imp = pd.DataFrame({'feature': feature_cols, 'importance': best_lgb.feature_importances_})
    
    # Save final models
    os.makedirs('models', exist_ok=True)
    best_lgb.booster_.save_model('models/lgb_elite.txt')
    best_cb.save_model('models/cb_elite.cbm')
    
    # FIXED BUG 7: return feature_cols
    return best_lgb, best_cb, imp, feature_cols
