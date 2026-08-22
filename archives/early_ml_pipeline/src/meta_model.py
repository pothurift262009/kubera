"""
═══════════════════════════════════════════════════════════════
🧠 META-MODEL v1 — The "Is it Profitable?" Engine
═══════════════════════════════════════════════════════════════
This module implements the Layer-2 Meta-Model that predicts the
probability of a trade candidate being profitable, given the
Base Model signal and market context.
"""

import pandas as pd  # type: ignore
import numpy as np   # type: ignore
import logging
import os
import joblib  # type: ignore
from typing import Optional, cast
import lightgbm as lgb  # type: ignore
from sklearn.model_selection import TimeSeriesSplit  # type: ignore
from sklearn.metrics import precision_score, roc_auc_score  # type: ignore
from sklearn.calibration import CalibratedClassifierCV  # type: ignore
try:
    from sklearn.frozen import FrozenEstimator # type: ignore
except ImportError:
    FrozenEstimator = None

logger = logging.getLogger(__name__)

class MetaModelTrainer:
    """
    Trains a second-level model to filter out false positives from Level 1.
    Target: 1 if future_return > cost, else 0.
    """

    def __init__(self, config: dict):
        self.config = config
        self.meta_cfg = config.get("meta_model", {})
        self.model: Optional[CalibratedClassifierCV] = None
        self.features = [
            "prob_base", "rank_base", "rel_strength",
            "volatility_20", "z_score_return",
            "rank_return", "rank_volume", "rank_momentum"
        ]
        self.calib_mean_p: Optional[float] = None
        self.calib_std_p: Optional[float] = None
        # Include regime if available
        if config["features"].get("use_regime_features", False):
            self.features.extend(["vol_regime", "trend_regime", "regime_score"])

    def create_meta_dataset(self, df: pd.DataFrame, y_prob: np.ndarray) -> pd.DataFrame:
        """
        Transforms base dataset + predictions into a meta-dataset.
        Only rows that pass the initial base selection are candidates.
        """
        meta_df = df.copy()
        meta_df["prob_base"] = y_prob

        # Add rank within timestamp (cross-sectional)
        meta_df["rank_base"] = meta_df.groupby("date")["prob_base"].rank(pct=True, ascending=False)

        # Meta Target: Is it actually profitable given strict Stage 17 cost bounds?
        # Since src/features.py already rigorously computes this in class 'target', we reuse it directly
        # to ensure unified learning objectives across Layer 1 and Layer 2.
        if "target" in meta_df.columns:
            meta_df["meta_target"] = meta_df["target"].astype(np.int8)
        else:
            cost = self.config.get("costs", {}).get("total_round_trip", 0.0006)
            k = 0.5
            vol_hurdle = (meta_df["volatility_20"] * k).clip(upper=0.015) if "volatility_20" in meta_df.columns else 0.002
            meta_df["meta_target"] = (meta_df["future_return"] > (cost + vol_hurdle)).astype(np.int8)

        # Filter to candidates that logically COULD be trades (e.g. prob > 0.4)
        # We train on a broad set of candidates to help the model learn the boundary
        base_threshold = self.meta_cfg.get("candidate_threshold", 0.45)
        meta_df = meta_df[meta_df["prob_base"] >= base_threshold].copy()

        logger.info(f"🧬 Meta-dataset created: {len(meta_df):,} candidates from {len(df):,} total rows.")
        return meta_df

    def train(self, meta_df: pd.DataFrame):
        """Trains the Meta-Model using a time-series split."""
        X = meta_df[self.features].fillna(0).replace([np.inf, -np.inf], 0)
        y = meta_df["meta_target"]

        # Time-series split
        n_splits = self.meta_cfg.get("n_splits", 3)
        tscv = TimeSeriesSplit(n_splits=n_splits)

        aucs = []

        # Base LightGBM with explicit imbalance weighting
        base_estimator = lgb.LGBMClassifier(
            n_estimators=self.meta_cfg.get("n_estimators", 150),
            max_depth=self.meta_cfg.get("max_depth", 5),
            learning_rate=0.05,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Fit calibrator which inherently fits the base estimator
            base_estimator.fit(X_train, y_train)
            if FrozenEstimator is not None:
                fold_model = CalibratedClassifierCV(FrozenEstimator(base_estimator), method='isotonic')
            else:
                fold_model = CalibratedClassifierCV(base_estimator, method='isotonic', cv='prefit')
            fold_model.fit(X_val, y_val)

            y_prob_meta = fold_model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_prob_meta)
            aucs.append(auc)
            logger.info(f"  Fold {fold+1}: Meta AUC = {auc:.4f}")

        logger.info(f"📊 Meta-Model CV AUC: {np.mean(aucs):.4f} (±{np.std(aucs):.4f})")

        # Final fit on all meta-data, wrapping in calibration (we use 20% of data for platt scaling)
        split_idx = int(len(X) * 0.8)
        X_t, X_v = X.iloc[:split_idx], X.iloc[split_idx:]
        y_t, y_v = y.iloc[:split_idx], y.iloc[split_idx:]

        base_estimator.fit(X_t, y_t)
        if FrozenEstimator is not None:
            self.model = CalibratedClassifierCV(FrozenEstimator(base_estimator), method='isotonic')
        else:
            self.model = CalibratedClassifierCV(base_estimator, method='isotonic', cv='prefit')
        self.model.fit(X_v, y_v)

        # Save Calibration baseline for distribution shift alerts
        y_prob_meta = self.model.predict_proba(X_v)[:, 1]
        self.calib_mean_p = float(np.mean(y_prob_meta))
        self.calib_std_p = float(np.std(y_prob_meta))
        logger.info(f"⚖️ Meta-Model Baseline Probs → Mean: {self.calib_mean_p:.3f} | Std: {self.calib_std_p:.3f}")

        # Importance (from the base LightGBM estimator)
        feat_imp = pd.Series(base_estimator.feature_importances_, index=self.features).sort_values(ascending=False)
        logger.info("🔥 Meta-Feature Importance:\n" + feat_imp.to_string())

        return self.model

    def predict(self, df: pd.DataFrame, y_prob: np.ndarray) -> np.ndarray:
        """Generates meta-probabilities for the given dataset."""
        if self.model is None:
            raise ValueError("Meta-Model not trained or loaded.")

        temp_df = df.copy()
        temp_df["prob_base"] = y_prob
        temp_df["rank_base"] = temp_df.groupby("date")["prob_base"].rank(pct=True, ascending=False)

        X = temp_df[self.features].fillna(0).replace([np.inf, -np.inf], 0)
        model = cast(CalibratedClassifierCV, self.model)
        meta_probs = model.predict_proba(X)[:, 1]

        # ⚠️ CALIBRATION STABILITY CHECK
        calib_mean = self.calib_mean_p
        if calib_mean is not None:
            curr_mean = float(np.mean(meta_probs))
            if abs(curr_mean - calib_mean) > 0.10:
                logger.warning(
                    f"⚠️ CALIBRATION SHIFT ALERT: OOS Meta Prob Mean ({curr_mean:.3f}) "
                    f"deviated >10% from Training Baseline ({calib_mean:.3f}). "
                    f"Recalibration via retraining highly recommended to maintain Execution Model integrity."
                )

        return meta_probs

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f"💾 Meta-Model saved to {path}")

    def load(self, path: str):
        if os.path.exists(path):
            self.model = joblib.load(path)
            logger.info(f"📂 Meta-Model loaded from {path}")
        else:
            logger.warning(f"⚠️ Meta-Model file not found at {path}")
