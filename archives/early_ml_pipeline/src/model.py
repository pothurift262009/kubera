import pandas as pd  # type: ignore
import numpy as np   # type: ignore
import os
import logging
import time
from typing import Optional, cast
import lightgbm as lgb  # type: ignore
from sklearn.model_selection import TimeSeriesSplit  # type: ignore
from sklearn.metrics import roc_auc_score, precision_score  # type: ignore
from sklearn.calibration import IsotonicRegression  # type: ignore
from src.utils.memory import log_memory, clear_memory  # type: ignore

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Production ensemble trainer optimized for 8GB RAM.
    Uses LightGBM with efficient data handling.
    """

    def __init__(self, config: dict):
        self.config = config
        self.model_cfg = config["model"]
        self.train_cfg = config["training"]
        self.params = dict(self.model_cfg["params"])
        # Force evaluation metric for probability ranking
        self.params["metric"] = "auc"

        self.lgb_model: Optional[lgb.Booster] = None
        self.calibrator: Optional[IsotonicRegression] = None
        self.feature_importance_ = None

    def train(self, df: pd.DataFrame, features: list, target: str = "target"):
        """Main training pipeline with memory profiling."""
        strategy = self.train_cfg.get("validation_strategy", "walk_forward")
        logger.info(f"🧠 MODEL TRAINING (Optimized): strategy={strategy}")
        log_memory("Pre-training")

        # Optional: Sampling for faster/low-mem iterations
        if self.train_cfg.get("use_sampling", False):
            sample_ratio = self.train_cfg.get("sample_ratio", 0.5)
            logger.info(f"  ⚠️ SAMPLING ENABLED: Using {sample_ratio*100:.1f}% of data")
            df = df.sample(frac=sample_ratio, random_state=42).sort_values("date")
            clear_memory("Post-sampling")

        if strategy == "walk_forward":
            return self._train_walk_forward(df, features, target)
        else:
            raise NotImplementedError("Single split training is not implemented in this optimized version.")

    def _train_walk_forward(self, df, features, target):
        df.sort_values("date", inplace=True)
        n_splits = self.train_cfg.get("n_splits", 3)
        purge_days = self.train_cfg.get("purge_gap", 6)

        # 2. PURGED DATE-BASED WALK-FORWARD (Institutional standard)
        dates = df["date"].dt.date.unique()
        dates.sort()

        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_metrics = []

        for fold, (train_date_idx, val_date_idx) in enumerate(tscv.split(dates)):
            if purge_days > 0:
                train_date_idx = train_date_idx[:-purge_days]

            train_dates = dates[train_date_idx]
            val_dates = dates[val_date_idx]

            logger.info(f"  ── Fold {fold + 1} Training ({len(train_dates)} days train, {len(val_dates)} days val)...")

            # Map back to full DF safely preventing mid-day symbol leakage
            train_fold = df[df["date"].dt.date.isin(train_dates)]
            val_fold = df[df["date"].dt.date.isin(val_dates)]

            metrics = self._fit(train_fold, val_fold, features, target,
                                fold_label=f"Fold-{fold+1}")
            fold_metrics.append(metrics)
            clear_memory(f"Post-Fold-{fold+1}")

        # Final model on full training set using chronological split
        split_date_idx = int(len(dates) * self.train_cfg.get("train_ratio", 0.7))
        train_dates_final = dates[:split_date_idx]
        test_dates_final = dates[split_date_idx:]

        train_final = df[df["date"].dt.date.isin(train_dates_final)]
        test_final = df[df["date"].dt.date.isin(test_dates_final)]

        logger.info(f"  🏁 Training FINAL model...")
        self._fit(train_final, test_final, features, target, fold_label="FINAL")

        return self.lgb_model, test_final

    def _fit(self, train_df, val_df, features, target, fold_label=""):
        """Memory-efficient fitting."""
        # 1. Extract and Downcast
        X_train = train_df[features].astype(np.float32)
        y_train = train_df[target].astype(np.int8)
        logger.info(f"Target mean (train): {y_train.mean():.4f}")
        X_val = val_df[features].astype(np.float32)
        y_val = val_df[target].astype(np.int8)

        log_memory(f"{fold_label} Data Extracted")

        # 2. Extract Sample Weights for Ranking-Aware Optimization
        w_train = np.abs(train_df["future_return"].values).astype(np.float32)
        w_val = np.abs(val_df["future_return"].values).astype(np.float32)

        # 3. LightGBM Dataset with custom sample weights
        d_train = lgb.Dataset(X_train, label=y_train, weight=w_train, free_raw_data=True,)
        d_val = lgb.Dataset(X_val, label=y_val, weight=w_val, reference=d_train, free_raw_data=True,)

        # Explicitly delete original DataFrames and Series
        del X_train, X_val
        clear_memory(f"{fold_label} DataFrames Deleted")

        params = dict(self.params)
        if self.train_cfg.get("use_scale_pos_weight", True):
            n_neg = np.sum(y_train == 0)
            n_pos = np.sum(y_train == 1)
            params["scale_pos_weight"] = n_neg / max(n_pos, 1)

        self.lgb_model = lgb.train(
            params, d_train,
            valid_sets=[d_val],
            valid_names=["valid"],
            num_boost_round=self.train_cfg.get("num_boost_round", 1000),
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=200)],
        )

        # 4. Predict & Cleanup
        y_prob = self.lgb_model.predict(val_df[features].astype(np.float32))

        # PROBABILITY CALIBRATION (Fit on holdout OOS split)
        if fold_label == "FINAL":
            logger.info("  ⚖️ Calibrating probabilities via Isotonic Regression...")
            self.calibrator = IsotonicRegression(out_of_bounds="clip")
            self.calibrator.fit(y_prob, y_val)
            y_prob_calib = self.calibrator.predict(y_prob)  # type: ignore
            y_prob_eval = y_prob_calib
        else:
            y_prob_eval = y_prob

        auc = roc_auc_score(y_val, y_prob_eval)
        # Assuming classification decision boundary ~0.7 after class balancing
        y_pred_bin = (y_prob_eval > 0.7).astype(np.int8)
        prec = precision_score(y_val, y_pred_bin, zero_division=0)

        logger.info(f"  [{fold_label}] AUC={auc:.4f} | Est. Precision(0.7)={prec:.3f}")

        self.feature_importance_ = pd.DataFrame({
            "feature": features,
            "importance": self.lgb_model.feature_importance(importance_type="gain"),
        }).sort_values("importance", ascending=False)

        # Per-regime profiling (efficient version)
        if "vol_regime" in val_df.columns:
            for reg in val_df["vol_regime"].unique():
                mask = (val_df["vol_regime"].values == reg)
                if y_val.values[mask].sum() > 0:
                    reg_auc = roc_auc_score(y_val.values[mask], y_prob[mask])
                    logger.info(f"    [{fold_label}] Vol Regime {reg} AUC: {reg_auc:.4f}")

        return {"auc": auc}

    def predict(self, df: pd.DataFrame, features: list) -> np.ndarray:
        """Memory-efficient calibrated prediction."""
        if self.lgb_model is None:
            raise ValueError("Model not trained.")
        X = df[features].astype(np.float32)
        model = cast(lgb.Booster, self.lgb_model)

        y_prob = model.predict(X).astype(np.float32)

        # Apply Isotonic Calibration to fix LightGBM scale_pos_weight distortion
        if self.calibrator is not None:
            y_prob = self.calibrator.predict(y_prob).astype(np.float32)  # type: ignore

        return y_prob

    def save_model(self, path: str):
        if self.lgb_model is not None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            model = cast(lgb.Booster, self.lgb_model)
            model.save_model(path)
            logger.info(f"💾 Model saved → {path}")

    def log_importance(self, top_n=20):
        import pandas as pd

        if not hasattr(self, "lgb_model") or self.lgb_model is None:
            return None

        importance = self.lgb_model.feature_importance()
        features = self.lgb_model.feature_name()

        df = pd.DataFrame({
            "feature": features,
            "importance": importance
        }).sort_values("importance", ascending=False)

        print("\nTop Features:")
        print(df.head(top_n))

        return df
