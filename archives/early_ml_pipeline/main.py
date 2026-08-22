"""
═══════════════════════════════════════════════════════════════
🚀 MAIN v2 — Production ML Pipeline Orchestrator
═══════════════════════════════════════════════════════════════
Pipeline stages:
  1. Data Ingestion
  2. Preprocessing
  3. Feature Engineering v2 (persistence, cross-sectional ranks)
  4. Regime Detection
  5. Target Labeling (path-consistent)
  6. Feature Selection (correlation + importance pruning)
  7. Model Training (LightGBM + LR ensemble, walk-forward CV)
  8. Probability Diagnostics
  9. PnL-Based Threshold Optimization
 10. Multi-Stage Signal Filtering
 11. Cost-Aware PnL Analysis
 12. Results Persistence

Usage:
  python main.py --config config.yaml
  python main.py --config config.yaml --skip-cache
═══════════════════════════════════════════════════════════════
"""

import yaml  # type: ignore
import logging
import os
import sys
import time
import argparse
import pandas as pd  # type: ignore
import numpy as np   # type: ignore

from src.data_loader import DataLoader  # type: ignore
from src.preprocessing import Preprocessor  # type: ignore
from src.features import FeatureEngineer  # type: ignore
from src.regime import RegimeDetector  # type: ignore
from src.feature_selection import FeatureSelector  # type: ignore
from src.model import ModelTrainer  # type: ignore
from src.evaluation import Evaluator  # type: ignore
from src.meta_model import MetaModelTrainer  # type: ignore
from src.feedback import FeedbackLoop  # type: ignore


def setup_logging(config: dict) -> logging.Logger:
    log_dir = config["data"]["log_dir"]
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"pipeline_v2_{time.strftime('%Y%m%d_%H%M%S')}.log")

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    root.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    logger = logging.getLogger("PIPELINE")
    logger.info(f"{'═' * 60}")
    logger.info("🚀 PRODUCTION ML PIPELINE v2 — SIGNAL QUALITY FOCUS")
    logger.info(f"   Log: {log_file}")
    logger.info(f"{'═' * 60}")
    return logger


def run_pipeline(config_path: str, skip_cache: bool = False):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger = setup_logging(config)
    t_start = time.time()

    os.makedirs(config["data"]["model_dir"], exist_ok=True)
    os.makedirs(config["data"].get("results_dir", "results"), exist_ok=True)

    try:
        # ═════════════════════════════════════════════════════
        # STAGE 1: DATA INGESTION
        # ═════════════════════════════════════════════════════
        logger.info("─" * 50)
        logger.info("📂 STAGE 1: DATA INGESTION")
        logger.info("─" * 50)

        loader = DataLoader(config)

        df_ready = None

        if not skip_cache:
            cached_df = loader.load_processed()
            if cached_df is not None:
                logger.info("⚡ Using cached v2 processed data")
                df_ready = cached_df
                engineer = FeatureEngineer(config)
                features = engineer.get_feature_list(df_ready)
                logger.info(f"  Features: {len(features)}")
                if "target" not in df_ready.columns:
                    logger.warning("Cache missing target. Re-processing...")
                    skip_cache = True
                    df_ready = None

        if skip_cache or df_ready is None:
            raw_df = loader.load_raw_csv()
            logger.info(f"  Dataset: {len(raw_df):,} rows × {raw_df['symbol'].nunique()} symbols")

            # ═════════════════════════════════════════════════
            # STAGE 2: PREPROCESSING
            # ═════════════════════════════════════════════════
            logger.info("─" * 50)
            logger.info("🏗️ STAGE 2: PREPROCESSING")
            logger.info("─" * 50)

            preprocessor = Preprocessor(config)
            cleaned_df = preprocessor.process(raw_df)
            del raw_df

            # ═════════════════════════════════════════════════
            # STAGE 3: FEATURE ENGINEERING v2
            # ═════════════════════════════════════════════════
            logger.info("─" * 50)
            logger.info("⚡ STAGE 3: FEATURE ENGINEERING v2")
            logger.info("─" * 50)

            engineer = FeatureEngineer(config)
            df_features = engineer.create_features(cleaned_df)
            del cleaned_df

            # ═════════════════════════════════════════════════
            # STAGE 4: REGIME DETECTION
            # ═════════════════════════════════════════════════
            logger.info("─" * 50)
            logger.info("🌊 STAGE 4: REGIME DETECTION")
            logger.info("─" * 50)

            regime_detector = RegimeDetector(config)
            df_features = regime_detector.detect_regimes(df_features)

            # ═════════════════════════════════════════════════
            # STAGE 5: TARGET LABELING
            # ═════════════════════════════════════════════════
            logger.info("─" * 50)
            logger.info("🎯 STAGE 5: TARGET LABELING (path-consistent)")
            logger.info("─" * 50)

            df_ready = engineer.create_targets(df_features)
            features = engineer.get_feature_list(df_ready)

            logger.info(f"  Final: {len(df_ready):,} rows | {len(features)} features")
            del df_features

            # Cache
            loader.save_processed(df_ready)

        # ═════════════════════════════════════════════════════
        # STAGE 6: INITIAL MODEL (for feature importance)
        # ═════════════════════════════════════════════════════
        logger.info("─" * 50)
        logger.info("🧠 STAGE 6: INITIAL MODEL TRAINING")
        logger.info("─" * 50)

        trainer = ModelTrainer(config)
        model, test_df = trainer.train(df_ready, features)

        # Get initial predictions for feature selection scoring
        importance_df = trainer.log_importance(top_n=20)

        # ═════════════════════════════════════════════════════
        # STAGE 7: FEATURE SELECTION
        # ═════════════════════════════════════════════════════
        logger.info("─" * 50)
        logger.info("🔧 STAGE 7: FEATURE SELECTION")
        logger.info("─" * 50)

        selector = FeatureSelector(config)
        selected_features = selector.select(df_ready, features, importance_df)

        # ═════════════════════════════════════════════════════
        # STAGE 8: RETRAIN WITH SELECTED FEATURES
        # ═════════════════════════════════════════════════════
        if len(selected_features) < len(features):
            logger.info("─" * 50)
            logger.info("🧠 STAGE 8: RETRAIN WITH SELECTED FEATURES")
            logger.info("─" * 50)

            trainer2 = ModelTrainer(config)
            model, test_df = trainer2.train(df_ready, selected_features)
            importance_df = trainer2.log_importance(top_n=20)
            trainer = trainer2
            features = selected_features
        else:
            logger.info("  No features pruned. Skipping retrain.")

        # Predictions on holdout
        y_prob = trainer.predict(test_df, features)
        y_true = test_df["target"].values
        future_returns = test_df["future_return"].values if "future_return" in test_df.columns else None

        logger.info(f"  Holdout: {len(test_df):,} rows | Pos: {y_true.sum():,} ({y_true.mean() * 100:.2f}%)")

        # Save model
        model_path = os.path.join(config["data"]["model_dir"], "trading_model_v2.lgb")
        trainer.save_model(model_path)

        # ═════════════════════════════════════════════════════
        # STAGE 9: PROBABILITY DIAGNOSTICS
        # ═════════════════════════════════════════════════════
        logger.info("─" * 50)
        logger.info("📊 STAGE 9: PROBABILITY DIAGNOSTICS")
        logger.info("─" * 50)

        evaluator = Evaluator(config)
        evaluator.probability_diagnostics(y_prob)

        # ═════════════════════════════════════════════════════
        # STAGE 10: PNL-BASED THRESHOLD OPTIMIZATION
        # ═════════════════════════════════════════════════════
        logger.info("─" * 50)
        logger.info("🔍 STAGE 10: PNL-BASED THRESHOLD OPTIMIZATION")
        logger.info("─" * 50)

        sweep_df = evaluator.threshold_sweep(y_true, y_prob, future_returns)
        optimal = evaluator.find_optimal_threshold(sweep_df)
        optimal_threshold = optimal["threshold"]

        # ═════════════════════════════════════════════════════
        # STAGE 11: SIGNAL EVALUATION
        # ═════════════════════════════════════════════════════
        logger.info("─" * 50)
        logger.info("📈 STAGE 11: SIGNAL EVALUATION")
        logger.info("─" * 50)

        eval_metrics = evaluator.evaluate_at_threshold(y_true, y_prob, optimal_threshold)

        # ═════════════════════════════════════════════════════
        # STAGE 12: MULTI-STAGE SIGNAL FILTERING
        # ═════════════════════════════════════════════════════
        logger.info("─" * 50)
        logger.info("🎛️ STAGE 12: MULTI-STAGE SIGNAL FILTERING")
        logger.info("─" * 50)

        filtered_signals = evaluator.multi_stage_filter(
            test_df, y_prob, base_threshold_percentile=0.85
        )

        # ═════════════════════════════════════════════════════
        # STAGE 13: COST-AWARE PNL ANALYSIS (BASE)
        # ═════════════════════════════════════════════════════
        logger.info("─" * 50)
        logger.info("💰 STAGE 13: COST-AWARE PNL ANALYSIS (BASE)")
        logger.info("─" * 50)
        pnl_metrics_base = evaluator.pnl_analysis(filtered_signals)

        # ═════════════════════════════════════════════════════
        # STAGE 14: META-MODEL (LAYER 2) — THE PROFIT ENGINE
        # ═════════════════════════════════════════════════════
        logger.info("─" * 50)
        logger.info("🧠 STAGE 14: META-MODEL TRAINING (LAYER 2)")
        logger.info("─" * 50)

        meta_trainer = MetaModelTrainer(config)

        # Isolate independent training block without the test holdout
        train_df = df_ready[~df_ready.index.isin(test_df.index)].copy()  # pyre-ignore

        from sklearn.model_selection import TimeSeriesSplit  # pyre-ignore
        import lightgbm as lgb  # pyre-ignore

        tscv_meta = TimeSeriesSplit(n_splits=3)  # pyre-ignore
        oos_preds = np.zeros(len(train_df))

        logger.info("  Generating OOS base predictions on Train block for Meta calibration...")
        for train_idx, val_idx in tscv_meta.split(train_df):
            X_tr, y_tr = train_df.iloc[train_idx][selected_features], train_df.iloc[train_idx]["target"]
            X_va, y_va = train_df.iloc[val_idx][selected_features], train_df.iloc[val_idx]["target"]
            d_tr = lgb.Dataset(X_tr, label=y_tr)
            d_va = lgb.Dataset(X_va, label=y_va, reference=d_tr)

            mdl = lgb.train({'objective': 'binary', 'metric': 'auc', 'verbosity': -1, 'learning_rate': 0.05, 'num_leaves': 31, 'bagging_fraction': 0.8},
                            d_tr, valid_sets=[d_tr, d_va], callbacks=[lgb.early_stopping(30, verbose=False)])
            oos_preds[val_idx] = mdl.predict(X_va)

        first_val_idx = next(tscv_meta.split(train_df))[1][0]
        meta_train_subset = train_df.iloc[first_val_idx:].copy()
        meta_oos_probs = oos_preds[first_val_idx:]

        # Train Meta exclusively on OOS
        meta_dataset = meta_trainer.create_meta_dataset(meta_train_subset, meta_oos_probs)
        logger.info(f"  Training meta-model on {len(meta_dataset):,} OOS trade candidates...")
        meta_trainer.train(meta_dataset)

        # Predict Meta Probs cleanly for the unseen evaluate set
        y_meta_prob = meta_trainer.predict(test_df, y_prob)
        meta_trainer.save(config["meta_model"].get("model_path", "models/meta_model.joblib"))

        # ═════════════════════════════════════════════════════
        # STAGE 15: PRODUCTION TRADE SELECTION (LAYER 3)
        # ═════════════════════════════════════════════════════
        logger.info("─" * 50)
        logger.info("🎯 STAGE 15: PRODUCTION TRADE SELECTION")
        logger.info("─" * 50)

        # Apply calibration if requested
        if config.get("calibrate_probs", False):
            y_prob = evaluator.calibrate_probs(y_true, y_prob, method="isotonic")

        # ═════════════════════════════════════════════════════
        # STAGE 16: PERFORMANCE FEEDBACK LOOP (Layer 4)
        # ═════════════════════════════════════════════════════
        logger.info("─" * 50)
        logger.info("♻️ STAGE 16: ADAPTIVE PERFORMANCE FEEDBACK LOOP")
        logger.info("─" * 50)

        feedback_engine = FeedbackLoop(config)
        final_signals = feedback_engine.simulate_live_trading(
            test_df, y_prob, y_meta_prob, evaluator
        )

        # Final PnL Analysis
        pnl_metrics = evaluator.pnl_analysis(final_signals)
        filtered_signals = final_signals # Update for exports

        # ═════════════════════════════════════════════════════
        # SAVE RESULTS
        # ═════════════════════════════════════════════════════
        results_dir = config["data"].get("results_dir", "results")

        sweep_df.to_csv(os.path.join(results_dir, "threshold_sweep_v2.csv"), index=False)
        if importance_df is not None:
            importance_df.to_csv(os.path.join(results_dir, "feature_importance_v2.csv"), index=False)
        if len(filtered_signals) > 0:
            save_cols = [c for c in ["date", "symbol", "close", "y_prob", "target",
                                      "future_return", "vol_regime", "trend_regime",
                                      "regime_score"] if c in filtered_signals.columns]
            filtered_signals[save_cols].to_csv(
                os.path.join(results_dir, "filtered_signals_v2.csv"), index=False
            )

        logger.info(f"  Results saved to {results_dir}/")

        # ═════════════════════════════════════════════════════
        # FINAL SUMMARY
        # ═════════════════════════════════════════════════════
        elapsed = (time.time() - t_start) / 60

        logger.info(f"\n{'═' * 60}")
        logger.info("✅ PIPELINE v2 COMPLETE — SUMMARY")
        logger.info(f"{'═' * 60}")
        logger.info(f"  Duration:           {elapsed:.2f} minutes")
        logger.info(f"  Dataset:            {len(df_ready):,} rows")  # type: ignore

        logger.info(f"  Features (final):   {len(features)}")
        logger.info(f"  Holdout:            {len(test_df):,} rows")
        logger.info(f"  Optimal Threshold:  {optimal_threshold:.3f}")
        logger.info(f"  Precision:          {eval_metrics['precision']:.4f}")
        logger.info(f"  Recall:             {eval_metrics['recall']:.4f}")
        logger.info(f"  F1:                 {eval_metrics['f1']:.4f}")
        logger.info(f"  Signals (raw):      {eval_metrics['n_signals']:,}")

        if filtered_signals is not None and len(filtered_signals) > 0:
            logger.info(f"  Signals (filtered): {len(filtered_signals):,}")
        if pnl_metrics:
            logger.info(f"  Win Rate:           {pnl_metrics.get('win_rate_pct', 0):.2f}%")
            logger.info(f"  Profit Factor:      {pnl_metrics.get('profit_factor_raw', 0):.3f} (raw) → {pnl_metrics.get('profit_factor_net', 0):.3f} (net)")
            logger.info(f"  Expectancy:         {pnl_metrics.get('expectancy_raw_pct', 0):.4f}% → {pnl_metrics.get('expectancy_net_pct', 0):.4f}% (net)")
            logger.info(f"  Max Drawdown:       {pnl_metrics.get('max_drawdown_pct', 0):.3f}%")
            logger.info(f"  Sharpe Ratio:       {pnl_metrics.get('sharpe_ratio', 0):.2f}")
        logger.info(f"  Model:              {model_path}")
        logger.info(f"{'═' * 60}")

        # ═════════════════════════════════════════════════════
        # MOMENTUM STRATEGY EXECUTION
        # ═════════════════════════════════════════════════════
        from src.strategies.momentum_strategy import run_momentum_strategy

        momentum_df = df_ready.copy()

        # Create forward returns
        momentum_df["f_ret_1"] = momentum_df.groupby("symbol")["close"].shift(-1) / momentum_df["close"] - 1
        momentum_df["f_ret_2"] = momentum_df.groupby("symbol")["close"].shift(-2) / momentum_df["close"] - 1
        momentum_df["f_ret_3"] = momentum_df.groupby("symbol")["close"].shift(-3) / momentum_df["close"] - 1

        run_momentum_strategy(momentum_df)

    except Exception as e:
        logger.exception(f"❌ CRITICAL FAILURE: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NIFTY50 Alpha Pipeline v2")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--skip-cache", action="store_true")
    args = parser.parse_args()
    run_pipeline(args.config, skip_cache=args.skip_cache)
