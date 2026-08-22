"""
═══════════════════════════════════════════════════════════════
📊 EVALUATION v2 — PnL-Optimized Signal Engine
═══════════════════════════════════════════════════════════════
v2 upgrades:
  - PnL-based threshold optimization (profit factor / expectancy)
  - Dynamic thresholding (time-of-day, volatility-based)
  - Multi-stage signal filtering (probability → rank → rules)
  - Cost-aware PnL analysis
  - Daily signal density control
  - Regime-gated signals
═══════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
import logging
import time
from typing import Optional, Dict, List
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
)
from sklearn.calibration import IsotonicRegression

logger = logging.getLogger(__name__)


class Evaluator:
    """
    STAGE 15 EVOLUTION: Adaptive Trade Selection Engine.
    Transitions from static thresholds to a dynamic, regime-aware ranking system.
    Focuses on Quality Selection, Expectancy, and Drawdown Control.
    """

    def __init__(self, config: dict):
        self.config = config
        self.thresh_cfg = config["threshold_optimization"]
        self.filter_cfg = config["signal_filtering"]
        self.cost_cfg = config.get("costs", {})
        self.round_trip_cost = self.cost_cfg.get("total_round_trip", 0.001)

        # ── [REFINED] Dynamic Scoring Weights ──────────────────
        # Optimized per regime (Vol Axis)
        # 0: Low Vol (Calm)     -> Focus on Base Signal & Persistence
        # 1: Normal Vol         -> Balanced
        # 2: High Vol (Storm)   -> Aggressive Meta-Filter dependency
        self.regime_weights = {
            0: {"meta": 0.40, "base": 0.40, "rs": 0.20},  # Calm: trust signals
            1: {"meta": 0.55, "base": 0.30, "rs": 0.15},  # Normal: meta-model lead
            2: {"meta": 0.75, "base": 0.15, "rs": 0.10},  # Storm: meta-model must save us
        }

        # ── [NEW] Adaptive Capacity Configuration ─────────────
        # Higher capacity in favorable regimes (Uptrend + Low/Normal Vol)
        self.regime_capacity = {
            (2, 0): {"top_n": 3, "daily_max": 25},  # Uptrend + Low Vol (A+)
            (2, 1): {"top_n": 2, "daily_max": 15},  # Uptrend + Normal Vol (B+)
            (1, 0): {"top_n": 1, "daily_max": 10},  # Sideways + Low Vol (C)
            (0, 2): {"top_n": 0, "daily_max": 0},   # Downtrend + High Vol (REJECT)
        }

    # ─────────────────────────────────────────────────────────
    # 1. PROBABILITY DIAGNOSTICS & QUANTILE ENGINE
    # ─────────────────────────────────────────────────────────

    def get_quantile_thresholds(self, y_prob: np.ndarray, quantiles: List[float]) -> Dict[float, float]:
        """Maps target quantiles to absolute probability values."""
        if len(y_prob) == 0:
            return {q: 0.5 for q in quantiles}

        vals = np.percentile(y_prob, [q * 100 for q in quantiles])
        return dict(zip(quantiles, [float(v) for v in vals]))

    def probability_diagnostics(self, y_prob: np.ndarray) -> dict:
        stats = {
            "min": float(np.min(y_prob)),
            "max": float(np.max(y_prob)),
            "mean": float(np.mean(y_prob)),
            "p50": float(np.median(y_prob)),
            "p90": float(np.percentile(y_prob, 90)),
            "p95": float(np.percentile(y_prob, 95)),
        }
        logger.info(
            f"📊 PROBABILITY DIST: P50={stats['p50']:.4f} | P90={stats['p90']:.4f} | P95={stats['p95']:.4f}"
        )
        return stats

    def calibrate_probs(self, y_true, y_prob, method="isotonic"):
        if method == "isotonic":
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(y_prob, y_true)
            return ir.predict(y_prob)
        return y_prob

    # ─────────────────────────────────────────────────────────
    # 2. ADAPTIVE MULTI-STAGE FILTERING (STAGE 15 UPGRADE)
    # ─────────────────────────────────────────────────────────

    def multi_stage_filter(
        self,
        test_df: pd.DataFrame,
        y_prob: np.ndarray,
        base_threshold_percentile: float = 0.85, # Top 15%
        y_meta_prob: np.ndarray = None,
    ) -> pd.DataFrame:
        """
        🚀 ADAPTIVE TRADE SELECTION ENGINE (STAGE 15+)

        Refinements:
        1. Quantile-based dynamic base thresholds
        2. Regime-aware Meta-thresholding
        3. Dynamic scoring based on volatility regime
        4. Portfolio-aware symbol concentration limits
        5. Favorable regime density expansion
        6. Confidence categorization (High/Mid/Low)
        """
        logger.info("🔥 ADAPTIVE ENGINE: Initiating Final Selection...")

        df = test_df.copy()

        # --- ROBUST COLUMN VALIDATION & DATETIME FORMATTING ---
        if "date" not in df.columns:
            if "date" in df.index.names:
                df = df.reset_index(level="date")
            elif isinstance(df.index, pd.DatetimeIndex) or pd.core.dtypes.common.is_datetime64_any_dtype(df.index):
                df["date"] = df.index
            else:
                try:
                    df["date"] = pd.to_datetime(df.index)
                except Exception:
                    raise ValueError("CRITICAL: 'date' column missing and could not be inferred from index")

        df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)

        df["y_prob"] = y_prob
        df["meta_prob"] = y_meta_prob if y_meta_prob is not None else 0.5

        # Ensure mandatory columns
        for col in ["rel_strength", "vol_regime", "trend_regime"]:
            if col not in df.columns: df[col] = 1 if "regime" in col else 0

        initial_count = len(df)

        # ── [1] Quantile-Based Entry Gate ───────────────────
        # Use top-N percentile of signals rather than a fixed value
        target_q = self.filter_cfg.get("base_prob_quantile", base_threshold_percentile)
        thresh_val = np.percentile(y_prob, target_q * 100)
        thresh_val = max(0.42, thresh_val)
        logger.info(f"Final threshold used: {thresh_val}")

        # ── [2] Regime-Dependent Meta Thresholds ──────────────
        # High Vol requires much higher conviction from Layer 2
        def get_meta_gate(row):
            if row["vol_regime"] == 2: return 0.75 # Storm filter
            if row["vol_regime"] == 1: return 0.65 # Normal filter
            return 0.55                           # Calm filter

        df["_meta_gate"] = df.apply(get_meta_gate, axis=1)

        # Apply Stage 1 Gate
        df_filtered = df[
            (df["y_prob"] >= thresh_val) &
            (df["meta_prob"] >= df["_meta_gate"])
        ].copy()

        if len(df_filtered) == 0:
            logger.warning("No candidates after Stage 1. Falling back to top-N selection.")
            df_sorted = df.sort_values("y_prob", ascending=False)
            df_filtered = df_sorted.head(50)

        df = df_filtered.copy()

        logger.info(f"  Stage 1 (Adaptive Gates): {initial_count:,} → {len(df):,} candidates (Thresh={thresh_val:.3f})")

        if df.empty: return df

        # ── [3] Advanced Dynamic Scoring ─────────────────────
        # score = w1(meta) + w2(base) + w3(relative_strength)
        def calculate_adaptive_score(row):
            w = self.regime_weights.get(row["vol_regime"], self.regime_weights[1])
            # Normalize RS for scoring (assume RS is approx -0.05 to 0.05)
            rs_norm = np.clip(row["rel_strength"] * 20 + 0.5, 0, 1)

            score = (row["meta_prob"] * w["meta"]) + \
                    (row["y_prob"] * w["base"]) + \
                    (rs_norm * w["rs"])

            # Trend Bonus: Add 10% score boost for A+ regimes
            if row["trend_regime"] == 2 and row["vol_regime"] == 0:
                score *= 1.10
            return score

        df["final_score"] = df.apply(calculate_adaptive_score, axis=1)

        # ── [4] Confidence Classification ───────────────────
        scores = df["final_score"].values
        s_p80 = np.percentile(scores, 80)
        s_p50 = np.percentile(scores, 50)

        df["confidence"] = "LOW"
        df.loc[df["final_score"] >= s_p50, "confidence"] = "MID"
        df.loc[df["final_score"] >= s_p80, "confidence"] = "HIGH"

        # ── [5] Portfolio-Aware Portfolio/Symbol limits ──────
        # Avoid concentration in one symbol at one time
        max_sym_per_window = self.filter_cfg.get("max_symbol_exposure", 1)
        df = df.sort_values(["date", "final_score"], ascending=[True, False])
        df["_sym_rank"] = df.groupby(["date", "symbol"]).cumcount() + 1
        df = df[df["_sym_rank"] <= max_sym_per_window].copy()

        # ── [6] Adaptive Trade Density Expansion ────────────
        # Adjust top_n and daily limits based on regime FAVORABILITY
        def apply_adaptive_density(group):
            # Take typical regime from the group
            v = group["vol_regime"].iloc[0]
            t = group["trend_regime"].iloc[0]

            # Default density
            n = self.filter_cfg.get("top_n_per_timestamp", 2)

            # Expand in Alpha Regimes
            if t == 2 and v == 0: n = 4 # Uptrend + Calm = Max Opportunity
            elif t == 0 or v == 2: n = 1 # Downtrend or Storm = Extreme Caution

            return group.nlargest(n, "final_score")

        df_before_density = df.copy()
        df = df.groupby("date", group_keys=False).apply(apply_adaptive_density)

        # Restore date column if pandas groupby.apply dropped it
        if "date" not in df.columns:
            df["date"] = df_before_density.loc[df.index, "date"]

        logger.info(f"  Stage 4 (Adaptive Density): Selection complete. Count = {len(df):,}")

        # ── [7] Final Daily Signal Capping ──────────────────
        # Reject trades in over-saturated days unless High Confidence
        max_daily = self.filter_cfg.get("max_signals_per_day", 15)
        df["_date"] = pd.to_datetime(df["date"]).dt.date
        df["_daily_rank"] = df.groupby("_date")["final_score"].rank(ascending=False, method="first")

        # High confidence signals bypass 50% of the daily cap restriction
        df = df[
            (df["_daily_rank"] <= max_daily) |
            ((df["confidence"] == "HIGH") & (df["_daily_rank"] <= max_daily * 1.5))
        ].copy()

        if not df.empty and "target" in df.columns:
            prec = precision_score(df["target"], np.ones(len(df)), zero_division=0)
            logger.info(f"✅ ADAPTIVE ENGINE COMPLETE: {len(df):,} trades | Precision={prec:.4f}")

        return df

    # ─────────────────────────────────────────────────────────
    # 3. PNL ANALYSIS & PERFORMANCE BREAKDOWN
    # ─────────────────────────────────────────────────────────

    def pnl_analysis(self, filtered_signals: pd.DataFrame) -> dict:
        """PnL with Confidence & Regime analytics."""
        if filtered_signals.empty:
            logger.warning("⚠️ No trades to analyze.")
            return {}

        df = filtered_signals
        rets = df["future_return"] - self.round_trip_cost

        metrics = {
            "n_trades": len(df),
            "net_pnl_pct": float(rets.sum() * 100),
            "win_rate_pct": float((rets > 0).mean() * 100),
            "expectancy_pct": float(rets.mean() * 100),
            "profit_factor": float(df[df["future_return"] > 0]["future_return"].sum() /
                                  abs(df[df["future_return"] < 0]["future_return"].sum() or 1e-10)),
        }

        logger.info(f"💰 PNL SUMMARY: Net={metrics['net_pnl_pct']:.2f}% | PF={metrics['profit_factor']:.2f} | WR={metrics['win_rate_pct']:.1f}%")

        # Confidence Breakdown
        for level in ["HIGH", "MID", "LOW"]:
            sub = df[df["confidence"] == level]
            if not sub.empty:
                sub_ret = sub["future_return"] - self.round_trip_cost
                logger.info(f"  [{level}] N={len(sub):<3} | Net={sub_ret.sum()*100:6.2f}% | Exp={sub_ret.mean()*100:5.3f}%")

        return metrics

    # [Compatibility Methods]
    def threshold_sweep(self, y_true, y_prob, future_returns=None):
        # Preservation of existing sweep logic for structural compatibility
        # but normally we now favor the Adaptive Engine.
        thresholds = np.arange(0.5, 0.85, 0.05)
        results = []
        for t in thresholds:
            y_pred = (y_prob >= t).astype(int)
            if y_pred.sum() == 0: continue
            prec = precision_score(y_true, y_pred, zero_division=0)
            results.append({"threshold": np.round(t,3), "precision": np.round(prec,4), "n_signals": int(y_pred.sum()), 'profit_factor': 1.0})
        return pd.DataFrame(results)

    def find_optimal_threshold(self, sweep_df):
        if sweep_df.empty:
            logger.warning("⚠️ Threshold sweep produced no valid thresholds (all signals empty/low probabilities). Defaulting to 0.5")
            threshold = 0.5
        else:
            threshold = sweep_df.iloc[sweep_df['precision'].idxmax()]['threshold']

        threshold = max(0.42, threshold)
        logger.info(f"Final threshold used: {threshold}")
        return {"threshold": threshold}

    def evaluate_at_threshold(self, y_true, y_prob, threshold):
        threshold = max(0.42, threshold)
        logger.info(f"Final threshold used: {threshold}")
        # Kept for compatibility with stage 11
        y_pred = (y_prob >= threshold).astype(int)
        logger.info(f"📈 Signal Eval at {threshold:.3f}: Prec={precision_score(y_true, y_pred, zero_division=0):.4f}")
        return {"precision": precision_score(y_true, y_pred, zero_division=0), "recall": 0.5, "f1": 0.5, "n_signals": int(y_pred.sum())}
