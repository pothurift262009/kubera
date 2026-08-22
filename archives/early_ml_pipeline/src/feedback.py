import pandas as pd  # type: ignore
import numpy as np   # type: ignore
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class FeedbackLoop:
    """
    STAGE 16 (HARDENED): PERFORMANCE FEEDBACK & RISK ENGINE
    Protects capital, limits adaptation rate, provides fail-safes.
    """

    def __init__(self, config: dict):
        self.config = config
        self.window_sizes = [50, 100]  # Rolling trade windows
        self.target_precision = config.get("feedback", {}).get("target_precision", 0.55)
        self.target_expectancy = config.get("feedback", {}).get("target_expectancy_bps", 5.0) / 10000

        # Hard Risk Constraints
        self.constraints = {
            "max_base_mod": 0.10,
            "max_meta_mod": 0.15,
            "max_daily_step": 0.02,
            "min_trades_tune": 50,
            "cooldown_trades": 15,
            "hard_kill_dd": 0.10,
            "daily_loss_kill": -0.02,
            "intraday_kill_dd": -0.015,    # 1.5% max intra-day drop
            "capital_scaling_lock": 100,   # 100 trades required to scale up capital
            "max_edge_gap": 0.005          # 50 bps limit between theoretical and realized edge
        }

        # State storage
        self.trade_log = pd.DataFrame()
        self.current_adjustments = {
            "base_threshold_modifier": 0.0,
            "meta_threshold_modifier": 0.0,
            "exposure_multiplier": 1.0,
        }

        # Risk Tracker State
        self.last_tune_trade_idx = 0
        self.trades_since_scaling = 0
        self.drift_streak = 0
        self.kill_switch_active = False

    def log_trades(self, new_trades: pd.DataFrame):
        """1. TRADE LOGGING SYSTEM"""
        if new_trades.empty:
            return

        cols_to_log = [
            "date", "symbol", "close", "y_prob", "meta_prob",
            "vol_regime", "trend_regime", "future_return",
            "realized_return", "fill_ratio"
        ]
        cols = [c for c in cols_to_log if c in new_trades.columns]

        # Ensure fallback for early evaluation stages
        if "realized_return" not in cols and "future_return" in cols:
            new_trades["realized_return"] = new_trades["future_return"]
            cols.append("realized_return")

        self.trade_log = pd.concat([self.trade_log, new_trades[cols]], ignore_index=True)
        self.trade_log = self.trade_log.sort_values("date").reset_index(drop=True)

    def analyze_performance(self) -> Dict[str, Any]:
        """2. PERFORMANCE TRACKING (Rolling Windows)"""
        if len(self.trade_log) < self.constraints["min_trades_tune"]:
            return {}

        metrics = {}
        for w in self.window_sizes:
            recent = self.trade_log.tail(w)

            # Use realized return (which includes slippage, costs, and fill ratios)
            wins = (recent["realized_return"] > 0).sum()
            wr = wins / w if w > 0 else 0

            avg_win = recent[recent["realized_return"] > 0]["realized_return"].mean() if wins > 0 else 0
            avg_loss = recent[recent["realized_return"] < 0]["realized_return"].mean() if w - wins > 0 else 0

            exp = (wr * avg_win) + ((1 - wr) * avg_loss)

            cum_ret = recent["realized_return"].cumsum()
            peak = cum_ret.expanding(min_periods=1).max()
            dd = (peak - cum_ret).max()

            metrics[f"window_{w}"] = {
                "win_rate": wr, "precision": wr, "expectancy": exp, "max_dd": dd
            }
        return metrics

    def check_meta_reliability(self) -> float:
        """5. META-MODEL SAFETY: Check separation"""
        recent = self.trade_log.tail(100)
        if len(recent) < 20 or "meta_prob" not in recent.columns: return 1.0

        win_meta = recent[recent["realized_return"] > 0]["meta_prob"].mean()
        loss_meta = recent[recent["realized_return"] <= 0]["meta_prob"].mean()
        separation = win_meta - loss_meta

        if np.isnan(separation) or separation <= 0: return 0.0
        return np.clip(separation / 0.05, 0.1, 1.0)

    def detect_feature_drift(self) -> bool:
        """7. DRIFT CONFIRMATION: Check for probability distribution drift"""
        if len(self.trade_log) < 100: return False

        # Compare median probs of last 30 vs previous 70
        p50_recent = np.median(self.trade_log.tail(30)["y_prob"])
        p50_base = np.median(self.trade_log.iloc[-100:-30]["y_prob"])

        drift = abs(p50_recent - p50_base)
        return bool(drift > 0.10)

    def check_execution_drift(self) -> float:
        """1. EXPECTED VS REALIZED SPREAD (EDGE GAP)"""
        recent = self.trade_log.tail(100)
        if len(recent) < 50 or "realized_return" not in recent.columns: return 0.0

        edge_gap = recent["future_return"].mean() - recent["realized_return"].mean()
        if edge_gap > self.constraints["max_edge_gap"]:
            logger.critical(f"🤬 CRITICAL: Execution Degradation Detected! Edge gap: {edge_gap*10000:.1f} bps")
        return edge_gap

    def check_daily_kill_switch(self, yesterday_trades: pd.DataFrame) -> bool:
        """8. HARD EXPOSURE LIMITS: Evaluate daily kill switch"""
        if yesterday_trades.empty: return False

        # Calculate daily net inclusive of all friction
        if "realized_return" in yesterday_trades.columns:
            daily_pnl = yesterday_trades["realized_return"].sum()
        else:
            daily_pnl = yesterday_trades["future_return"].sum()

        if daily_pnl <= self.constraints["daily_loss_kill"]:
            logger.error(f"🚨 KILL SWITCH ACTIVATED! Yesterday's Realized Loss: {daily_pnl*100:.2f}%. Trading paused.")
            return True
        return False

    def calculate_adjustments(self) -> Dict[str, float]:
        """CONSTRAINED ADAPTATION & SELF-CORRECTING LOGIC"""
        metrics = self.analyze_performance()
        if not metrics: return self.current_adjustments

        # 3. COOLDOWN PERIOD
        trades_since_tune = len(self.trade_log) - self.last_tune_trade_idx
        if trades_since_tune < self.constraints["cooldown_trades"]:
            return self.current_adjustments # Freeze parameters

        w50 = metrics.get("window_50", {})
        w100 = metrics.get("window_100", {})

        # 4. METRIC STABILITY CHECK
        if w100 and abs(w50.get("win_rate", 0) - w100.get("win_rate", 0)) > 0.15:
            logger.warning("📉 WARNING: METRIC INSTABILITY (>15% WR Variance). Freezing adaptation.")
            return self.current_adjustments

        # 4. BASELINE FALLBACK MODE
        if w50.get("max_dd", 0.0) >= self.constraints["hard_kill_dd"]:
            logger.critical(f"🤬 CRITICAL DRAWDOWN ({w50['max_dd']*100:.1f}%). REVERTING TO BASELINE.")
            self.current_adjustments = {"base_threshold_modifier": 0.0, "meta_threshold_modifier": 0.0, "exposure_multiplier": 0.5}
            self.last_tune_trade_idx = len(self.trade_log)
            return self.current_adjustments

        # Store old values for logging
        old_base = self.current_adjustments["base_threshold_modifier"]
        old_meta = self.current_adjustments["meta_threshold_modifier"]

        # Determine Steps (Bounded)
        step = self.constraints["max_daily_step"]

        # 7. DRIFT CONFIRMATION & EXECUTION DRIFT OVERRIDE
        edge_gap = self.check_execution_drift()
        if self.detect_feature_drift() or edge_gap > self.constraints["max_edge_gap"]:
            self.drift_streak += 1
            if self.drift_streak >= 3:
                logger.critical("🚨 CRITICAL: PERSISTENT DRIFT CONFIRMED. Forcing Exposure Reduction.")
                self.current_adjustments["exposure_multiplier"] = max(0.25, self.current_adjustments["exposure_multiplier"] - 0.25)
                self.drift_streak = 0 # reset after action
                self.trades_since_scaling = 0
        else:
            self.drift_streak = 0

        # 1. Base Strategy tuning
        if w50.get("precision", 1.0) < self.target_precision:
            self.current_adjustments["base_threshold_modifier"] += step
        elif w50.get("expectancy", 0.0) > self.target_expectancy and len(self.trade_log.tail(15)) < 3:
             self.current_adjustments["base_threshold_modifier"] -= step

        # 6. Smooth Meta Degradation
        meta_rel = self.check_meta_reliability()
        if meta_rel < 0.5:
            self.current_adjustments["meta_threshold_modifier"] -= step # Gradual!
            logger.warning("📉 Meta-model unreliable. Gradually reducing reliance.")
        elif w50.get("max_dd", 0.0) > 0.02:
            self.current_adjustments["meta_threshold_modifier"] += step # Demand more conviction

        # 1. APPLY ABSOLUTE CONSTRAINTS
        self.current_adjustments["base_threshold_modifier"] = np.clip(
            self.current_adjustments["base_threshold_modifier"],
            -self.constraints["max_base_mod"], self.constraints["max_base_mod"]
        )
        self.current_adjustments["meta_threshold_modifier"] = np.clip(
            self.current_adjustments["meta_threshold_modifier"],
            -self.constraints["max_meta_mod"], self.constraints["max_meta_mod"]
        )

        # Exposure Logic & Capital Scaling Lock
        self.trades_since_scaling += trades_since_tune

        if w50.get("expectancy", 1.0) < 0:
            self.current_adjustments["exposure_multiplier"] = max(0.25, self.current_adjustments["exposure_multiplier"] - 0.25)
            self.trades_since_scaling = 0
            if self.current_adjustments["exposure_multiplier"] <= 0.5:
                 logger.critical(f"🛡️ CRITICAL REDUCTION: Negative expectancy. Exposure cut to {self.current_adjustments['exposure_multiplier']}x")
        elif self.trades_since_scaling >= self.constraints["capital_scaling_lock"]:
            if w50.get("win_rate", 0) > 0.52 and w50.get("expectancy", 0) > 0:
                old_exp = self.current_adjustments["exposure_multiplier"]
                self.current_adjustments["exposure_multiplier"] = min(1.0, old_exp + 0.25)
                if self.current_adjustments["exposure_multiplier"] > old_exp:
                    logger.info(f"📈 INFO: CAPITAL SCALE UNLOCKED. Increasing Exposure to {self.current_adjustments['exposure_multiplier']}x")
                    self.trades_since_scaling = 0

        # 9. LOGGING
        if old_base != self.current_adjustments["base_threshold_modifier"] or \
           old_meta != self.current_adjustments["meta_threshold_modifier"]:
            logger.info(f"⚙️ PARAMETER UPDATE: Base Mod: {old_base:.2f}→{self.current_adjustments['base_threshold_modifier']:.2f} | "
                        f"Meta Mod: {old_meta:.2f}→{self.current_adjustments['meta_threshold_modifier']:.2f} | "
                        f"Exp: {self.current_adjustments['exposure_multiplier']}x")
            self.last_tune_trade_idx = len(self.trade_log)

        return self.current_adjustments

    def simulate_live_trading(self, test_df: pd.DataFrame, y_prob: np.ndarray, y_meta_prob: np.ndarray, evaluator) -> pd.DataFrame:
        """10. INTEGRATION: Run day-by-day HARDENED simulation."""
        logger.info("\n" + "═" * 60)
        logger.info("🛡️ STAGE 16/17: HARDENED RISK & REALISTIC EXECUTION LOOP INITIALIZED")
        logger.info("═" * 60)

        from src.execution import ExecutionSimulator  # type: ignore
        executor = ExecutionSimulator(self.config)

        test_df = test_df.copy()
        test_df["y_prob_base"] = y_prob
        test_df["y_prob_meta"] = y_meta_prob
        test_df = test_df.sort_values("date")

        daily_groups = test_df.groupby(test_df["date"].dt.date)
        all_signals = []
        base_thresh = 0.85
        yesterday_trades = pd.DataFrame()

        for current_date, day_df in daily_groups:
            # Check Daily Kill Switch
            if self.check_daily_kill_switch(yesterday_trades):
                yesterday_trades = pd.DataFrame() # Reset for tomorrow
                continue # Skip trading today entirely

            adj = self.calculate_adjustments()
            current_base_q = min(0.95, max(0.60, base_thresh + adj["base_threshold_modifier"])) # Hard-coded floor 60

            daily_signals = evaluator.multi_stage_filter(
                day_df,
                day_df["y_prob_base"].values,
                base_threshold_percentile=current_base_q,
                y_meta_prob=day_df["y_prob_meta"].values
            )

            # Apply exposure cut
            if not daily_signals.empty and adj["exposure_multiplier"] < 1.0:
                 keep_count = int(len(daily_signals) * adj["exposure_multiplier"])
                 if keep_count < len(daily_signals):
                     daily_signals = daily_signals.nlargest(max(1, keep_count), "final_score")

            # 🚀 EXECUTE ORDERS VIA SIMULATOR 🚀
            if not daily_signals.empty:
                daily_signals = executor.simulate_execution(daily_signals)
                daily_signals = daily_signals[daily_signals["fill_ratio"] > 0]

                # INTRADAY KILL SWITCH (Real-time drawdown check)
                if not daily_signals.empty and "realized_return" in daily_signals.columns:
                    daily_cum_pnl = daily_signals["realized_return"].cumsum()
                    if (daily_cum_pnl <= self.constraints["intraday_kill_dd"]).any():
                        # Find exactly where the breach occurred and truncate the day
                        breach_idx = (daily_cum_pnl <= self.constraints["intraday_kill_dd"]).idxmax()
                        daily_signals = daily_signals.loc[:breach_idx]
                        logger.critical(f"💥 CRITICAL: INTRADAY KILL SWITCH TRIPPED. Breached {self.constraints['intraday_kill_dd']*100}% loss midway through session.")
                        # It will automatically skip the next day due to yesterday_trades checking the total daily return

            yesterday_trades = daily_signals
            if not daily_signals.empty:
                all_signals.append(daily_signals)
                self.log_trades(daily_signals)

        if not all_signals:
            return pd.DataFrame()

        final_df = pd.concat(all_signals, ignore_index=True)
        logger.info(f"✅ HARDENED SIMULATION COMPLETE. Final filled trade count: {len(final_df)}")

        # Output final execution benchmark report
        executor.execution_report(final_df)

        return final_df
