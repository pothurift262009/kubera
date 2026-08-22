"""
═══════════════════════════════════════════════════════════════
🔧 FEATURE SELECTION — Remove Noise, Keep Signal
═══════════════════════════════════════════════════════════════
Strategy:
  1. Correlation filter: remove one of each highly-correlated pair
  2. Importance filter: drop features below min importance threshold
  3. Hard cap on max features

WHY: More features ≠ more alpha. Redundant/noisy features
     dilute the model's ability to learn real patterns and
     increase overfitting risk in financial data.
═══════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
import logging
import time
from typing import List, Optional, Any, cast

logger = logging.getLogger(__name__)


class FeatureSelector:
    """
    Production feature selector that prunes noise while preserving signal.
    """

    def __init__(self, config: dict):
        self.config = config
        self.fs_cfg = config.get("feature_selection", {})
        self.corr_threshold = self.fs_cfg.get("correlation_threshold", 0.92)
        self.min_importance_pct = self.fs_cfg.get("min_importance_pct", 0.5)
        self.max_features = self.fs_cfg.get("max_features", 40)
        self.selected_features_: Optional[List[str]] = None

    def select(
        self,
        df: pd.DataFrame,
        features: list,
        importance_df: pd.DataFrame = None,
    ) -> list:
        """
        Multi-stage feature selection pipeline.
        Returns pruned feature list.
        """
        if not self.fs_cfg.get("enabled", True):
            logger.info("🔧 Feature selection DISABLED. Using all features.")
            self.selected_features_ = features
            return features

        logger.info(f"🔧 FEATURE SELECTION: starting with {len(features)} features")
        t0 = time.time()

        current = list(features)

        # ── Stage 1: Correlation Filter ──────────────────────
        current = self._correlation_filter(df, current)

        # ── Stage 2: Importance Filter ──────────────────────
        if importance_df is not None:
            current = self._importance_filter(current, importance_df)

        # ── Stage 3: Hard Cap ───────────────────────────────
        if len(current) > self.max_features:
            if importance_df is not None:
                # Keep top-N by importance
                imp_filtered = importance_df[
                    importance_df["feature"].isin(current)
                ].sort_values("importance", ascending=False)
                current = imp_filtered["feature"].head(self.max_features).tolist()
                logger.info(
                    f"  Hard cap: {len(current)} features"
                )
            else:
                # Cast to avoid "list[Error]" hallucination
                current = cast(Any, current)[:self.max_features]

        protected_features = [
            "rank_return", "rank_volume",
            "rank_momentum", "z_score_return"
        ]

        for f in protected_features:
            if f not in current and f in df.columns:
                current.append(f)

        self.selected_features_ = current
        elapsed = time.time() - t0
        logger.info(
            f"✅ FEATURE SELECTION COMPLETE: {len(features)} → {len(current)} features "
            f"| {elapsed:.1f}s"
        )
        logger.info(f"  Selected: {current}")

        return current

    def _correlation_filter(self, df: pd.DataFrame, features: list) -> list:
        """
        Removes one feature from each pair with correlation > threshold.
        Keeps the feature with higher variance (more informative).
        """
        logger.info(
            f"  [1] Correlation filter (threshold={self.corr_threshold})..."
        )

        # Compute correlation on a sample for speed
        sample_size = min(200_000, len(df))
        sample = df[features].sample(n=sample_size, random_state=42)
        corr_matrix = sample.corr().abs()

        # Find highly correlated pairs
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        to_drop = set()
        for col in upper.columns:
            high_corr = upper.index[upper[col] > self.corr_threshold].tolist()
            for corr_col in high_corr:
                if corr_col not in to_drop:
                    # Drop the one with lower variance
                    col_var = cast(Any, sample[col]).var()
                    corr_var = cast(Any, sample[corr_col]).var()
                    if col_var > corr_var:
                        to_drop.add(corr_col)
                    else:
                        to_drop.add(col)

        remaining = [f for f in features if f not in to_drop]
        if to_drop:
            logger.info(
                f"    Removed {len(to_drop)} correlated features: "
                f"{sorted(to_drop)}"
            )
        logger.info(f"    Remaining: {len(remaining)}")

        return remaining

    def _importance_filter(
        self, features: list, importance_df: pd.DataFrame
    ) -> list:
        """
        Removes features contributing less than min_importance_pct
        of total importance.
        """
        logger.info(
            f"  [2] Importance filter (min={self.min_importance_pct}%)..."
        )

        # Filter to current feature set
        imp = importance_df[importance_df["feature"].isin(features)].copy()

        if len(imp) == 0:
            return features

        total_imp = imp["importance"].sum()
        if total_imp <= 0:
            return features

        imp["pct"] = imp["importance"] / total_imp * 100

        # Keep features above threshold
        keep = imp[imp["pct"] >= self.min_importance_pct]["feature"].tolist()
        dropped = [f for f in features if f not in keep]

        if dropped:
            logger.info(
                f"    Removed {len(dropped)} low-importance features: "
                f"{dropped}"
            )
        logger.info(f"    Remaining: {len(keep)}")

        return keep
