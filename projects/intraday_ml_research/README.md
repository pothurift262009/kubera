# Intraday ML Research

This iteration evaluates whether technical, volume, opening-range, and cross-sectional features can rank short-horizon NSE long opportunities.

## Pipeline

1. Load one-minute OHLCV data and resample to five-minute bars.
2. Generate per-symbol technical and intraday features.
3. Add cross-sectional ranks at each timestamp.
4. Create volatility-adaptive triple-barrier labels.
5. Train an XGBoost classifier on chronological splits.
6. Score the holdout period and pass candidates to the research backtester.

## Entry points

- `main.py`: research pipeline.
- `sanity_check.py`: backtester behavior check.
- `analyze_backtest.py`: exploratory analysis script.

The source is preserved from the original experiment. Review the root validation notes before interpreting results; it needs a frozen, session-aware walk-forward evaluation before it should be used to make performance claims.
