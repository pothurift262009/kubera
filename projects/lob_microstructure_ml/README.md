# LOB Microstructure ML Research

This iteration combines OHLCV with level-1 to level-5 limit-order-book observations to investigate short-horizon intraday classification.

## Components

- `lob_processing.py`: chunked LOB processing and five-minute resampling.
- `feature_engineering.py`: technical, regime, momentum, and time features.
- `labeling.py`: asymmetric triple-barrier labels.
- `model.py`: LightGBM/CatBoost ensemble experiment.
- `backtest.py`: cross-sectional capital-allocation backtest.

`main.py` contains original local dataset paths and is retained as experiment history. Replace them with configuration before running it elsewhere.
