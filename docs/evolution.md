# Evolution of Kubera

Kubera began as a set of practical intraday trading experiments and developed through three distinct questions.

## 1. Can transparent rules generate disciplined signals?

The initial work focused on opening-range breakout, VWAP reversion, and gap momentum ideas. The archived `rule_based_v3` and `rule_based_v10` modules show the progression from signal rules to dashboard, paper-trading, broker, risk, and backtesting components.

## 2. Can the feature set support model-driven ranking?

The early ML pipeline introduced data cleaning, cross-sectional features, regime detection, feature selection, LightGBM models, threshold exploration, and result persistence. It remains an archive because its evaluation and selection logic needs a clean, frozen validation protocol before its results can be treated as evidence.

`projects/intraday_ml_research` is the cleaner next iteration: it resamples minute data to five-minute bars, builds technical and relative-strength features, creates ATR-based labels, trains a calibrated XGBoost classifier, and generates research backtests.

## 3. Can order-book information add short-horizon context?

`projects/lob_microstructure_ml` extends the research to level-1 through level-5 order-book data. It computes microprice, order-flow imbalance, queue imbalance, depth, spread, and momentum features before training an ensemble.

## 4. Can research outputs be surfaced safely?

`projects/kite_execution_platform` separates the execution concern from model research. It contains an opening-range-breakout simulator, a Streamlit UI, position sizing, a SQLite trade ledger, and Kite adapter boundaries. Its live components should be treated as development code and used only after an independent paper-trading and risk review.
