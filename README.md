# Kubera

**An evolving intraday-equity research portfolio for the Indian market.**

Kubera documents a progression from transparent rule-based strategies, through machine-learning research, to paper/live execution interfaces. It is a software and research portfolio—not investment advice and not a claim of profitability.

## Project evolution

| Stage | Location | What it demonstrates |
| --- | --- | --- |
| Rule-based strategies | `archives/rule_based_v3`, `archives/rule_based_v10` | VWAP mean reversion, gap momentum, historical backtesting, paper-trading experiments |
| Early ML exploration | `archives/early_ml_pipeline` | Feature engineering, regime filters, LightGBM experiments, threshold analysis |
| Intraday ML research | `projects/intraday_ml_research` | 1-minute ingestion, 5-minute features, ATR-based labels, calibrated XGBoost, research backtests |
| LOB microstructure research | `projects/lob_microstructure_ml` | Limit-order-book features, triple-barrier labels, LightGBM/CatBoost ensemble research |
| Live bridge experiment | `archives/kite_bridge_experiment` | Incomplete Kite signal bridge, preserved with its dry-run log |
| Execution platform | `projects/kite_execution_platform` | ORB strategy, candle-level simulator, SQLite trade ledger, Streamlit dashboard, Kite integration boundaries |

Read [the evolution notes](docs/evolution.md) and [validation notes](docs/validation.md) before interpreting any result or running an experiment.

## Results and findings

Every backtest output this project produced is published in [`results/`](results/),
including the ones that lost money. [`docs/findings.md`](docs/findings.md) documents
what each iteration showed and what an audit of my own results turned up.

| Iteration | Trades | Net P&L | Return |
|---|---:|---:|---:|
| VWAP reversion (v3) | 42 | +Rs 3,286 | +3.3% |
| Gap momentum (v10) | 166 | -Rs 5,882 | -5.9% |
| Opening range breakout | 1,492 | -Rs 96,927 | -96.9% |

The headline finding is that transaction costs, not signal quality, decided all
three: the ORB strategy's gross loss was Rs 6,626 and its costs were Rs 90,300.
A separately reported 1.28 Sharpe for the ML pipeline **does not survive audit** --
the corrected read is an ROC-AUC of 0.54, and the reasons are documented in full.

These are diagnoses, not performance claims. Read the findings before the numbers.

## Repository map

```text
kubera/
├── projects/      # Current, independently runnable research/platform modules
├── archives/      # Earlier iterations, kept as history
├── results/       # Published backtest outputs, trade ledgers, and charts
└── docs/          # Methodology, findings, and validation notes
```

Each project owns its own configuration and dependencies. They are intentionally separate: their data contracts, targets, and evaluation logic are not identical.

## Safety and reproducibility

- Kite credentials belong in environment variables, never in source.
- Datasets, trained models, and databases are not committed. Charts and run logs are — they are the evidence behind the results.
- Live order placement is out of scope. The execution code has not had an independent risk review.
- No figure in `results/` is reproducible until the items in [`docs/findings.md`](docs/findings.md) are fixed.

## Where to start

- [`docs/findings.md`](docs/findings.md) — what each iteration showed, and the audit that retracted a previously reported Sharpe.
- [`results/`](results/) — every backtest output, including the ones that lost money.
- `projects/kite_execution_platform` — candle-by-candle ORB simulator, risk sizing, SQLite ledger, Streamlit dashboard.
- `projects/intraday_ml_research` — feature engineering and time-series validation.
- `projects/lob_microstructure_ml` — order-book features and the ensemble.

Modules under `archives/` are historical experiments, not maintained code. `docs/validation.md` covers known limitations.
