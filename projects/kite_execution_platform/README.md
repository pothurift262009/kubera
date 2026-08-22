# Kite Execution Platform Prototype

This project is the execution-oriented branch of Kubera. It includes an opening-range-breakout strategy, a candle-by-candle backtest engine, position sizing, a SQLite ledger, a Streamlit dashboard, and Kite broker adapter boundaries.

## Commands

```bash
python main.py backtest
python main.py dashboard
```

Configure a local CSV path in `config.yaml`. Keep API credentials outside this repository and use paper trading before any broker integration. The execution code is retained as a portfolio artifact and requires additional risk review before live use.
