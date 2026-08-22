"""
Kite Trading Platform v3
Usage:
  python main.py dashboard          # launch UI
  python main.py dashboard --port 8502
  python main.py backtest           # CLI backtest (full data)
  python main.py live               # headless live trading (no UI)
"""
import argparse, os, sys, yaml
sys.path.insert(0, os.path.dirname(__file__))
from src.utils.logger import setup_logger

logger = setup_logger("Main")


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def cmd_backtest(config):
    from src.data.loader import DataLoader
    from src.backtest.engine import BacktestEngine
    from src.backtest.metrics import compute

    logger.info("Loading data...")
    stock_data = DataLoader(config["data"]["csv_path"]).load()

    logger.info("Running backtest...")
    engine  = BacktestEngine(config, stock_data)
    results = engine.run(log_cb=lambda m: None)

    if "error" in results:
        print(f"\n❌  {results['error']}")
        return

    print("\n" + "═"*52)
    print("  BACKTEST RESULTS")
    print("═"*52)
    print(f"  Trades        : {results['total_trades']}")
    print(f"  Win Rate      : {results['win_rate']}%")
    print(f"  Total P&L     : ₹{results['total_pnl']:,.0f}")
    print(f"  Return        : {results['return_pct']}%")
    print(f"  Profit Factor : {results['profit_factor']}")
    print(f"  Max Drawdown  : {results['max_drawdown_pct']}%")
    print(f"  Final Capital : ₹{results['final_capital']:,.0f}")
    print("═"*52)


def cmd_live(config):
    """Headless live trading — runs in terminal, survives laptop sleep via nohup."""
    from src.execution.auto_trader import AutoTrader
    from src.strategy.vwap_reversion import HIGH_BETA_STOCKS

    logger.info("Starting headless AutoTrader...")
    trader = AutoTrader(config, list(HIGH_BETA_STOCKS))
    trader.start()

    import time
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt — stopping trader")
        trader.stop()


def cmd_dashboard(port=8501):
    os.system(f"streamlit run dashboard/app.py --server.port {port}")


def main():
    parser = argparse.ArgumentParser(description="Kite Trading Platform v3")
    parser.add_argument("command", choices=["backtest","dashboard","live"])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--port",   type=int, default=8501)
    args   = parser.parse_args()
    config = load_config(args.config)

    if args.command == "backtest":   cmd_backtest(config)
    elif args.command == "dashboard": cmd_dashboard(args.port)
    elif args.command == "live":      cmd_live(config)


if __name__ == "__main__":
    main()
