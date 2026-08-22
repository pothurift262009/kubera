"""
Kite Trading Platform — CLI
Usage:
    python main.py backtest              # run backtest, print results
    python main.py dashboard             # launch Streamlit UI
    python main.py dashboard --port 8502 # custom port
"""
import argparse
import os
import sys
import yaml

sys.path.insert(0, os.path.dirname(__file__))

from src.utils.logger import setup_logger

logger = setup_logger("Main")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def cmd_backtest(config: dict):
    from src.data.loader import DataLoader
    from src.backtest.engine import BacktestEngine
    from src.backtest.metrics import compute

    logger.info("Loading data…")
    stock_data = DataLoader(config["data"]["csv_path"]).load()

    logger.info("Running backtest…")
    engine  = BacktestEngine(config, stock_data)
    results = engine.run()

    if "error" in results:
        print(f"\n❌  {results['error']}")
        return

    print("\n" + "═" * 52)
    print("  BACKTEST RESULTS")
    print("═" * 52)
    print(f"  Total Trades    : {results['total_trades']}")
    print(f"  Win Rate        : {results['win_rate']}%")
    print(f"  Total P&L       : ₹{results['total_pnl']:,.0f}")
    print(f"  Return          : {results['return_pct']}%")
    print(f"  Profit Factor   : {results['profit_factor']}")
    print(f"  Max Drawdown    : {results['max_drawdown_pct']}%")
    print(f"  Final Capital   : ₹{results['final_capital']:,.0f}")
    print("═" * 52)

    metrics = compute(results["trades_df"], results["equity_curve"], config["capital"]["total"])
    print("\n  DETAILED METRICS")
    print("─" * 52)
    for k, v in metrics.items():
        print(f"  {k:<22}: {v}")
    print()


def cmd_dashboard(port: int = 8501):
    os.system(f"streamlit run dashboard/app.py --server.port {port}")


def main():
    parser = argparse.ArgumentParser(
        description="Kite Trading Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n  python main.py backtest\n  python main.py dashboard"
    )
    parser.add_argument("command", choices=["backtest", "dashboard"],
                        help="Command to run")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to config file (default: config.yaml)")
    parser.add_argument("--port", type=int, default=8501,
                        help="Streamlit port for dashboard (default: 8501)")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.command == "backtest":
        cmd_backtest(config)
    elif args.command == "dashboard":
        cmd_dashboard(args.port)


if __name__ == "__main__":
    main()
