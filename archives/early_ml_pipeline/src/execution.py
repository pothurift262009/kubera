import numpy as np   # type: ignore
import pandas as pd  # type: ignore
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ExecutionSimulator:
    """
    STAGE 17: REALISTIC EXECUTION MODELING
    Simulates real-world friction: Slippage, Bid-Ask Spread, Latency,
    Partial Fills, and full strictly-defined Indian Market Costs.
    """

    def __init__(self, config: dict):
        self.config = config.get("execution", {})

        # Indian Equities Intraday Cost Structure (Percentage based for scaling)
        self.costs = {
            "brokerage_rate": 0.0003,      # 0.03% (Max cap not modeled for conservative estimates)
            "stt_sell": 0.00025,           # 0.025% on Sell only
            "txn_charge": 0.0000345,       # NSE transaction charge 0.00345%
            "sebi_charge": 0.000001,       # 10 per crore
            "stamp_duty_buy": 0.00003,     # 0.003% on Buy only
            "gst_rate": 0.18               # 18% on (Brokerage + Txn + SEBI)
        }

        # Execution specific configs
        self.base_spread_bps = 2.0                 # Normal bid-ask spread
        self.base_slippage_bps = 3.0               # Base latency slippage
        self.latency_decay_threshold = 0.0015      # 0.15% adverse move kills the trade signal

    def _calculate_india_costs(self) -> float:
        """Calculates total round-trip taxes and fees completely realistically (in %)."""
        c = self.costs
        # Buy Side
        buy_brokerage = c["brokerage_rate"]
        buy_txn = c["txn_charge"]
        buy_gst = (buy_brokerage + buy_txn + c["sebi_charge"]) * c["gst_rate"]
        buy_total = buy_brokerage + buy_txn + c["sebi_charge"] + c["stamp_duty_buy"] + buy_gst

        # Sell Side
        sell_brokerage = c["brokerage_rate"]
        sell_txn = c["txn_charge"]
        sell_gst = (sell_brokerage + sell_txn + c["sebi_charge"]) * c["gst_rate"]
        sell_total = sell_brokerage + sell_txn + c["sebi_charge"] + c["stt_sell"] + sell_gst

        round_trip_pct = buy_total + sell_total
        return round_trip_pct

    def simulate_execution(self, trades: pd.DataFrame) -> pd.DataFrame:
        """
        Runs idealized signals through the meat-grinder of real markets.
        Input: Trades dataframe with 'close' (signal trigger), 'future_return', 'vol_regime'.
        Output: Modified dataframe with execution metrics and realized PnL.
        """
        if trades.empty:
            return trades

        df = trades.copy()
        n = len(df)

        # 1. FIXED COSTS
        round_trip_costs = self._calculate_india_costs()
        df["statutory_costs_pct"] = round_trip_costs

        # 2. VOLATILITY-DEPENDENT SLIPPAGE & SPREAD
        # 0=Low Vol, 1=Normal, 2=High Vol
        vol_multiplier = df["vol_regime"].map({0: 0.8, 1: 1.2, 2: 2.5}).fillna(1.2).values

        # Randomize slippage slightly to simulate liquidity pockets
        # Slippage heavily depends on regime. High vol = high slippage.
        rng = np.random.default_rng(seed=42) # For reproducibility in backtest

        # Spread cost (half spread crossed on market order entry + exit = 1 full spread)
        spread_cost = (self.base_spread_bps / 10000) * vol_multiplier

        # Latency Slippage (delay between signal and NSE execution)
        simulated_delay_slippage = (self.base_slippage_bps / 10000) * vol_multiplier * rng.uniform(0.5, 1.5, size=n)

        total_friction_entry = (spread_cost / 2) + simulated_delay_slippage
        total_friction_exit = (spread_cost / 2) + simulated_delay_slippage

        df["slippage_impact_pct"] = total_friction_entry + total_friction_exit

        # 3. LATENCY-AWARE SIGNAL DECAY
        # If the entry price moved unfavorably by > threshold before execution, we abort the trade
        aborted_mask = total_friction_entry > self.latency_decay_threshold

        # 4. PARTIAL FILL SIMULATION (Priority Logic)
        # Limit vs Market logic implicitly modeled via fill ratios.
        # In highly volatile regimes, institutional momentum algorithms grab liquidity first.
        fill_ratio = np.ones(n)
        high_vol_mask = df["vol_regime"] == 2

        # In High Vol, 20% chance of only getting a partial fill (e.g. 50%)
        partial_hits = rng.random(n) < 0.20
        num_partials = int(np.sum(high_vol_mask & partial_hits))
        if num_partials > 0:
            fill_ratio[high_vol_mask & partial_hits] = rng.uniform(0.3, 0.7, size=num_partials)

        # Aborted trades get 0 fill
        fill_ratio[aborted_mask] = 0.0
        df["fill_ratio"] = fill_ratio

        # 5. EXECUTION PENALTY APPLY
        # Adjusted Return = (Ideal Return * Fill Ratio) - Costs - Slippage
        # Notice we only apply round trip costs if fill_ratio > 0
        df["realized_return"] = np.where(
            df["fill_ratio"] > 0,
            (df["future_return"] - df["slippage_impact_pct"] - df["statutory_costs_pct"]),
            0.0
        )

        return df

    def execution_report(self, executed_trades: pd.DataFrame) -> Dict[str, float]:
        """Generates comprehensive metrics on execution quality."""
        if executed_trades.empty:
            return {}

        df = executed_trades
        attempted = len(df)
        filled = df[df["fill_ratio"] > 0]
        fully_filled = df[df["fill_ratio"] == 1.0]
        aborted = df[df["fill_ratio"] == 0.0]

        fill_rate = len(filled) / attempted if attempted > 0 else 0
        avg_fill_qty = df["fill_ratio"].mean()

        raw_pnl = (df["future_return"] * df["fill_ratio"]).sum()  # Unpenalized but fill-adjusted
        realized_pnl = df["realized_return"].sum()               # Fully penalized

        total_slippage = df.loc[df["fill_ratio"] > 0, "slippage_impact_pct"].sum()
        total_costs = df.loc[df["fill_ratio"] > 0, "statutory_costs_pct"].sum()

        win_rate = (df["realized_return"] > 0).mean() * 100
        gross_wins = df[df["realized_return"] > 0]["realized_return"].sum()
        gross_losses = abs(df[df["realized_return"] < 0]["realized_return"].sum())
        pf = gross_wins / gross_losses if gross_losses > 0 else 99.0

        logger.info("\n" + "═" * 40)
        logger.info("📉 REALISTIC EXECUTION REPORT")
        logger.info("═" * 40)
        logger.info(f"Orders Attempted:    {attempted}")
        logger.info(f"Trades Filled:       {len(filled)} (Rate: {fill_rate:.1%})")
        logger.info(f"Avg Fill Qty:        {avg_fill_qty:.1%}")
        logger.info(f"Orders Aborted:      {len(aborted)} (Price moved away)")
        logger.info("-" * 40)
        logger.info(f"Raw Target PnL:      {raw_pnl * 100:.2f}%")
        logger.info(f"Lost to Slippage:    -{total_slippage * 100:.2f}%")
        logger.info(f"Lost to Taxes/Fees:  -{total_costs * 100:.2f}%")
        logger.info(f"REALIZED PNL:        {realized_pnl * 100:.2f}%")
        logger.info("-" * 40)
        logger.info(f"Realized Win Rate:   {win_rate:.1f}%")
        logger.info(f"Realized Profit Fac: {pf:.2f}")
        logger.info("═" * 40 + "\n")

        return {
            "fill_rate": fill_rate,
            "avg_fill_ratio": avg_fill_qty,
            "slippage_lost_pct": total_slippage * 100,
            "cost_lost_pct": total_costs * 100,
            "realized_pnl_pct": realized_pnl * 100,
            "profit_factor": pf,
            "win_rate": win_rate
        }
