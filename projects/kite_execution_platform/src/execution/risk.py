"""
RiskManager — position sizing + portfolio-level guardrails.
"""
from src.utils.logger import setup_logger

logger = setup_logger("RiskManager")


class RiskManager:
    def __init__(self, config: dict):
        cfg                  = config["strategy"]
        self.max_positions   = cfg["max_positions"]
        self.risk_pct        = cfg["risk_pct"]
        self.capital         = config["capital"]["total"]
        self.open_positions  = {}          # {symbol: position_dict}

    # ------------------------------------------------------------------ #

    @property
    def slots_available(self) -> int:
        return self.max_positions - len(self.open_positions)

    def can_trade(self) -> bool:
        return self.slots_available > 0

    def position_size(self, entry: float, sl: float) -> int:
        risk_amt  = self.capital * self.risk_pct
        risk_unit = abs(entry - sl)
        if risk_unit == 0:
            return 0
        qty_risk    = int(risk_amt / risk_unit)
        qty_capital = int((self.capital / self.max_positions) / entry)
        qty = min(qty_risk, qty_capital)
        logger.info(f"Sizing: entry={entry} sl={sl} → qty={qty}")
        return qty

    def add_position(self, symbol: str, position: dict):
        self.open_positions[symbol] = position
        logger.info(f"Position added: {symbol}  open={len(self.open_positions)}/{self.max_positions}")

    def close_position(self, symbol: str, exit_price: float, exit_reason: str):
        pos = self.open_positions.pop(symbol, None)
        if pos is None:
            return 0.0
        if pos["direction"] == "LONG":
            pnl = (exit_price - pos["entry_price"]) * pos["quantity"]
        else:
            pnl = (pos["entry_price"] - exit_price) * pos["quantity"]
        self.capital += pnl
        logger.info(f"Closed {symbol} [{exit_reason}]  pnl=₹{pnl:,.0f}  capital=₹{self.capital:,.0f}")
        return pnl

    def daily_reset(self):
        self.open_positions.clear()
