"""
KiteBroker — thin wrapper around kiteconnect for order placement,
position management, and authentication.
"""
import yaml
from kiteconnect import KiteConnect
from src.utils.logger import setup_logger

logger = setup_logger("KiteBroker")


class KiteBroker:
    def __init__(self, config: dict):
        self.config = config
        self.kite   = KiteConnect(api_key=config["kite"]["api_key"])
        if config["kite"].get("access_token"):
            self.kite.set_access_token(config["kite"]["access_token"])

    # ------------------------------------------------------------------ #
    #  Auth
    # ------------------------------------------------------------------ #

    def login_url(self) -> str:
        return self.kite.login_url()

    def generate_session(self, request_token: str) -> str:
        data = self.kite.generate_session(
            request_token, api_secret=self.config["kite"]["api_secret"]
        )
        token = data["access_token"]
        self.kite.set_access_token(token)
        # Persist to config.yaml
        self.config["kite"]["access_token"] = token
        with open("config.yaml", "w") as f:
            yaml.dump(self.config, f)
        logger.info("Access token saved to config.yaml")
        return token

    # ------------------------------------------------------------------ #
    #  Orders
    # ------------------------------------------------------------------ #

    def buy(self, symbol: str, qty: int) -> str:
        return self._place(symbol, qty, self.kite.TRANSACTION_TYPE_BUY)

    def sell(self, symbol: str, qty: int) -> str:
        return self._place(symbol, qty, self.kite.TRANSACTION_TYPE_SELL)

    def place_sl_order(self, symbol: str, direction: str, qty: int, sl_price: float) -> str:
        tx = self.kite.TRANSACTION_TYPE_SELL if direction == "LONG" else self.kite.TRANSACTION_TYPE_BUY
        trigger = round(sl_price, 1)
        # Limit slightly beyond trigger so order fills
        limit   = round(trigger * (0.995 if direction == "LONG" else 1.005), 1)
        oid = self.kite.place_order(
            tradingsymbol   = symbol,
            exchange        = self.kite.EXCHANGE_NSE,
            transaction_type= tx,
            quantity        = qty,
            order_type      = self.kite.ORDER_TYPE_SL,
            trigger_price   = trigger,
            price           = limit,
            product         = self.kite.PRODUCT_MIS,
            variety         = self.kite.VARIETY_REGULAR,
        )
        logger.info(f"SL order placed: {symbol} trigger={trigger} | {oid}")
        return oid

    def cancel_order(self, order_id: str):
        self.kite.cancel_order(variety=self.kite.VARIETY_REGULAR, order_id=order_id)
        logger.info(f"Cancelled order {order_id}")

    # ------------------------------------------------------------------ #
    #  Data / Status
    # ------------------------------------------------------------------ #

    def get_positions(self) -> list:
        return self.kite.positions().get("day", [])

    def get_orders(self) -> list:
        return self.kite.orders()

    def get_ltp(self, symbols: list) -> dict:
        instruments = [f"NSE:{s}" for s in symbols]
        raw = self.kite.ltp(instruments)
        return {k.replace("NSE:", ""): v["last_price"] for k, v in raw.items()}

    def get_instrument_tokens(self, symbols: list) -> dict:
        """Returns {symbol: instrument_token}"""
        instruments = self.kite.instruments("NSE")
        result = {}
        for inst in instruments:
            if inst["tradingsymbol"] in symbols:
                result[inst["tradingsymbol"]] = inst["instrument_token"]
        return result

    def square_off_all(self):
        """Square off every open MIS position."""
        for pos in self.get_positions():
            qty = pos["quantity"]
            if qty == 0:
                continue
            sym = pos["tradingsymbol"]
            if qty > 0:
                self.sell(sym, abs(qty))
            else:
                self.buy(sym, abs(qty))
            logger.info(f"Squared off {sym} qty={qty}")

    # ------------------------------------------------------------------ #

    def _place(self, symbol: str, qty: int, tx_type) -> str:
        oid = self.kite.place_order(
            tradingsymbol   = symbol,
            exchange        = self.kite.EXCHANGE_NSE,
            transaction_type= tx_type,
            quantity        = qty,
            order_type      = self.kite.ORDER_TYPE_MARKET,
            product         = self.kite.PRODUCT_MIS,
            variety         = self.kite.VARIETY_REGULAR,
        )
        logger.info(f"Order: {tx_type} {qty} {symbol} | {oid}")
        return oid
