import sqlite3
import os
import pandas as pd


DB_PATH = "data/trades.db"


def get_conn():
    os.makedirs("data", exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db():
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                date         TEXT,
                symbol       TEXT,
                direction    TEXT,
                entry_price  REAL,
                exit_price   REAL,
                quantity     INTEGER,
                sl           REAL,
                target       REAL,
                pnl          REAL,
                pnl_pct      REAL,
                exit_reason  TEXT,
                entry_time   TEXT,
                exit_time    TEXT,
                mode         TEXT
            )
        """)


def log_trade(trade: dict):
    init_db()
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO trades
                (date, symbol, direction, entry_price, exit_price, quantity,
                 sl, target, pnl, pnl_pct, exit_reason, entry_time, exit_time, mode)
            VALUES
                (:date, :symbol, :direction, :entry_price, :exit_price, :quantity,
                 :sl, :target, :pnl, :pnl_pct, :exit_reason, :entry_time, :exit_time, :mode)
        """, trade)


def get_trades(mode: str = None) -> pd.DataFrame:
    init_db()
    query = "SELECT * FROM trades"
    params = []
    if mode:
        query += " WHERE mode = ?"
        params.append(mode)
    query += " ORDER BY entry_time DESC"
    with get_conn() as conn:
        return pd.read_sql_query(query, conn, params=params)


def get_equity_curve(mode: str = "backtest") -> pd.DataFrame:
    init_db()
    with get_conn() as conn:
        df = pd.read_sql_query(
            "SELECT date, SUM(pnl) as daily_pnl FROM trades WHERE mode=? GROUP BY date ORDER BY date",
            conn, params=[mode]
        )
    if not df.empty:
        df["cumulative_pnl"] = df["daily_pnl"].cumsum()
    return df


def clear_backtest_trades():
    init_db()
    with get_conn() as conn:
        conn.execute("DELETE FROM trades WHERE mode='backtest'")
