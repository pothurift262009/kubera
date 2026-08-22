"""
Kite Trading Platform — Dashboard v3
Features:
  - Backtest with date range presets (1M, 3M, 6M, 1Y, 3Y, All)
  - Verbose live log window during backtest
  - Fully automatic live trading tab
  - Trade log with equity curve
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import yaml
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, timedelta, datetime

from src.data.loader import DataLoader
from src.backtest.engine import BacktestEngine
from src.backtest.metrics import compute as compute_metrics
from src.utils import db

st.set_page_config(page_title="Kite Trader", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")
DARK = "plotly_dark"


@st.cache_resource
def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


@st.cache_resource(show_spinner="Loading market data…")
def load_stock_data(csv_path):
    return DataLoader(csv_path).load()


def equity_chart(ec, initial):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ec["date"], y=ec["capital"], mode="lines",
        fill="tozeroy", line=dict(color="#00d4aa", width=2),
        fillcolor="rgba(0,212,170,0.08)", name="Capital"))
    fig.add_hline(y=initial, line_dash="dash", line_color="gray",
                  annotation_text="Start", annotation_position="right")
    fig.update_layout(template=DARK, height=350,
                      xaxis_title="Date", yaxis_title="Capital (₹)", margin=dict(t=10))
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 Kite Trader")
    st.caption("VWAP Mean Reversion — v3")
    st.markdown("---")
    page = st.radio("Navigate", ["🔬 Backtest", "⚡ Auto Trading", "📋 Trade Log"])

config = load_config()

# ══════════════════════════════════════════════════════════════════════════
#  BACKTEST
# ══════════════════════════════════════════════════════════════════════════
if page == "🔬 Backtest":
    st.header("Backtest — VWAP Mean Reversion")

    # ── Date range selector ───────────────────────────────────────────
    st.subheader("📅 Date Range")
    preset = st.radio("Quick select", ["1 Month", "3 Months", "6 Months",
                                        "1 Year", "3 Years", "All Data"],
                      horizontal=True, index=3)

    today = date.today()
    preset_map = {
        "1 Month":   today - timedelta(days=30),
        "3 Months":  today - timedelta(days=90),
        "6 Months":  today - timedelta(days=180),
        "1 Year":    today - timedelta(days=365),
        "3 Years":   today - timedelta(days=365*3),
        "All Data":  date(2000, 1, 1),
    }
    default_start = preset_map[preset]

    col_d1, col_d2 = st.columns(2)
    start_date = col_d1.date_input("From", value=default_start)
    end_date   = col_d2.date_input("To",   value=today)

    st.markdown("---")

    # ── Strategy params ───────────────────────────────────────────────
    with st.expander("⚙️ Strategy Parameters", expanded=True):
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        capital   = r1c1.number_input("Capital (₹)",       value=100_000, step=10_000, min_value=10_000)
        max_pos   = r1c2.selectbox("Max positions / day",  [1, 2, 3], index=1)
        risk_pct  = r1c3.slider("Risk per trade (%)",      0.5, 5.0, 3.0, 0.5) / 100
        deviation = r1c4.slider("VWAP Deviation % trigger",0.5, 2.5, 1.5, 0.1)

        r2c1, r2c2, r2c3 = st.columns(3)
        rsi_os = r2c1.slider("RSI Oversold",   20, 45, 32, 1)
        rsi_ob = r2c2.slider("RSI Overbought", 55, 80, 68, 1)
        sl_ratio = r2c3.slider("SL ratio",     0.3, 1.5, 1.0, 0.1)

    run = st.button("▶  Run Backtest", type="primary", use_container_width=True)

    if run:
        cfg = {**config}
        cfg["capital"]["total"]               = capital
        cfg["strategy"]["max_positions"]      = max_pos
        cfg["strategy"]["risk_pct"]           = risk_pct
        cfg["strategy"]["vwap_deviation_pct"] = deviation
        cfg["strategy"]["rsi_oversold"]       = rsi_os
        cfg["strategy"]["rsi_overbought"]     = rsi_ob
        cfg["strategy"]["sl_ratio"]           = sl_ratio

        try:
            stock_data = load_stock_data(cfg["data"]["csv_path"])
        except FileNotFoundError:
            st.error(f"CSV not found: {cfg['data']['csv_path']}")
            st.stop()

        # ── Live log window ───────────────────────────────────────────
        st.markdown("### 📋 Live Backtest Log")
        log_container = st.empty()
        log_lines     = []

        def append_log(msg: str):
            log_lines.append(msg)
            log_container.code("\n".join(log_lines[-60:]), language=None)

        bar    = st.progress(0, text="Starting…")
        engine = BacktestEngine(cfg, stock_data)

        results = engine.run(
            start_date = start_date,
            end_date   = end_date,
            progress_cb= lambda p: bar.progress(p, text=f"Running… {p*100:.0f}%"),
            log_cb     = append_log,
        )
        bar.empty()

        if "error" in results:
            st.error(results["error"])
            st.stop()

        st.success(f"✅  Backtest complete — {results['total_trades']} trades | "
                   f"{start_date} → {end_date}")

        # ── KPIs ──────────────────────────────────────────────────────
        k1,k2,k3,k4,k5,k6 = st.columns(6)
        k1.metric("Total P&L",    f"₹{results['total_pnl']:,.0f}", f"{results['return_pct']}%")
        k2.metric("Win Rate",     f"{results['win_rate']}%")
        k3.metric("Profit Factor", results["profit_factor"])
        k4.metric("Max Drawdown", f"{results['max_drawdown_pct']}%")
        k5.metric("Avg Win",      f"₹{results['avg_win']:,.0f}")
        k6.metric("Avg Loss",     f"₹{results['avg_loss']:,.0f}")

        st.plotly_chart(equity_chart(results["equity_curve"], capital), use_container_width=True)

        cl, cr = st.columns(2)
        with cl:
            st.subheader("P&L Distribution")
            fig = px.histogram(results["trades_df"], x="pnl", nbins=40,
                               color_discrete_sequence=["#00d4aa"], template=DARK)
            fig.add_vline(x=0, line_dash="dash", line_color="red")
            fig.update_layout(height=300, margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)
        with cr:
            st.subheader("Exit Reasons")
            counts = results["trades_df"]["exit_reason"].value_counts()
            fig = px.pie(values=counts.values, names=counts.index, template=DARK,
                         color_discrete_sequence=["#00d4aa","#ff4c4c","#ffd700"], hole=0.4)
            fig.update_layout(height=300, margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)

        m = compute_metrics(results["trades_df"], results["equity_curve"], capital)
        st.subheader("Detailed Metrics")
        st.dataframe(pd.DataFrame(m.items(), columns=["Metric","Value"]),
                     use_container_width=True, hide_index=True)

        st.subheader("Trade Log")
        cols = ["date","symbol","direction","entry_price","exit_price",
                "quantity","pnl","pnl_pct","exit_reason"]
        st.dataframe(results["trades_df"][cols], use_container_width=True, hide_index=True)
        st.download_button("⬇ Download trades CSV",
                           results["trades_df"].to_csv(index=False),
                           f"backtest_{start_date}_{end_date}.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════
#  AUTO TRADING
# ══════════════════════════════════════════════════════════════════════════
elif page == "⚡ Auto Trading":
    st.header("⚡ Automatic Live Trading")
    st.warning("⚠️  This places REAL orders with REAL money via Kite Connect API.")

    # ── Auth ──────────────────────────────────────────────────────────
    st.subheader("🔑 Kite Authentication")
    api_key = config["kite"].get("api_key", "")

    if not api_key or api_key == "YOUR_API_KEY":
        st.error("Set your api_key and api_secret in config.yaml first.")
    else:
        if not config["kite"].get("access_token"):
            login_url = f"https://kite.zerodha.com/connect/login?api_key={api_key}"
            st.markdown(f"**Step 1:** [Click here to login to Kite →]({login_url})")
            st.markdown("**Step 2:** After login, copy the `request_token` from the redirect URL")
            req_token = st.text_input("Paste request_token here")
            if st.button("Generate Access Token") and req_token:
                try:
                    from src.execution.broker import KiteBroker
                    broker = KiteBroker(config)
                    broker.generate_session(req_token)
                    st.success("✅ Access token saved. Refresh the page.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Auth failed: {e}")
        else:
            st.success("✅ Authenticated with Kite")

    is_auth = bool(config["kite"].get("access_token"))

    # ── Controls ──────────────────────────────────────────────────────
    st.subheader("🎛️ Trading Controls")
    if "auto_trader" not in st.session_state:
        st.session_state.auto_trader = None

    running = st.session_state.auto_trader is not None and \
              st.session_state.auto_trader.get_status()["running"]

    c1, c2 = st.columns(2)
    with c1:
        if not running:
            if st.button("▶  START Auto Trading", type="primary",
                         disabled=not is_auth, use_container_width=True):
                try:
                    from src.execution.auto_trader import AutoTrader
                    from src.strategy.vwap_reversion import HIGH_BETA_STOCKS
                    trader = AutoTrader(config, list(HIGH_BETA_STOCKS))
                    trader.start()
                    st.session_state.auto_trader = trader
                    st.success("🟢 Auto Trading started!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to start: {e}")
        else:
            st.success("🟢 Auto Trading ACTIVE")

    with c2:
        if running:
            if st.button("⏹  STOP & Square Off All", type="secondary",
                         use_container_width=True):
                try:
                    st.session_state.auto_trader.stop()
                    from src.execution.broker import KiteBroker
                    KiteBroker(config).square_off_all()
                    st.session_state.auto_trader = None
                    st.warning("Stopped and squared off all positions.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Stop error: {e}")

    # ── Live Status ───────────────────────────────────────────────────
    if running:
        status = st.session_state.auto_trader.get_status()

        st.markdown("---")
        s1, s2, s3 = st.columns(3)
        s1.metric("Day P&L",        f"₹{status['daily_pnl']:+,.0f}")
        s2.metric("Open Positions",  len(status["positions"]))
        s3.metric("Signals Fired",   len(status["signals_fired"]))

        if status["positions"]:
            st.subheader("Open Positions")
            pos_rows = []
            for sym, pos in status["positions"].items():
                pos_rows.append({
                    "Symbol":    sym,
                    "Direction": pos["direction"],
                    "Entry":     pos["entry_price"],
                    "SL":        pos["sl"],
                    "Target":    pos["target"],
                    "Qty":       pos["quantity"],
                    "Since":     pos["entry_time"],
                })
            st.dataframe(pd.DataFrame(pos_rows), use_container_width=True, hide_index=True)

        # Kite positions
        st.subheader("Kite Positions (live)")
        try:
            from src.execution.broker import KiteBroker
            positions = KiteBroker(config).get_positions()
            open_pos  = [p for p in positions if p["quantity"] != 0]
            if open_pos:
                pdf = pd.DataFrame(open_pos)[["tradingsymbol","quantity",
                                               "average_price","last_price","pnl"]]
                st.dataframe(pdf, use_container_width=True, hide_index=True)
            else:
                st.info("No open positions in Kite")
        except Exception as e:
            st.warning(f"Could not fetch Kite positions: {e}")

        # Today's completed trades
        today_trades = db.get_trades(mode="live")
        if not today_trades.empty:
            st.subheader("Today's Completed Trades")
            st.dataframe(today_trades[["symbol","direction","entry_price",
                                       "exit_price","quantity","pnl","exit_reason"]],
                         use_container_width=True, hide_index=True)

        # Live log from file
        st.subheader("Live Log")
        today_str = datetime.now().strftime("%Y%m%d")
        log_path  = f"logs/{today_str}_AutoTrader.log"
        if os.path.exists(log_path):
            with open(log_path) as f:
                lines = f.readlines()[-80:]
            st.code("".join(lines), language=None)
        else:
            st.info("Log file not created yet. Waiting for first candle...")

        if st.button("🔄 Refresh Status"):
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════
#  TRADE LOG
# ══════════════════════════════════════════════════════════════════════════
elif page == "📋 Trade Log":
    st.header("Trade History")

    c1, _, c3 = st.columns([2, 3, 1])
    mode_filter = c1.selectbox("Mode", ["All", "backtest", "live"])
    if c3.button("🔄 Refresh"):
        st.rerun()

    mode   = None if mode_filter == "All" else mode_filter
    trades = db.get_trades(mode=mode)

    if trades.empty:
        st.info("No trades yet. Run a backtest first.")
    else:
        wins = trades[trades["pnl"] > 0]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total P&L",    f"₹{trades['pnl'].sum():,.0f}")
        m2.metric("Win Rate",     f"{len(wins)/len(trades)*100:.1f}%")
        m3.metric("Total Trades",  len(trades))
        loss_sum = trades[trades["pnl"] < 0]["pnl"].sum()
        pf = round(wins["pnl"].sum() / abs(loss_sum), 2) if loss_sum else "∞"
        m4.metric("Profit Factor", pf)

        ec = db.get_equity_curve(mode="backtest" if mode != "live" else "live")
        if not ec.empty:
            fig = go.Figure(go.Scatter(x=ec["date"], y=ec["cumulative_pnl"],
                mode="lines", line=dict(color="#00d4aa", width=2),
                fill="tozeroy", fillcolor="rgba(0,212,170,0.08)"))
            fig.update_layout(template=DARK, height=260, title="Cumulative P&L",
                              xaxis_title="Date", yaxis_title="₹", margin=dict(t=40))
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(trades[["date","symbol","direction","entry_price","exit_price",
                              "quantity","pnl","exit_reason","mode"]],
                     use_container_width=True, hide_index=True)
        st.download_button("⬇ Download CSV",
                           trades.to_csv(index=False), "trades.csv", "text/csv")
