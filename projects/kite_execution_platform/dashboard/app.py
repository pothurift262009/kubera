"""
Kite Trading Platform — Streamlit Dashboard
Run:  streamlit run dashboard/app.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import yaml
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from src.data.loader import DataLoader
from src.backtest.engine import BacktestEngine
from src.backtest.metrics import compute as compute_metrics
from src.utils import db

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Kite Trader",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

DARK = "plotly_dark"

# ── Helpers ──────────────────────────────────────────────────────────────────
@st.cache_resource
def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

@st.cache_resource(show_spinner="Loading market data…")
def load_stock_data(csv_path):
    return DataLoader(csv_path).load()

def equity_chart(ec: pd.DataFrame, initial: float):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ec["date"], y=ec["capital"],
        mode="lines", fill="tozeroy",
        line=dict(color="#00d4aa", width=2),
        fillcolor="rgba(0,212,170,0.08)",
        name="Capital",
    ))
    fig.add_hline(y=initial, line_dash="dash", line_color="gray",
                  annotation_text="Start", annotation_position="right")
    fig.update_layout(template=DARK, height=380,
                      xaxis_title="Date", yaxis_title="Capital (₹)",
                      margin=dict(t=20))
    return fig

def pnl_histogram(trades_df: pd.DataFrame):
    fig = px.histogram(trades_df, x="pnl", nbins=40,
                       color_discrete_sequence=["#00d4aa"], template=DARK,
                       labels={"pnl": "Trade P&L (₹)"})
    fig.add_vline(x=0, line_dash="dash", line_color="red")
    fig.update_layout(height=320, margin=dict(t=20))
    return fig

def exit_pie(trades_df: pd.DataFrame):
    counts = trades_df["exit_reason"].value_counts()
    fig = px.pie(values=counts.values, names=counts.index, template=DARK,
                 color_discrete_sequence=["#00d4aa", "#ff4c4c", "#ffd700"],
                 hole=0.4)
    fig.update_layout(height=320, margin=dict(t=20))
    return fig

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 Kite Trader")
    st.markdown("---")
    page = st.radio("Navigate", ["🔬 Backtest", "⚡ Live Trading", "📋 Trade Log"])

config = load_config()

# ══════════════════════════════════════════════════════════════════════════════
#  BACKTEST
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔬 Backtest":
    st.header("Backtest Engine")

    with st.expander("⚙️ Strategy Parameters", expanded=True):
        c1, c2, c3, c4, c5 = st.columns(5)
        capital   = c1.number_input("Capital (₹)",       value=100_000, step=10_000, min_value=10_000)
        orb_min   = c2.selectbox("ORB Period (min)",      [15, 30, 45], index=0)
        rr        = c3.selectbox("Risk : Reward",         [1.5, 2.0, 2.5, 3.0], index=1)
        risk_pct  = c4.slider("Risk per trade (%)",       0.5, 3.0, 2.0, 0.5) / 100
        max_pos   = c5.slider("Max positions / day",      1, 5, 3)

    run = st.button("▶  Run Backtest", type="primary", use_container_width=True)

    if run:
        cfg = {**config}
        cfg["capital"]["total"]           = capital
        cfg["strategy"]["orb_minutes"]    = orb_min
        cfg["strategy"]["rr_ratio"]       = rr
        cfg["strategy"]["risk_pct"]       = risk_pct
        cfg["strategy"]["max_positions"]  = max_pos

        try:
            stock_data = load_stock_data(cfg["data"]["csv_path"])
        except FileNotFoundError:
            st.error(f"CSV not found: {cfg['data']['csv_path']}\n\nUpdate `csv_path` in config.yaml.")
            st.stop()

        bar = st.progress(0, text="Running…")
        engine  = BacktestEngine(cfg, stock_data)
        results = engine.run(progress_cb=lambda p: bar.progress(p, text=f"{p*100:.0f}%"))
        bar.empty()

        if "error" in results:
            st.error(results["error"])
            st.stop()

        st.success(f"✅  {results['total_trades']} trades executed")

        # KPI row
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Total P&L",      f"₹{results['total_pnl']:,.0f}", f"{results['return_pct']}%")
        k2.metric("Win Rate",        f"{results['win_rate']}%")
        k3.metric("Profit Factor",   results["profit_factor"])
        k4.metric("Max Drawdown",    f"{results['max_drawdown_pct']}%")
        k5.metric("Avg Win",         f"₹{results['avg_win']:,.0f}")
        k6.metric("Avg Loss",        f"₹{results['avg_loss']:,.0f}")

        st.plotly_chart(equity_chart(results["equity_curve"], capital), use_container_width=True)

        col_l, col_r = st.columns(2)
        with col_l:
            st.subheader("P&L Distribution")
            st.plotly_chart(pnl_histogram(results["trades_df"]), use_container_width=True)
        with col_r:
            st.subheader("Exit Reasons")
            st.plotly_chart(exit_pie(results["trades_df"]), use_container_width=True)

        st.subheader("Performance Metrics")
        m = compute_metrics(results["trades_df"], results["equity_curve"], capital)
        mdf = pd.DataFrame(m.items(), columns=["Metric", "Value"])
        st.dataframe(mdf, use_container_width=True, hide_index=True)

        st.subheader("Trade Log (last 100)")
        show_cols = ["date", "symbol", "direction", "entry_price",
                     "exit_price", "quantity", "pnl", "pnl_pct", "exit_reason"]
        st.dataframe(results["trades_df"][show_cols].tail(100),
                     use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  LIVE TRADING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚡ Live Trading":
    st.header("Live Trading")
    st.warning("⚠️  Live trading uses real money. Verify backtest results before proceeding.")

    col_auth, col_ctrl = st.columns(2)

    with col_auth:
        st.subheader("Kite Authentication")
        api_key = config["kite"].get("api_key", "")
        if not api_key or api_key == "YOUR_API_KEY":
            st.error("Set your API key in config.yaml first.")
        else:
            if not config["kite"].get("access_token"):
                login_url = f"https://kite.zerodha.com/connect/login?api_key={api_key}"
                st.markdown(f"**Step 1:** [Login to Kite →]({login_url})")
                req_token = st.text_input("**Step 2:** Paste request_token from redirect URL")
                if st.button("Generate Access Token") and req_token:
                    try:
                        from src.execution.broker import KiteBroker
                        broker = KiteBroker(config)
                        broker.generate_session(req_token)
                        st.success("✅ Token saved. Refresh the page.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Auth failed: {e}")
            else:
                st.success("✅ Authenticated with Kite")

    with col_ctrl:
        st.subheader("Trading Controls")
        if "live_running" not in st.session_state:
            st.session_state.live_running  = False
            st.session_state.live_trader   = None

        is_auth = bool(config["kite"].get("access_token"))

        if not st.session_state.live_running:
            if st.button("▶  Start Live Trading", type="primary", disabled=not is_auth):
                try:
                    from src.execution.live_trader import LiveTrader
                    from src.data.loader import DataLoader
                    stock_data = DataLoader(config["data"]["csv_path"]).load()
                    symbols    = list(stock_data.keys())[:47]
                    trader     = LiveTrader(config, symbols)
                    trader.start()
                    st.session_state.live_trader  = trader
                    st.session_state.live_running = True
                    st.success("🟢 Live trader started")
                    st.rerun()
                except Exception as e:
                    st.error(f"Start failed: {e}")
        else:
            st.success("🟢 Live Trading Active")
            if st.button("⏹  Stop & Square Off All", type="secondary"):
                try:
                    st.session_state.live_trader.stop()
                    from src.execution.broker import KiteBroker
                    KiteBroker(config).square_off_all()
                except Exception as e:
                    st.error(f"Square off error: {e}")
                st.session_state.live_running = False
                st.session_state.live_trader  = None
                st.rerun()

    # Live positions table
    if st.session_state.live_running and is_auth:
        st.subheader("Open Positions")
        try:
            from src.execution.broker import KiteBroker
            positions = KiteBroker(config).get_positions()
            open_pos  = [p for p in positions if p["quantity"] != 0]
            if open_pos:
                pdf = pd.DataFrame(open_pos)[["tradingsymbol", "quantity",
                                               "average_price", "last_price", "pnl"]]
                st.dataframe(pdf, use_container_width=True, hide_index=True)
            else:
                st.info("No open positions")
        except Exception as e:
            st.warning(f"Could not fetch positions: {e}")

        # Today's live trades from DB
        today_trades = db.get_trades(mode="live")
        if not today_trades.empty:
            st.subheader("Today's Live Trades")
            st.dataframe(today_trades[["symbol", "direction", "entry_price",
                                       "exit_price", "quantity", "pnl", "exit_reason"]],
                         use_container_width=True, hide_index=True)
            st.metric("Today P&L", f"₹{today_trades['pnl'].sum():,.0f}")


# ══════════════════════════════════════════════════════════════════════════════
#  TRADE LOG
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Trade Log":
    st.header("Trade History")

    c1, c2, c3 = st.columns([2, 2, 1])
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
        m1.metric("Total P&L",   f"₹{trades['pnl'].sum():,.0f}")
        m2.metric("Win Rate",    f"{len(wins)/len(trades)*100:.1f}%")
        m3.metric("Total Trades", len(trades))
        m4.metric("Profit Factor",
                  round(wins["pnl"].sum() / abs(trades[trades["pnl"]<0]["pnl"].sum()), 2)
                  if trades[trades["pnl"]<0]["pnl"].sum() != 0 else "∞")

        ec = db.get_equity_curve(mode="backtest" if mode != "live" else "live")
        if not ec.empty:
            fig = go.Figure(go.Scatter(
                x=ec["date"], y=ec["cumulative_pnl"],
                mode="lines", line=dict(color="#00d4aa", width=2),
                fill="tozeroy", fillcolor="rgba(0,212,170,0.08)"
            ))
            fig.update_layout(template=DARK, height=280, title="Cumulative P&L",
                              xaxis_title="Date", yaxis_title="P&L (₹)", margin=dict(t=40))
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            trades[["date", "symbol", "direction", "entry_price",
                    "exit_price", "quantity", "pnl", "exit_reason", "mode"]],
            use_container_width=True, hide_index=True
        )

        csv = trades.to_csv(index=False)
        st.download_button("⬇ Download CSV", csv, "trades.csv", "text/csv")
