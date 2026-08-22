"""
Kite Trading Platform — v10 VWAP Mean Reversion (1-min data)
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

st.set_page_config(page_title="Kite Trader v10", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")
DARK = "plotly_dark"


@st.cache_resource
def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


@st.cache_resource(show_spinner="Loading 1-min market data… (this may take 30-60s)")
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


with st.sidebar:
    st.title("📈 Kite Trader")
    st.caption("VWAP Mean Reversion — 1-min — v10")
    st.markdown("---")
    page = st.radio("Navigate", ["🔬 Backtest", "📄 Paper Trading",
                                  "⚡ Live Trading", "📋 Trade Log"])

config = load_config()

# ══════════════════════════════════════════════════════════════════════════
#  BACKTEST
# ══════════════════════════════════════════════════════════════════════════
if page == "🔬 Backtest":
    st.header("Backtest — VWAP Mean Reversion (1-min)")
    st.info("💡 Price stretches ≥ deviation% from VWAP + RSI confirmation → fade back to VWAP. "
            "1-min data = more signals, same edge.")

    st.subheader("📅 Date Range")
    preset = st.radio("Quick select",
                      ["1 Month","3 Months","6 Months","1 Year","3 Years","All Data"],
                      horizontal=True, index=5)
    today = date.today()
    preset_map = {
        "1 Month":  today - timedelta(days=30),
        "3 Months": today - timedelta(days=90),
        "6 Months": today - timedelta(days=180),
        "1 Year":   today - timedelta(days=365),
        "3 Years":  today - timedelta(days=365*3),
        "All Data": date(2000, 1, 1),
    }
    col_d1, col_d2 = st.columns(2)
    start_date = col_d1.date_input("From", value=preset_map[preset])
    end_date   = col_d2.date_input("To",   value=today)
    st.markdown("---")

    with st.expander("⚙️ Strategy Parameters", expanded=True):
        r1c1,r1c2,r1c3,r1c4 = st.columns(4)
        capital   = r1c1.number_input("Capital (₹)",        value=100_000, step=10_000, min_value=10_000)
        max_pos   = r1c2.selectbox("Max positions / day",   [1,2,3,4,5], index=2)
        risk_pct  = r1c3.slider("Risk per trade (%)",       0.5, 5.0, 3.0, 0.5) / 100
        deviation = r1c4.slider("VWAP Deviation % trigger", 0.5, 2.5, 1.2, 0.1)

        r2c1,r2c2,r2c3 = st.columns(3)
        rsi_os   = r2c1.slider("RSI Oversold",  20, 45, 35, 1)
        rsi_ob   = r2c2.slider("RSI Overbought",55, 80, 65, 1)
        sl_ratio = r2c3.slider("SL ratio",      0.3, 1.5, 0.8, 0.1)

    if st.button("▶  Run Backtest", type="primary", use_container_width=True):
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
            st.error(f"File not found: {cfg['data']['csv_path']}")
            st.stop()

        st.markdown("### 📋 Live Backtest Log")
        log_container = st.empty()
        log_lines     = []

        def append_log(msg):
            log_lines.append(msg)
            log_container.code("\n".join(log_lines[-60:]), language=None)

        bar     = st.progress(0, text="Starting…")
        engine  = BacktestEngine(cfg, stock_data)
        results = engine.run(
            start_date  = start_date,
            end_date    = end_date,
            progress_cb = lambda p: bar.progress(p, text=f"Running… {p*100:.0f}%"),
            log_cb      = append_log,
        )
        bar.empty()

        if "error" in results:
            st.error(results["error"])
            st.stop()

        st.success(f"✅  {results['total_trades']} trades | {start_date} → {end_date}")

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
            fig = px.histogram(results["trades_df"], x="pnl", nbins=50,
                               color_discrete_sequence=["#00d4aa"], template=DARK)
            fig.add_vline(x=0, line_dash="dash", line_color="red")
            fig.update_layout(height=300, margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)
        with cr:
            st.subheader("Exit Reasons")
            counts = results["trades_df"]["exit_reason"].value_counts()
            fig = px.pie(values=counts.values, names=counts.index, template=DARK,
                         color_discrete_sequence=["#00d4aa","#ff4c4c","#ffd700","#888"],
                         hole=0.4)
            fig.update_layout(height=300, margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("P&L by Symbol")
        sym_pnl = results["trades_df"].groupby("symbol")["pnl"].agg(
            Total="sum", Trades="count",
            WinRate=lambda x: f"{(x>0).mean()*100:.0f}%"
        ).sort_values("Total", ascending=False).reset_index()
        st.dataframe(sym_pnl, use_container_width=True, hide_index=True)

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
                           f"backtest_v10_{start_date}_{end_date}.csv", "text/csv")

# ══════════════════════════════════════════════════════════════════════════
#  PAPER TRADING
# ══════════════════════════════════════════════════════════════════════════
elif page == "📄 Paper Trading":
    st.header("📄 Paper Trading")
    st.info("Live Kite data. No real orders placed.")
    api_key = config["kite"].get("api_key","")
    if not api_key or api_key == "YOUR_API_KEY":
        st.error("Set api_key in config.yaml first.")
        st.stop()
    if not config["kite"].get("access_token"):
        login_url = f"https://kite.zerodha.com/connect/login?api_key={api_key}"
        st.markdown(f"[Login to Kite →]({login_url})")
        req_token = st.text_input("Paste request_token")
        if st.button("Generate Token") and req_token:
            try:
                from src.execution.broker import KiteBroker
                KiteBroker(config).generate_session(req_token)
                st.success("✅ Saved.")
                st.rerun()
            except Exception as e:
                st.error(f"Auth failed: {e}")
        st.stop()
    st.success("✅ Authenticated")
    is_running = (st.session_state.get("paper_trader") is not None and
                  st.session_state["paper_trader"].get_status()["running"])
    c1, c2 = st.columns(2)
    with c1:
        if not is_running:
            if st.button("▶  START Paper Trading", type="primary", use_container_width=True):
                try:
                    from src.execution.paper_trader import PaperTrader
                    from src.strategy.vwap_reversion import HIGH_BETA_STOCKS
                    trader = PaperTrader(config, list(HIGH_BETA_STOCKS))
                    trader.start()
                    st.session_state["paper_trader"] = trader
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed: {e}")
        else:
            st.success("🟢 Paper Trading ACTIVE")
    with c2:
        if is_running:
            if st.button("⏹  Stop", type="secondary", use_container_width=True):
                st.session_state["paper_trader"].stop()
                st.session_state["paper_trader"] = None
                st.rerun()
    if is_running:
        status = st.session_state["paper_trader"].get_status()
        s1,s2,s3 = st.columns(3)
        s1.metric("Virtual Capital", f"₹{status['capital']:,.0f}")
        s2.metric("Day P&L",         f"₹{status['daily_pnl']:+,.0f}")
        s3.metric("Trades Today",     status.get("trade_count", 0))
        today_str = datetime.now().strftime("%Y%m%d")
        log_path  = f"logs/{today_str}_PaperTrader.log"
        if os.path.exists(log_path):
            with open(log_path) as f:
                lines = f.readlines()[-60:]
            st.code("".join(lines), language=None)
        if st.button("🔄 Refresh"):
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════
#  LIVE TRADING
# ══════════════════════════════════════════════════════════════════════════
elif page == "⚡ Live Trading":
    st.header("⚡ Live Trading")
    st.warning("⚠️  Real orders. Real money. Paper trade first.")
    api_key = config["kite"].get("api_key","")
    if not api_key or api_key == "YOUR_API_KEY":
        st.error("Set api_key in config.yaml first.")
        st.stop()
    if not config["kite"].get("access_token"):
        login_url = f"https://kite.zerodha.com/connect/login?api_key={api_key}"
        st.markdown(f"[Login to Kite →]({login_url})")
        req_token = st.text_input("Paste request_token")
        if st.button("Generate Token") and req_token:
            try:
                from src.execution.broker import KiteBroker
                KiteBroker(config).generate_session(req_token)
                st.success("✅ Saved.")
                st.rerun()
            except Exception as e:
                st.error(f"Auth failed: {e}")
        st.stop()
    st.success("✅ Authenticated")
    is_running = (st.session_state.get("auto_trader") is not None and
                  st.session_state["auto_trader"].get_status()["running"])
    c1, c2 = st.columns(2)
    with c1:
        if not is_running:
            if st.button("▶  START Live Trading", type="primary", use_container_width=True):
                try:
                    from src.execution.auto_trader import AutoTrader
                    from src.strategy.vwap_reversion import HIGH_BETA_STOCKS
                    trader = AutoTrader(config, list(HIGH_BETA_STOCKS))
                    trader.start()
                    st.session_state["auto_trader"] = trader
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed: {e}")
        else:
            st.success("🟢 Live Trading ACTIVE")
    with c2:
        if is_running:
            if st.button("⏹  STOP & Square Off", type="secondary", use_container_width=True):
                try:
                    st.session_state["auto_trader"].stop()
                    from src.execution.broker import KiteBroker
                    KiteBroker(config).square_off_all()
                    st.session_state["auto_trader"] = None
                    st.rerun()
                except Exception as e:
                    st.error(f"Stop error: {e}")
    if is_running:
        status = st.session_state["auto_trader"].get_status()
        s1,s2,s3 = st.columns(3)
        s1.metric("Capital",        f"₹{status['capital']:,.0f}")
        s2.metric("Day P&L",        f"₹{status['daily_pnl']:+,.0f}")
        s3.metric("Open Positions",  len(status["positions"]))
        today_str = datetime.now().strftime("%Y%m%d")
        log_path  = f"logs/{today_str}_AutoTrader.log"
        if os.path.exists(log_path):
            with open(log_path) as f:
                lines = f.readlines()[-60:]
            st.subheader("Live Log")
            st.code("".join(lines), language=None)
        if st.button("🔄 Refresh"):
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════
#  TRADE LOG
# ══════════════════════════════════════════════════════════════════════════
elif page == "📋 Trade Log":
    st.header("Trade History")
    c1,_,c3 = st.columns([2,3,1])
    mode_filter = c1.selectbox("Mode", ["All","backtest","paper","live"])
    if c3.button("🔄 Refresh"):
        st.rerun()
    mode   = None if mode_filter == "All" else mode_filter
    trades = db.get_trades(mode=mode)
    if trades.empty:
        st.info("No trades yet.")
    else:
        wins     = trades[trades["pnl"] > 0]
        loss_sum = trades[trades["pnl"] < 0]["pnl"].sum()
        pf       = round(wins["pnl"].sum() / abs(loss_sum), 2) if loss_sum else "∞"
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Total P&L",    f"₹{trades['pnl'].sum():,.0f}")
        m2.metric("Win Rate",     f"{len(wins)/len(trades)*100:.1f}%")
        m3.metric("Total Trades",  len(trades))
        m4.metric("Profit Factor", pf)
        ec = db.get_equity_curve(mode="backtest" if mode_filter in ["All","backtest"] else mode_filter)
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