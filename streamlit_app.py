import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from datetime import date, timedelta
from app.config import get_finnhub_key
from app import data as md
from app import catalysts as cat
from app import backtest as bt

st.set_page_config(page_title="Moonshot Biotech", page_icon="🚀", layout="wide")
st.title("🚀 Moonshot Biotech Dashboard — Backtest+")

# Sidebar (backtest-focused for brevity)
with st.sidebar:
    st.header("Universe")
    source = st.radio("Backend", ["Finnhub (preferred)", "yfinance / custom"], index=1)
    api_key = st.text_input("Finnhub API Key (optional)", value=get_finnhub_key(), type="password")
    up_universe = st.file_uploader("Upload tickers CSV (col: symbol)", type=["csv"], key="uni")
    pasted = st.text_area("...or paste tickers separated by comma/newline", height=70)
    sheet_url = st.text_input("Google Sheet URL (optional)")

    st.markdown("---")
    st.subheader("Backtest Window")
    start_date = st.date_input("Start date", value=(date.today() - timedelta(days=365*2)))
    end_date = st.date_input("End date", value=date.today())
    rebalance = st.selectbox("Rebalance", ["Daily", "Weekly"], index=1)

    st.markdown("---")
    st.subheader("Selection")
    top_n = st.slider("Top N per rebalance", 5, 50, 20, step=1)
    min_score = st.slider("Min score threshold", 0.0, 1.0, 0.4, 0.05)
    max_syms = st.slider("Max tickers (cap for speed)", 20, 300, 120, step=10)

    st.markdown("---")
    st.subheader("Risk & Costs")
    sl_pct = st.slider("Stop-loss (%)", 0.0, 20.0, 8.0, 0.5) / 100.0
    tp_pct = st.slider("Take-profit (%)", 0.0, 50.0, 15.0, 0.5) / 100.0
    cost_bps = st.slider("Transaction cost (bps of turnover)", 0, 100, 10, 1)

    st.markdown("---")
    st.subheader("Catalysts (optional)")
    up = st.file_uploader("Upload catalysts CSV", type=["csv"], key="cat")
    events_df = None
    if up:
        try:
            events_df = cat.load_catalysts_csv(up)
            st.success(f"Catalysts loaded: {len(events_df)} rows")
        except Exception as e:
            st.error(f"Failed to parse catalysts CSV: {e}")

# Build universe
@st.cache_data(ttl=3600)
def load_universe_finnhub(api_key: str) -> pd.DataFrame:
    return md.list_us_symbols_finnhub(api_key)

@st.cache_data(ttl=3600)
def load_universe_yf(seed_path: str, up_df, pasted_text, sheet_url) -> pd.DataFrame:
    return md.list_symbols_from_inputs(seed_path, up_df, pasted_text, sheet_url)

if source == "Finnhub (preferred)" and api_key:
    universe = load_universe_finnhub(api_key)
    if universe is None or universe.empty:
        st.warning("Finnhub universe empty/failed — using seed/pasted/uploaded via yfinance.")
        universe = load_universe_yf("data/biotech_seed.csv", pd.read_csv(up_universe) if up_universe else None, pasted, sheet_url)
else:
    universe = load_universe_yf("data/biotech_seed.csv", pd.read_csv(up_universe) if up_universe else None, pasted, sheet_url)

if universe is None or universe.empty:
    st.error("No symbols available. Upload/paste tickers or try again later.")
    st.stop()

st.write(f"Universe size for backtest: {len(universe)} symbols")

# Run backtest
run_bt = st.button("Run Backtest")
if run_bt:
    with st.spinner("Running backtest…"):
        scores, closes, ohlc_map = bt.build_rank_table(
            symbols=list(universe["symbol"].tolist()),
            start=str(start_date), end=str(end_date),
            weights=bt.PRESET_WEIGHTS["Balanced"], events_df=events_df, max_symbols=int(max_syms)
        )
        if scores.empty:
            st.error("No historical data fetched. Try a different date range or smaller universe.")
        else:
            freq = "D" if rebalance == "Daily" else "W"
            wts = bt.rebalance_weights(scores, top_n=int(top_n), min_score=float(min_score), freq=freq)
            port, rets = bt.simulate_portfolio(wts, closes, ohlc_map, sl_pct=float(sl_pct), tp_pct=float(tp_pct), cost_bps=float(cost_bps))

            m = bt.metrics_from_returns(port)
            def fmt_pct(x): return ("—" if x is None or pd.isna(x) else f"{x*100:.1f}%")
            cols = st.columns(6)
            with cols[0]: st.metric("Total Return", fmt_pct(m.get("Total Return")))
            with cols[1]: st.metric("CAGR", fmt_pct(m.get("CAGR")))
            with cols[2]: st.metric("Volatility", fmt_pct(m.get("Volatility")))
            with cols[3]: st.metric("Sharpe (rf=0)", f"{m.get('Sharpe (rf=0)', float('nan')):.2f}" if m.get('Sharpe (rf=0)')==m.get('Sharpe (rf=0)') else "—")
            with cols[4]: st.metric("Max Drawdown", fmt_pct(m.get("Max Drawdown")))
            with cols[5]: st.metric("Hit Rate", fmt_pct(m.get("Hit Rate")))

            eq = (1 + port).cumprod()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=eq.index, y=eq.values, mode="lines", name="Equity"))
            fig.update_layout(margin=dict(l=0,r=0,b=0,t=30), height=320)
            st.plotly_chart(fig, use_container_width=True)

            # Trade log
            tlog = bt.trade_log_from_weights(wts, rets)
            if not tlog.empty:
                tlog = tlog.sort_values("entry_date")
                tlog["return_pct"] = (tlog["return"] * 100).round(2)
                st.subheader("Trade Log (entries/exits)")
                st.dataframe(tlog, use_container_width=True, height=300)
                st.download_button("Download trade log", data=tlog.to_csv(index=False), file_name="trade_log.csv")

# Parameter sweep
st.markdown("---")
st.subheader("Parameter Sweep (quick scan)")
colA, colB, colC, colD = st.columns(4)
with colA:
    presets = st.multiselect("Weight presets", list(bt.PRESET_WEIGHTS.keys()), default=["Balanced","Momentum-heavy","Catalyst-heavy"])
with colB:
    topn_list = st.multiselect("Top N list", [10,15,20,25,30], default=[15,20,30])
with colC:
    score_list = st.multiselect("Min score list", [0.3,0.4,0.5,0.6], default=[0.4,0.5])
with colD:
    freq_list = st.multiselect("Rebalance", ["Daily","Weekly"], default=["Weekly"])

max_syms_sweep = st.slider("Max tickers for sweep", 20, 200, 80, step=10)
run_sweep = st.button("Run Sweep")
if run_sweep:
    with st.spinner("Sweeping presets × topN × thresholds × freq…"):
        res = bt.sweep(
            symbols=list(universe["symbol"].tolist()),
            start=str(start_date), end=str(end_date),
            presets=presets, topn_list=topn_list, score_list=score_list, freq_list=freq_list,
            max_symbols=int(max_syms_sweep)
        )
        if res.empty:
            st.warning("No results. Try widening the date range or increasing max tickers.")
        else:
            # Pretty-format percentages
            show = res.copy()
            for k in ["Total Return","CAGR","Volatility","Max Drawdown","Hit Rate"]:
                show[k] = (show[k] * 100).round(1).astype(str) + "%"
            show["Sharpe (rf=0)"] = res["Sharpe (rf=0)"].round(2)
            st.dataframe(show, use_container_width=True, height=380)
            st.download_button("Download sweep results", data=res.to_csv(index=False), file_name="sweep_results.csv")

st.caption("Backtests approximate SL/TP using daily high/low vs prior close (order unknown). Costs applied on turnover at rebalance. Research use only, not financial advice.")
