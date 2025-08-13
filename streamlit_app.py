import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from datetime import date, timedelta
from app.config import get_finnhub_key, get_secret
from app import data as md
from app import indicators as ind
from app import scoring as ms
from app import catalysts as cat
from app import news as newsmod
from app import backtest as bt

st.set_page_config(page_title="Moonshot Biotech", page_icon="🚀", layout="wide")
st.title("🚀 Moonshot Biotech — Scanner & Backtest")

# Sidebar
with st.sidebar:
    st.header("Data Source")
    source = st.radio("Backend", ["Finnhub (preferred)", "yfinance / custom"], index=1)
    api_key = st.text_input("Finnhub API Key (optional)", value=get_finnhub_key(), type="password")
    resolution = st.selectbox("Candle Resolution", ["D","60","30","15","5","1"], index=0)
    lookback_days = st.slider("Lookback (days)", 60, 365, 180)

    st.markdown("---")
    st.subheader("Universe (optional)")
    up_universe = st.file_uploader("Upload tickers CSV (col: symbol)", type=["csv"], key="uni")
    pasted = st.text_area("...or paste tickers separated by comma/newline", height=70)
    sheet_url = st.text_input("Google Sheet URL (optional)")

    st.markdown("---")
    st.subheader("Watchlists")
    wl_choice = st.selectbox("Choose watchlist", ["All (seed)", "XBI (seed)", "IBB (seed)", "SMID (seed)"], index=0)

    st.markdown("---")
    st.subheader("Selection View")
    focus_mode = st.selectbox("Auto-select by", ["Moonshot Score", "% Change (today)", "Volume Spike vs 30d"], index=0)
    top_k = st.slider("Rows to show", 20, 200, 100, step=10)

    st.markdown("---")
    st.subheader("Weights")
    w_price = st.slider("Price Breakout", 0, 100, 21)
    w_rsi   = st.slider("RSI Momentum", 0, 100, 25)
    w_macd  = st.slider("MACD Signal", 0, 100, 20)
    w_vol   = st.slider("Volume Spike", 0, 100, 20)
    w_cat   = st.slider("Catalyst Proximity", 0, 100, 10)
    weights = {"price_breakout": w_price/100.0, "rsi_momentum": w_rsi/100.0, "macd_signal": w_macd/100.0, "volume_spike": w_vol/100.0, "catalyst_proximity": w_cat/100.0}

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
    universe = load_universe_finnhub(api_key); used_source = "finnhub"
    if universe is None or universe.empty:
        st.warning("Finnhub universe empty/failed — using yfinance/custom.")
        universe = load_universe_yf("data/biotech_seed.csv", pd.read_csv(up_universe) if up_universe else None, pasted, sheet_url); used_source="yfinance/custom"
else:
    universe = load_universe_yf("data/biotech_seed.csv", pd.read_csv(up_universe) if up_universe else None, pasted, sheet_url); used_source="yfinance/custom"

# Watchlist override
if wl_choice == "XBI (seed)":
    wl = pd.read_csv("data/watchlist_xbi_seed.csv")
elif wl_choice == "IBB (seed)":
    wl = pd.read_csv("data/watchlist_ibb_seed.csv")
elif wl_choice == "SMID (seed)":
    wl = pd.read_csv("data/watchlist_smid_seed.csv")
else:
    wl = None
if wl is not None and not wl.empty:
    universe = pd.DataFrame({"symbol": wl["symbol"].astype(str).str.upper().unique().tolist()})

if universe is None or universe.empty:
    st.error("No symbols available. Upload/paste tickers or try again later."); st.stop()

st.write(f"Universe size: {len(universe)} symbols (source: {used_source}; watchlist: {wl_choice})")

# Live scan
rows = []
symbols = list(universe["symbol"].tolist())
progress = st.progress(0, text="Scanning live…")
for i, sym in enumerate(symbols, start=1):
    try:
        if source == "Finnhub (preferred)" and api_key:
            df = md.get_candles_finnhub(api_key, sym, resolution=resolution, lookback_days=int(lookback_days)); ds="finnhub"
        else:
            df = md.get_candles_yf(sym, resolution=resolution, lookback_days=int(lookback_days)); ds="yfinance"
        if df is None or df.empty or len(df) < 35: progress.progress(i/len(symbols)); continue
        c = df["c"]; v = df["v"]; sma10 = c.rolling(10, min_periods=10).mean()
        price_breakout = float(c.iloc[-1] > 1.05 * (sma10.iloc[-1] if not pd.isna(sma10.iloc[-1]) else c.iloc[-1]))
        from app.indicators import rsi, macd, volume_spike
        rsi_val = rsi(c, 14).iloc[-1]; _,_,hist = macd(c)
        macd_hist = hist.iloc[-1] if not pd.isna(hist.iloc[-1]) else 0.0
        vol_spike = volume_spike(v, 30).iloc[-1] if len(v)>=30 else 0.0
        catalyst_component = 0.0 if events_df is None else 0.0  # simple live placeholder
        pct_change = float((c.iloc[-1]/c.iloc[-2]-1.0)*100.0) if len(c)>=2 else 0.0
        rows.append({"symbol":sym,"price":float(c.iloc[-1]),"pct_change":pct_change,"data_source":ds,
                     "price_breakout":price_breakout,"rsi":float(rsi_val) if not pd.isna(rsi_val) else np.nan,
                     "macd_hist":float(macd_hist),"vol_spike":float(vol_spike) if not pd.isna(vol_spike) else 0.0,
                     "catalyst_component":float(catalyst_component)})
    except Exception: pass
    progress.progress(i/len(symbols))

scan_df = pd.DataFrame(rows)
if scan_df.empty:
    st.warning("No live data returned. Try daily candles, or a smaller universe.")
else:
    scan_scored = ms.apply_scoring(scan_df, weights)
    if focus_mode == "Moonshot Score":
        view = scan_scored.sort_values("moonshot_score", ascending=False).head(top_k)
    elif focus_mode == "% Change (today)":
        view = scan_scored.sort_values("pct_change", ascending=False).head(top_k)
    else:
        view = scan_scored.sort_values("vol_spike", ascending=False).head(top_k)
    st.subheader("Scanner")
    st.dataframe(view.reset_index(drop=True), use_container_width=True)
    st.download_button("Download scanner CSV", data=view.to_csv(index=False), file_name="scanner.csv")

tabs = st.tabs(["Backtest", "News"])

with tabs[0]:
    st.subheader("Backtest")
    colA, colB, colC = st.columns(3)
    with colA: start_date = st.date_input("Start date", value=(date.today()-timedelta(days=365*2)))
    with colB: end_date   = st.date_input("End date", value=date.today())
    with colC: rebalance  = st.selectbox("Rebalance", ["Daily","Weekly"], index=1)
    top_n = st.slider("Top N per rebalance", 5, 50, 20, step=1)
    min_score = st.slider("Min score threshold", 0.0, 1.0, 0.4, 0.05)
    sl_pct = st.slider("Stop-loss (%)", 0.0, 20.0, 8.0, 0.5)/100.0
    tp_pct = st.slider("Take-profit (%)", 0.0, 50.0, 15.0, 0.5)/100.0
    cost_bps = st.slider("Transaction cost (bps of turnover)", 0, 100, 10, 1)
    max_syms = st.slider("Max tickers (speed cap)", 20, 300, min(120, len(universe)), step=10)
    run_bt = st.button("Run Backtest")
    if run_bt:
        with st.spinner("Crunching…"):
            scores, closes, ohlc_map = bt.build_rank_table(list(universe["symbol"].tolist()), str(start_date), str(end_date), weights, max_symbols=int(max_syms))
            if scores.empty or closes.empty:
                st.error("No historical data fetched. Try another date range or smaller universe.")
            else:
                freq = "D" if rebalance=="Daily" else "W"
                wts = bt.rebalance_weights(scores, int(top_n), float(min_score), freq=freq)
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
                eq = (1+port).cumprod(); fig = go.Figure(); fig.add_trace(go.Scatter(x=eq.index, y=eq.values, mode="lines", name="Equity"))
                fig.update_layout(margin=dict(l=0,r=0,b=0,t=30), height=320); st.plotly_chart(fig, use_container_width=True)
                tlog = bt.trade_log_from_weights(wts, rets)
                if not tlog.empty:
                    tlog = tlog.sort_values("entry_date"); tlog["return_pct"] = (tlog["return"]*100).round(2)
                    st.subheader("Trade Log"); st.dataframe(tlog, use_container_width=True, height=320)
                    st.download_button("Download trade log", data=tlog.to_csv(index=False), file_name="trade_log.csv")

    st.markdown("---")
    st.subheader("Parameter Sweep")
    colA, colB, colC, colD = st.columns(4)
    with colA: presets = st.multiselect("Weight presets", list(bt.PRESET_WEIGHTS.keys()), default=list(bt.PRESET_WEIGHTS.keys()))
    with colB: topn_list = st.multiselect("Top N list", [10,15,20,25,30], default=[15,20,30])
    with colC: score_list = st.multiselect("Min score list", [0.3,0.4,0.5,0.6], default=[0.4,0.5])
    with colD: freq_list = st.multiselect("Rebalance", ["Daily","Weekly"], default=["Weekly"])
    max_syms_sweep = st.slider("Max tickers for sweep", 20, 200, 80, step=10)
    run_sweep = st.button("Run Sweep")
    if run_sweep:
        with st.spinner("Sweeping…"):
            res = bt.sweep(list(universe["symbol"].tolist()), str(start_date), str(end_date), presets, topn_list, score_list, freq_list, max_symbols=int(max_syms_sweep))
            if res.empty: st.warning("No results. Try widening the window or adding tickers.")
            else:
                show = res.copy()
                for k in ["Total Return","CAGR","Volatility","Max Drawdown","Hit Rate"]:
                    show[k] = (show[k]*100).round(1).astype(str) + "%"
                show["Sharpe (rf=0)"] = res["Sharpe (rf=0)"].round(2)
                st.dataframe(show, use_container_width=True, height=360)
                st.download_button("Download sweep", data=res.to_csv(index=False), file_name="sweep_results.csv")

with tabs[1]:
    st.subheader("Biotech News")
    news_df = newsmod.fetch_biotech_headlines(max_items=80)
    if news_df is not None and not news_df.empty:
        for _, r in news_df.head(30).iterrows():
            st.markdown(f"- [{r['title']}]({r['link']}) — *{r['source']}*")
    else:
        st.info("No headlines fetched yet.")

st.caption("Main file = streamlit_app.py. If Finnhub fails or is omitted, the scanner falls back to yfinance. Backtests approximate SL/TP using daily high/low vs prior close. Research use only, not financial advice.")
