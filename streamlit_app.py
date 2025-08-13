import os
import time
import pandas as pd
import numpy as np
import streamlit as st
from app.config import get_finnhub_key
from app import data as md
from app import indicators as ind
from app import scoring as ms
from app import catalysts as cat

st.set_page_config(page_title="Moonshot Biotech", page_icon="🚀", layout="wide")
st.title("🚀 Moonshot Biotech Dashboard")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Finnhub API Key", value=get_finnhub_key(), type="password")
    resolution = st.selectbox("Candle Resolution", ["D","60","30","15","5","1"], index=0)
    lookback_days = st.slider("Lookback window (days)", 60, 365, 180)
    st.markdown("---")
    st.subheader("Moonshot Weights")
    w_price = st.slider("Price Breakout", 0, 100, 21)
    w_rsi = st.slider("RSI Momentum", 0, 100, 25)
    w_macd = st.slider("MACD Signal", 0, 100, 20)
    w_vol = st.slider("Volume Spike", 0, 100, 20)
    w_cat = st.slider("Catalyst Proximity", 0, 100, 10)
    weights = {
        "price_breakout": w_price/100.0,
        "rsi_momentum": w_rsi/100.0,
        "macd_signal": w_macd/100.0,
        "volume_spike": w_vol/100.0,
        "catalyst_proximity": w_cat/100.0
    }
    st.markdown("---")
    st.subheader("Catalysts (optional)")
    up = st.file_uploader("Upload catalysts CSV", type=["csv"])
    events_df = None
    if up:
        try:
            events_df = cat.load_catalysts_csv(up)
            st.success(f"Catalysts loaded: {len(events_df)} rows")
        except Exception as e:
            st.error(f"Failed to parse catalysts CSV: {e}")

st.caption("Tip: Add your Finnhub key in Streamlit Cloud → Secrets.")

if not api_key:
    st.info("Enter your Finnhub API key in the sidebar or set it in secrets.")
    st.stop()

@st.cache_data(ttl=3600)
def load_universe(api_key: str) -> pd.DataFrame:
    df = md.list_us_symbols(api_key)
    return df

universe = load_universe(api_key)
st.write(f"Universe size: {len(universe)} symbols" if len(universe) else "Universe is empty (check API limits).")

sample = st.number_input("Scan first N symbols", min_value=10, max_value=int(max(10, len(universe) or 10)), value=150, step=10)
symbols = list(universe["symbol"].head(int(sample)))

progress = st.progress(0, text="Fetching & computing…")
rows = []
for i, sym in enumerate(symbols, start=1):
    try:
        candles = md.get_candles(api_key, sym, resolution=resolution, lookback_days=int(lookback_days))
        if candles.empty or len(candles) < 35:
            continue
        c = candles["c"]
        v = candles["v"]
        sma10 = ind.sma(c, 10)
        price_breakout = float(c.iloc[-1] > 1.05 * sma10.iloc[-1]) if not np.isnan(sma10.iloc[-1]) else 0.0
        rsi = ind.rsi(c, 14).iloc[-1]
        macd_line, signal_line, hist = ind.macd(c)
        macd_hist = hist.iloc[-1]
        vol_spike = ind.volume_spike(v, 30).iloc[-1] if len(v) >= 30 else 0.0
        catalyst_component = cat.catalyst_proximity_component(events_df, sym) if events_df is not None else 0.0
        rows.append({
            "symbol": sym,
            "price": float(c.iloc[-1]),
            "price_breakout": price_breakout,
            "rsi": float(rsi) if not np.isnan(rsi) else np.nan,
            "macd_hist": float(macd_hist) if not np.isnan(macd_hist) else 0.0,
            "vol_spike": float(vol_spike) if not np.isnan(vol_spike) else 0.0,
            "catalyst_component": float(catalyst_component)
        })
    except Exception:
        pass
    progress.progress(i / max(len(symbols), 1))

df = pd.DataFrame(rows)
if df.empty:
    st.warning("No data returned. Try lower N or daily candles to reduce rate limits.")
    st.stop()

scored = ms.apply_scoring(df, weights).sort_values("moonshot_score", ascending=False)
st.subheader("Candidates")
st.dataframe(scored.head(100), use_container_width=True)
st.download_button("Download CSV", data=scored.to_csv(index=False), file_name="moonshot_candidates.csv")

st.markdown("""
**Scoring Notes**
- price_breakout = close > 1.05 × 10‑day SMA  
- rsi = 14‑period RSI; >55 up‑weights, <30 light bounce potential  
- macd_hist > 0 favors bullish momentum  
- vol_spike = volume / 30‑day average (capped)  
- catalyst_component = decays from 1 → 0 at 45 days out  
""")
