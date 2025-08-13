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
    st.header("Data Source")
    source = st.radio("Select data backend", ["Finnhub (preferred)", "yfinance / custom"])
    api_key = st.text_input("Finnhub API Key (optional)", value=get_finnhub_key(), type="password")
    resolution = st.selectbox("Candle Resolution", ["D","60","30","15","5","1"], index=0)
    lookback_days = st.slider("Lookback window (days)", 60, 365, 180)

    st.markdown("---")
    st.subheader("Universe (yfinance/custom only)")
    up_universe = st.file_uploader("Upload tickers CSV (col: symbol)", type=["csv"], key="uni")
    pasted = st.text_area("...or paste tickers separated by comma/newline", height=100)
    st.caption("If empty, a small built-in biotech seed list is used.")

    st.markdown("---")
    st.subheader("Moonshot Weights")
    w_price = st.slider("Price Breakout", 0, 100, 21)
    w_rsi   = st.slider("RSI Momentum", 0, 100, 25)
    w_macd  = st.slider("MACD Signal", 0, 100, 20)
    w_vol   = st.slider("Volume Spike", 0, 100, 20)
    w_cat   = st.slider("Catalyst Proximity", 0, 100, 10)
    weights = {
        "price_breakout": w_price/100.0,
        "rsi_momentum":   w_rsi/100.0,
        "macd_signal":    w_macd/100.0,
        "volume_spike":   w_vol/100.0,
        "catalyst_proximity": w_cat/100.0
    }

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

st.caption("Tip: If Finnhub fails, the app will try yfinance automatically.")

@st.cache_data(ttl=3600)
def load_universe_finnhub(api_key: str) -> pd.DataFrame:
    return md.list_us_symbols_finnhub(api_key)

@st.cache_data(ttl=3600)
def load_universe_yf(seed_path: str, up_df) -> pd.DataFrame:
    return md.list_symbols_from_inputs(seed_path, up_df, pasted)

def get_candles(symbol: str, api_key: str, resolution: str, lookback_days: int, prefer_finnhub: bool):
    if prefer_finnhub and api_key:
        try:
            df = md.get_candles_finnhub(api_key, symbol, resolution=resolution, lookback_days=lookback_days)
            if df is not None and not df.empty:
                return df, "finnhub"
        except Exception:
            pass
    # fallback
    try:
        df = md.get_candles_yf(symbol, resolution=resolution, lookback_days=lookback_days)
        if df is not None and not df.empty:
            return df, "yfinance"
    except Exception:
        pass
    return pd.DataFrame(), "none"

# Build universe
if source == "Finnhub (preferred)" and api_key:
    universe = load_universe_finnhub(api_key)
    used_source = "finnhub"
    if universe is None or universe.empty:
        st.warning("Finnhub universe empty/failed — falling back to yfinance/custom.")
        universe = load_universe_yf("data/biotech_seed.csv", pd.read_csv(up_universe) if up_universe else None)
        used_source = "yfinance/custom"
else:
    universe = load_universe_yf("data/biotech_seed.csv", pd.read_csv(up_universe) if up_universe else None)
    used_source = "yfinance/custom"

if universe is None or universe.empty:
    st.error("No symbols available. Upload/paste a ticker list or try again.")
    st.stop()

st.write(f"Universe size: {len(universe)} symbols (source: {used_source})")

sample = st.number_input("Scan first N symbols", min_value=5, max_value=int(max(5, len(universe) or 5)), value=min(150, len(universe)), step=5)
symbols = list(universe["symbol"].head(int(sample)))

progress = st.progress(0, text="Fetching & computing…")
rows = []
for i, sym in enumerate(symbols, start=1):
    df, ds = get_candles(sym, api_key, resolution, int(lookback_days), prefer_finnhub=(source.startswith("Finnhub")))
    if df is None or df.empty or len(df) < 35:
        progress.progress(i / max(len(symbols), 1))
        continue
    try:
        c = df["c"]; v = df["v"]
        sma10 = ind.sma(c, 10)
        price_breakout = float(c.iloc[-1] > 1.05 * (sma10.iloc[-1] if not pd.isna(sma10.iloc[-1]) else c.iloc[-1]))
        rsi = ind.rsi(c, 14).iloc[-1]
        macd_line, signal_line, hist = ind.macd(c)
        macd_hist = hist.iloc[-1] if not pd.isna(hist.iloc[-1]) else 0.0
        vol_spike = ind.volume_spike(v, 30).iloc[-1] if len(v) >= 30 else 0.0
        catalyst_component = cat.catalyst_proximity_component(events_df, sym) if events_df is not None else 0.0
        rows.append({
            "symbol": sym,
            "price": float(c.iloc[-1]),
            "data_source": ds,
            "price_breakout": price_breakout,
            "rsi": float(rsi) if not pd.isna(rsi) else np.nan,
            "macd_hist": float(macd_hist),
            "vol_spike": float(vol_spike) if not pd.isna(vol_spike) else 0.0,
            "catalyst_component": float(catalyst_component)
        })
    except Exception:
        pass
    progress.progress(i / max(len(symbols), 1))

df = pd.DataFrame(rows)
if df.empty:
    st.warning("No data returned. Reduce N, use daily candles, or try again to avoid rate limits.")
    st.stop()

scored = ms.apply_scoring(df, weights).sort_values("moonshot_score", ascending=False)
st.subheader("Candidates")
st.dataframe(scored.head(100), use_container_width=True)
st.download_button("Download CSV", data=scored.to_csv(index=False), file_name="moonshot_candidates.csv")

st.markdown("""
**Notes**
- If Finnhub is selected and available, the app uses Finnhub; otherwise it automatically falls back to **yfinance**.
- You can upload or paste tickers to define a custom universe when using yfinance.
- Scoring:
  - price_breakout = close > 1.05 × 10‑day SMA
  - rsi = 14‑period RSI; >55 up‑weights, <30 light bounce potential
  - macd_hist > 0 favors bullish momentum
  - vol_spike = volume / 30‑day average (capped)
  - catalyst_component = decays from 1 → 0 at 45 days out
""")
