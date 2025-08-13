# Moonshot Biotech – Streamlit App

A Streamlit dashboard to surface potential **moonshot** moves in U.S.-listed biotech & pharma stocks.
It uses the Finnhub API for symbols, quotes and candles, computes technical indicators (RSI, MACD, MAs),
lets you tune weights, and (optionally) merges **FDA/catalyst** events you upload as a CSV.

---

## Features
- Pull the *full* U.S. biotech+pharma universe from Finnhub (not just top 20).
- Real-time quotes and intraday/daily candles.
- Technical indicators: RSI, MACD, SMA/EMA, volume spike vs 30-day avg.
- Configurable **Moonshot Score** (default weights below).
- Upload a catalysts CSV (PDUFA, trial readouts, etc.) and auto-join by ticker & date proximity.
- Alert table + downloadable CSV of candidates.
- Deployable to Streamlit Community Cloud or Hugging Face Spaces for a free URL.

### Default Moonshot Scoring Weights
- Price Breakout (>5% above 10‑day avg): **21%**
- RSI Momentum (oversold/overbought): **25%**
- MACD Signal (bullish cross, momentum): **20%**
- Volume Spike (vs 30‑day avg): **20%**
- Catalyst Proximity (upcoming FDA events): **10%**

You can customize these in the sidebar.

---

## Getting Started (Local)

1) **Install Python 3.10+** and **git**.
2) Create a virtual env and install requirements:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3) Set your **Finnhub API key**:
   - Copy `.env.example` to `.env` and add your key, **or**
   - Put it in `.streamlit/secrets.toml` as shown below.

`.env.example`:
```
FINNHUB_API_KEY=YOUR_KEY_HERE
```

`.streamlit/secrets.toml`:
```toml
FINNHUB_API_KEY = "YOUR_KEY_HERE"
```

4) Run the app:
```bash
streamlit run streamlit_app.py
```

The terminal will show a local URL (e.g., http://localhost:8501) and a public URL for temporary sharing (if enabled by Streamlit).

---

## Deploy (Free URL)

### Option A) **Streamlit Community Cloud** (easiest)
1. Push this folder to a **public GitHub repo**.
2. Go to https://share.streamlit.io, connect GitHub, choose the repo & `streamlit_app.py` as the entry point.
3. Set **Secrets** in the app settings:
   ```toml
   FINNHUB_API_KEY = "YOUR_KEY_HERE"
   ```
4. Click **Deploy** → you get a public URL.

### Option B) **Hugging Face Spaces**
1. Create a new *Space* → Type: **Streamlit**.
2. Upload your repo (or connect to GitHub).
3. Add a *Secret* named `FINNHUB_API_KEY`.
4. The Space builds and gives you a public URL.

---

## Catalysts CSV Format
Upload a CSV with at least:
```
ticker,event_date,event_type,notes,source
```
- `event_date` in `YYYY-MM-DD`.
- Example in `data/sample_catalysts.csv`.

---

## Notes
- Heavy universes & intraday candles can rate-limit Finnhub on free tiers; consider batching & caching.
- If you prefer yfinance for backfill, you can add it alongside Finnhub.
- This project is meant as **research tooling**, not financial advice.
