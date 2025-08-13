# Moonshot Biotech – Streamlit App
A Streamlit dashboard to surface potential **moonshot** moves in U.S.-listed biotech & pharma stocks.
It uses the Finnhub API for symbols, quotes and candles, computes technical indicators (RSI, MACD, MAs),
lets you tune weights, and (optionally) merges **FDA/catalyst** events you upload as a CSV.

## Features
- Pull the *full* U.S. biotech+pharma universe from Finnhub.
- Real-time quotes and intraday/daily candles.
- Technical indicators: RSI, MACD, SMA/EMA, volume spike vs 30-day avg.
- Configurable **Moonshot Score**.
- Upload a catalysts CSV (PDUFA, trial readouts, etc.) and auto-join by ticker & date proximity.
- Alert table + downloadable CSV of candidates.
- Deployable to Streamlit Community Cloud or Hugging Face Spaces for a free URL.

### Default Moonshot Scoring Weights
- Price Breakout: **21%**
- RSI Momentum: **25%**
- MACD Signal: **20%**
- Volume Spike: **20%**
- Catalyst Proximity: **10%**

## Getting Started (Local)
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy (Free URL)
### Streamlit Community Cloud
- Push to GitHub → share.streamlit.io → New app (main file: `streamlit_app.py`).
- Add secret:
```toml
FINNHUB_API_KEY = "YOUR_KEY"
```

## Catalysts CSV Format
```
ticker,event_date,event_type,notes,source
```
