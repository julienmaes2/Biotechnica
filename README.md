# Moonshot Biotech – Streamlit App (Finnhub + yfinance fallback)
A Streamlit dashboard to surface potential **moonshot** moves in U.S.-listed biotech & pharma stocks.

## Data sources
- **Finnhub (preferred)** for complete US symbol universe & candles — uses your API key.
- **Automatic fallback to yfinance** when Finnhub is unavailable. You can also **upload/paste** tickers.

## Quick Deploy (Streamlit Cloud)
1) Push these files to a **public GitHub repo**.
2) Deploy at share.streamlit.io → Main file: `streamlit_app.py`.
3) Add secret:
```toml
FINNHUB_API_KEY = "YOUR_KEY"
```
