# Moonshot Biotech — Full Bundle
Streamlit app with:
- Finnhub preferred, automatic yfinance fallback
- Beginner/Pro scanner with watchlists, favorites, news panel
- Backtest tab with SL/TP, transaction costs, trade log, parameter sweep

## Deploy (Streamlit Cloud)
- Main file: `streamlit_app.py`
- Python: `.streamlit/runtime.txt` is 3.10
- Secrets (optional):
```toml
FINNHUB_API_KEY="YOUR_KEY"
TELEGRAM_BOT_TOKEN=""
TELEGRAM_CHAT_ID=""
SENDGRID_API_KEY=""
EMAIL_FROM=""
EMAIL_TO=""
```
