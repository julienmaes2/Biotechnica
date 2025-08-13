import time
import pandas as pd
import requests
from datetime import datetime, timedelta, timezone

BASE = "https://finnhub.io/api/v1"

def _auth(api_key: str) -> dict:
    return {"token": api_key}

def list_us_symbols(api_key: str, industry_filters=("Biotechnology","Pharmaceuticals")) -> pd.DataFrame:
    """List U.S. symbols, filter to biotech/pharma by Finnhub's 'finnhubIndustry' field."""
    url = f"{BASE}/stock/symbol"
    params = {"exchange": "US", **_auth(api_key)}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    if df.empty:
        return df
    if "finnhubIndustry" in df.columns:
        df = df[df["finnhubIndustry"].isin(industry_filters)].copy()
    # Keep relevant columns
    keep = ["symbol","description","finnhubIndustry","type","currency"]
    return df[keep].drop_duplicates()

def get_quote(api_key: str, symbol: str) -> dict:
    url = f"{BASE}/quote"
    params = {"symbol": symbol, **_auth(api_key)}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json()

def get_candles(api_key: str, symbol: str, resolution: str = "D", lookback_days: int = 120) -> pd.DataFrame:
    """Get candle data for `symbol`. resolution: 1,5,15,30,60, D, W, M"""
    to_ts = int(time.time())
    from_ts = to_ts - lookback_days * 86400
    url = f"{BASE}/stock/candle"
    params = {"symbol": symbol, "resolution": resolution, "from": from_ts, "to": to_ts, **_auth(api_key)}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    if data.get("s") != "ok":
        return pd.DataFrame()
    df = pd.DataFrame({
        "t": pd.to_datetime(data["t"], unit="s", utc=True),
        "o": data["o"], "h": data["h"], "l": data["l"], "c": data["c"], "v": data["v"]
    })
    df = df.set_index("t").sort_index()
    return df
