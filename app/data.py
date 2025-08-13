import time
import pandas as pd
import requests

BASE = "https://finnhub.io/api/v1"

def _auth(api_key: str) -> dict:
    return {"token": api_key}

def list_us_symbols(api_key: str, industry_filters=("Biotechnology","Pharmaceuticals")) -> pd.DataFrame:
    """Return US-listed symbols. If 'finnhubIndustry' is missing, skip industry filtering safely."""
    url = f"{BASE}/stock/symbol"
    params = {"exchange": "US", **_auth(api_key)}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    if df.empty:
        return df

    # Normalize columns we care about
    keep = [c for c in ["symbol","description","finnhubIndustry","type","currency"] if c in df.columns]
    df = df[keep].drop_duplicates()

    # Filter by industry IF the column exists; otherwise return all US symbols
    if "finnhubIndustry" in df.columns:
        df = df[df["finnhubIndustry"].isin(industry_filters)].copy()

    # Basic sanity: keep only 'Common Stock' or empty type if present
    if "type" in df.columns:
        df = df[(df["type"].isin(["Common Stock","","EQS"])) | df["type"].isna()]

    return df.reset_index(drop=True)

def get_candles(api_key: str, symbol: str, resolution: str = "D", lookback_days: int = 120) -> pd.DataFrame:
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
