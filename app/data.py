import time
import pandas as pd
import requests
import yfinance as yf

BASE = "https://finnhub.io/api/v1"

def _auth(api_key: str) -> dict:
    return {"token": api_key}

# ---------- FINNHUB (preferred) ----------
def list_us_symbols_finnhub(api_key: str, industry_filters=("Biotechnology","Pharmaceuticals")) -> pd.DataFrame:
    """Return US-listed symbols. If 'finnhubIndustry' is missing, skip industry filter."""
    url = f"{BASE}/stock/symbol"
    params = {"exchange": "US", **_auth(api_key)}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    if df.empty:
        return df
    # keep safe subset
    keep = [c for c in ["symbol","description","finnhubIndustry","type","currency"] if c in df.columns]
    df = df[keep].drop_duplicates()
    if "finnhubIndustry" in df.columns:
        df = df[df["finnhubIndustry"].isin(industry_filters)].copy()
    if "type" in df.columns:
        df = df[(df["type"].isin(["Common Stock","","EQS"])) | df["type"].isna()]
    return df.reset_index(drop=True)

def get_candles_finnhub(api_key: str, symbol: str, resolution: str = "D", lookback_days: int = 120) -> pd.DataFrame:
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
    return df.set_index("t").sort_index()

# ---------- YFINANCE (fallback) ----------
_YF_RES_MAP = {
    "D": ("1d", None),
    "60": ("60m", "60d"),
    "30": ("30m", "60d"),
    "15": ("15m", "60d"),
    "5":  ("5m",  "60d"),
    "1":  ("1m",  "7d"),
}

def list_symbols_from_inputs(seed_csv_path: str, uploaded_df: pd.DataFrame = None, pasted: str = None) -> pd.DataFrame:
    """Return a DataFrame with a single 'symbol' column from uploaded CSV, textarea, or seed file."""
    if uploaded_df is not None and not uploaded_df.empty:
        col = [c for c in uploaded_df.columns if c.lower() == "symbol"]
        if col:
            syms = uploaded_df[col[0]].astype(str).str.upper().str.strip().unique().tolist()
            return pd.DataFrame({"symbol": syms})
    if pasted:
        syms = [s.strip().upper() for s in pasted.replace("\n", ",").split(",") if s.strip()]
        if syms:
            return pd.DataFrame({"symbol": list(dict.fromkeys(syms))})
    # fallback to seed
    seed = pd.read_csv(seed_csv_path)
    seed["symbol"] = seed["symbol"].astype(str).str.upper().str.strip()
    return seed[["symbol"]].drop_duplicates().reset_index(drop=True)

def get_candles_yf(symbol: str, resolution: str = "D", lookback_days: int = 120) -> pd.DataFrame:
    interval, period = _YF_RES_MAP.get(resolution, ("1d", None))
    if period is None:
        start = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=lookback_days+5)
        df = yf.download(symbol, start=start, progress=False, interval=interval, auto_adjust=False)
    else:
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns={"Open":"o","High":"h","Low":"l","Close":"c","Volume":"v"})
    df.index = pd.to_datetime(df.index, utc=True)
    return df[["o","h","l","c","v"]].sort_index()
