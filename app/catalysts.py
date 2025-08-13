import pandas as pd
from datetime import datetime, timedelta

def load_catalysts_csv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, encoding="utf-8", engine="python")
    # Normalize
    cols = {c.lower(): c for c in df.columns}
    # Ensure expected columns exist
    for required in ["ticker","event_date"]:
        if required not in [c.lower() for c in df.columns]:
            raise ValueError("Catalysts CSV must include at least: ticker,event_date")
    # Parse date
    df["event_date"] = pd.to_datetime(df[[c for c in df.columns if c.lower()=="event_date"][0]]).dt.date
    # Uppercase tickers
    df["ticker"] = df[[c for c in df.columns if c.lower()=="ticker"][0]].astype(str).str.upper()
    return df

def catalyst_proximity_component(events_df: pd.DataFrame, ticker: str, today=None, lookahead_days=45):
    if events_df is None or events_df.empty:
        return 0.0
    if today is None:
        today = pd.Timestamp.today().date()
    sub = events_df[events_df["ticker"] == ticker]
    if sub.empty:
        return 0.0
    nearest = (sub["event_date"] - pd.Timestamp(today).date()).apply(lambda d: d.days)
    nearest = nearest[nearest >= 0]
    if nearest.empty:
        return 0.0
    days = nearest.min()
    # Decay: 1.0 at 0 days, 0.0 at >= lookahead_days
    return max(0.0, 1.0 - days / lookahead_days)
