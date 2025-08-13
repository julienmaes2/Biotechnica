import pandas as pd
def load_catalysts_csv(path: str) -> pd.DataFrame:
    try: df = pd.read_csv(path)
    except Exception: df = pd.read_csv(path, encoding="utf-8", engine="python")
    cols_lower = [c.lower() for c in df.columns]
    if "ticker" not in cols_lower or "event_date" not in cols_lower:
        raise ValueError("Catalysts CSV must include at least: ticker,event_date")
    df["event_date"] = pd.to_datetime(df[df.columns[cols_lower.index("event_date")]]).dt.date
    df["ticker"] = df[df.columns[cols_lower.index("ticker")]].astype(str).str.upper()
    if "event_type" not in [c.lower() for c in df.columns]: df["event_type"] = ""
    if "notes" not in [c.lower() for c in df.columns]: df["notes"] = ""
    return df
def catalyst_proximity_component(events_df: pd.DataFrame, ticker: str, today=None, lookahead_days=45):
    if events_df is None or events_df.empty: return 0.0
    if today is None: today = pd.Timestamp.today().date()
    sub = events_df[events_df["ticker"] == ticker]
    if sub.empty: return 0.0
    nearest = (sub["event_date"] - pd.Timestamp(today).date()).apply(lambda d: d.days)
    nearest = nearest[nearest >= 0]
    if nearest.empty: return 0.0
    days = nearest.min(); return max(0.0, 1.0 - days / lookahead_days)
