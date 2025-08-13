import numpy as np
import pandas as pd
def moonshot_score_row_like(price_breakout, rsi_val, macd_hist, vol_spike, catalyst_component, weights):
    if pd.isna(rsi_val): rsi_component = 0.0
    else:
        if rsi_val >= 55: rsi_component = min((rsi_val - 55) / 25, 1.0)
        elif rsi_val <= 30: rsi_component = 0.5 * min((30 - rsi_val) / 30, 1.0)
        else: rsi_component = 0.0
    macd_up = 1.0 if (pd.notna(macd_hist) and macd_hist > 0) else 0.0
    vol_component = 0.0
    if pd.notna(vol_spike): vol_component = min(max(vol_spike - 1.0, 0.0), 2.0) / 2.0
    return (weights["price_breakout"]*float(price_breakout) +
            weights["rsi_momentum"]*rsi_component +
            weights["macd_signal"]*macd_up +
            weights["volume_spike"]*vol_component +
            weights["catalyst_proximity"]*float(catalyst_component or 0.0))
def apply_scoring(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    df = df.copy()
    def _row(r): return moonshot_score_row_like(r.get("price_breakout",0), r.get("rsi",float("nan")), r.get("macd_hist",0), r.get("vol_spike",0), r.get("catalyst_component",0), weights)
    df["moonshot_score"] = df.apply(_row, axis=1)
    return df
