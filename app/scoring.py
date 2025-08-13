import numpy as np
import pandas as pd
def moonshot_score_row(row, weights):
    price_breakout = float(row.get("price_breakout", 0))
    rsi_val = row.get("rsi", np.nan)
    if np.isnan(rsi_val):
        rsi_component = 0
    else:
        if rsi_val >= 55:
            rsi_component = min((rsi_val - 55) / 25, 1.0)
        elif rsi_val <= 30:
            rsi_component = 0.5 * min((30 - rsi_val) / 30, 1.0)
        else:
            rsi_component = 0.0
    macd_hist = row.get("macd_hist", 0.0)
    macd_up = 1.0 if macd_hist > 0 else 0.0
    vol_spike = row.get("vol_spike", 0.0)
    vol_component = min(max(vol_spike - 1.0, 0.0), 2.0) / 2.0
    catalyst_component = row.get("catalyst_component", 0.0)
    score = (
        weights["price_breakout"] * price_breakout +
        weights["rsi_momentum"] * rsi_component +
        weights["macd_signal"] * macd_up +
        weights["volume_spike"] * vol_component +
        weights["catalyst_proximity"] * catalyst_component
    )
    return score
def apply_scoring(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    df = df.copy()
    df["moonshot_score"] = df.apply(lambda r: moonshot_score_row(r, weights), axis=1)
    return df
