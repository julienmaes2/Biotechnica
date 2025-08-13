import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from app import indicators as ind
from app import data as data

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["c"]; v = df["v"]
    sma10 = ind.sma(c, 10)
    price_breakout = (c > 1.05 * sma10).astype(float)
    rsi = ind.rsi(c, 14)
    macd_line, signal_line, hist = ind.macd(c)
    macd_hist = hist
    vol_spike = ind.volume_spike(v, 30)
    return pd.DataFrame({"price_breakout":price_breakout,"rsi":rsi,"macd_hist":macd_hist,"vol_spike":vol_spike}, index=df.index)

def daily_score_from_features(feat: pd.DataFrame, weights: Dict[str,float]) -> pd.Series:
    w = weights; rsi = feat["rsi"]
    rsi_component = pd.Series(0.0, index=feat.index)
    rsi_component = rsi_component.mask(rsi >= 55, ((rsi - 55)/25).clip(0,1))
    rsi_component = rsi_component.mask(rsi <= 30, 0.5 * ((30 - rsi)/30).clip(0,1))
    macd_up = (feat["macd_hist"] > 0).astype(float)
    vol_component = ((feat["vol_spike"] - 1.0).clip(0,2.0)) / 2.0
    return (w["price_breakout"]*feat["price_breakout"].fillna(0.0) +
            w["rsi_momentum"]*rsi_component.fillna(0.0) +
            w["macd_signal"]*macd_up.fillna(0.0) +
            w["volume_spike"]*vol_component.fillna(0.0))

def build_rank_table(symbols: List[str], start: str, end: str, weights: Dict[str, float], max_symbols=200):
    rank_map = {}; close_map = {}; ohlc_map = {}
    for sym in symbols[:max_symbols]:
        hist = data.get_history_yf(sym, start, end)
        if hist is None or hist.empty or len(hist) < 40: continue
        feat = compute_features(hist)
        score = daily_score_from_features(feat, weights)
        rank_map[sym] = score; close_map[sym] = hist["c"]; ohlc_map[sym] = hist[["o","h","l","c"]]
    if not rank_map: return pd.DataFrame(), pd.DataFrame(), {}
    scores = pd.DataFrame(rank_map).sort_index()
    closes = pd.DataFrame(close_map).reindex(scores.index).sort_index()
    return scores, closes, ohlc_map

def rebalance_weights(scores: pd.DataFrame, top_n: int, min_score: float, freq: str = "D"):
    rebal_dates = scores.index if freq=="D" else scores.resample("W-FRI").last().dropna(how="all").index
    weights = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
    for d in rebal_dates:
        row = scores.loc[d].dropna()
        sel = row[row >= min_score].sort_values(ascending=False).head(top_n)
        if len(sel)==0: continue
        w = 1.0 / len(sel); weights.loc[d, sel.index] = w
    weights = weights.replace(0.0, np.nan).ffill().fillna(0.0)
    return weights.shift(1).fillna(0.0)  # apply next day

def apply_sl_tp_to_returns(ohlc_row: pd.Series, prev_close: float, sl_pct: float, tp_pct: float) -> float:
    if pd.isna(prev_close) or prev_close==0 or ohlc_row is None or ohlc_row.empty: return 0.0
    h = ohlc_row["h"]; l = ohlc_row["l"]; c = ohlc_row["c"]
    tp_price = prev_close * (1 + tp_pct); sl_price = prev_close * (1 - sl_pct)
    ret = (c/prev_close) - 1.0
    hit_tp = (h >= tp_price); hit_sl = (l <= sl_price)
    if hit_tp and hit_sl: return tp_pct if ret >= 0 else -sl_pct
    if hit_tp: return tp_pct
    if hit_sl: return -sl_pct
    return ret

def turnover_from_weights(wts: pd.DataFrame) -> pd.Series:
    tw = wts.diff().abs().sum(axis=1).fillna(0.0)
    if len(wts)>0: tw.iloc[0] = wts.iloc[0].abs().sum()
    return tw

def simulate_portfolio(weights: pd.DataFrame, closes: pd.DataFrame, ohlc_map: Dict[str, pd.DataFrame],
                       sl_pct: float = 0.0, tp_pct: float = 9.99, cost_bps: float = 0.0):
    assets = closes.columns
    rets = pd.DataFrame(index=closes.index, columns=assets, data=0.0)
    prev_close = closes.shift(1)
    for sym in assets:
        if sym not in ohlc_map: continue
        ohlc = ohlc_map[sym].reindex(closes.index)
        prev = prev_close[sym]
        rets[sym] = [apply_sl_tp_to_returns(ohlc.loc[d], prev.loc[d], sl_pct, tp_pct) if (d in ohlc.index and d in prev.index) else 0.0 for d in closes.index]
    rets = rets.fillna(0.0)
    weights, rets = weights.align(rets, join="inner", axis=0)
    port = (weights * rets).sum(axis=1)
    if cost_bps and cost_bps>0:
        tw = turnover_from_weights(weights); port = port - tw*(cost_bps/10000.0)
    return port, rets

def trade_log_from_weights(weights: pd.DataFrame, rets: pd.DataFrame) -> pd.DataFrame:
    log = []; prev_w = weights.shift(1).fillna(0.0); active = {}
    for d in weights.index:
        w = weights.loc[d]; pw = prev_w.loc[d]
        for sym in weights.columns:
            if pw.get(sym,0)>0 and w.get(sym,0)==0:
                tr = active.get(sym, {"entry_date": d, "cum": 0.0, "bars": 0}); tr["exit_date"] = d
                log.append({"symbol":sym,"entry_date":tr["entry_date"],"exit_date":tr["exit_date"],"bars":tr["bars"],"return":tr["cum"]}); active.pop(sym,None)
        for sym in weights.columns:
            if pw.get(sym,0)==0 and w.get(sym,0)>0: active[sym] = {"entry_date": d, "cum": 0.0, "bars": 0}
        for sym in list(active.keys()):
            r = rets.loc[d, sym] if sym in rets.columns else 0.0
            active[sym]["cum"] = (1+active[sym]["cum"])*(1+r)-1.0; active[sym]["bars"] += 1
    last_d = weights.index[-1] if len(weights.index) else None
    for sym, tr in active.items():
        log.append({"symbol":sym,"entry_date":tr["entry_date"],"exit_date":last_d,"bars":tr["bars"],"return":tr["cum"]})
    return pd.DataFrame(log)

def metrics_from_returns(port: pd.Series):
    if port is None or port.empty: return {}
    ann = 252; total_return = (1.0 + port).prod() - 1.0; days = len(port); years = days/ann
    cagr = (1.0 + total_return)**(1/years) - 1.0 if years>0 else np.nan
    vol = port.std(ddof=0) * (ann ** 0.5); sharpe = (port.mean()*ann)/vol if vol>0 else np.nan
    eq = (1 + port).cumprod(); peak = eq.cummax(); dd = eq/peak - 1.0; max_dd = dd.min(); hit_rate = (port>0).mean()
    return {"Total Return": total_return, "CAGR": cagr, "Volatility": vol, "Sharpe (rf=0)": sharpe, "Max Drawdown": max_dd, "Hit Rate": hit_rate}

def annual_returns(port: pd.Series):
    if port is None or port.empty: return pd.Series(dtype=float)
    eq = (1 + port).cumprod(); by_year = eq.resample("Y").last().pct_change().dropna(); by_year.index = by_year.index.year; return by_year

PRESET_WEIGHTS = {
    "Balanced": {"price_breakout":0.21,"rsi_momentum":0.25,"macd_signal":0.20,"volume_spike":0.20,"catalyst_proximity":0.10},
    "Momentum-heavy": {"price_breakout":0.30,"rsi_momentum":0.30,"macd_signal":0.25,"volume_spike":0.10,"catalyst_proximity":0.05},
    "Catalyst-heavy": {"price_breakout":0.10,"rsi_momentum":0.15,"macd_signal":0.15,"volume_spike":0.10,"catalyst_proximity":0.50},
}
def sweep(symbols, start, end, presets, topn_list, score_list, freq_list, max_symbols=150):
    results = []
    for preset in presets:
        weights = PRESET_WEIGHTS.get(preset, PRESET_WEIGHTS["Balanced"])
        scores, closes, ohlc_map = build_rank_table(symbols, start, end, weights, max_symbols=max_symbols)
        if scores.empty: continue
        for top_n in topn_list:
            for min_score in score_list:
                for freq in freq_list:
                    wts = rebalance_weights(scores, int(top_n), float(min_score), "D" if freq.lower().startswith("d") else "W")
                    port, rets = simulate_portfolio(wts, closes, ohlc_map, sl_pct=0.0, tp_pct=9.99, cost_bps=0.0)
                    m = metrics_from_returns(port)
                    results.append({"preset":preset,"top_n":top_n,"min_score":min_score,"rebalance":freq, **m})
    return pd.DataFrame(results)
