# quantum_mantra_power.py - FINAL PRODUCTION READY SINGLE-FILE
"""
Quantum M.A.N.T.R.A. - All-Time Best Smart Signal Engine (Single File)
=======================================================================
- Fresh data each run; no caching of old sheets without manual refresh
- Full cross-sectional percentiles for all numeric metrics
- Regime-aware dynamic weighting
- Anomaly overrides (breakouts, exhaustion, divergence)
- Robust numeric parsing (K/M/B/L/Cr suffixes)
- No default mismatches in UI widgets
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io, re, logging, warnings
from datetime import datetime
from typing import Dict
import requests
from requests.adapters import HTTPAdapter, Retry

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# LOGGER
# ─────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger("quantum_mantra_power")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(h)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
class Config:
    SHEET_URL       = (
        "https://docs.google.com/spreadsheets/d/"
        "1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk/"
        "export?format=csv&gid=2026492216"
    )
    REQUEST_TIMEOUT = 30
    RETRIES         = 3
    BACKOFF_FACTOR  = 0.3

    # Percentile columns to compute
    PCT_COLS = [
        "ret_1d","ret_3d","ret_7d","ret_30d","ret_3m","ret_6m","ret_1y","ret_3y","ret_5y",
        "from_low_pct","from_high_pct",
        "vol_ratio_1d_90d","vol_ratio_7d_90d","vol_ratio_30d_90d","rvol",
        "eps_change_pct","pe",
    ]

    # Base weights
    BASE_WEIGHTS: Dict[str,float] = {
        "momentum":   0.25,
        "trend":      0.15,
        "value":      0.15,
        "volume":     0.15,
        "fundamental":0.15,
        "longevity":  0.15,
    }

    # Signal thresholds
    SIGNAL_THRESHOLDS = {
        "QUANTUM_BUY":  0.90,
        "STRONG_BUY":   0.75,
        "BUY":          0.60,
        "WATCH":        0.45,
        "NEUTRAL":      0.30,
        "AVOID":        0.15,
        "STRONG_AVOID": 0.00,
    }

    # Suffix multipliers
    SUFFIXES = {"K":1e3,"M":1e6,"B":1e9,"L":1e5,"Cr":1e7}

_num_re = re.compile(r"^\s*([+-]?[\d,\.]+)\s*([KMBlCr%]+)?\s*$", re.IGNORECASE)

def parse_numeric(x, default=0.0):
    if pd.isna(x): return default
    s = str(x).replace("₹","").strip()
    m = _num_re.match(s)
    if not m: return default
    num, suf = m.groups()
    try:
        val = float(num.replace(",",""))
    except:
        return default
    if suf and "%" in suf:
        return val
    if suf:
        suf = suf.capitalize()
        return val * Config.SUFFIXES.get(suf,1)
    return val

def create_session():
    s = requests.Session()
    r = Retry(total=Config.RETRIES, backoff_factor=Config.BACKOFF_FACTOR,
              status_forcelist=[429,500,502,503,504], allowed_methods=["GET"])
    s.mount("https://", HTTPAdapter(max_retries=r))
    return s

def load_data():
    """Fetch fresh sheet data and clean."""
    try:
        resp = create_session().get(Config.SHEET_URL, timeout=Config.REQUEST_TIMEOUT)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
    except Exception as e:
        logger.error("Fetch error: %s", e)
        st.error("🔴 Data fetch failed")
        return pd.DataFrame()

    # drop unnamed
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]

    # normalize names
    df.columns = df.columns.str.strip().str.lower().str.replace(r"\s+","_",regex=True)

    # parse numerics
    for c in df.columns:
        if df[c].dtype == object:
            sample = df[c].dropna().astype(str).head(5)
            if sample.map(lambda v: bool(_num_re.match(v))).any():
                df[c] = df[c].map(lambda v: parse_numeric(v,np.nan))

    # clean ticker
    if "ticker" in df:
        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
        df = df[df["ticker"]!=""]

    return df

def compute_percentiles(df):
    """Add percentile columns for each PCT_COL."""
    for c in Config.PCT_COLS:
        if c in df:
            df[f"{c}_pct"] = df[c].rank(pct=True, na_option="keep").fillna(0.5)
        else:
            df[f"{c}_pct"] = 0.5
    return df

def detect_regime(df):
    """Market regime via median ret_30d_pct."""
    m = df["ret_30d_pct"].median()
    return "momentum" if m>0.6 else "reversion"

def score_groups(df):
    """Compute group scores 0–1."""
    # Momentum
    w = {"ret_1d_pct":0.4,"ret_3d_pct":0.3,"ret_7d_pct":0.2,"ret_30d_pct":0.1}
    df["momentum_score"] = sum(df[k]*v for k,v in w.items() if k in df).clip(0,1)

    # Trend
    mats = ["sma_20d","sma_50d","sma_200d"]
    trend = [(df["price"]>df[m]).astype(float) for m in mats if m in df]
    df["trend_score"] = (sum(trend)/len(trend)).clip(0,1) if trend else 0.5

    # Value
    df["value_score"] = ((df["from_low_pct_pct"]) + (1-df["from_high_pct_pct"]))/2
    df["value_score"] = df["value_score"].clip(0,1)

    # Volume
    df["volume_score"] = np.mean([
        df["vol_ratio_1d_90d_pct"], df["vol_ratio_7d_90d_pct"],
        df["vol_ratio_30d_90d_pct"], df["rvol_pct"]
    ], axis=0).clip(0,1)

    # Fundamental
    eps = df["eps_change_pct_pct"]
    pe  = 1 - df["pe_pct"]
    tier_map = {'5↓':0,'5↑':0.2,'15↑':0.4,'35↑':0.6,'55↑':0.8,'75↑':0.9,'95↑':1}
    et = df["eps_tier"].map(tier_map).fillna(0.5)
    df["fundamental_score"] = np.average([eps,pe,et],axis=0,weights=[0.5,0.3,0.2]).clip(0,1)

    # Longevity
    df["longevity_score"] = np.mean([df["ret_1y_pct"],df["ret_3y_pct"],df["ret_5y_pct"]],axis=0).clip(0,1)

    return df

def apply_dynamic_weights(df, regime):
    """Adjust base weights by regime and compute combined score."""
    mul = {g:1 for g in Config.BASE_WEIGHTS}
    if regime=="momentum":
        mul.update({"momentum":1.3,"value":0.8,"longevity":0.8})
    w = {g:Config.BASE_WEIGHTS[g]*mul[g] for g in Config.BASE_WEIGHTS}
    tot = sum(w.values())
    df["quantum_score"] = (
        df["momentum_score"]*w["momentum"] +
        df["trend_score"]   *w["trend"] +
        df["value_score"]   *w["value"] +
        df["volume_score"]  *w["volume"] +
        df["fundamental_score"]*w["fundamental"] +
        df["longevity_score"]*w["longevity"]
    )/tot
    df["quantum_score"] = df["quantum_score"].clip(0,1)
    return df

def apply_overrides(df):
    """Anomaly & exhaustion overrides."""
    df["special_setup"] = "NONE"
    # Volume explosion
    m = (df["vol_ratio_1d_90d_pct"]>0.99)&(df["quantum_score"]>0.7)
    df.loc[m, "quantum_score"] = (df.loc[m,"quantum_score"]+0.05).clip(0,1)
    df.loc[m, "special_setup"] = "VOLUME_EXPLOSION"
    # Convergence
    m = (df["momentum_score"]>0.8)&(df["volume_score"]>0.8)&(df["trend_score"]>0.8)
    df.loc[m, "quantum_score"] = (df.loc[m,"quantum_score"]+0.1).clip(0,1)
    df.loc[m, "special_setup"] = "QUANTUM_CONVERGENCE"
    # Exhaustion
    m = (df["ret_1d_pct"]>0.95)&(df["ret_3d_pct"]>0.95)&(df["volume_score"]<0.3)
    df.loc[m, "quantum_score"] *= 0.4
    df.loc[m, "special_setup"] = "EXHAUSTION"
    # Earnings divergence
    m = (df["quantum_score"]>0.8)&(df["eps_change_pct"]<0)
    df.loc[m, "quantum_score"] = (df.loc[m,"quantum_score"]-0.1).clip(0,1)
    df.loc[m, "special_setup"] = "EARNINGS_DIVERGENCE"
    return df

def map_signals(df):
    df["signal"] = "NEUTRAL"
    options = list(Config.SIGNAL_THRESHOLDS.items())
    for sig,thr in sorted(options, key=lambda x: x[1], reverse=True):
        df.loc[df["quantum_score"]>=thr, "signal"] = sig
    # enforce caps
    df.loc[df["special_setup"]=="EXHAUSTION","signal"] = "AVOID"
    df.loc[df["special_setup"]=="EARNINGS_DIVERGENCE","signal"] = "WATCH"
    return df

def analyze(df):
    if df.empty: return df
    df = compute_percentiles(df)
    regime = detect_regime(df)
    df = score_groups(df)
    df = apply_dynamic_weights(df, regime)
    df = apply_overrides(df)
    df = map_signals(df)
    return df.sort_values("quantum_score", ascending=False)

def main():
    st.set_page_config(page_title="Quantum M.A.N.T.R.A.",layout="wide")
    st.title("🌌 Quantum M.A.N.T.R.A. – Smart Signal Engine")

    df = load_data()
    if df.empty:
        return

    df = analyze(df)

    # Sidebar filters
    st.sidebar.header("Filters")
    signal_options = sorted(df["signal"].unique())
    desired = ["QUANTUM_BUY","STRONG_BUY","BUY"]
    default_signals = [s for s in desired if s in signal_options]
    sigs = st.sidebar.multiselect("Signal", signal_options, default=default_signals)
    df = df[df["signal"].isin(sigs)]

    minq = st.sidebar.slider("Min Quantum Score", 0.0, 1.0, 0.5, 0.05)
    df = df[df["quantum_score"] >= minq]

    st.metric("Total Stocks", len(df))

    st.dataframe(df[[
        "ticker","company_name","signal","quantum_score","special_setup",
        "price","from_low_pct","from_high_pct"
    ]].head(200), use_container_width=True)

    fig = go.Figure(go.Pie(
        labels=df["signal"].value_counts().index,
        values=df["signal"].value_counts().values,
        hole=0.4
    ))
    st.plotly_chart(fig, use_container_width=True)

if __name__=="__main__":
    main()
