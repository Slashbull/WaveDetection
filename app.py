# ─── Part 1/3: Imports, Configuration & Data Loading ─────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import logging
from typing import List
from datetime import datetime

# Suppress non-critical warnings
import warnings
warnings.filterwarnings('ignore')

# ─── Logger Setup ────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [EDGE] %(levelname)s: %(message)s"))
    logger.addHandler(handler)

# ─── Configuration ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EDGE Protocol",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

SHEET_ID = "1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk"
GID = "2026492216"
SHEET_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID}"

EDGE_THRESHOLDS = {
    "EXPLOSIVE": 85,     # Top 1% - Bet 10%
    "STRONG": 70,        # Top 5% - Bet 5%
    "MODERATE": 50,      # Top 10% - Bet 2%
    "WATCH": 30          # Monitor
}

# ─── Data Loader ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_data() -> pd.DataFrame:
    """
    Fetches the CSV from Google Sheets, cleans and types all relevant columns.
    Returns an empty DataFrame on failure (and logs the error).
    """
    try:
        resp = requests.get(SHEET_URL, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        df.columns = [c.strip() for c in df.columns]

        # Helper to clean ₹ and commas
        def clean_numeric(col: pd.Series) -> pd.Series:
            return (
                col.astype(str)
                   .str.replace("₹", "", regex=False)
                   .str.replace(",", "", regex=False)
                   .str.strip()
                   .replace(["", "NA", "N/A", "nan"], np.nan)
                   .pipe(pd.to_numeric, errors="coerce")
            )

        # Price & SMA columns
        for col in ["price","low_52w","high_52w","sma_20d","sma_50d","sma_200d","prev_close"]:
            if col in df:
                df[col] = clean_numeric(df[col])

        # Return columns
        for col in ["ret_1d","ret_3d","ret_7d","ret_30d","ret_3m","ret_6m","ret_1y","ret_3y","ret_5y"]:
            if col in df:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Volume columns
        for col in ["volume_1d","volume_7d","volume_30d","volume_3m","volume_90d","volume_180d"]:
            if col in df:
                df[col] = clean_numeric(df[col]).fillna(0)

        # Volume ratios (remove % sign)
        for col in [
            "vol_ratio_1d_90d","vol_ratio_7d_90d","vol_ratio_30d_90d",
            "vol_ratio_1d_180d","vol_ratio_7d_180d","vol_ratio_30d_180d","vol_ratio_90d_180d"
        ]:
            if col in df:
                df[col] = clean_numeric(df[col])

        # Fundamental columns
        for col in ["pe","eps_current","eps_last_qtr","eps_change_pct","eps_duplicate"]:
            if col in df:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Market cap
        if "market_cap" in df:
            df["market_cap_num"] = (
                df["market_cap"].astype(str)
                                  .str.replace("₹","",regex=False)
                                  .str.replace(",","",regex=False)
                                  .str.replace(" Cr","",regex=False)
                                  .pipe(pd.to_numeric, errors="coerce")
            )

        # Other numerics
        for col in ["from_low_pct","from_high_pct","rvol"]:
            if col in df:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Drop unlabeled or zero‑price rows
        if "ticker" in df:
            df = df[df["ticker"].notna()]
        if "price" in df:
            df = df[df["price"] > 0]

        logger.info(f"Loaded {len(df)} rows with {len(df.columns)} cols")
        return df

    except Exception as e:
        logger.error(f"Data load failed: {e}", exc_info=True)
        return pd.DataFrame()
        
# ─── Part 2/3: EDGE Calculation Engine ────────────────────────────────────────

def calculate_volume_acceleration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Secret weapon: Δ between 30d/90d and 30d/180d volume ratios.
    Classifies loading/distribution and assigns percentile.
    """
    df = df.copy()
    df["volume_acceleration"] = 0.0
    df["vol_accel_status"]    = "NO_DATA"
    df["vol_accel_percentile"]= 50.0

    if {"vol_ratio_30d_90d","vol_ratio_30d_180d"}.issubset(df.columns):
        df["volume_acceleration"] = (
            df["vol_ratio_30d_90d"].fillna(0) 
          - df["vol_ratio_30d_180d"].fillna(0)
        )
        mask = df["volume_acceleration"].notna()
        bins = [-np.inf,-10,0,10,20,30,np.inf]
        labels = [
            "EXODUS","DISTRIBUTION","NEUTRAL",
            "ACCUMULATION","HEAVY_ACCUMULATION","INSTITUTIONAL_LOADING"
        ]
        df.loc[mask, "vol_accel_status"]     = pd.cut(df.loc[mask,"volume_acceleration"], bins=bins, labels=labels)
        df.loc[mask, "vol_accel_percentile"] = df.loc[mask,"volume_acceleration"].rank(pct=True)*100

    return df

def calculate_momentum_divergence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Short‑term vs long‑term return divergence, flagged for breakout / stealth patterns.
    """
    df = df.copy()
    short = [c for c in ["ret_1d","ret_3d","ret_7d"] if c in df]
    long  = [c for c in ["ret_30d","ret_3m"] if c in df]

    df["short_momentum"] = df[short].fillna(0).mean(axis=1) if short else 0
    df["long_momentum"]  = df[long].fillna(0).mean(axis=1)  if long  else 0
    df["momentum_divergence"] = df["short_momentum"] - df["long_momentum"]
    df["divergence_pattern"]  = "NEUTRAL"

    if "volume_acceleration" in df:
        df.loc[
            (df["momentum_divergence"] > 5) & (df["volume_acceleration"] > 0),
            "divergence_pattern"
        ] = "EXPLOSIVE_BREAKOUT"
        df.loc[
            (df["momentum_divergence"] > 0) & (df["volume_acceleration"] > 10),
            "divergence_pattern"
        ] = "MOMENTUM_BUILDING"
        df.loc[
            (df["momentum_divergence"] < 0) & (df["volume_acceleration"] > 20),
            "divergence_pattern"
        ] = "STEALTH_ACCUMULATION"

    return df

def calculate_risk_reward(df: pd.DataFrame) -> pd.DataFrame:
    """
    Upside vs volatility proxy → risk_reward_ratio, support_distance.
    """
    df = df.copy()
    if {"price","high_52w","low_52w"}.issubset(df.columns):
        df["upside_potential"]  = ((df["high_52w"] - df["price"]) / df["price"] * 100).clip(0,200)
        df["recent_volatility"] = ((df["high_52w"] - df["low_52w"]) / df["price"] * 25).clip(1,50)
        df["risk_reward_ratio"] = (df["upside_potential"] / df["recent_volatility"]).clip(0,10)
        df["support_distance"]  = ((df["price"] - df["low_52w"]) / df["price"] * 100).clip(0,100)
    return df

def calculate_time_arbitrage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Finds long‑term winners on short‑term weakness and quality in selloffs.
    """
    df = df.copy()
    if {"ret_1y","ret_3y","ret_30d","from_high_pct"}.issubset(df.columns):
        df["long_term_annual"]        = df["ret_3y"] / 3
        df["time_arbitrage_opportunity"] = (
            (df["ret_1y"] > df["long_term_annual"]) &
            (df["ret_30d"].between(-10,5))
        )
        df["quality_selloff"] = (
            (df["ret_1y"] < 0) &
            (df["ret_3y"] > 100) &
            (df["from_high_pct"] < -30)
        )
    return df

def calculate_edge_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combines volume accel, divergence, R:R, fundamentals & trend bonuses into 0–100.
    """
    df = df.copy()
    df["edge_score"] = 0.0

    # Volume accel (40%)
    if "volume_acceleration" in df:
        va = df["volume_acceleration"].fillna(0)
        df["vol_accel_score"] = pd.cut(
            va, bins=[-np.inf,0,10,20,30,np.inf],
            labels=[0,25,50,75,100]
        ).astype(float)
        df.loc[va<0,"vol_accel_score"] = 0
        df["edge_score"] += df["vol_accel_score"] * 0.4

    # Momentum div (25%)
    if {"momentum_divergence","short_momentum"}.issubset(df.columns):
        ms = pd.Series(0,index=df.index)
        mask1 = (df["momentum_divergence"]>0)&(df["volume_acceleration"]>0)
        ms.loc[mask1] = 60
        mask2 = (df["momentum_divergence"]>5)&(df["short_momentum"]>0)
        ms.loc[mask2] = 80
        mask3 = (df["momentum_divergence"]<0)&(df["volume_acceleration"]>20)
        ms.loc[mask3] = 100
        df["momentum_score"] = ms
        df["edge_score"] += ms * 0.25

    # Risk/Reward (20%)
    if "risk_reward_ratio" in df:
        df["rr_score"] = (df["risk_reward_ratio"]*20).clip(0,100)
        df["edge_score"] += df["rr_score"] * 0.2

    # Fundamentals (15%)
    fs = pd.Series(0,index=df.index)
    count = 0
    if "eps_change_pct" in df:
        eps = df["eps_change_pct"].fillna(0)
        s = pd.cut(eps,bins=[-np.inf,0,15,30,np.inf],labels=[0,30,60,100]).astype(float)
        fs += s; count += 1
    if "pe" in df:
        pe = df["pe"].fillna(50)
        s = pd.cut(pe,bins=[-np.inf,5,10,25,40,np.inf],labels=[0,50,100,50,0]).astype(float)
        fs += s; count += 1
    if count>0:
        df["fundamental_score"] = fs/count
        df["edge_score"] += df["fundamental_score"]*0.15
    else:
        df["edge_score"] /= 0.85  # redistribute

    # Trend bonus
    if {"price","sma_50d","sma_200d"}.issubset(df.columns):
        bonus = ((df["price"]>df["sma_50d"])&(df["price"]>df["sma_200d"])).astype(float)*5
        df["edge_score"] = (df["edge_score"]+bonus).clip(0,100)

    # Room‑to‑run bonus
    if "from_high_pct" in df:
        fh = df["from_high_pct"].fillna(0)
        rb = pd.Series(0,index=df.index)
        rb.loc[fh.between(-40,-15)] = 5
        rb.loc[fh.between(-35,-20)] = 10
        df["edge_score"] = (df["edge_score"]+rb).clip(0,100)

    # Categorize
    df["edge_category"] = pd.cut(
        df["edge_score"],
        bins=[-0.1,30,50,70,85,100.1],
        labels=["NO_EDGE","WATCH","MODERATE","STRONG","EXPLOSIVE"]
    )
    return df

def calculate_position_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Suggests position percent, stops & targets based on edge & support.
    """
    df = df.copy()
    pos_map = {"EXPLOSIVE":10,"STRONG":5,"MODERATE":2,"WATCH":0,"NO_EDGE":0}
    if "edge_category" in df:
        df["suggested_position_pct"] = df["edge_category"].map(pos_map).fillna(0)

    if {"price","low_52w","sma_50d"}.issubset(df.columns):
        df["stop_loss"] = np.maximum(
            df["price"]*0.93,
            np.maximum(df["sma_50d"]*0.98, df["low_52w"]*1.02)
        )
        df["stop_loss_pct"] = ((df["stop_loss"]-df["price"])/df["price"]*100).round(2)

    if "upside_potential" in df:
        df["target_1"]     = df["price"]*(1+df["upside_potential"]*0.25/100)
        df["target_2"]     = df["price"]*(1+df["upside_potential"]*0.50/100)
        df["target_1_pct"] = ((df["target_1"]-df["price"])/df["price"]*100).round(2)
        df["target_2_pct"] = ((df["target_2"]-df["price"])/df["price"]*100).round(2)

    return df

# ─── Part 3/3: Visualization, Diagnostics & Streamlit App ─────────────────

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List

# Bring in our loader & engines
from __main__ import load_data  # if running as single file; else import from your module
from __main__ import (
    calculate_volume_acceleration, calculate_momentum_divergence,
    calculate_risk_reward, calculate_time_arbitrage,
    calculate_edge_scores, calculate_position_metrics
)

# ─── Visualization Components ───────────────────────────────────────────────

def create_edge_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Bar or histogram of edge categories/scores."""
    if "edge_category" in df:
        counts = df["edge_category"].value_counts().sort_index()
        fig = go.Figure([go.Bar(x=counts.index, y=counts.values, text=counts.values, textposition="auto")])
        fig.update_layout(title="EDGE Distribution", xaxis_title="Category", yaxis_title="Count", height=400)
        return fig
    # fallback
    fig = go.Figure([go.Histogram(x=df["edge_score"], nbinsx=20)])
    fig.update_layout(title="EDGE Score Histogram", xaxis_title="Score", yaxis_title="Count", height=400)
    return fig

def create_volume_acceleration_scatter(df: pd.DataFrame) -> go.Figure:
    """Scatter of vol‑accel vs short‑momentum colored by edge."""
    valid = df.dropna(subset=["edge_score","volume_acceleration","short_momentum"]).nlargest(100,"edge_score")
    fig = go.Figure()
    colors = {"EXPLOSIVE":"red","STRONG":"orange","MODERATE":"yellow","WATCH":"grey","NO_EDGE":"lightgrey"}
    for cat,col in colors.items():
        sub = valid[valid["edge_category"]==cat]
        if not sub.empty:
            fig.add_trace(go.Scatter(
                x=sub["volume_acceleration"], y=sub["short_momentum"], mode="markers+text",
                name=cat, text=sub["ticker"], textposition="top center",
                marker=dict(size=sub["edge_score"]/5, color=col, line=dict(width=1, color="black")),
                customdata=sub["edge_score"]
            ))
    fig.add_hline(0); fig.add_vline(0)
    fig.update_layout(
        title="Volume Acceleration Map",
        xaxis_title="Volume Accel (%)", yaxis_title="Short Momentum (%)", height=600
    )
    return fig

def create_edge_radar(stock: dict) -> go.Figure:
    """Radar of the 5 edge components for a single stock."""
    cats  = ["Volume","Momentum","Risk/Reward","Fundamental","Trend"]
    vals  = [
        stock.get("vol_accel_score",0),
        stock.get("momentum_score",0),
        stock.get("rr_score",0),
        stock.get("fundamental_score",0),
        100 if stock.get("price",0)>stock.get("sma_200d",0) else 0
    ]
    fig = go.Figure(go.Scatterpolar(r=vals,theta=cats,fill="toself"))
    fig.update_layout(title=f"{stock.get('ticker')} EDGE Radar", polar=dict(radialaxis=dict(range=[0,100])), height=400)
    return fig

def diagnose_data_issues(df: pd.DataFrame) -> List[str]:
    """Quick checks for missing or invalid columns."""
    issues = []
    for col in ["vol_ratio_30d_90d","vol_ratio_30d_180d","price","ticker"]:
        if col not in df:
            issues.append(f"Missing: {col}")
    if "price" in df:
        bad = (df["price"]<=0).sum()
        if bad>0: issues.append(f"{bad} rows with non‑positive price")
    return issues

# ─── Main Streamlit App ────────────────────────────────────────────────────

def main() -> None:
    st.title("⚡ EDGE Protocol – Finding What Others Can't See")

    df = load_data()
    if df.empty:
        st.error("❌ Could not load data; please try again later.")
        return

    # Sidebar diagnostics
    with st.sidebar:
        st.header("🔍 Debug Info")
        st.write("Rows:", len(df), "Cols:", len(df.columns))
        iss = diagnose_data_issues(df)
        if iss:
            st.error("Issues:\n• " + "\n• ".join(iss))

    # Calculate all components
    df = calculate_volume_acceleration(df)
    df = calculate_momentum_divergence(df)
    df = calculate_risk_reward(df)
    df = calculate_time_arbitrage(df)
    df = calculate_edge_scores(df)
    df = calculate_position_metrics(df)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔥 Explosive","📊 Distribution","📈 Map","📚 How It Works"])

    with tab1:
        exp = df[df["edge_category"]=="EXPLOSIVE"].nlargest(10,"edge_score")
        st.subheader("Top Explosive Opportunities")
        st.dataframe(exp[["ticker","edge_score","volume_acceleration","short_momentum"]], use_container_width=True)
        if not exp.empty:
            st.plotly_chart(create_edge_radar(exp.iloc[0].to_dict()), use_container_width=True)

    with tab2:
        st.subheader("EDGE Distribution")
        st.plotly_chart(create_edge_distribution_chart(df), use_container_width=True)

    with tab3:
        st.subheader("Volume Acceleration Map")
        st.plotly_chart(create_volume_acceleration_scatter(df), use_container_width=True)

    with tab4:
        st.markdown("""
        **How EDGE Protocol Works**

        1. **Volume Acceleration** (40%): Δ(30d/90d) – Δ(30d/180d)
        2. **Momentum Divergence** (25%): short vs long returns
        3. **Risk/Reward** (20%): Upside vs volatility
        4. **Fundamentals** (15%): EPS growth & PE
        ---
        > **Position Sizing:**
        > - Explosive ≥85 → 10%  
        > - Strong ≥70 → 5%  
        > - Moderate ≥50 → 2%
        """)
    st.caption(f"Version 1.0 • Data refreshed every 5min • {datetime.now().date()}")

if __name__ == "__main__":
    main()

