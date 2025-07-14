# app.py (M.A.N.T.R.A. Streamlit Dashboard — FINAL VERSION)
"""
Streamlit dashboard UI for M.A.N.T.R.A.
Uses core.py for loading, engine.py for analytics, filters.py for filtering.
Ultra-clean, minimal, robust. No code bloat.
"""

import streamlit as st
import pandas as pd
import io
from datetime import datetime

from core import load_and_process
from engine import (
    run_signal_engine, run_decision_engine, run_anomaly_detector,
    compute_edge_signals, find_edges, run_sector_mapper, auto_detect_regime
)
from filters import (
    apply_smart_filters, get_unique_tags, get_unique_sectors, get_unique_categories
)

# ============================================================================
# PAGE CONFIG & HEADER
# ============================================================================

st.set_page_config(
    page_title="M.A.N.T.R.A. — Stock Dashboard",
    page_icon="📈",
    layout="wide",
)

st.title("📈 M.A.N.T.R.A. — Indian Stock Intelligence Engine")
st.caption("Decisions, Not Guesses. Data-Driven Edge Only.")

# ============================================================================
# LOAD DATA (caching handled by core)
# ============================================================================

with st.sidebar:
    st.markdown("### 🩺 Data Health")
    stocks_df, sector_df, summary = load_and_process()
    st.metric("Stocks", summary.get("total_stocks", 0))
    st.metric("Sectors", summary.get("total_sectors", 0))
    st.metric("Blanks", summary.get("quality_analysis", {}).get("null_analysis", {}).get("total_nulls", 0))
    st.metric("Duplicates", summary.get("quality_analysis", {}).get("duplicate_analysis", {}).get("duplicate_tickers", 0))
    st.caption(f"Source: {summary.get('source','')}")
    st.caption(f"Data Hash: {summary.get('data_hash','')}")
    st.caption(f"Last Reload: {datetime.now():%Y-%m-%d %H:%M}")

# ============================================================================
# REGIME SELECTOR
# ============================================================================

auto_regime = auto_detect_regime(stocks_df)
regimes = ["balanced", "momentum", "value", "growth", "volume"]
regime = st.sidebar.selectbox(
    "Market Regime",
    regimes,
    index=regimes.index(auto_regime) if auto_regime in regimes else 0,
    help="Regime affects how stocks are scored (momentum, value, etc.)"
)

# ============================================================================
# DATA PIPELINE
# ============================================================================

with st.spinner("Computing scores & analytics..."):
    df = run_signal_engine(stocks_df, sector_df, regime=regime)
    df = run_decision_engine(df)
    df = run_anomaly_detector(df)
    df = compute_edge_signals(df)
    sector_scores = run_sector_mapper(sector_df)

# ============================================================================
# SIDEBAR FILTERS
# ============================================================================

# Tag
tags = get_unique_tags(df) or ["Buy", "Watch", "Avoid"]
selected_tags = st.sidebar.multiselect("Tags", tags, default=["Buy"])

# Score
min_score = st.sidebar.slider("Min Final Score", 0, 100, 60)

# Category
categories = get_unique_categories(df) or ["All"]
selected_categories = st.sidebar.multiselect("Category", categories, default=categories)

# Sector
sector_list = get_unique_sectors(sector_scores) or ["All"]
selected_sectors = st.sidebar.multiselect("Sector", sector_list, default=sector_list)

# DMA, EPS, Exclude near-high, Anomalies
dma_option = st.sidebar.selectbox("DMA Filter", ["No filter", "Above 50D", "Above 200D"])
eps_only = st.sidebar.checkbox("Strong EPS Only", False)
exclude_high = st.sidebar.checkbox("Exclude Near 52W High", True)
anomaly_only = st.sidebar.checkbox("Anomalies Only", False)

# Strategy
preset = st.sidebar.selectbox(
    "Strategy Preset",
    ["None", "High Momentum", "Low PE + EPS Jumpers", "Base Buy Zones", "Volume Spike"]
)

# Search & Sort
search_ticker = st.sidebar.text_input("Search Ticker").strip().upper()
sort_by = st.sidebar.selectbox("Sort By", ["final_score", "momentum_score", "value_score", "eps_score", "volume_score"])
ascending = st.sidebar.checkbox("Ascending Sort", False)

# Export
export_fmt = st.sidebar.radio("Export", ["CSV", "Excel"], index=0)

# ============================================================================
# FILTERED DATA
# ============================================================================

filtered = apply_smart_filters(
    df,
    selected_tags=selected_tags,
    min_score=min_score,
    selected_sectors=selected_sectors,
    selected_categories=selected_categories,
    dma_option=dma_option,
    eps_only=eps_only,
    exclude_high=exclude_high,
    anomaly_only=anomaly_only,
    preset=preset,
    search_ticker=search_ticker,
    sort_by=sort_by,
    ascending=ascending
)

# ============================================================================
# MAIN KPIs
# ============================================================================

k1, k2, k3 = st.columns(3)
k1.metric("Total Stocks", len(df))
k2.metric("Buy Tags", int(df["tag"].eq("Buy").sum()) if "tag" in df else 0)
k3.metric("Anomalies", int(df["anomaly"].sum()) if "anomaly" in df else 0)

# ============================================================================
# TOP 10 BUY IDEAS
# ============================================================================

st.subheader("🎯 Top 10 Buy Ideas")
if "tag" in filtered and "final_score" in filtered:
    top10 = filtered[filtered["tag"] == "Buy"].nlargest(10, "final_score")
else:
    top10 = pd.DataFrame()
if top10.empty:
    st.info("No 'Buy' ideas found with filters.")
else:
    cols = st.columns(5)
    for i, (_, row) in enumerate(top10.iterrows()):
        with cols[i % 5]:
            st.metric(
                label=f"{row['ticker']} — {row['company_name'][:18]}",
                value=f"{row['final_score']:.1f}",
                delta=f"₹{row.get('target_price', 0):.0f}"
            )

# ============================================================================
# MAIN TAB LAYOUT
# ============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📋 All Opportunities", "🔥 Sector Rotation", "🚨 Anomalies", "🪄 Edge Finder"
])

# --- Tab 1: All Opportunities ---
with tab1:
    st.subheader("All Opportunities Table")
    if filtered.empty:
        st.warning("No stocks match your filters. Adjust your criteria.")
    else:
        st.dataframe(filtered, use_container_width=True)
        if export_fmt == "CSV":
            st.download_button("Download CSV", filtered.to_csv(index=False).encode(), "opportunities.csv")
        else:
            buf = io.BytesIO()
            filtered.to_excel(buf, index=False)
            st.download_button("Download Excel", buf.getvalue(), "opportunities.xlsx")

# --- Tab 2: Sector Rotation ---
with tab2:
    st.subheader("Sector Heatmap / Rotation")
    if sector_scores.empty:
        st.warning("No sector data.")
    else:
        st.dataframe(sector_scores, use_container_width=True)
        if "sector_score" in sector_scores and "sector" in sector_scores:
            st.bar_chart(sector_scores.set_index("sector")["sector_score"])

# --- Tab 3: Anomalies ---
with tab3:
    st.subheader("Anomaly Detector")
    anomalies = df[df["anomaly"]] if "anomaly" in df else pd.DataFrame()
    if anomalies.empty:
        st.success("No anomalies detected in current regime.")
    else:
        st.dataframe(anomalies, use_container_width=True)

# --- Tab 4: Edge Finder ---
with tab4:
    st.subheader("Edge Finder — Alpha Opportunities")
    edge_df = find_edges(filtered) if not filtered.empty else pd.DataFrame()
    if edge_df.empty:
        st.info("No special edges found in current filter.")
    else:
        st.dataframe(edge_df, use_container_width=True)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.caption(f"Last updated: {datetime.now():%Y-%m-%d %H:%M}")
st.caption("All logic is 100% data-driven. This is your personal edge.")

# END OF FILE
