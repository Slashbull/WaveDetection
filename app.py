# app.py — FINAL (locked)  ❖  M.A.N.T.R.A. 2025-07-14
"""
Streamlit dashboard — ultra-simple UI, world-class UX
• One-page layout (no navigation)
• Sidebar: data source, sheet overrides, CSV upload, live filters (tag, sector, category, price tier, search)
• Main: KPI tiles + interactive table + edge-prob chart + download
• Depends on core.py (orchestrator) only — keeps UI layer dead-simple
Run:
    streamlit run app.py
"""

from __future__ import annotations
import streamlit as st, pandas as pd
import core, config

CFG = config.CONFIG

# ─────────────────────────────────────────────────────────────────────────────
# Page setup
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=CFG.APP_NAME,
    page_icon=CFG.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title(f"{CFG.APP_ICON} {CFG.APP_NAME}")
st.caption("Personal Indian Stock Intelligence — Decisions, not guesses.")

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar – data source controls
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.header("Data Source")
mode = st.sidebar.radio("Load from", ("Google Sheet", "Upload CSV"))

if mode == "Google Sheet":
    sheet_id = st.sidebar.text_input("Sheet ID", value="")
    gid      = st.sidebar.text_input("gid", value="")
    if st.sidebar.button("Fetch ✨"):
        with st.spinner("Downloading & scoring…"):
            df_view = core.run(sheet_id or None, gid or None, top=1000)
            st.session_state["df_view"] = df_view

else:
    file = st.sidebar.file_uploader("Upload Watchlist CSV", type="csv")
    if file is not None:
        with st.spinner("Loading & scoring…"):
            df_view = core.run(csv=file, top=1000)
            st.session_state["df_view"] = df_view

# No data yet
if "df_view" not in st.session_state:
    st.info("⬅️  Load your Watchlist data via the sidebar.")
    st.stop()

df = st.session_state["df_view"].copy()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar – live filters
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.header("Filters")
all_tags = df.tag.unique().tolist()
sel_tags = st.sidebar.multiselect("Tag", all_tags, default=all_tags)

all_sectors   = sorted(df.sector.dropna().unique())
all_category  = sorted(df.category.dropna().unique())
all_price_tier= sorted(df.price_tier.dropna().unique())

sel_sector   = st.sidebar.multiselect("Sector", all_sectors)
sel_category = st.sidebar.multiselect("Category", all_category)
sel_price    = st.sidebar.multiselect("Price tier", all_price_tier)
search       = st.sidebar.text_input("Search ticker contains")

mask  = df.tag.isin(sel_tags)
if sel_sector:   mask &= df.sector.isin(sel_sector)
if sel_category: mask &= df.category.isin(sel_category)
if sel_price:    mask &= df.price_tier.isin(sel_price)
if search:       mask &= df.ticker.str.contains(search, case=False, na=False)

filtered = df[mask].reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
# KPIs
# ─────────────────────────────────────────────────────────────────────────────
col1,col2,col3 = st.columns(3)
col1.metric("BUY",   int((filtered.tag == "BUY").sum()))
col2.metric("WATCH", int((filtered.tag == "WATCH").sum()))
col3.metric("AVOID", int((filtered.tag == "AVOID").sum()))

# ─────────────────────────────────────────────────────────────────────────────
# Table
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Watchlist (filtered)")
st.dataframe(filtered, use_container_width=True, height=480)

# ─────────────────────────────────────────────────────────────────────────────
# Edge-probability bar chart (top 20)
# ─────────────────────────────────────────────────────────────────────────────
if not filtered.empty:
    st.subheader("Top Edge Probabilities — Top 20")
    chart_df = filtered.head(20).set_index("ticker")["edge_prob"]
    st.bar_chart(chart_df)

# ─────────────────────────────────────────────────────────────────────────────
# Download button
# ─────────────────────────────────────────────────────────────────────────────
st.download_button(
    label="Download filtered CSV",
    data=filtered.to_csv(index=False).encode(),
    file_name="mantra_filtered_watchlist.csv",
    mime="text/csv",
)

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.caption(f"{CFG.APP_VERSION}  •  Powered by adaptive_tag_model.py")
