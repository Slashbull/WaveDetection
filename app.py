from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Imports & Config
# ─────────────────────────────────────────────────────────────────────────────
import io
import json
import math
import warnings
from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from scipy import stats

warnings.filterwarnings("ignore")

# Google Sheet Configuration
# Replace SHEET_ID and GID_WATCHLIST with your actual Google Sheet details
SHEET_ID = "1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk"
GID_WATCHLIST = "2026492216" # This GID corresponds to a specific sheet/tab within your Google Sheet
SHEET_URL = (
    f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID_WATCHLIST}"
)

# UI CONSTANTS
PAGE_TITLE = "EDGE Protocol – Volume‑Acceleration Intelligence"

# Define weighting profiles for different trading styles [cite: ⚡ EDGE Protocol System - COMPLETE]
PROFILE_PRESETS = {
    "Balanced": (0.40, 0.25, 0.20, 0.15),
    "Swing": (0.50, 0.30, 0.20, 0.00), # Higher Volume Accel & Momentum
    "Positional": (0.40, 0.25, 0.25, 0.10), # Slightly more emphasis on R/R and Fundamentals
    "Momentum‑only": (0.60, 0.30, 0.10, 0.00), # Heavily weighted towards Volume Accel & Momentum
    "Breakout": (0.45, 0.40, 0.15, 0.00), # Strong emphasis on Momentum and Volume Accel for breakout confirmation
    "Long‑Term": (0.25, 0.25, 0.15, 0.35), # Higher weight for Fundamentals
}

# Define EDGE Score thresholds for classification [cite: ⚡ EDGE Protocol System - COMPLETE]
EDGE_THRESHOLDS = {
    "EXPLOSIVE": 85,
    "STRONG": 70,
    "MODERATE": 50,
    "WATCH": 0 # Default for anything below MODERATE
}

MIN_STOCKS_PER_SECTOR = 4 # Minimum number of stocks in a sector to avoid thin sector alerts

# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def winsorise_series(s: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    """
    Winsorises a pandas Series to cap outliers at specified quantiles.
    This helps to reduce the impact of extreme values on calculations.
    """
    if s.empty or not pd.api.types.is_numeric_dtype(s):
        return s
    lo, hi = s.quantile([lower_q, upper_q])
    return s.clip(lo, hi)


def calc_atr20(price: pd.Series) -> pd.Series:
    """
    Calculates a proxy for Average True Range (ATR) over 20 periods
    if only close prices are available. ATR is a measure of volatility.
    """
    # Using rolling standard deviation scaled by sqrt(2) as a proxy for ATR
    # This is a simplification; true ATR requires high, low, close prices.
    return price.rolling(20).std().fillna(method="bfill") * math.sqrt(2)


@lru_cache(maxsize=1)
def load_sheet() -> pd.DataFrame:
    """
    Loads data from the specified Google Sheet URL, performs initial cleaning,
    type conversions, and derives necessary columns.
    Uses lru_cache to avoid re-fetching data on every Streamlit rerun.
    """
    try:
        resp = requests.get(SHEET_URL, timeout=30)
        resp.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        raw = pd.read_csv(io.BytesIO(resp.content))

        # Standardise column headers for easier access
        raw.columns = (
            raw.columns.str.strip()
            .str.lower()
            .str.replace("%", "pct")
            .str.replace(" ", "_")
        )

        df = raw.copy()

        # Clean and convert numeric columns that might contain non-numeric characters
        for col in df.columns:
            if df[col].dtype == object: # Only process object (string) columns
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace(r"[₹,%]", "", regex=True) # Remove currency, percentage, and comma symbols
                    .str.replace(",", "")
                    .replace({"nan": np.nan, "": np.nan}) # Convert 'nan' string and empty strings to actual NaN
                )
                # Attempt to convert to numeric, coercing errors to NaN
                df[col] = pd.to_numeric(df[col], errors="ignore") # 'ignore' keeps non-numeric as object

        # Convert specific columns to numeric after cleaning
        numeric_cols_to_convert = [
            'market_cap', 'volume_1d', 'volume_7d', 'volume_30d', 'volume_90d', 'volume_180d',
            'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d',
            'vol_ratio_1d_180d', 'vol_ratio_7d_180d', 'vol_ratio_30d_180d', 'vol_ratio_90d_180d',
            'price', 'ret_1d', 'low_52w', 'high_52w', 'from_low_pct', 'from_high_pct',
            'sma_20d', 'sma_50d', 'sma_200d', 'ret_3d', 'ret_7d', 'ret_30d', 'ret_3m',
            'ret_6m', 'ret_1y', 'ret_3y', 'ret_5y', 'rvol', 'prev_close', 'pe',
            'eps_current', 'eps_last_qtr', 'eps_change_pct', 'year'
        ]
        for col in numeric_cols_to_convert:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # For percentage ratios that were cleaned (e.g., 'vol_ratio_1d_180d'), convert to decimal
        # These columns were already converted to numeric, now ensure they are decimals if they represent percentages
        pct_ratio_cols = ['vol_ratio_1d_180d', 'vol_ratio_7d_180d', 'vol_ratio_30d_180d', 'vol_ratio_90d_180d']
        for col in pct_ratio_cols:
            if col in df.columns:
                # Assuming the numeric value is still in percentage form (e.g., 50 for 50%)
                # Only divide by 100 if the values are large (e.g., >1 or < -1) indicating percentage
                # This is a heuristic, adjust if your raw data format is different
                df[col] = np.where(df[col].abs() > 1, df[col] / 100.0, df[col])


        # Winsorise numeric columns to handle extreme outliers
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].apply(winsorise_series, axis=0)

        # Fill remaining NaNs for critical columns with sensible defaults
        df.fillna({
            'price': df['prev_close'].fillna(1.0), # Fallback to prev_close, then 1.0
            'prev_close': df['price'].fillna(1.0), # Fallback to price, then 1.0
            'low_52w': df['price'] * 0.5, # Assume 50% below current price if missing
            'high_52w': df['price'] * 1.5, # Assume 50% above current price if missing
            'sma_20d': df['price'], 'sma_50d': df['price'], 'sma_200d': df['price'],
            'volume_1d': 0, 'volume_7d': 0, 'volume_30d': 0, 'volume_90d': 0, 'volume_180d': 0,
            'vol_ratio_1d_90d': 0.0, 'vol_ratio_7d_90d': 0.0, 'vol_ratio_30d_90d': 0.0,
            'vol_ratio_1d_180d': 0.0, 'vol_ratio_7d_180d': 0.0, 'vol_ratio_30d_180d': 0.0, 'vol_ratio_90d_180d': 0.0,
            'rvol': 1.0, # Neutral relative volume
            'pe': df['pe'].median() if not df['pe'].isnull().all() else 30.0, # Use median or default
            'eps_current': 0.0, 'eps_last_qtr': 0.0, 'eps_change_pct': 0.0,
            'market_cap': df['market_cap'].median() if not df['market_cap'].isnull().all() else 1000.0, # Default to 1000 Cr
            'year': df['year'].median() if not df['year'].isnull().all() else 2000.0,
            'from_low_pct': 0.0, 'from_high_pct': 0.0,
            'ret_1d': 0.0, 'ret_3d': 0.0, 'ret_7d': 0.0, 'ret_30d': 0.0,
            'ret_3m': 0.0, 'ret_6m': 0.0, 'ret_1y': 0.0, 'ret_3y': 0.0, 'ret_5y': 0.0,
        }, inplace=True)

        # Derived columns
        if "price" in df.columns:
            df["atr_20"] = calc_atr20(df["price"]) # Calculate ATR proxy

        # 30‑day ₹ volume proxy (price*volume_30d) if missing
        if "volume_30d" in df.columns and "price" in df.columns:
            df["rs_volume_30d"] = df["volume_30d"] * df["price"]
        else:
            df["rs_volume_30d"] = 0 # Default if columns are missing

        return df

    except requests.exceptions.RequestException as e:
        st.error(f"Network or data fetching error: {e}. Please check your internet connection or Google Sheet URL/permissions.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading or preprocessing data: {e}. Please ensure the Google Sheet data format is as expected.")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Sector statistics with fallback
# ─────────────────────────────────────────────────────────────────────────────

def sector_stats(df: pd.DataFrame, sector: str) -> Tuple[pd.Series, pd.Series, int]:
    """
    Calculates mean and standard deviation for a given sector.
    Falls back to full market statistics if the sector has too few stocks.
    """
    sector_df = df[df["sector"] == sector]
    n = len(sector_df)

    if n < MIN_STOCKS_PER_SECTOR:
        # Fallback - use full market statistics if sector is too small
        sector_df = df
        n = len(sector_df)

    mean = sector_df.mean(numeric_only=True)
    # Replace zero standard deviation with a small number to avoid division by zero
    std = sector_df.std(numeric_only=True).replace(0, 1e-6)
    return mean, std, n


# ─────────────────────────────────────────────────────────────────────────────
# Scoring blocks - Each block contributes to the overall EDGE score
# ─────────────────────────────────────────────────────────────────────────────

def score_vol_accel(row: pd.Series) -> float:
    """
    Scores Volume Acceleration based on the difference between 30d/90d and 30d/180d volume ratios.
    This is the "secret weapon" for detecting institutional accumulation. [cite: ⚡ EDGE Protocol System - COMPLETE]
    """
    # Ensure required columns are present and not NaN
    if pd.isna(row.get("vol_ratio_30d_90d")) or pd.isna(row.get("vol_ratio_30d_180d")):
        return np.nan # Return NaN if data is missing for this score

    # Calculate the difference in volume ratios (acceleration)
    # A positive delta means recent 30-day volume is accelerating relative to longer periods.
    delta = row["vol_ratio_30d_90d"] - row["vol_ratio_30d_180d"]

    # Map delta to a 0-100 score using a normal CDF (Cumulative Distribution Function)
    # This assumes delta is somewhat normally distributed around 0.
    # The divisor (0.2) acts as a scaling factor; tune this based on data distribution.
    pct = stats.norm.cdf(delta / 0.2) * 100

    # Apply a "Pattern bonus" for high conviction trades [cite: ⚡ EDGE Protocol System - COMPLETE]
    # If volume acceleration is strong (delta >= 20%) AND price is consolidating
    # (e.g., more than 10% below 52-week high), it's a "gold" signal.
    if (
        delta >= 0.20 # 20% acceleration (since vol_ratio is decimal)
        and not pd.isna(row.get("from_high_pct"))
        and row["from_high_pct"] <= -10 # Price is at least 10% below 52-week high
    ):
        pct = min(pct + 5, 100) # Add a bonus, capping at 100

    return pct


def score_momentum(row: pd.Series, df: pd.DataFrame) -> float:
    """
    Scores momentum based on short-term returns relative to sector peers.
    Aims at "catching turns early". [cite: ⚡ EDGE Protocol System - COMPLETE]
    """
    # Identify relevant return columns
    ret_cols = [c for c in df.columns if c.startswith("ret_") and c.endswith("d")]
    if not ret_cols:
        return np.nan

    # Get sector-specific mean and standard deviation for return columns
    mean, std, n = sector_stats(df, row["sector"])
    
    # Ensure standard deviation is not zero to prevent division errors
    valid_std = std[ret_cols].replace(0, 1e-6)

    # Calculate Z-scores for returns relative to sector mean
    # Z-score measures how many standard deviations an observation is from the mean.
    z_scores = (row[ret_cols] - mean[ret_cols]) / valid_std

    # Calculate the mean of Z-scores for overall momentum
    raw_momentum_score = z_scores.mean()

    # Map raw score to 0-100 using CDF, clipping to ensure bounds
    return np.clip(stats.norm.cdf(raw_momentum_score) * 100, 0, 100)


def score_risk_reward(row: pd.Series) -> float:
    """
    Scores the risk/reward profile of a stock based on its current price,
    52-week high/low, and ATR. [cite: ⚡ EDGE Protocol System - COMPLETE]
    """
    required = ["price", "low_52w", "high_52w", "atr_20"]
    if any(pd.isna(row.get(c)) for c in required):
        return np.nan

    # Calculate potential upside and downside based on 52-week range
    upside = row["high_52w"] - row["price"]
    downside = row["price"] - row["low_52w"]

    # Use ATR to normalize the risk/reward difference. ATR is a measure of volatility.
    atr = row["atr_20"] if row["atr_20"] and row["atr_20"] > 0 else 1 # Avoid division by zero

    # Risk/Reward metric: (Upside - Downside) / ATR
    # A higher positive value indicates a more favorable risk/reward.
    rr_metric = (upside - downside) / atr

    # Map metric to 0-100 using CDF. The divisor (4) is a tuning parameter.
    return np.clip(stats.norm.cdf(rr_metric / 4) * 100, 0, 100)


def score_fundamentals(row: pd.Series, df: pd.DataFrame) -> float:
    """
    Scores fundamentals based on EPS change and PE ratio. [cite: ⚡ EDGE Protocol System - COMPLETE]
    """
    # If both EPS change and PE are missing, return NaN
    if pd.isna(row.get("eps_change_pct")) and pd.isna(row.get("pe")):
        return np.nan

    eps_score = np.nan
    if not pd.isna(row.get("eps_change_pct")):
        # Clip EPS change to a reasonable range (-50% to 100%) and map to 0-100
        eps_score = np.clip(row["eps_change_pct"], -50, 100)
        # Scale EPS score: -50 -> 0, 100 -> 100. Linear scaling for simplicity.
        eps_score = (eps_score + 50) / 1.5 # (100 - (-50)) / 100 = 1.5

    pe_score = np.nan
    if not pd.isna(row.get("pe")) and 0 < row["pe"] <= 100:
        # Lower PE is generally better for value. Map PE 0-100 to score 100-0.
        pe_score = 100 - (row["pe"] / 100 * 100)
    elif not pd.isna(row.get("pe")) and row["pe"] > 100: # Very high PE, score very low
        pe_score = 0
    elif not pd.isna(row.get("pe")) and row["pe"] <= 0: # Negative or zero PE, score very low
        pe_score = 0

    scores = [s for s in [eps_score, pe_score] if not pd.isna(s)]
    if not scores:
        return np.nan # If no valid fundamental scores, return NaN
    return np.mean(scores)


# ─────────────────────────────────────────────────────────────────────────────
# Edge score wrapper - Combines individual scores into a final EDGE score
# ─────────────────────────────────────────────────────────────────────────────

def compute_scores(df: pd.DataFrame, weights: Tuple[float, float, float, float]) -> pd.DataFrame:
    """
    Computes the overall EDGE score for each stock by combining individual component scores
    with adaptive weighting. [cite: ⚡ EDGE Protocol System - COMPLETE]
    """
    df = df.copy() # Work on a copy to avoid modifying the original DataFrame

    # Calculate individual component scores
    df["vol_score"] = df.apply(score_vol_accel, axis=1)
    df["mom_score"] = df.apply(score_momentum, axis=1, df=df)
    df["rr_score"] = df.apply(score_risk_reward, axis=1)
    df["fund_score"] = df.apply(score_fundamentals, axis=1, df=df)

    # Define the columns that hold the component scores
    block_cols = ["vol_score", "mom_score", "rr_score", "fund_score"]

    # Adaptive weighting: Renormalise weights row-wise based on available scores
    # If a component score is NaN for a specific stock, its weight is redistributed
    # proportionally among the other available components for that stock. [cite: ⚡ EDGE Protocol System - COMPLETE]
    w_array = np.array(weights)
    out_scores = []
    for idx, row in df.iterrows():
        active_mask = ~row[block_cols].isna() # Identify which scores are not NaN
        if not active_mask.any(): # If all scores are NaN for a stock
            out_scores.append(np.nan)
            continue

        # Get the weights for the active (non-NaN) scores
        active_weights = w_array[active_mask]
        # Normalize these active weights so they sum to 1
        norm_w = active_weights / active_weights.sum()
        
        # Calculate the weighted sum of active scores
        edge = (row[block_cols][active_mask] * norm_w).sum()
        out_scores.append(edge)

    df["EDGE"] = out_scores # Assign the final EDGE score

    # Classify the EDGE Score into categories (EXPLOSIVE, STRONG, MODERATE, WATCH) [cite: ⚡ EDGE Protocol System - COMPLETE]
    conditions = [
        df["EDGE"] >= EDGE_THRESHOLDS["EXPLOSIVE"],
        df["EDGE"] >= EDGE_THRESHOLDS["STRONG"],
        df["EDGE"] >= EDGE_THRESHOLDS["MODERATE"],
    ]
    choices = ["EXPLOSIVE", "STRONG", "MODERATE"]
    df["tag"] = np.select(conditions, choices, default="WATCH")

    # Calculate position sizing based on EDGE Classification [cite: ⚡ EDGE Protocol System - COMPLETE]
    df['position_size_pct'] = df['tag'].apply(
        lambda x: (
            0.10 if x == "EXPLOSIVE" else
            0.05 if x == "STRONG" else
            0.02 if x == "MODERATE" else
            0.00
        )
    )

    # Calculate dynamic stop losses and profit targets [cite: ⚡ EDGE Protocol System - COMPLETE]
    # This is a simplified example. In a real system, these would be more sophisticated.
    df['dynamic_stop'] = df['price'] * 0.95 # 5% below current price
    df['target1'] = df['price'] * 1.05 # 5% above current price
    df['target2'] = df['price'] * 1.10 # 10% above current price

    # Ensure stop is not below 52w low, and targets are not above 52w high
    df['dynamic_stop'] = np.maximum(df['dynamic_stop'], df['low_52w'])
    df['target1'] = np.minimum(df['target1'], df['high_52w'])
    df['target2'] = np.minimum(df['target2'], df['high_52w'])


    return df


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit UI - Renders the web application
# ─────────────────────────────────────────────────────────────────────────────

# New function to calculate volume acceleration and classification
def calculate_volume_acceleration_and_classify(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates volume acceleration metrics and classifies accumulation/distribution.
    This was previously embedded in the main scoring function but is needed earlier.
    """
    df = df.copy() # Work on a copy

    # Calculate Average Daily Volume for respective periods
    df['avg_vol_30d'] = df['volume_30d'] / 30.0
    df['avg_vol_90d'] = df['volume_90d'] / 90.0
    df['avg_vol_180d'] = df['volume_180d'] / 180.0

    # Calculate Volume Ratios (percentage change)
    # Handle division by zero by using np.where or replacing zero denominators with NaN then filling
    df['vol_ratio_30d_90d_calc'] = np.where(df['avg_vol_90d'] != 0,
                                         (df['avg_vol_30d'] / df['avg_vol_90d'] - 1) * 100, 0)
    df['vol_ratio_30d_180d_calc'] = np.where(df['avg_vol_180d'] != 0,
                                          (df['avg_vol_30d'] / df['avg_vol_180d'] - 1) * 100, 0)
    df['vol_ratio_90d_180d_calc'] = np.where(df['avg_vol_180d'] != 0,
                                          (df['avg_vol_90d'] / df['avg_vol_180d'] - 1) * 100, 0)

    # Volume Acceleration: Checks if recent accumulation (30d) is accelerating faster than longer periods (90d, 180d)
    df['volume_acceleration'] = df['vol_ratio_30d_90d_calc'] - df['vol_ratio_30d_180d_calc']

    # Classify based on volume acceleration and current ratios
    def classify_volume(row):
        ratio_30_90 = row['vol_ratio_30d_90d_calc']
        ratio_30_180 = row['vol_ratio_30d_180d_calc']
        acceleration = row['volume_acceleration']

        if acceleration > 20 and ratio_30_90 > 5 and ratio_30_180 > 5:
            return "Institutional Loading"
        elif acceleration > 5 and ratio_30_90 > 0 and ratio_30_180 > 0:
            return "Heavy Accumulation"
        elif ratio_30_90 > 0 and ratio_30_180 > 0:
            return "Accumulation"
        elif ratio_30_90 < 0 and ratio_30_180 < 0 and acceleration < -5:
            return "Exodus"
        elif ratio_30_90 < 0 and ratio_30_180 < 0:
            return "Distribution"
        else:
            return "Neutral"

    df['volume_classification'] = df.apply(classify_volume, axis=1)
    return df


def render_ui():
    """
    Renders the Streamlit user interface for the EDGE Protocol application.
    """
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    st.title(PAGE_TITLE)
    st.markdown("Your unfair advantage: **Volume acceleration data** showing if accumulation is ACCELERATING (not just high).")

    # Sidebar controls for user settings
    with st.sidebar:
        st.header("Settings")
        # Allow user to select a predefined weighting profile
        profile_name = st.radio("Profile", list(PROFILE_PRESETS.keys()), index=0, help="Select a weighting profile for the EDGE score components.")
        weights = PROFILE_PRESETS[profile_name] # Get weights based on selected profile

        # Slider for minimum EDGE score to display
        min_edge = st.slider("Min EDGE Score for Display", 0, 100, 50, 1, help="Only show stocks with an EDGE score above this value.")
        
        # Checkbox to include small/micro cap stocks
        show_smallcaps = st.checkbox("Include small/micro caps", value=False, help="Uncheck to filter out nano/micro cap stocks based on 'category' column.")

    # Load and preprocess data from Google Sheet
    df = load_sheet()

    if df.empty:
        st.error("No data available to process. Please check the data source and try again.")
        return # Exit if data loading failed

    # Filter out small/micro caps if checkbox is unchecked
    if not show_smallcaps and "category" in df.columns:
        df = df[~df["category"].astype(str).str.contains("nano|micro", case=False, na=False)]

    # Filter out stocks with very low 30-day rupee volume (liquidity filter)
    # Assumes 'rs_volume_30d' is in Rupees and 1e7 is 1 Crore (10 million)
    if "rs_volume_30d" in df.columns:
        df = df[df["rs_volume_30d"] >= 1e7] # Filter for sufficient liquidity


    # --- IMPORTANT FIX ---
    # Ensure volume acceleration and classification are calculated *before* computing overall scores
    # as these columns are used in the display_df and stock deep dive.
    df_processed = calculate_volume_acceleration_and_classify(df.copy())
    
    # Compute all EDGE scores and classifications
    df_scored = compute_scores(df_processed, weights)

    # Filter by minimum EDGE score set by the user
    df_filtered_by_min_edge = df_scored[df_scored["EDGE"] >= min_edge].copy()


    # Low‑N alert for concentrated EDGE signals
    explosive_df = df_filtered_by_min_edge[df_filtered_by_min_edge["tag"] == "EXPLOSIVE"]
    if not explosive_df.empty:
        # Count stocks per sector in the original df to check for thin sectors
        sector_counts = df_scored["sector"].value_counts()
        
        # Identify explosive signals coming from sectors with fewer than MIN_STOCKS_PER_SECTOR
        low_n_explosive = explosive_df[explosive_df["sector"].map(sector_counts) < MIN_STOCKS_PER_SECTOR]
        
        if len(explosive_df) > 0 and len(low_n_explosive) / len(explosive_df) > 0.4:
            st.sidebar.warning(
                f"⚠️  Edge concentration alert: {len(low_n_explosive)} / {len(explosive_df)} EXPLOSIVE signals come from thin sectors (less than {MIN_STOCKS_PER_SECTOR} stocks)."
            )

    # Tabs for different analysis views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Daily EDGE Signals", "📈 Volume Acceleration Insights", "🔥 Sector Heatmap", "🔍 Stock Deep Dive", "⚙️ Raw Data & Logs"])

    with tab1:
        st.header("Daily EDGE Signals")
        st.markdown("Find the highest conviction trades here based on the EDGE Protocol's comprehensive scoring. [cite: ⚡ EDGE Protocol System - COMPLETE]")

        # Filter by EDGE Classification for display in this tab
        selected_edge_class_display = st.multiselect(
            "Filter by EDGE Classification for Display:",
            options=["EXPLOSIVE", "STRONG", "MODERATE", "WATCH"], # Explicit order
            default=["EXPLOSIVE", "STRONG"] # Default to high conviction [cite: ⚡ EDGE Protocol System - COMPLETE]
        )

        display_df = df_filtered_by_min_edge[df_filtered_by_min_edge["tag"].isin(selected_edge_class_display)].sort_values("EDGE", ascending=False)

        if not display_df.empty:
            st.dataframe(
                display_df[[
                    "ticker", "company_name", "sector", "tag", "EDGE",
                    "vol_score", "mom_score", "rr_score", "fund_score",
                    "price", "position_size_pct", "dynamic_stop", "target1", "target2",
                    "vol_ratio_30d_90d_calc", "vol_ratio_30d_180d_calc", # Use the calculated % diffs
                    "volume_acceleration", # This column should now exist
                    "volume_classification" # This column should now exist
                ]].style.background_gradient(cmap='RdYlGn', subset=['EDGE']).format({
                    "EDGE": "{:.2f}",
                    "vol_score": "{:.2f}", "mom_score": "{:.2f}", "rr_score": "{:.2f}", "fund_score": "{:.2f}",
                    "price": "₹{:.2f}",
                    "position_size_pct": "{:.2%}", # [cite: ⚡ EDGE Protocol System - COMPLETE]
                    "dynamic_stop": "₹{:.2f}", "target1": "₹{:.2f}", "target2": "₹{:.2f}", # [cite: ⚡ EDGE Protocol System - COMPLETE]
                    "vol_ratio_30d_90d_calc": "{:.2f}%", "vol_ratio_30d_180d_calc": "{:.2f}%", # Format as percentage
                    "volume_acceleration": "{:.2f}%" # [cite: ⚡ EDGE Protocol System - COMPLETE]
                }),
                use_container_width=True
            )

            # Export functionality for filtered signals
            csv = display_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Export Filtered Signals to CSV",
                data=csv,
                file_name=f"edge_signals_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
            )
        else:
            st.info("No stocks match the selected EDGE Classification filters or minimum EDGE score.")

    with tab2:
        st.header("Volume Acceleration Insights")
        st.markdown("Visualize the relationship between Volume Acceleration (difference in 30d/90d and 30d/180d ratios) and Distance from 52-Week High. [cite: ⚡ EDGE Protocol System - COMPLETE]")
        st.markdown("Look for stocks with high positive `Volume Acceleration` and negative `% from 52W High` (i.e., consolidating price with accelerating accumulation) – this is where the 'gold' is found. [cite: ⚡ EDGE Protocol System - COMPLETE]")

        if "volume_acceleration" in df_scored.columns and "from_high_pct" in df_scored.columns:
            # Use the already calculated 'volume_acceleration' column directly for the y-axis
            fig2 = px.scatter(
                df_scored,
                x="from_high_pct",
                y="volume_acceleration", # Use the existing volume_acceleration column
                color="tag",
                size="EDGE", # Size points by overall EDGE score
                hover_data=["ticker", "company_name", "sector", "EDGE", "vol_score", "mom_score", "volume_classification"],
                title="Volume Acceleration vs. Distance from 52-Week High",
                labels={
                    "from_high_pct": "% From 52-Week High (Lower is better for consolidation)",
                    "volume_acceleration": "Volume Acceleration (30d/90d - 30d/180d % Diff)"
                },
                color_discrete_map={ # Consistent colors
                    "EXPLOSIVE": "#FF4B4B", # Red
                    "STRONG": "#FFA500",    # Orange
                    "MODERATE": "#FFD700",  # Gold/Yellow
                    "WATCH": "#1F77B4"      # Blue
                }
            )
            fig2.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
            fig2.add_vline(x=-10, line_dash="dash", line_color="green", annotation_text="< -10% from High (Consolidation Zone)")
            fig2.add_hline(y=20, line_dash="dash", line_color="red", annotation_text="> 20% Volume Acceleration (Strong)")

            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Required columns for Volume Acceleration Scatter plot are missing.")

    with tab3:
        st.header("Sector Heatmap (Average EDGE Score)")
        st.markdown("Visualize the average EDGE score across different sectors. Sectors with higher average scores might indicate broader opportunities. Opacity indicates sectors with fewer stocks, suggesting less reliable averages.")

        # Aggregate data by sector
        agg = df_scored.groupby("sector").agg(
            edge_mean=("EDGE", "mean"),
            n=("EDGE", "size")
        ).reset_index()
        
        # Add opacity based on number of stocks in sector
        agg["opacity"] = np.where(agg["n"] < MIN_STOCKS_PER_SECTOR, 0.4, 1.0)

        fig = px.treemap(agg, path=["sector"], values="n", color="edge_mean",
                         range_color=(0, 100), # Ensure color scale is 0-100 for EDGE scores
                         color_continuous_scale=px.colors.sequential.Viridis, # Choose a color scale
                         title="Average EDGE Score by Sector"
                        )
        fig.update_traces(marker=dict(opacity=agg["opacity"]))
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.header("Stock Deep Dive (Radar Chart)")
        st.markdown("Select an individual stock to see its detailed EDGE component breakdown and all raw metrics. [cite: ⚡ EDGE Protocol System - COMPLETE]")
        
        # Ensure only stocks with valid calculated scores are available for selection
        available_stocks = df_scored[df_scored['EDGE'].notnull()].sort_values('company_name')['ticker'].tolist()
        
        if available_stocks:
            selected_ticker = st.selectbox("Select Ticker:", available_stocks)
            selected_stock_row = df_scored[df_scored['ticker'] == selected_ticker].iloc[0]
            
            # Plot radar chart for selected stock
            plot_stock_radar_chart(selected_stock_row)

            st.subheader(f"Detailed Metrics for {selected_stock_row['company_name']} ({selected_stock_row['ticker']})")
            
            # Display key metrics using st.metric
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Price", f"₹{selected_stock_row['price']:.2f}")
                st.metric("EDGE Score", f"{selected_stock_row['EDGE']:.2f}")
                st.metric("Classification", selected_stock_row['tag'])
            with col2:
                st.metric("Volume Accel. Diff", f"{selected_stock_row['volume_acceleration']:.2f}%") # [cite: ⚡ EDGE Protocol System - COMPLETE]
                st.metric("Volume Classification", selected_stock_row['volume_classification']) # This column should now exist
                st.metric("Position Size", f"{selected_stock_row['position_size_pct']:.2%}") # [cite: ⚡ EDGE Protocol System - COMPLETE]
            with col3:
                st.metric("Dynamic Stop", f"₹{selected_stock_row['dynamic_stop']:.2f}") # [cite: ⚡ EDGE Protocol System - COMPLETE]
                st.metric("Target 1", f"₹{selected_stock_row['target1']:.2f}") # [cite: ⚡ EDGE Protocol System - COMPLETE]
                st.metric("Target 2", f"₹{selected_stock_row['target2']:.2f}") # [cite: ⚡ EDGE Protocol System - COMPLETE]

            st.markdown("---")
            st.subheader("All Raw & Calculated Data")
            # Display all columns for the selected stock, formatted
            st.dataframe(
                selected_stock_row.to_frame().T.style.format(
                    {
                        'market_cap': "₹{:,.0f} Cr",
                        'price': "₹{:.2f}",
                        'ret_1d': "{:.2f}%", 'ret_3d': "{:.2f}%", 'ret_7d': "{:.2f}%", 'ret_30d': "{:.2f}%",
                        'ret_3m': "{:.2f}%", 'ret_6m': "{:.2f}%", 'ret_1y': "{:.2f}%",
                        'ret_3y': "{:.2f}%", 'ret_5y': "{:.2f}%",
                        'low_52w': "₹{:.2f}", 'high_52w': "₹{:.2f}",
                        'from_low_pct': "{:.2f}%", 'from_high_pct': "{:.2f}%",
                        'sma_20d': "₹{:.2f}", 'sma_50d': "₹{:.2f}", 'sma_200d': "₹{:.2f}",
                        'volume_1d': "{:,.0f}", 'volume_7d': "{:,.0f}", 'volume_30d': "{:,.0f}",
                        'volume_90d': "{:,.0f}", 'volume_180d': "{:,.0f}",
                        'vol_ratio_1d_90d': "{:.2%}", 'vol_ratio_7d_90d': "{:.2%}", 'vol_ratio_30d_90d': "{:.2%}",
                        'vol_ratio_1d_180d': "{:.2%}", 'vol_ratio_7d_180d': "{:.2%}", 'vol_ratio_30d_180d': "{:.2%}", 'vol_ratio_90d_180d': "{:.2%}",
                        'vol_ratio_30d_90d_calc': "{:.2f}%", 'vol_ratio_30d_180d_calc': "{:.2f}%",
                        'rvol': "{:.2f}",
                        'prev_close': "₹{:.2f}",
                        'pe': "{:.2f}",
                        'eps_current': "{:.2f}", 'eps_last_qtr': "{:.2f}", 'eps_change_pct': "{:.2f}%",
                        'atr_20': "₹{:.2f}", 'rs_volume_30d': "₹{:,.0f}",
                        'vol_score': "{:.2f}", 'mom_score': "{:.2f}", 'rr_score': "{:.2f}", 'fund_score': "{:.2f}",
                        'EDGE': "{:.2f}",
                        'dynamic_stop': "₹{:.2f}", 'target1': "₹{:.2f}", 'target2': "₹{:.2f}",
                        'volume_acceleration': "{:.2f}%",
                        'volume_classification': "{}",
                        'delta_accel': "{:.2f}%" # Assuming this is the delta_accel for the plot
                    }
                ),
                use_container_width=True
            )
        else:
            st.info("No stocks available for deep dive or data issues.")


    with tab5:
        st.header("Raw Data and Logs")
        st.markdown("Review the raw loaded data and the configurable parameters of the EDGE Protocol. This tab is useful for debugging and understanding the data.")
        
        st.subheader("Raw Data (First 10 Rows after initial load and cleaning)")
        st.dataframe(df.head(10), use_container_width=True)

        st.subheader("EDGE Thresholds & Position Sizing")
        st.json(EDGE_THRESHOLDS)

        st.subheader("Current Weighting of EDGE Components")
        st.write("The weights below are applied to the four core components to calculate the final EDGE Score. These weights adapt if fundamental data is missing.")
        st.markdown(f"""
            * **Volume Acceleration ({weights[0]*100:.0f}%)**: Your secret weapon, detecting institutional accumulation acceleration. [cite: ⚡ EDGE Protocol System - COMPLETE]
            * **Momentum Divergence ({weights[1]*100:.0f}%)**: Catching turns early and confirming price action. [cite: ⚡ EDGE Protocol System - COMPLETE]
            * **Risk/Reward Mathematics ({weights[2]*100:.0f}%)**: Ensuring favorable trade setups. [cite: ⚡ EDGE Protocol System - COMPLETE]
            * **Fundamentals ({weights[3]*100:.0f}%)**: Adaptive weighting. Redistributed if EPS/PE data is missing. [cite: ⚡ EDGE Protocol System - COMPLETE]
        """)
        if weights[3] == 0: # If fundamental weight became 0 due to adaptive weighting
            st.warning("Note: Fundamental (EPS/PE) data was largely missing or invalid, so its weight has been redistributed.")

        st.subheader("Full Processed Data (First 5 Rows)")
        st.dataframe(df_scored.head(5), use_container_width=True)

        # Export full processed data
        csv_full = df_scored.to_csv(index=False).encode('utf-8')
        st.download_button("Download Full Processed CSV", csv_full, "edge_protocol_full_output.csv", "text/csv")
        
        st.write(f"Data last processed: {pd.Timestamp.utcnow().strftime('%Y‑%m‑%d %H:%M:%S UTC')}")


if __name__ == "__main__":
    render_ui()
