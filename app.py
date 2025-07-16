# app.py - THE ULTIMATE TRADING EDGE SYSTEM
"""
EDGE Protocol - Finding What Others Can't See
=============================================
Your unfair advantage: Volume acceleration data showing if accumulation
is ACCELERATING (not just high). This finds institutional moves early.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import io
from datetime import datetime
import warnings
warnings.filterwarnings('ignore') # Suppress warnings for cleaner output

# ============================================================================
# CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="EDGE Protocol",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Data source configuration
# IMPORTANT: REPLACE THIS URL with your Google Sheet's "Published to web" CSV link.
# Go to File -> Share -> Publish to web -> Choose the specific sheet -> Select CSV -> Publish -> Copy the URL.
[cite_start]SHEET_ID = "1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk" # Your sheet ID [cite: 1]
[cite_start]GID = "2026492216" # The GID for the specific sheet tab within your spreadsheet [cite: 2]
SHEET_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID}"

# EDGE thresholds for classification and suggested position sizing
EDGE_THRESHOLDS = {
    'EXPLOSIVE': 85,     # Top 1% - Suggested position size: 10%
    'STRONG': 70,        # Top 5% - Suggested position size: 5%
    'MODERATE': 50,      # Top 10% - Suggested position size: 2%
    'WATCH': 30          # Monitor stocks with this edge, no immediate position
}

# ============================================================================
# DATA LOADING & PREPARATION
# ============================================================================

@st.cache_data(ttl=300) # Cache data for 5 minutes to reduce repeated API calls
def load_data():
    """
    Loads data from the configured Google Sheet URL, performs robust type conversions,
    and initial data cleaning.
    """
    try:
        # Fetch data from the published Google Sheet URL
        response = requests.get(SHEET_URL, timeout=30)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

        # Read CSV data into a Pandas DataFrame
        df = pd.read_csv(io.StringIO(response.text))

        # Clean column names by stripping leading/trailing whitespace
        df.columns = [col.strip() for col in df.columns]

        # Define a dictionary of columns and their specific cleaning/conversion functions
        # This makes the data preparation process more organized and robust.
        conversions = {
            # Price and SMA related columns (currency/comma cleaning, then numeric)
            [cite_start]'price': lambda x: pd.to_numeric(x.astype(str).str.replace('₹', '').str.replace(',', '').str.strip(), errors='coerce'), [cite: 6]
            [cite_start]'low_52w': lambda x: pd.to_numeric(x.astype(str).str.replace('₹', '').str.replace(',', '').str.strip(), errors='coerce'), [cite: 7]
            [cite_start]'high_52w': lambda x: pd.to_numeric(x.astype(str).str.replace('₹', '').str.replace(',', '').str.strip(), errors='coerce'), [cite: 9]
            [cite_start]'sma_20d': lambda x: pd.to_numeric(x.astype(str).str.replace('₹', '').str.replace(',', '').str.strip(), errors='coerce'), [cite: 12]
            [cite_start]'sma_50d': lambda x: pd.to_numeric(x.astype(str).str.replace('₹', '').str.replace(',', '').str.strip(), errors='coerce'), [cite: 13]
            [cite_start]'sma_200d': lambda x: pd.to_numeric(x.astype(str).str.replace('₹', '').str.replace(',', '').str.strip(), errors='coerce'), [cite: 14]
            'prev_close': lambda x: pd.to_numeric(x.astype(str).str.replace('₹', '').str.replace(',', '').str.strip(), errors='coerce'),

            # Return percentage columns (direct numeric conversion)
            [cite_start]'ret_1d': lambda x: pd.to_numeric(x, errors='coerce'), [cite: 6]
            [cite_start]'ret_3d': lambda x: pd.to_numeric(x, errors='coerce'), [cite: 15]
            [cite_start]'ret_7d': lambda x: pd.to_numeric(x, errors='coerce'), [cite: 16]
            [cite_start]'ret_30d': lambda x: pd.to_numeric(x, errors='coerce'), [cite: 17]
            [cite_start]'ret_3m': lambda x: pd.to_numeric(x, errors='coerce'), [cite: 18]
            [cite_start]'ret_6m': lambda x: pd.to_numeric(x, errors='coerce'), [cite: 19]
            [cite_start]'ret_1y': lambda x: pd.to_numeric(x, errors='coerce'), [cite: 20]
            [cite_start]'ret_3y': lambda x: pd.to_numeric(x, errors='coerce'), [cite: 21]
            [cite_start]'ret_5y': lambda x: pd.to_numeric(x, errors='coerce'), [cite: 22]

            # Volume columns (clean commas, then numeric, fill NaN with 0 as volume can be zero)
            [cite_start]'volume_1d': lambda x: pd.to_numeric(x.astype(str).str.replace('₹', '').str.replace(',', '').str.strip(), errors='coerce').fillna(0), [cite: 23]
            [cite_start]'volume_7d': lambda x: pd.to_numeric(x.astype(str).str.replace('₹', '').str.replace(',', '').str.strip(), errors='coerce').fillna(0), [cite: 24]
            [cite_start]'volume_30d': lambda x: pd.to_numeric(x.astype(str).str.replace('₹', '').str.replace(',', '').str.strip(), errors='coerce').fillna(0), [cite: 25]
            'volume_3m': lambda x: pd.to_numeric(x.astype(str).str.replace('₹', '').str.replace(',', '').str.strip(), errors='coerce').fillna(0),
            [cite_start]'volume_90d': lambda x: pd.to_numeric(x.astype(str).str.replace(',', '').str.strip(), errors='coerce').fillna(0), [cite: 26]
            [cite_start]'volume_180d': lambda x: pd.to_numeric(x.astype(str).str.replace(',', '').str.strip(), errors='coerce').fillna(0), [cite: 27]

            # Volume ratio columns (handle %, commas, various null representations, fill NaN with 0)
            'vol_ratio_1d_90d': lambda x: pd.to_numeric(x.astype(str).str.replace('%', '').str.strip().replace(['', '-', 'NA', 'N/A', 'nan', 'NaN'], np.nan), errors='coerce').fillna(0),
            'vol_ratio_7d_90d': lambda x: pd.to_numeric(x.astype(str).str.replace('%', '').str.strip().replace(['', '-', 'NA', 'N/A', 'nan', 'NaN'], np.nan), errors='coerce').fillna(0),
            'vol_ratio_30d_90d': lambda x: pd.to_numeric(x.astype(str).str.replace('%', '').str.strip().replace(['', '-', 'NA', 'N/A', 'nan', 'NaN'], np.nan), errors='coerce').fillna(0),
            'vol_ratio_1d_180d': lambda x: pd.to_numeric(x.astype(str).str.replace('%', '').str.strip().replace(['', '-', 'NA', 'N/A', 'nan', 'NaN'], np.nan), errors='coerce').fillna(0),
            'vol_ratio_7d_180d': lambda x: pd.to_numeric(x.astype(str).str.replace('%', '').str.strip().replace(['', '-', 'NA', 'N/A', 'nan', 'NaN'], np.nan), errors='coerce').fillna(0),
            'vol_ratio_30d_180d': lambda x: pd.to_numeric(x.astype(str).str.replace('%', '').str.strip().replace(['', '-', 'NA', 'N/A', 'nan', 'NaN'], np.nan), errors='coerce').fillna(0),
            'vol_ratio_90d_180d': lambda x: pd.to_numeric(x.astype(str).str.replace('%', '').str.strip().replace(['', '-', 'NA', 'N/A', 'nan', 'NaN'], np.nan), errors='coerce').fillna(0),

            # Fundamental data columns (direct numeric conversion)
            'pe': lambda x: pd.to_numeric(x, errors='coerce'),
            'eps_current': lambda x: pd.to_numeric(x, errors='coerce'),
            'eps_last_qtr': lambda x: pd.to_numeric(x, errors='coerce'),
            'eps_change_pct': lambda x: pd.to_numeric(x, errors='coerce'),
            'eps_duplicate': lambda x: pd.to_numeric(x, errors='coerce'),

            # Other numeric columns
            [cite_start]'market_cap': lambda x: pd.to_numeric(x.astype(str).str.replace('₹', '').str.replace(',', '').str.replace(' Cr', '').str.strip(), errors='coerce'), [cite: 4]
            [cite_start]'from_low_pct': lambda x: pd.to_numeric(x, errors='coerce'), [cite: 8]
            [cite_start]'from_high_pct': lambda x: pd.to_numeric(x, errors='coerce'), [cite: 11]
            'rvol': lambda x: pd.to_numeric(x, errors='coerce')
        }

        # Apply conversions to DataFrame columns that exist
        for col, func in conversions.items():
            if col in df.columns:
                df[col] = func(df[col])

        # Ensure 'market_cap_num' exists for calculations
        if 'market_cap' in df.columns:
            df.rename(columns={'market_cap': 'market_cap_num'}, inplace=True)
        else: # Create a dummy column if original market_cap is missing
            df['market_cap_num'] = np.nan

        # Filter out rows with invalid tickers or non-positive prices, crucial for analysis
        if 'ticker' in df.columns:
            df = df[df['ticker'].notna() & (df['ticker'] != '')]
        if 'price' in df.columns:
            df = df[df['price'] > 0]

        # Reset index after filtering
        return df.reset_index(drop=True)

    except requests.exceptions.Timeout:
        st.error("Data loading timed out (30 seconds). Please check your internet connection or Google Sheet server status.")
        return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to load data from Google Sheet: {str(e)}. Please ensure the sheet is published to the web as CSV and the URL in the code is correct.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An unexpected error occurred during data loading and preparation: {str(e)}. Please check your data format.")
        return pd.DataFrame()

# ============================================================================
# EDGE CALCULATION ENGINE - THE CORE LOGIC
# ============================================================================

def calculate_volume_acceleration(df):
    """
    Calculates the 'secret weapon' - volume acceleration.
    This is the difference between 30-day volume ratio vs 90-day avg and
    30-day volume ratio vs 180-day avg.
    """
    df['volume_acceleration'] = np.nan # Initialize with NaN
    df['vol_accel_status'] = 'NO_DATA'
    df['vol_accel_percentile'] = np.nan

    # Check for the primary required columns
    if 'vol_ratio_30d_90d' in df.columns and 'vol_ratio_30d_180d' in df.columns:
        # Perform calculation only where both ratios are available and not NaN
        valid_ratios_mask = df['vol_ratio_30d_90d'].notna() & df['vol_ratio_30d_180d'].notna()
        df.loc[valid_ratios_mask, 'volume_acceleration'] = \
            df.loc[valid_ratios_mask, 'vol_ratio_30d_90d'] - df.loc[valid_ratios_mask, 'vol_ratio_30d_180d']

        # Classify acceleration status based on thresholds
        valid_accel_mask = df['volume_acceleration'].notna()
        if valid_accel_mask.sum() > 0: # Ensure there are valid values to classify
            df.loc[valid_accel_mask, 'vol_accel_status'] = pd.cut(
                df.loc[valid_accel_mask, 'volume_acceleration'],
                bins=[-np.inf, -10, 0, 10, 20, 30, np.inf], # Define buckets for classification
                labels=['EXODUS', 'DISTRIBUTION', 'NEUTRAL', 'ACCUMULATION', 'HEAVY_ACCUMULATION', 'INSTITUTIONAL_LOADING'],
                right=False # Bins are inclusive on the left, exclusive on the right
            ).astype(str) # Convert to string to avoid potential CategoricalDtype issues in Streamlit display

            # Calculate percentile rank of volume acceleration
            df.loc[valid_accel_mask, 'vol_accel_percentile'] = df.loc[valid_accel_mask, 'volume_acceleration'].rank(pct=True, method='average') * 100
            df['vol_accel_percentile'].fillna(50, inplace=True) # Default percentile to 50 for NaNs

    elif 'vol_ratio_30d_90d' in df.columns:
        # Fallback if only the 30d/90d ratio is available (less ideal, but better than nothing)
        st.warning("Only 'vol_ratio_30d_90d' is available for volume acceleration. Calculation will be a proxy.")
        df['volume_acceleration'] = df['vol_ratio_30d_90d'].fillna(0) # Use this as a proxy

        valid_proxy_mask = df['volume_acceleration'].notna()
        if valid_proxy_mask.sum() > 0:
            df.loc[valid_proxy_mask, 'vol_accel_status'] = pd.cut(
                df.loc[valid_proxy_mask, 'volume_acceleration'],
                bins=[-np.inf, -50, -20, 0, 20, 50, np.inf],
                labels=['EXODUS', 'DISTRIBUTION', 'NEUTRAL', 'ACCUMULATION', 'HEAVY_ACCUMULATION', 'INSTITUTIONAL_LOADING'],
                right=False
            ).astype(str)
            df.loc[valid_proxy_mask, 'vol_accel_percentile'] = df.loc[valid_proxy_mask, 'volume_acceleration'].rank(pct=True, method='average') * 100
            df['vol_accel_percentile'].fillna(50, inplace=True)
    else:
        # If neither critical volume ratio column exists, log a warning
        st.warning("Neither 'vol_ratio_30d_90d' nor 'vol_ratio_30d_180d' found. Volume acceleration cannot be calculated.")
        # Ensure these columns are initialized to prevent downstream errors
        df['volume_acceleration'] = np.nan
        df['vol_accel_status'] = 'NO_DATA'
        df['vol_accel_percentile'] = np.nan

    return df

def calculate_momentum_divergence(df):
    """
    Detects momentum acceleration patterns by comparing short-term and long-term momentum.
    """
    df['short_momentum'] = 0.0
    df['long_momentum'] = 0.0
    df['momentum_divergence'] = 0.0
    df['divergence_pattern'] = 'NEUTRAL'

    # Short-term momentum: average of 1-day, 3-day, and 7-day returns
    [cite_start]short_cols = ['ret_1d', 'ret_3d', 'ret_7d'] [cite: 6, 15, 16]
    available_short = [col for col in short_cols if col in df.columns]
    if available_short:
        df['short_momentum'] = df[available_short].mean(axis=1, skipna=True).fillna(0)

    # Long-term momentum: average of 30-day and 3-month returns
    [cite_start]long_cols = ['ret_30d', 'ret_3m'] [cite: 17, 18]
    available_long = [col for col in long_cols if col in df.columns]
    if available_long:
        df['long_momentum'] = df[available_long].mean(axis=1, skipna=True).fillna(0)

    # Momentum divergence: short-term minus long-term momentum
    df['momentum_divergence'] = df['short_momentum'] - df['long_momentum']

    # Classify divergence patterns based on volume acceleration and momentum divergence
    if 'volume_acceleration' in df.columns:
        # Use .fillna(0) for comparison to ensure no NaN issues in boolean masks
        vol_accel_safe = df['volume_acceleration'].fillna(0)
        mom_div_safe = df['momentum_divergence'].fillna(0)

        # EXPLOSIVE_BREAKOUT: Strong positive short-term momentum and positive volume acceleration
        mask_explosive = (mom_div_safe > 5) & (vol_accel_safe > 0)
        df.loc[mask_explosive, 'divergence_pattern'] = 'EXPLOSIVE_BREAKOUT'

        # MOMENTUM_BUILDING: Moderate positive momentum divergence and significant volume acceleration
        mask_building = (mom_div_safe > 0) & (vol_accel_safe > 10)
        df.loc[mask_building, 'divergence_pattern'] = 'MOMENTUM_BUILDING'

        # STEALTH_ACCUMULATION: Negative momentum divergence (price might be stagnant/down) but strong volume acceleration (smart money buying into weakness)
        mask_stealth = (mom_div_safe < 0) & (vol_accel_safe > 20)
        df.loc[mask_stealth, 'divergence_pattern'] = 'STEALTH_ACCUMULATION'

    return df

def calculate_risk_reward(df):
    """
    Calculates mathematical edge in risk/reward based on price, 52-week high/low.
    """
    df['upside_potential'] = 0.0
    df['recent_volatility'] = 0.0
    df['risk_reward_ratio'] = 0.0
    df['support_distance'] = 0.0

    [cite_start]if all(col in df.columns for col in ['price', 'high_52w', 'low_52w']): [cite: 6, 7, 9]
        # Mask for valid prices to prevent division by zero or negative prices
        valid_prices_mask = (df['price'] > 0)

        # Upside potential: percentage distance from current price to 52-week high
        df.loc[valid_prices_mask, 'upside_potential'] = \
            ((df['high_52w'] - df['price']) / df['price'] * 100).clip(0, 200).fillna(0) # Cap at 200% for extreme cases

        # Recent volatility (simplified): based on 52-week range, scaled down
        # Dividing by 4 serves as a proxy for quarterly volatility
        df.loc[valid_prices_mask, 'recent_volatility'] = \
            ((df['high_52w'] - df['low_52w']) / df['price'] * 100 / 4).clip(1, 50).fillna(1) # Min volatility of 1% to avoid division by zero later

        # Risk/Reward ratio: Upside potential divided by (2 * recent volatility)
        # The factor of 2 applies a penalty for volatility as perceived risk
        valid_rr_mask = valid_prices_mask & (df['recent_volatility'] > 0) # Ensure no division by zero
        df.loc[valid_rr_mask, 'risk_reward_ratio'] = \
            (df['upside_potential'] / (2 * df['recent_volatility'])).clip(0, 10).fillna(0) # Cap at 10x for practical purposes

        # Support level distance: percentage distance from current price to 52-week low
        df.loc[valid_prices_mask, 'support_distance'] = \
            ((df['price'] - df['low_52w']) / df['price'] * 100).clip(0, 100).fillna(0)

    return df

def calculate_time_arbitrage(df):
    """
    Identifies quality stocks that might be experiencing a temporary pullback,
    presenting a "time arbitrage" opportunity.
    """
    df['long_term_annual'] = 0.0
    df['time_arbitrage_opportunity'] = False
    df['quality_selloff'] = False

    [cite_start]if all(col in df.columns for col in ['ret_1y', 'ret_3y', 'ret_30d', 'from_high_pct']): [cite: 11, 17, 20, 21]
        # Long-term winner taking a break: Good long-term returns but recent moderate pullback
        df['long_term_annual'] = (df['ret_3y'] / 3).fillna(0) # Annualize 3-year return
        df['time_arbitrage_opportunity'] = (
            (df['ret_1y'] > df['long_term_annual']) & # 1-year return is stronger than annualized 3-year (suggests recent strength)
            (df['ret_30d'] < 5) & # Recent 30-day return is slightly positive or flat
            (df['ret_30d'] > -10) # Not a severe recent drop
        ).fillna(False) # Fill NaNs to ensure boolean column

        # Quality in selloff: Strong multi-year performance but current significant drawdown
        df['quality_selloff'] = (
            (df['ret_1y'] < 0) & # Negative 1-year return (underperforming recently)
            (df['ret_3y'] > 100) & # Strong 3-year return (long-term winner)
            (df['from_high_pct'] < -30) # Currently more than 30% down from its 52-week high
        ).fillna(False)

    return df

def calculate_edge_scores(df):
    """
    Calculates the final composite EDGE score based on weighted components.
    This is the heart of the system.
    """
    df['edge_score'] = 0.0 # Initialize as float
    df['vol_accel_score'] = 0.0
    df['momentum_score'] = 0.0
    df['rr_score'] = 0.0
    df['fundamental_score'] = 0.0

    # 1. Volume Acceleration Score (40% weight) - The most crucial component
    if 'volume_acceleration' in df.columns:
        vol_accel = df['volume_acceleration'].fillna(0) # Treat NaN volume acceleration as neutral (0 for scoring)
        df.loc[vol_accel > 0, 'vol_accel_score'] = 25
        df.loc[vol_accel > 10, 'vol_accel_score'] = 50
        df.loc[vol_accel > 20, 'vol_accel_score'] = 75
        df.loc[vol_accel > 30, 'vol_accel_score'] = 100 # Max score for extreme acceleration
        df.loc[vol_accel < -20, 'vol_accel_score'] = 0 # Penalize significant negative acceleration
        df['edge_score'] += df['vol_accel_score'] * 0.40 # Apply 40% weight

    # 2. Momentum Divergence Score (25% weight)
    if 'momentum_divergence' in df.columns and 'volume_acceleration' in df.columns and 'short_momentum' in df.columns:
        mom_div_safe = df['momentum_divergence'].fillna(0)
        vol_accel_safe = df['volume_acceleration'].fillna(0)
        short_mom_safe = df['short_momentum'].fillna(0)

        # Base score for positive divergence with any positive volume
        mask1 = (mom_div_safe > 0) & (vol_accel_safe > 0)
        df.loc[mask1, 'momentum_score'] = 60

        # Higher score for strong momentum acceleration
        mask2 = (mom_div_safe > 5) & (short_mom_safe > 0)
        df.loc[mask2, 'momentum_score'] = 80

        # Highest score for the valuable 'stealth accumulation' pattern
        mask3 = (mom_div_safe < 0) & (vol_accel_safe > 20)
        df.loc[mask3, 'momentum_score'] = 100
        df['edge_score'] += df['momentum_score'] * 0.25 # Apply 25% weight

    # 3. Risk/Reward Score (20% weight)
    if 'risk_reward_ratio' in df.columns:
        # Scale the ratio (0-10) to a score (0-100)
        df['rr_score'] = (df['risk_reward_ratio'].fillna(0) * 10).clip(0, 100) # Assuming ratio max of 10, scale to 100
        df['edge_score'] += df['rr_score'] * 0.20 # Apply 20% weight

    # 4. Fundamental Score (15% weight)
    fundamental_score_component = pd.Series(0.0, index=df.index)
    fundamental_factors_count = 0
    current_fundamental_weight = 0.15 # Initial weight for fundamental factors

    if 'eps_change_pct' in df.columns:
        eps_data = df['eps_change_pct'].fillna(0)
        eps_score_temp = pd.Series(0.0, index=df.index)
        eps_score_temp[eps_data > 0] = 30
        eps_score_temp[eps_data > 15] = 60
        eps_score_temp[eps_data > 30] = 100
        fundamental_score_component += eps_score_temp
        fundamental_factors_count += 1

    if 'pe' in df.columns:
        pe_data = df['pe'].fillna(50) # Use a neutral PE if missing
        pe_score_temp = pd.Series(0.0, index=df.index)
        pe_score_temp[(pe_data > 5) & (pe_data < 40)] = 50 # Reasonable PE range
        pe_score_temp[(pe_data > 10) & (pe_data < 25)] = 100 # Optimal PE range
        fundamental_score_component += pe_score_temp
        fundamental_factors_count += 1

    if fundamental_factors_count > 0:
        df['fundamental_score'] = fundamental_score_component / fundamental_factors_count
        df['edge_score'] += df['fundamental_score'] * current_fundamental_weight
    else:
        # If no fundamental data, redistribute its weight proportionally to other categories
        # Remaining weight = 1 - current_fundamental_weight = 0.85
        # The sum of technical weights is 0.40 + 0.25 + 0.20 = 0.85
        # So, we just scale up the current edge_score
        if df['edge_score'].sum() > 0:
            df['edge_score'] = (df['edge_score'] / 0.85 * 1.0).clip(0, 100) # Scale to max 100

    # Bonus multipliers for trend alignment
    [cite_start]if all(col in df.columns for col in ['price', 'sma_50d', 'sma_200d']): [cite: 6, 13, 14]
        price_data = df['price'].fillna(0)
        sma50_data = df['sma_50d'].fillna(price_data) # If SMA is missing, assume it's at price
        sma200_data = df['sma_200d'].fillna(price_data)

        # 5 point bonus if price is above both 50-day and 200-day SMAs (strong uptrend)
        trend_bonus = ((price_data > sma50_data) & (price_data > sma200_data)).astype(int) * 5
        df['edge_score'] = (df['edge_score'] + trend_bonus).clip(0, 100)

    # Additional bonus for stocks with room to run (not overextended near 52-week high)
    [cite_start]if 'from_high_pct' in df.columns: [cite: 11]
        room_bonus = pd.Series(0.0, index=df.index)
        from_high = df['from_high_pct'].fillna(0) # Treat NaN as 0 (no drawdown)

        # Give points if stock is in a "buyable dip" range from its highs
        room_bonus[(from_high < -15) & (from_high > -40)] = 5 # Good dip
        room_bonus[(from_high < -20) & (from_high > -35)] = 10 # Optimal dip range
        df['edge_score'] = (df['edge_score'] + room_bonus).clip(0, 100)

    # Final NaN handling for edge_score: ensure all NaNs are 0
    df['edge_score'] = df['edge_score'].fillna(0)

    # Final classification into EDGE categories using the defined thresholds
    df['edge_category'] = pd.cut(
        df['edge_score'],
        bins=[-0.1, EDGE_THRESHOLDS['WATCH'], EDGE_THRESHOLDS['MODERATE'], EDGE_THRESHOLDS['STRONG'], EDGE_THRESHOLDS['EXPLOSIVE'], 100.1],
        labels=['NO_EDGE', 'WATCH', 'MODERATE', 'STRONG', 'EXPLOSIVE'],
        right=False # Bins are inclusive on the left, exclusive on the right (except the last one)
    ).astype(str) # Convert to string to avoid CategoricalDtype issues in Streamlit

    return df

def calculate_position_metrics(df):
    """
    Calculates suggested position sizing, stop loss, and target prices based on EDGE scores.
    """
    # Initialize all new columns with default/NaN values
    df['suggested_position_pct'] = 0.0
    df['stop_loss'] = np.nan
    df['stop_loss_pct'] = np.nan
    df['target_1'] = np.nan
    df['target_2'] = np.nan
    df['target_1_pct'] = np.nan
    df['target_2_pct'] = np.nan

    # Map EDGE category to suggested position percentage
    position_map = {
        'EXPLOSIVE': 10,
        'STRONG': 5,
        'MODERATE': 2,
        'WATCH': 0,
        'NO_EDGE': 0
    }
    if 'edge_category' in df.columns:
        df['suggested_position_pct'] = df['edge_category'].map(position_map).fillna(0).astype(float)

    # Stop loss calculation
    [cite_start]if all(col in df.columns for col in ['price', 'low_52w', 'sma_50d']): [cite: 6, 7, 13]
        valid_price_mask = (df['price'] > 0).fillna(False) # Ensure price is valid for calculation

        # Dynamic stop loss: Max of (7% below current price, 2% below 50-day SMA, 2% above 52-week low)
        # This provides multiple levels of support as potential stop points.
        # Fill SMA/low NaNs with a value that doesn't affect the 'max' if that specific data is missing
        sl_from_price = df.loc[valid_price_mask, 'price'] * 0.93
        sl_from_sma50 = df.loc[valid_price_mask, 'sma_50d'].fillna(df.loc[valid_price_mask, 'price']) * 0.98
        sl_from_low52w = df.loc[valid_price_mask, 'low_52w'].fillna(df.loc[valid_price_mask, 'price']) * 1.02

        # Combine these and assign back to stop_loss for valid entries
        df.loc[valid_price_mask, 'stop_loss'] = np.maximum.reduce([sl_from_price, sl_from_sma50, sl_from_low52w])

        # Ensure stop loss is never higher than the current price (a common edge case for bad data)
        df.loc[(df['stop_loss'].notna()) & (df['stop_loss'] > df['price']), 'stop_loss'] = df['price'] * 0.93

        # Calculate stop loss percentage
        valid_sl_calc_mask = valid_price_mask & df['stop_loss'].notna()
        df.loc[valid_sl_calc_mask, 'stop_loss_pct'] = \
            ((df.loc[valid_sl_calc_mask, 'stop_loss'] - df.loc[valid_sl_calc_mask, 'price']) / df.loc[valid_sl_calc_mask, 'price'] * 100).round(2)
        df['stop_loss_pct'].fillna(0, inplace=True) # Fill remaining NaNs for display

    # Target calculation
    [cite_start]if 'upside_potential' in df.columns and 'price' in df.columns: [cite: 6]
        valid_target_calc_mask = (df['price'] > 0).fillna(False) & df['upside_potential'].notna()

        # Target 1: 25% of upside potential from current price
        df.loc[valid_target_calc_mask, 'target_1'] = \
            df.loc[valid_target_calc_mask, 'price'] * (1 + df.loc[valid_target_calc_mask, 'upside_potential'] * 0.25 / 100)
        # Target 2: 50% of upside potential from current price
        df.loc[valid_target_calc_mask, 'target_2'] = \
            df.loc[valid_target_calc_mask, 'price'] * (1 + df.loc[valid_target_calc_mask, 'upside_potential'] * 0.50 / 100)

        # Calculate target percentages
        df.loc[valid_target_calc_mask, 'target_1_pct'] = \
            ((df.loc[valid_target_calc_mask, 'target_1'] - df.loc[valid_target_calc_mask, 'price']) / df.loc[valid_target_calc_mask, 'price'] * 100).round(2)
        df.loc[valid_target_calc_mask, 'target_2_pct'] = \
            ((df.loc[valid_target_calc_mask, 'target_2'] - df.loc[valid_target_calc_mask, 'price']) / df.loc[valid_target_calc_mask, 'price'] * 100).round(2)

        df['target_1_pct'].fillna(0, inplace=True)
        df['target_2_pct'].fillna(0, inplace=True)

    return df

# ============================================================================
# VISUALIZATION COMPONENTS
# ============================================================================

def create_edge_distribution_chart(df):
    """Visualizes the distribution of stocks across different EDGE categories."""
    if 'edge_category' in df.columns and not df.empty:
        # Define a consistent order for categories to ensure predictable chart display
        category_order = ['EXPLOSIVE', 'STRONG', 'MODERATE', 'WATCH', 'NO_EDGE']
        # Count occurrences and reindex to ensure all categories are present, even if with 0 count
        edge_counts = df['edge_category'].value_counts().reindex(category_order, fill_value=0)

        # Define consistent colors for each category
        colors_map = {
            'EXPLOSIVE': '#f5576c',  # Red-pink for highest conviction
            'STRONG': '#ffaa00',    # Orange for strong
            'MODERATE': '#ffdd44',  # Yellow for moderate
            'WATCH': '#888888',     # Gray for watch
            'NO_EDGE': '#cccccc'    # Light gray for no edge
        }
        # Get colors in the defined order, only for present categories
        bar_colors = [colors_map[cat] for cat in edge_counts.index]

        if len(edge_counts) > 0:
            fig = go.Figure(data=[go.Bar(
                x=edge_counts.index,
                y=edge_counts.values,
                text=edge_counts.values, # Display count on bars
                textposition='auto',
                marker_color=bar_colors
            )])

            fig.update_layout(
                title="EDGE Distribution Across Market",
                xaxis_title="EDGE Category",
                yaxis_title="Number of Stocks",
                height=400,
                template="plotly_white", # Clean white background theme
                # Adjust margins for better fit
                margin=dict(l=50, r=50, t=50, b=50)
            )
            return fig
    # Fallback chart if no categories or edge scores are available or DataFrame is empty
    if 'edge_score' in df.columns and not df['edge_score'].empty:
        fig = go.Figure(data=[go.Histogram(
            x=df['edge_score'],
            nbinsx=20, # Number of bins for the histogram
            marker_color='lightblue',
            hovertemplate='Score: %{x}<br>Count: %{y}<extra></extra>' # Custom hover info
        )])

        fig.update_layout(
            title="EDGE Score Distribution (Fallback)",
            xaxis_title="EDGE Score",
            yaxis_title="Number of Stocks",
            height=400,
            template="plotly_white",
            margin=dict(l=50, r=50, t=50, b=50)
        )
        return fig

    return None # Return None if no meaningful chart can be created

def create_volume_acceleration_scatter(df):
    """
    Creates the 'Secret Weapon' scatter plot: Volume Acceleration vs. Short-term Momentum.
    Highlights different EDGE categories.
    """
    # Filter for stocks with valid data for the plot, and limit to top N stocks for clarity
    # Ensure all columns used for plotting are non-NaN and edge_score > 0
    plot_df = df[
        (df['edge_score'].notna()) & (df['edge_score'] > 0) &
        df['volume_acceleration'].notna() &
        df['short_momentum'].notna() &
        df['ticker'].notna() # Ensure ticker is present for labels
    ].nlargest(200, 'edge_score').copy() # Increased to 200 for more visibility

    if len(plot_df) < 5: # Need a minimum number of points for a meaningful scatter plot
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5,
            text="Insufficient data for Volume Acceleration Map. Need more qualified stocks.",
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            title="Volume Acceleration Map - Data Insufficient",
            height=600,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            template="plotly_white"
        )
        return fig

    fig = go.Figure()

    # Define colors for each category (consistent with bar chart)
    colors_map = {
        'EXPLOSIVE': '#f5576c',
        'STRONG': '#ffaa00',
        'MODERATE': '#ffdd44',
        'WATCH': '#888888',
        'NO_EDGE': '#cccccc'
    }

    # Plot each category separately to control legend, color, and hover info
    # Sort categories to ensure consistent legend order
    category_order_for_plot = ['EXPLOSIVE', 'STRONG', 'MODERATE', 'WATCH', 'NO_EDGE']
    for category in category_order_for_plot:
        cat_stocks = plot_df[plot_df['edge_category'] == category]
        if not cat_stocks.empty:
            fig.add_trace(go.Scatter(
                x=cat_stocks['volume_acceleration'],
                y=cat_stocks['short_momentum'],
                mode='markers+text', # Show markers and text (ticker)
                name=category, # Name for legend
                text=cat_stocks['ticker'], # Text to display (ticker symbol)
                textposition="top center",
                textfont=dict(size=8),
                marker=dict(
                    size=(cat_stocks['edge_score'] / 10) + 5, # Scale marker size by edge score, with a minimum base size
                    color=colors_map.get(category, '#cccccc'), # Use defined colors, default to light gray
                    line=dict(width=1, color='black') # Black border for markers
                ),
                hovertemplate='<b>%{text}</b><br>Vol Accel: %{x:.1f}%<br>Momentum: %{y:.1f}%<br>Edge Score: %{customdata:.1f}<extra></extra>',
                customdata=cat_stocks['edge_score'] # Custom data for hover (edge score)
            ))

    # Add quadrant lines for visual guidance (x=0 for Vol Accel, y=0 for Momentum)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, annotation_text="Neutral Momentum", annotation_position="bottom right")
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5, annotation_text="Neutral Volume Accel", annotation_position="top left")

    # Add quadrant labels for interpretation
    # Dynamically position labels based on actual data range for better visibility
    # Use max/min of plotted data, or fallback values if the range is too small
    max_vol_accel = plot_df['volume_acceleration'].max() if not plot_df['volume_acceleration'].empty else 30
    min_vol_accel = plot_df['volume_acceleration'].min() if not plot_df['volume_acceleration'].empty else -30
    max_short_momentum = plot_df['short_momentum'].max() if not plot_df['short_momentum'].empty else 20
    min_short_momentum = plot_df['short_momentum'].min() if not plot_df['short_momentum'].empty else -20

    fig.add_annotation(x=max_vol_accel * 0.7, y=max_short_momentum * 0.7,
                       text="🔥 EXPLOSIVE ZONE", showarrow=False, font=dict(size=14, color="red"))
    fig.add_annotation(x=max_vol_accel * 0.7, y=min_short_momentum * 0.7 if min_short_momentum < 0 else -10,
                       text="🏦 STEALTH ACCUMULATION", showarrow=False, font=dict(size=14, color="green"))
    fig.add_annotation(x=min_vol_accel * 0.7 if min_vol_accel < 0 else -20, y=max_short_momentum * 0.7,
                       text="⚠️ PROFIT TAKING", showarrow=False, font=dict(size=14, color="orange"))
    fig.add_annotation(x=min_vol_accel * 0.7 if min_vol_accel < 0 else -20, y=min_short_momentum * 0.7 if min_short_momentum < 0 else -10,
                       text="💀 AVOID", showarrow=False, font=dict(size=14, color="gray"))

    fig.update_layout(
        title="Volume Acceleration Map - Your SECRET EDGE",
        xaxis_title="Volume Acceleration (30d/90d vs 30d/180d)",
        yaxis_title="Short-term Momentum (%)",
        height=600,
        showlegend=True,
        hovermode="closest", # Shows hover info for the closest point
        template="plotly_white"
    )

    return fig

def create_edge_radar(stock_data):
    """
    Creates a radar chart to visualize the individual components contributing to a stock's EDGE score.
    """
    # Define categories for the radar chart
    categories = ['Volume\nAcceleration', 'Momentum\nDivergence', 'Risk/Reward',
                  'Fundamental\nStrength', 'Trend\nAlignment']

    # Safely retrieve the component scores from stock_data, defaulting to 0 if not present
    values = [
        stock_data.get('vol_accel_score', 0),
        stock_data.get('momentum_score', 0),
        stock_data.get('rr_score', 0),
        stock_data.get('fundamental_score', 0),
        # Trend alignment: A bonus score of 100 if price is above 200-day SMA, else 0
        100 if stock_data.get('price', 0) > stock_data.get('sma_200d', -1) else 0 # Use -1 for SMA_200d default to ensure 0 value if missing or invalid
    ]

    fig = go.Figure(data=go.Scatterpolar(
        r=values, # Radial coordinates (scores)
        theta=categories, # Angular coordinates (categories)
        fill='toself', # Fill the area defined by the trace
        name='EDGE Components',
        marker_color='#4CAF50', # Pleasant green color for markers
        line_color='#2E8B57' # Darker green for the line
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100], # Set radial axis range from 0 to 100 for scores
                gridcolor='lightgray',
                linecolor='gray'
            ),
            angularaxis=dict(
                linecolor='gray'
            )
        ),
        showlegend=False, # No legend needed for single trace
        title=f"<b>{stock_data.get('ticker', 'Stock')}</b> - EDGE Analysis", # Bold ticker name
        height=400,
        margin=dict(l=50, r=50, t=80, b=50), # Adjust margins for better fit and title visibility
        template="plotly_white"
    )

    return fig

# ============================================================================
# MAIN STREAMLIT APPLICATION FUNCTION
# ============================================================================

def main():
    # Custom CSS for a vibrant and professional Streamlit interface
    st.markdown("""
    <style>
    .main-header {
        font-size: 3.5em;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4); /* Gradient text color */
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
        padding-top: 10px;
    }
    .sub-header {
        font-size: 1.5em;
        color: #666;
        text-align: center;
        margin-bottom: 30px;
    }
    .edge-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); /* Blue-purple gradient */
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin: 10px 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .explosive-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); /* Pink to Red gradient */
    }
    .stMetric {
        background-color: #f0f2f6; /* Light gray background for metrics */
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); /* Subtle shadow */
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 1.2em;
        padding: 10px 20px;
    }
    .stAlert {
        border-radius: 10px;
    }
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden; /* Ensures borders are rounded correctly */
    }
    </style>
    """, unsafe_allow_html=True)

    # Main Application Header
    st.markdown('<h1 class="main-header">⚡ EDGE Protocol</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Finding What Others Can\'t See</p>', unsafe_allow_html=True)

    # Load and process data with a spinner for user feedback
    with st.spinner("Calculating EDGE across stocks... This might take a moment based on data size and connection."):
        df = load_data()

        # If data loading fails or returns empty DataFrame, display error and stop execution
        if df.empty:
            st.error("Failed to load or process data. Please ensure the Google Sheet is correctly published to the web as CSV and the URL in the code is accurate. Check the 'How It Works' tab under 'Data & System Health Check' for diagnostics.")
            st.stop() # Halts the app execution gracefully

        # --- Sidebar for Debugging and Data Insights ---
        with st.sidebar:
            st.markdown("### ⚙️ Debug & Data Insights")
            st.write(f"Total rows loaded: **{len(df)}**")
            st.write(f"Total columns: **{len(df.columns)}**")

            # Check for presence of critical columns needed for calculations
            critical_cols_check = ['vol_ratio_30d_90d', 'vol_ratio_30d_180d', 'price', 'ticker', 'high_52w', 'low_52w']
            missing_critical = [col for col in critical_cols_check if col not in df.columns]
            if missing_critical:
                st.error(f"🚨 Missing critical columns for full functionality: {', '.join(missing_critical)}")
            else:
                st.success("✅ All essential columns are present.")

            # Option to display a raw data sample for debugging
            if st.checkbox("Show Raw Data Sample"):
                st.write("First 5 rows with key columns:")
                sample_cols_display = ['ticker', 'price', 'vol_ratio_30d_90d', 'vol_ratio_30d_180d', 'ret_7d']
                # Filter to only show columns that actually exist in the DataFrame
                available_sample_cols = [col for col in sample_cols_display if col in df.columns]
                if available_sample_cols:
                    st.dataframe(df[available_sample_cols].head())
                else:
                    st.info("No relevant sample columns available for display.")

            # Option to test calculations for a specific stock
            if st.checkbox("Test specific stock calculations"):
                test_ticker_input = st.text_input("Enter ticker (e.g., RELIANCE):", value="RELIANCE").upper()
                if test_ticker_input and test_ticker_input in df['ticker'].values:
                    test_stock_data_pre_calc = df[df['ticker'] == test_ticker_input].iloc[0] # Get before main calculations
                    st.write(f"**Raw Data for {test_ticker_input}:**")
                    st.json({
                        col: test_stock_data_pre_calc.get(col, 'N/A')
                        for col in ['price', 'vol_ratio_30d_90d', 'vol_ratio_30d_180d', 'ret_7d', 'ret_30d', 'pe', 'eps_change_pct']
                        if col in test_stock_data_pre_calc
                    })
                else:
                    st.warning(f"Ticker '{test_ticker_input}' not found in data or invalid input.")

        # --- Perform all EDGE calculations ---
        df = calculate_volume_acceleration(df)
        df = calculate_momentum_divergence(df)
        df = calculate_risk_reward(df)
        df = calculate_time_arbitrage(df)
        df = calculate_edge_scores(df)
        df = calculate_position_metrics(df)

        # Fallback scoring: If the main 'edge_score' calculation yields no meaningful results (e.g., all NaNs, all zeros),
        # apply a simplified scoring to ensure some functionality. This is a safeguard.
        if 'edge_score' not in df.columns or df['edge_score'].isnull().all() or df['edge_score'].sum() == 0:
            st.warning("⚠️ Main EDGE scoring failed or resulted in all zeros. Applying simplified scoring. Results may vary.")
            df['edge_score'] = 0.0 # Reset to float for simplified calculations

            # Simple momentum score based on 7-day and 30-day returns
            if 'ret_7d' in df.columns and 'ret_30d' in df.columns:
                df['simple_momentum_score'] = (
                    (df['ret_7d'].fillna(0) > 3).astype(int) * 20 +
                    (df['ret_30d'].fillna(0) > 5).astype(int) * 20 +
                    (df['ret_7d'].fillna(0) > df['ret_30d'].fillna(0)/4.3).astype(int) * 20 # Short-term outperforming long-term
                )
                df['edge_score'] += df['simple_momentum_score']
            else:
                st.info("Missing return data for simple momentum scoring.")

            # Simple value score based on distance from 52-week high
            if 'from_high_pct' in df.columns:
                df['simple_value_score'] = pd.Series(0.0, index=df.index)
                df.loc[(df['from_high_pct'].fillna(0) < -20) & (df['from_high_pct'].fillna(0) > -40), 'simple_value_score'] = 20 # 20-40% below high is a good dip
                df['edge_score'] += df['simple_value_score']
            else:
                st.info("Missing 'from_high_pct' for simple value scoring.")

            # Simple trend score based on SMAs
            if all(col in df.columns for col in ['price', 'sma_50d', 'sma_200d']):
                df['simple_trend_score'] = (
                    (df['price'].fillna(0) > df['sma_50d'].fillna(0)).astype(int) * 10 +
                    (df['price'].fillna(0) > df['sma_200d'].fillna(0)).astype(int) * 10
                )
                df['edge_score'] += df['simple_trend_score']
            else:
                st.info("Missing SMA data for simple trend scoring.")

            # Re-classify with simplified scoring
            df['edge_category'] = pd.cut(
                df['edge_score'],
                bins=[-0.1, 20, 40, 60, 80, 100.1],
                labels=['NO_EDGE', 'WATCH', 'MODERATE', 'STRONG', 'EXPLOSIVE'],
                right=False
            ).astype(str) # Ensure string conversion

            # Drop temporary simple score columns
            df.drop(columns=['simple_momentum_score', 'simple_value_score', 'simple_trend_score'], errors='ignore', inplace=True)

        # More detailed debug info in sidebar after calculations
        with st.sidebar:
            st.markdown("---")
            st.markdown("### 📊 Calculated Metrics Summary")
            if 'volume_acceleration' in df.columns and not df['volume_acceleration'].empty:
                vol_accel_stats = df['volume_acceleration'].describe(percentiles=[]).round(2)
                st.write("\n**Volume Acceleration Stats:**")
                st.write(f"Mean: {vol_accel_stats.get('mean', np.nan):.2f}%")
                st.write(f"Max: {vol_accel_stats.get('max', np.nan):.2f}%")
                st.write(f"Min: {vol_accel_stats.get('min', np.nan):.2f}%")
                st.write(f"Valid values: {df['volume_acceleration'].notna().sum()}")
            else:
                st.info("Volume acceleration data not available for summary.")

            if 'edge_score' in df.columns and not df['edge_score'].empty:
                edge_stats = df['edge_score'].describe(percentiles=[]).round(2)
                st.write("\n**EDGE Score Stats:**")
                st.write(f"Mean: {edge_stats.get('mean', np.nan):.2f}")
                st.write(f"Max: {edge_stats.get('max', np.nan):.2f}")
                st.write(f"Count > 50: {(df['edge_score'] > 50).sum()}")
                st.write(f"Count > 70: {(df['edge_score'] > 70).sum()}")
                st.write(f"Count > 85: {(df['edge_score'] > 85).sum()}")
            else:
                st.info("EDGE score data not available for summary.")


    # Ensure edge_category is string type for filtering reliability
    df['edge_category'] = df['edge_category'].astype(str)

    # Filter stocks into categories based on calculated 'edge_category'
    explosive_stocks = df[df['edge_category'] == 'EXPLOSIVE'].sort_values('edge_score', ascending=False)
    strong_stocks = df[df['edge_category'] == 'STRONG'].sort_values('edge_score', ascending=False)
    moderate_stocks = df[df['edge_category'] == 'MODERATE'].sort_values('edge_score', ascending=False)

    # --- Summary Metrics (Top of the page) ---
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("🔥 EXPLOSIVE EDGE", len(explosive_stocks),
                 help="Highest conviction opportunities (Top 1%). Suggested position: 10% of capital.")

    with col2:
        st.metric("💎 STRONG EDGE", len(strong_stocks),
                 help="High conviction opportunities (Top 5%). Suggested position: 5% of capital.")

    with col3:
        st.metric("📈 MODERATE EDGE", len(moderate_stocks),
                 help="Good opportunities (Top 10%). Suggested position: 2% of capital.")

    with col4:
        # Average Volume Acceleration for high-edge stocks, if available
        if 'volume_acceleration' in df.columns and not df['volume_acceleration'].empty:
            high_edge_df_for_avg = df[df['edge_score'] > 70].copy()
            if not high_edge_df_for_avg['volume_acceleration'].empty:
                avg_vol_accel = high_edge_df_for_avg['volume_acceleration'].mean()
                st.metric("🔍 Avg Vol Accel (High Edge)", f"{avg_vol_accel:.1f}%",
                         help="Average Volume Acceleration for stocks with EDGE score > 70. This is a key indicator of institutional activity.")
            else:
                st.metric("🔍 Avg Vol Accel (High Edge)", "N/A", help="No high EDGE stocks to calculate average volume acceleration.")
        else:
            st.metric("🔍 Vol Accel", "N/A", help="Volume acceleration data is not available.")


    # --- Main Content Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔥 Explosive Opportunities",
        "📊 EDGE Analysis",
        "📈 Market Map",
        "📚 How It Works"
    ])

    with tab1:
        st.markdown("### 🔥 Today's EXPLOSIVE EDGE Opportunities")

        if len(explosive_stocks) > 0:
            top_pick = explosive_stocks.iloc[0] # Get the very top pick

            # Display the top pick prominently in a custom card
            ticker = top_pick.get('ticker', 'UNKNOWN')
            company_name = top_pick.get('company_name', 'N/A')
            edge_score = top_pick.get('edge_score', 0)
            vol_accel = top_pick.get('volume_acceleration', 0)
            vol_status = top_pick.get('vol_accel_status', 'Unknown')

            st.markdown(f"""
            <div class="edge-card explosive-card">
            <h2 style='margin:0'>🏆 TOP EXPLOSIVE EDGE: {ticker} ({company_name})</h2>
            <h1 style='margin:10px 0'>EDGE SCORE: {edge_score:.1f}/100</h1>
            <p style='font-size:18px'>Volume Acceleration: {vol_accel:.1f}% ({vol_status})</p>
            </div>
            """, unsafe_allow_html=True)

            # Detailed analysis of the top pick, side-by-side layout
            col1_detail, col2_detail = st.columns([2, 1])

            with col1_detail:
                st.markdown("#### 📊 Why This Has EXPLOSIVE EDGE:")

                # Volume Intelligence section
                st.success(f"""
                **🔍 Volume Intelligence (YOUR SECRET WEAPON):**
                - **Acceleration**: **{top_pick.get('volume_acceleration', 0):.1f}%** ({top_pick.get('vol_accel_status', 'Unknown')})
                - **30d vs 90d Avg Vol**: {top_pick.get('vol_ratio_30d_90d', 0):.1f}%
                - **30d vs 180d Avg Vol**: {top_pick.get('vol_ratio_30d_180d', 0):.1f}%
                - **Interpretation**: This shows **AGGRESSIVE ACCUMULATION**, indicating smart money positioning before a major move.
                """)

                # Momentum Analysis section
                st.info(f"""
                **📈 Momentum Analysis:**
                - **Short-term Momentum (avg 1-7d)**: {top_pick.get('short_momentum', 0):.1f}%
                - **Long-term Momentum (avg 30d-3m)**: {top_pick.get('long_momentum', 0):.1f}%
                - **Divergence Pattern**: {top_pick.get('divergence_pattern', 'Analyzing...')}
                - **Interpretation**: Identifies if momentum is building, breaking out, or if there's **stealth accumulation** (price stagnant/down but institutions buying).
                """)

                # Risk/Reward Setup section
                st.warning(f"""
                **🎯 Risk/Reward Setup:**
                - **Current Price**: ₹{top_pick.get('price', 0):.2f}
                - **Suggested Stop Loss**: ₹{top_pick.get('stop_loss', 0):.2f} ({top_pick.get('stop_loss_pct', 0):.1f}%)
                - **Target 1**: ₹{top_pick.get('target_1', 0):.2f} (+{top_pick.get('target_1_pct', 0):.1f}%)
                - **Target 2**: ₹{top_pick.get('target_2', 0):.2f} (+{top_pick.get('target_2_pct', 0):.1f}%)
                - **Calculated R:R Ratio**: 1:{top_pick.get('risk_reward_ratio', 0):.1f}
                - **Interpretation**: Ensures a favorable risk-to-reward before committing capital.
                """)

            with col2_detail:
                # Display the radar chart for the top pick
                fig_radar = create_edge_radar(top_pick)
                st.plotly_chart(fig_radar, use_container_width=True)

            # --- All Explosive Opportunities Table ---
            st.markdown("### 🔥 All EXPLOSIVE EDGE Stocks")

            # Define columns to display for the table
            display_cols_explosive = [
                'ticker', 'company_name', 'edge_score', 'volume_acceleration',
                'short_momentum', 'risk_reward_ratio', 'price',
                'suggested_position_pct', 'stop_loss', 'target_1', 'target_2'
            ]
            # Filter to only include columns that actually exist in the DataFrame
            available_cols_explosive = [col for col in display_cols_explosive if col in explosive_stocks.columns]

            if len(explosive_stocks) > 0 and len(available_cols_explosive) >= 3:
                # Limit to top 20 explosive stocks for display
                display_df_explosive = explosive_stocks[available_cols_explosive].head(20)

                # Define formatting for numeric columns
                format_dict_explosive = {
                    'edge_score': '{:.1f}',
                    'volume_acceleration': '{:.1f}%',
                    'short_momentum': '{:.1f}%',
                    'risk_reward_ratio': '{:.1f}x',
                    'price': '₹{:.2f}',
                    'suggested_position_pct': '{:.1f}%',
                    'stop_loss': '₹{:.2f}',
                    'target_1': '₹{:.2f}',
                    'target_2': '₹{:.2f}'
                }
                # Display DataFrame with formatting and conditional styling
                st.dataframe(
                    display_df_explosive.style.format(format_dict_explosive).background_gradient(
                        subset=['edge_score'], cmap='Reds' # Color code by edge score
                    ),
                    use_container_width=True,
                    height=400
                )
            else:
                st.info("No EXPLOSIVE EDGE opportunities found today. The market might be consolidating or lacking strong signals.")

        else: # If no Explosive stocks at all
            st.info("No EXPLOSIVE EDGE opportunities found today. The market might be consolidating or lacking strong signals. Please check the 'STRONG EDGE' category below for the next best opportunities.")
            # Provide a fallback view of top momentum stocks if no high-edge signals
            if 'ret_7d' in df.columns and 'price' in df.columns:
                st.markdown("### 📈 Top Momentum Stocks (Alternative View)")
                momentum_stocks_fallback = df[
                    (df['ret_7d'] > 5) &
                    (df['price'] > 0) &
                    (df['ret_7d'] < 30) # Not too overextended
                ].sort_values('ret_7d', ascending=False).head(10)

                if not momentum_stocks_fallback.empty:
                    display_cols_momentum = ['ticker', 'company_name', 'price', 'ret_7d', 'ret_30d', 'from_high_pct', 'pe']
                    available_cols_momentum = [col for col in display_cols_momentum if col in momentum_stocks_fallback.columns]
                    st.dataframe(
                        momentum_stocks_fallback[available_cols_momentum],
                        use_container_width=True,
                        height=300
                    )
                    st.info("""
                    💡 These stocks show strong momentum but might lack the volume acceleration confirmation of true EDGE signals.
                    Consider these for watchlist only.
                    """)
                else:
                    st.info("Market is in consolidation. No strong momentum detected.")


        # --- STRONG EDGE Opportunities Table ---
        st.markdown("### 💎 STRONG EDGE Opportunities")

        if len(strong_stocks) > 0:
            display_cols_strong = [
                'ticker', 'company_name', 'edge_score', 'volume_acceleration',
                'momentum_divergence', 'price', 'suggested_position_pct'
            ]
            available_cols_strong = [col for col in display_cols_strong if col in strong_stocks.columns]

            if len(strong_stocks) > 0 and len(available_cols_strong) >= 3:
                format_dict_strong = {
                    'edge_score': '{:.1f}',
                    'volume_acceleration': '{:.1f}%',
                    'momentum_divergence': '{:.1f}%',
                    'price': '₹{:.2f}',
                    'suggested_position_pct': '{:.1f}%'
                }
                st.dataframe(
                    strong_stocks[available_cols_strong].head(10).style.format(format_dict_strong).background_gradient(
                        subset=['edge_score'], cmap='Oranges' # Color code by edge score
                    ),
                    use_container_width=True,
                    height=300
                )
            else:
                st.warning("Limited data available to display STRONG EDGE stocks.")
        else:
            st.info("""
            📊 No STRONG EDGE signals detected in current market conditions.
            This indicates the market might be in a consolidation phase,
            or lacking the distinct volume acceleration patterns the EDGE Protocol identifies.
            """)


    with tab2:
        st.markdown("### 📊 Deep EDGE Analysis")

        # Display EDGE distribution chart
        fig_dist = create_edge_distribution_chart(df)
        if fig_dist:
            st.plotly_chart(fig_dist, use_container_width=True)
        else:
            st.info("No EDGE score distribution data available for plotting. Check data loading.")

        # Display Detected Patterns Across Market
        st.markdown("### 🎯 Detected Patterns Across Market")

        if 'divergence_pattern' in df.columns:
            # Filter out 'NEUTRAL' and 'NO_DATA' patterns as they are not "detected patterns"
            pattern_counts = df['divergence_pattern'].value_counts()
            pattern_counts = pattern_counts[~pattern_counts.index.isin(['NEUTRAL', 'NO_DATA'])].sort_values(ascending=False)

            if len(pattern_counts) > 0:
                cols_patterns = st.columns(3) # Create 3 columns for metrics
                for i, (pattern, count) in enumerate(pattern_counts.items()):
                    with cols_patterns[i % 3]: # Distribute metrics across columns
                        emoji = "🚀" if "EXPLOSIVE" in pattern else ("📈" if "MOMENTUM" in pattern else ("🏦" if "STEALTH" in pattern else "💡"))
                        st.metric(f"{emoji} {pattern.replace('_', ' ').title()}", count) # Format label
            else:
                st.info("No significant momentum divergence patterns detected in the market today.")
        else:
            st.warning("Momentum divergence pattern data is not available for analysis.")

        # Sector Edge Analysis
        st.markdown("### 🏭 Edge by Sector")

        if 'sector' in df.columns and 'edge_score' in df.columns:
            # Group by sector and calculate average EDGE score, average volume acceleration, and stock count
            sector_edge = df.dropna(subset=['sector', 'edge_score']).groupby('sector').agg(
                edge_score_mean=('edge_score', 'mean'),
                volume_acceleration_mean=('volume_acceleration', lambda x: x.mean() if x.notna().any() else np.nan),
                stock_count=('ticker', 'count')
            ).sort_values('edge_score_mean', ascending=False)

            # Filter out sectors with very few stocks for more meaningful averages (e.g., minimum 5 stocks)
            sector_edge = sector_edge[sector_edge['stock_count'] >= 5].head(15) # Display top 15 sectors

            if not sector_edge.empty:
                fig_sector = go.Figure(data=[go.Bar(
                    x=sector_edge.index,
                    y=sector_edge['edge_score_mean'],
                    text=sector_edge['edge_score_mean'].round(1),
                    textposition='auto',
                    marker_color=sector_edge['edge_score_mean'],
                    marker_colorscale='Viridis' # Use a color scale for visual appeal
                )])

                fig_sector.update_layout(
                    title="Average EDGE Score by Sector (Top 15 with 5+ Stocks)",
                    xaxis_title="Sector",
                    yaxis_title="Average EDGE Score",
                    height=450,
                    template="plotly_white",
                    margin=dict(l=50, r=50, t=50, b=50)
                )
                st.plotly_chart(fig_sector, use_container_width=True)
            else:
                st.info("Not enough data to perform sector-wise EDGE analysis or too few stocks per sector.")
        else:
            st.warning("Sector or Edge Score data is missing, cannot generate Sector Edge Analysis.")

    with tab3: # Market Map Tab
        st.markdown("### 📈 Market EDGE Map")

        # Display the Volume Acceleration Scatter plot
        fig_scatter = create_volume_acceleration_scatter(df)
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Overall Market Statistics
        st.markdown("### 📊 Overall Market Statistics")

        col_market1, col_market2, col_market3, col_market4 = st.columns(4)

        with col_market1:
            if 'volume_acceleration' in df.columns and not df['volume_acceleration'].empty:
                avg_vol_accel_overall = df['volume_acceleration'].mean()
                st.metric("Avg Volume Acceleration (All Stocks)", f"{avg_vol_accel_overall:.1f}%", help="Average volume acceleration across all analyzed stocks.")
            else:
                st.metric("Avg Volume Acceleration", "N/A")

        with col_market2:
            if 'volume_acceleration' in df.columns:
                positive_accel_count = (df['volume_acceleration'] > 0).sum()
                st.metric("Stocks with Positive Accumulation", positive_accel_count, help="Number of stocks showing positive volume acceleration (accumulation).")
            else:
                st.metric("Positive Accel Stocks", "N/A")

        with col_market3:
            if 'edge_score' in df.columns:
                high_edge_count = (df['edge_score'] > 70).sum()
                st.metric("High EDGE Stocks (>70 Score)", high_edge_count, help="Number of stocks with an EDGE score greater than 70 (Strong or Explosive).")
            else:
                st.metric("High EDGE Stocks", "N/A")

        with col_market4:
            if not df.empty and 'edge_category' in df.columns:
                explosive_pct_market = (len(explosive_stocks) / len(df)) * 100 if len(df) > 0 else 0
                st.metric("Explosive Opportunities (%)", f"{explosive_pct_market:.2f}%", help="Percentage of total analyzed stocks that are classified as 'Explosive EDGE'.")
            else:
                st.metric("Explosive %", "N/A")

    with tab4: # How It Works Tab
        st.markdown("""
        ### 📚 How EDGE Protocol Works

        #### 🔍 The Secret Weapon: Volume Acceleration

        While everyone watches price and basic volume, EDGE Protocol uses **Volume Acceleration** -
        comparing 30-day average daily volume ratios against both 90-day AND 180-day averages. This reveals:

        - **Is buying pressure INCREASING or just high?**
        - **Are institutions ACCELERATING their accumulation?**
        - **Is smart money positioning BEFORE the big price move?**

        This unique calculation provides an **UNFAIR ADVANTAGE** by identifying early shifts in institutional behavior.

        #### 📊 The 4-Layer EDGE System:

        Each stock is scored across four critical dimensions to generate a comprehensive EDGE score:

        1.  **Volume Acceleration (40% Weight)**
            -   **Your UNFAIR ADVANTAGE**: This is the core of the EDGE Protocol. It measures if buying/selling volume is *accelerating* relative to longer-term averages.
            -   **Positive acceleration**: Strong indicator of smart money accumulation.
            -   **Extreme acceleration (>30%)**: Often precedes significant price moves.

        2.  **Momentum Divergence (25% Weight)**
            -   Compares **short-term momentum** (1-7 days) against **long-term momentum** (30 days - 3 months).
            -   **Positive divergence**: Stock is beginning to accelerate.
            -   **Stealth accumulation**: Price is flat or down (negative momentum divergence), but volume acceleration is high (institutions are buying quietly). This is a highly valuable pattern.

        3.  **Risk/Reward Analysis (20% Weight)**
            -   A mathematical assessment of potential upside versus defined risk.
            -   Considers **upside potential** (distance to 52-week high) and **recent volatility**.
            -   Prioritizes trades with a **3:1 or better Risk/Reward ratio** for favorable outcomes.

        4.  **Fundamental Quality (15% Weight)**
            -   Assesses core company health based on available data.
            -   Factors include **EPS (Earnings Per Share) growth momentum** and **reasonable P/E (Price-to-Earnings) valuation**.
            -   **Adaptive Weighting**: If fundamental data is unavailable or incomplete, its weight is intelligently redistributed among the other technical factors to ensure a robust score.

        ---
        #### 🎯 Position Sizing Guidelines:

        Based on the final **EDGE Score**, the system provides suggested capital allocation:

        -   **EXPLOSIVE EDGE (85-100)**: Allocate **10%** of your trading capital. Represents maximum conviction signals.
        -   **STRONG EDGE (70-85)**: Allocate **5%** of your trading capital. High-conviction opportunities.
        -   **MODERATE EDGE (50-70)**: Allocate **2%** of your trading capital. Good quality opportunities for smaller positions.
        -   **WATCH (30-50)**: Monitor these stocks. They show some potential but lack the full EDGE criteria for immediate action.
        -   **NO EDGE (<30)**: Avoid or ignore these stocks for now; they do not meet the system's criteria.

        ---
        #### ⚡ Daily Workflow:

        1.  **Morning Check**: Begin by reviewing the **"Explosive Opportunities"** tab for the highest conviction trades.
        2.  **Entry Strategy**: Use the suggested current price for entry and the calculated position sizes based on EDGE score.
        3.  **Risk Management**: Always set **stop-loss orders** at the calculated levels to protect your capital.
        4.  **Profit Taking**: Utilize the **two-tier target system** for structured profit realization.
        5.  **Ongoing Monitoring**: Keep an eye on your open positions for any decay in their EDGE score or approach to stop levels.

        #### 🏆 Why This System Works:

        -   **Data Advantage**: Leverages unique volume analysis that most traders overlook.
        -   **Early Detection**: Designed to identify significant moves *before* they become widely obvious.
        -   **Integrated Risk Control**: Every signal comes with predefined risk and reward parameters.
        -   **Market Adaptability**: The system is robust and designed to perform across various market conditions.
        -   **Pattern-Based**: Built on insights derived from observed institutional trading behaviors.
        """)

        # Add data validation section
        st.markdown("---")
        st.markdown("### 🔍 Data & System Health Check")

        col_health1, col_health2, col_health3 = st.columns(3)

        with col_health1:
            st.markdown("**Core Data Coverage:**")
            critical_cols_display = {
                'vol_ratio_30d_90d': 'Volume Ratio 30d/90d',
                'vol_ratio_30d_180d': 'Volume Ratio 30d/180d',
                'price': 'Current Price',
                'ret_7d': '7-Day Return',
                'from_high_pct': 'Distance from High',
                'edge_score': 'Calculated EDGE Score'
            }

            for col, name in critical_cols_display.items():
                if col in df.columns:
                    non_null_count = df[col].notna().sum()
                    pct_coverage = (non_null_count / len(df) * 100) if len(df) > 0 else 0
                    if pct_coverage > 90:
                        st.success(f"✅ {name}: **{pct_coverage:.0f}%** covered")
                    elif pct_coverage > 70:
                        st.warning(f"⚠️ {name}: **{pct_coverage:.0f}%** covered (fair)")
                    else:
                        st.error(f"❌ {name}: **{pct_coverage:.0f}%** covered (low)")
                else:
                    st.error(f"❌ {name}: **Missing Column**")

        with col_health2:
            st.markdown("**Overall Data Quality Metrics:**")
            st.write(f"**Total Stocks Analyzed**: {len(df)}")
            if 'edge_score' in df.columns and not df['edge_score'].empty:
                stocks_with_edge = (df['edge_score'] > 0).sum()
                st.write(f"**Stocks with any EDGE (>0)**: {stocks_with_edge}")
                if stocks_with_edge > 0:
                    st.write(f"**Average EDGE Score**: {df['edge_score'].mean():.1f}")
            if 'volume_acceleration' in df.columns and not df['volume_acceleration'].empty:
                positive_vol_accel = (df['volume_acceleration'] > 0).sum()
                st.write(f"**Positive Vol Acceleration Stocks**: {positive_vol_accel}")
            st.write(f"**Last Data Update**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}")

        with col_health3:
            st.markdown("**Signal Generation Health:**")
            total_signals_generated = len(explosive_stocks) + len(strong_stocks) + len(moderate_stocks)
            st.write(f"**Total Actionable Signals**: {total_signals_generated}")
            if total_signals_generated > 20:
                st.success("✅ System generating healthy number of signals.")
            elif total_signals_generated > 5:
                st.warning("⚠️ Limited signals today. Market might be quiet.")
            else:
                st.error("❌ Very few signals. Consider checking data source validity.")
            st.write("---")
            st.markdown("**Next Data Cache Refresh:**")
            st.info("Approximately 5 minutes")

    # --- Download Section ---
    st.markdown("---")
    st.markdown("### 💾 Export EDGE Data")

    col_download1, col_download2, col_download3 = st.columns(3)

    with col_download1:
        if len(explosive_stocks) > 0:
            csv_explosive = explosive_stocks.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="🔥 Download Explosive EDGE (.csv)",
                data=csv_explosive,
                file_name=f"explosive_edge_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                type="primary"
            )

    with col_download2:
        if len(strong_stocks) > 0:
            csv_strong = strong_stocks.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="💎 Download Strong EDGE (.csv)",
                data=csv_strong,
                file_name=f"strong_edge_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )

    with col_download3:
        # Export all stocks with an edge score >= WATCH threshold
        edge_stocks_to_export = df[df['edge_score'] >= EDGE_THRESHOLDS['WATCH']].sort_values('edge_score', ascending=False)
        if len(edge_stocks_to_export) > 0:
            csv_full_analysis = edge_stocks_to_export.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📊 Download Full EDGE Analysis (.csv)",
                data=csv_full_analysis,
                file_name=f"edge_analysis_full_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No stocks with a 'WATCH' or higher EDGE score to export.")

    # --- Footer with Disclaimer ---
    st.markdown("---")
    st.caption("""
    **Disclaimer:** **EDGE Protocol** - Finding What Others Can't See.
    This tool is for informational and educational purposes only. It is not financial advice.
    Your SECRET WEAPON: Volume Acceleration reveals institutional behavior before price moves.

    Position sizes are suggestions based on edge strength and are not guarantees.
    Always conduct your own due diligence and consult with a financial professional before making any investment decisions.
    Never risk more capital than you can afford to lose. Past performance and identified patterns do not guarantee future results.

    Version: 1.0 FINAL (Enhanced) | Data updates every 5 minutes (cached).
    """)

# Entry point for the Streamlit application
if __name__ == "__main__":
    main()
```
