To make the `edge_protocol.py` code a complete, ready, bug-free, and error-free ultimate trading edge system, I'll focus on ensuring robust data handling, refining calculations to prevent edge cases, and enhancing the overall user experience within Streamlit.

Here's the refined code:

```python
# edge_protocol.py - THE ULTIMATE TRADING EDGE SYSTEM
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
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="EDGE Protocol",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Data source
SHEET_ID = "1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk"
GID = "2026492216"
SHEET_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID}"

# EDGE thresholds
EDGE_THRESHOLDS = {
    'EXPLOSIVE': 85,     # Top 1% - Bet 10%
    'STRONG': 70,        # Top 5% - Bet 5%
    'MODERATE': 50,      # Top 10% - Bet 2%
    'WATCH': 30          # Monitor
}

# ============================================================================
# DATA LOADING & PREPARATION
# ============================================================================

@st.cache_data(ttl=300) # Cache data for 5 minutes
def load_data():
    """
    Load and prepare data from the Google Sheet, performing all necessary
    type conversions and initial cleaning.
    """
    try:
        response = requests.get(SHEET_URL, timeout=30)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        df = pd.read_csv(io.StringIO(response.text))

        # Clean column names by stripping whitespace
        df.columns = [col.strip() for col in df.columns]

        # Define columns and their cleaning/conversion logic
        conversions = {
            'price': lambda x: pd.to_numeric(x.astype(str).str.replace('₹', '').str.replace(',', '').str.strip(), errors='coerce'),
            'low_52w': lambda x: pd.to_numeric(x.astype(str).str.replace('₹', '').str.replace(',', '').str.strip(), errors='coerce'),
            'high_52w': lambda x: pd.to_numeric(x.astype(str).str.replace('₹', '').str.replace(',', '').str.strip(), errors='coerce'),
            'sma_20d': lambda x: pd.to_numeric(x.astype(str).str.replace('₹', '').str.replace(',', '').str.strip(), errors='coerce'),
            'sma_50d': lambda x: pd.to_numeric(x.astype(str).str.replace('₹', '').str.replace(',', '').str.strip(), errors='coerce'),
            'sma_200d': lambda x: pd.to_numeric(x.astype(str).str.replace('₹', '').str.replace(',', '').str.strip(), errors='coerce'),
            'prev_close': lambda x: pd.to_numeric(x.astype(str).str.replace('₹', '').str.replace(',', '').str.strip(), errors='coerce'),

            'ret_1d': lambda x: pd.to_numeric(x, errors='coerce'),
            'ret_3d': lambda x: pd.to_numeric(x, errors='coerce'),
            'ret_7d': lambda x: pd.to_numeric(x, errors='coerce'),
            'ret_30d': lambda x: pd.to_numeric(x, errors='coerce'),
            'ret_3m': lambda x: pd.to_numeric(x, errors='coerce'),
            'ret_6m': lambda x: pd.to_numeric(x, errors='coerce'),
            'ret_1y': lambda x: pd.to_numeric(x, errors='coerce'),
            'ret_3y': lambda x: pd.to_numeric(x, errors='coerce'),
            'ret_5y': lambda x: pd.to_numeric(x, errors='coerce'),

            'volume_1d': lambda x: pd.to_numeric(x.astype(str).str.replace('₹', '').str.replace(',', '').str.strip(), errors='coerce').fillna(0),
            'volume_7d': lambda x: pd.to_numeric(x.astype(str).str.replace('₹', '').str.replace(',', '').str.strip(), errors='coerce').fillna(0),
            'volume_30d': lambda x: pd.to_numeric(x.astype(str).str.replace('₹', '').str.replace(',', '').str.strip(), errors='coerce').fillna(0),
            'volume_3m': lambda x: pd.to_numeric(x.astype(str).str.replace('₹', '').str.replace(',', '').str.strip(), errors='coerce').fillna(0),
            'volume_90d': lambda x: pd.to_numeric(x.astype(str).str.replace(',', '').str.strip(), errors='coerce').fillna(0),
            'volume_180d': lambda x: pd.to_numeric(x.astype(str).str.replace(',', '').str.strip(), errors='coerce').fillna(0),

            'vol_ratio_1d_90d': lambda x: pd.to_numeric(x.astype(str).str.replace('%', '').str.strip().replace(['', '-', 'NA', 'N/A', 'nan', 'NaN'], np.nan), errors='coerce').fillna(0),
            'vol_ratio_7d_90d': lambda x: pd.to_numeric(x.astype(str).str.replace('%', '').str.strip().replace(['', '-', 'NA', 'N/A', 'nan', 'NaN'], np.nan), errors='coerce').fillna(0),
            'vol_ratio_30d_90d': lambda x: pd.to_numeric(x.astype(str).str.replace('%', '').str.strip().replace(['', '-', 'NA', 'N/A', 'nan', 'NaN'], np.nan), errors='coerce').fillna(0),
            'vol_ratio_1d_180d': lambda x: pd.to_numeric(x.astype(str).str.replace('%', '').str.strip().replace(['', '-', 'NA', 'N/A', 'nan', 'NaN'], np.nan), errors='coerce').fillna(0),
            'vol_ratio_7d_180d': lambda x: pd.to_numeric(x.astype(str).str.replace('%', '').str.strip().replace(['', '-', 'NA', 'N/A', 'nan', 'NaN'], np.nan), errors='coerce').fillna(0),
            'vol_ratio_30d_180d': lambda x: pd.to_numeric(x.astype(str).str.replace('%', '').str.strip().replace(['', '-', 'NA', 'N/A', 'nan', 'NaN'], np.nan), errors='coerce').fillna(0),

            'pe': lambda x: pd.to_numeric(x, errors='coerce'),
            'eps_current': lambda x: pd.to_numeric(x, errors='coerce'),
            'eps_last_qtr': lambda x: pd.to_numeric(x, errors='coerce'),
            'eps_change_pct': lambda x: pd.to_numeric(x, errors='coerce'),
            'eps_duplicate': lambda x: pd.to_numeric(x, errors='coerce'),

            'market_cap': lambda x: pd.to_numeric(x.astype(str).str.replace('₹', '').str.replace(',', '').str.replace(' Cr', '').str.strip(), errors='coerce'),
            'from_low_pct': lambda x: pd.to_numeric(x, errors='coerce'),
            'from_high_pct': lambda x: pd.to_numeric(x, errors='coerce'),
            'rvol': lambda x: pd.to_numeric(x, errors='coerce')
        }

        for col, func in conversions.items():
            if col in df.columns:
                df[col] = func(df[col])

        # Rename market_cap for consistency if 'market_cap' was the raw column
        if 'market_cap' in df.columns and 'market_cap_num' not in df.columns:
            df.rename(columns={'market_cap': 'market_cap_num'}, inplace=True)
            df['market_cap_clean'] = df['market_cap_num'] # Keep consistent with original intent

        # Filter out rows with invalid tickers or prices
        if 'ticker' in df.columns:
            df = df[df['ticker'].notna() & (df['ticker'] != '')]
        if 'price' in df.columns:
            df = df[df['price'] > 0]

        return df.reset_index(drop=True)

    except requests.exceptions.Timeout:
        st.error("Data loading timed out. Please check your internet connection or try again later.")
        return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to load data from Google Sheet: {str(e)}. Please ensure the sheet is published and accessible.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An unexpected error occurred during data loading and preparation: {str(e)}")
        return pd.DataFrame()

# ============================================================================
# EDGE CALCULATION ENGINE
# ============================================================================

def calculate_volume_acceleration(df):
    """
    Calculate volume acceleration based on 30-day volume ratios against
    90-day and 180-day averages.
    """
    df['volume_acceleration'] = np.nan # Initialize with NaN
    df['vol_accel_status'] = 'NO_DATA'
    df['vol_accel_percentile'] = np.nan

    # Ensure required columns exist and are numeric
    if 'vol_ratio_30d_90d' in df.columns and 'vol_ratio_30d_180d' in df.columns:
        # Fill NaNs with 0 for calculation purposes where it makes sense,
        # but ensure the final acceleration calculation handles missing values.
        df['vol_ratio_30d_90d_calc'] = df['vol_ratio_30d_90d'].fillna(0)
        df['vol_ratio_30d_180d_calc'] = df['vol_ratio_30d_180d'].fillna(0)

        # The core calculation: difference between two ratios
        # Only calculate if both underlying ratios are not NaN in the original data
        valid_ratios_mask = df['vol_ratio_30d_90d'].notna() & df['vol_ratio_30d_180d'].notna()
        df.loc[valid_ratios_mask, 'volume_acceleration'] = \
            df.loc[valid_ratios_mask, 'vol_ratio_30d_90d_calc'] - df.loc[valid_ratios_mask, 'vol_ratio_30d_180d_calc']

        # Classify acceleration only for valid calculations
        valid_accel_mask = df['volume_acceleration'].notna()
        if valid_accel_mask.sum() > 0:
            df.loc[valid_accel_mask, 'vol_accel_status'] = pd.cut(
                df.loc[valid_accel_mask, 'volume_acceleration'],
                bins=[-np.inf, -10, 0, 10, 20, 30, np.inf],
                labels=['EXODUS', 'DISTRIBUTION', 'NEUTRAL', 'ACCUMULATION', 'HEAVY_ACCUMULATION', 'INSTITUTIONAL_LOADING'],
                right=False # Ensure bins are inclusive on the left
            ).astype(str) # Convert to string to avoid CategoricalDtype issues later

            # Percentile rank for valid acceleration values
            df.loc[valid_accel_mask, 'vol_accel_percentile'] = df.loc[valid_accel_mask, 'volume_acceleration'].rank(pct=True, method='average') * 100
            df['vol_accel_percentile'].fillna(50, inplace=True) # Fill NaNs for percentile with 50 (neutral)

    elif 'vol_ratio_30d_90d' in df.columns:
        # Fallback if only 30d/90d ratio is available (less accurate acceleration)
        st.warning("Only 'vol_ratio_30d_90d' available. Volume acceleration calculation will be a proxy.")
        df['volume_acceleration'] = df['vol_ratio_30d_90d'].fillna(0)

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
        st.warning("Neither 'vol_ratio_30d_90d' nor 'vol_ratio_30d_180d' found. Volume acceleration cannot be calculated.")

    # Clean up temporary columns
    df.drop(columns=['vol_ratio_30d_90d_calc', 'vol_ratio_30d_180d_calc'], errors='ignore', inplace=True)
    return df

def calculate_momentum_divergence(df):
    """Detect momentum acceleration patterns."""
    df['short_momentum'] = 0.0
    df['long_momentum'] = 0.0
    df['momentum_divergence'] = 0.0
    df['divergence_pattern'] = 'NEUTRAL'

    # Short-term momentum (1-7 days)
    short_cols = ['ret_1d', 'ret_3d', 'ret_7d']
    available_short = [col for col in short_cols if col in df.columns]
    if available_short:
        df['short_momentum'] = df[available_short].mean(axis=1, skipna=True).fillna(0)

    # Long-term momentum (30d-3m)
    long_cols = ['ret_30d', 'ret_3m']
    available_long = [col for col in long_cols if col in df.columns]
    if available_long:
        df['long_momentum'] = df[available_long].mean(axis=1, skipna=True).fillna(0)

    # Divergence analysis
    df['momentum_divergence'] = df['short_momentum'] - df['long_momentum']

    if 'volume_acceleration' in df.columns:
        # Define masks for patterns, handling NaNs safely
        # Ensure volume_acceleration and momentum_divergence are treated as numeric for comparison
        vol_accel_safe = df['volume_acceleration'].fillna(0)
        mom_div_safe = df['momentum_divergence'].fillna(0)

        # Explosive breakout pattern: strong positive momentum divergence and positive volume acceleration
        mask_explosive = (mom_div_safe > 5) & (vol_accel_safe > 0)
        df.loc[mask_explosive, 'divergence_pattern'] = 'EXPLOSIVE_BREAKOUT'

        # Momentum building pattern: positive momentum divergence and significant volume acceleration
        mask_building = (mom_div_safe > 0) & (vol_accel_safe > 10)
        df.loc[mask_building, 'divergence_pattern'] = 'MOMENTUM_BUILDING'

        # Stealth accumulation pattern: negative momentum divergence but strong volume acceleration (smart money buying into weakness)
        mask_stealth = (mom_div_safe < 0) & (vol_accel_safe > 20)
        df.loc[mask_stealth, 'divergence_pattern'] = 'STEALTH_ACCUMULATION'

    return df

def calculate_risk_reward(df):
    """Calculate mathematical edge in risk/reward."""
    df['upside_potential'] = 0.0
    df['recent_volatility'] = 0.0
    df['risk_reward_ratio'] = 0.0
    df['support_distance'] = 0.0

    if all(col in df.columns for col in ['price', 'high_52w', 'low_52w']):
        # Ensure 'price' is not zero to avoid division by zero
        valid_prices_mask = (df['price'] > 0)

        # Upside potential: percentage distance from current price to 52-week high
        df.loc[valid_prices_mask, 'upside_potential'] = \
            ((df['high_52w'] - df['price']) / df['price'] * 100).clip(0, 200).fillna(0)

        # Recent volatility (simplified): based on 52-week range, divided by 4 for a quarterly proxy
        df.loc[valid_prices_mask, 'recent_volatility'] = \
            ((df['high_52w'] - df['low_52w']) / df['price'] * 100 / 4).clip(1, 50).fillna(1) # Min volatility 1%

        # Risk/Reward ratio: Upside potential vs. (2 * recent volatility) - simplified risk proxy
        # Avoid division by zero for recent_volatility
        valid_rr_mask = valid_prices_mask & (df['recent_volatility'] > 0)
        df.loc[valid_rr_mask, 'risk_reward_ratio'] = \
            (df['upside_potential'] / (2 * df['recent_volatility'])).clip(0, 10).fillna(0)

        # Support level (simplified): percentage distance from current price to 52-week low
        df.loc[valid_prices_mask, 'support_distance'] = \
            ((df['price'] - df['low_52w']) / df['price'] * 100).clip(0, 100).fillna(0)

    return df

def calculate_time_arbitrage(df):
    """Find quality stocks in temporary weakness."""
    df['long_term_annual'] = 0.0
    df['time_arbitrage_opportunity'] = False
    df['quality_selloff'] = False

    if all(col in df.columns for col in ['ret_1y', 'ret_3y', 'ret_30d', 'from_high_pct']):
        # Long-term winner taking a break: Good long-term returns but recent pullback
        df['long_term_annual'] = (df['ret_3y'] / 3).fillna(0) # Annualize 3-year return
        df['time_arbitrage_opportunity'] = (
            (df['ret_1y'] > df['long_term_annual']) &
            (df['ret_30d'] < 5) &
            (df['ret_30d'] > -10)
        ).fillna(False) # Handle NaNs in boolean mask

        # Quality in selloff: Significant 3-year returns but currently far from high
        df['quality_selloff'] = (
            (df['ret_1y'] < 0) & # Negative 1-year return
            (df['ret_3y'] > 100) & # Strong 3-year return
            (df['from_high_pct'] < -30) # Significant drawdown from 52-week high
        ).fillna(False)

    return df

def calculate_edge_scores(df):
    """Master EDGE score calculation."""
    df['edge_score'] = 0.0 # Initialize as float
    df['vol_accel_score'] = 0.0
    df['momentum_score'] = 0.0
    df['rr_score'] = 0.0
    df['fundamental_score'] = 0.0

    # 1. Volume Acceleration Score (40% weight)
    if 'volume_acceleration' in df.columns:
        vol_accel = df['volume_acceleration'].fillna(0) # Treat NaN as 0 for scoring
        df.loc[vol_accel > 0, 'vol_accel_score'] = 25
        df.loc[vol_accel > 10, 'vol_accel_score'] = 50
        df.loc[vol_accel > 20, 'vol_accel_score'] = 75
        df.loc[vol_accel > 30, 'vol_accel_score'] = 100
        df.loc[vol_accel < -20, 'vol_accel_score'] = 0 # Significant negative acceleration reduces score

        df['edge_score'] += df['vol_accel_score'] * 0.4

    # 2. Momentum Divergence Score (25% weight)
    if 'momentum_divergence' in df.columns and 'volume_acceleration' in df.columns and 'short_momentum' in df.columns:
        mom_div_safe = df['momentum_divergence'].fillna(0)
        vol_accel_safe = df['volume_acceleration'].fillna(0)
        short_mom_safe = df['short_momentum'].fillna(0)

        # Positive divergence with positive volume acceleration
        mask1 = (mom_div_safe > 0) & (vol_accel_safe > 0)
        df.loc[mask1, 'momentum_score'] = 60

        # Strong acceleration in short-term momentum
        mask2 = (mom_div_safe > 5) & (short_mom_safe > 0)
        df.loc[mask2, 'momentum_score'] = 80

        # Hidden accumulation (negative momentum divergence but strong volume acceleration)
        mask3 = (mom_div_safe < 0) & (vol_accel_safe > 20)
        df.loc[mask3, 'momentum_score'] = 100

        df['edge_score'] += df['momentum_score'] * 0.25

    # 3. Risk/Reward Score (20% weight)
    if 'risk_reward_ratio' in df.columns:
        # Clip score to 0-100, treating NaN as 0
        df['rr_score'] = (df['risk_reward_ratio'].fillna(0) * 20).clip(0, 100)
        df['edge_score'] += df['rr_score'] * 0.2

    # 4. Fundamental Score (15% weight)
    fundamental_score_component = pd.Series(0.0, index=df.index)
    fundamental_factors_count = 0

    if 'eps_change_pct' in df.columns:
        eps_data = df['eps_change_pct'].fillna(0) # Treat missing EPS as 0 change
        eps_score_temp = pd.Series(0.0, index=df.index)
        eps_score_temp[eps_data > 0] = 30
        eps_score_temp[eps_data > 15] = 60
        eps_score_temp[eps_data > 30] = 100
        fundamental_score_component += eps_score_temp
        fundamental_factors_count += 1

    if 'pe' in df.columns:
        pe_data = df['pe'].fillna(50) # Assume a neutral/high PE if missing
        pe_score_temp = pd.Series(0.0, index=df.index)
        pe_score_temp[(pe_data > 5) & (pe_data < 40)] = 50 # Reasonable PE range
        pe_score_temp[(pe_data > 10) & (pe_data < 25)] = 100 # Optimal PE range
        fundamental_score_component += pe_score_temp
        fundamental_factors_count += 1

    if fundamental_factors_count > 0:
        df['fundamental_score'] = fundamental_score_component / fundamental_factors_count
        df['edge_score'] += df['fundamental_score'] * 0.15
    else:
        # Redistribute weight if no fundamental data is available
        # Scale up existing edge score components by dividing by the sum of their weights (0.4 + 0.25 + 0.2 = 0.85)
        # This prevents edge_score from being artificially low if fundamentals are missing
        if df['edge_score'].sum() > 0: # Only if there's some base score
            df['edge_score'] = df['edge_score'] / 0.85
            df['edge_score'] = df['edge_score'].clip(0, 100) # Ensure it doesn't exceed 100

    # Bonus multipliers for trend alignment
    if all(col in df.columns for col in ['price', 'sma_50d', 'sma_200d']):
        price_data = df['price'].fillna(0)
        sma50_data = df['sma_50d'].fillna(price_data) # If SMA is missing, assume it's at price
        sma200_data = df['sma_200d'].fillna(price_data)

        # 5 point bonus if price is above both 50d and 200d SMAs
        trend_bonus = ((price_data > sma50_data) & (price_data > sma200_data)).astype(int) * 5
        df['edge_score'] = (df['edge_score'] + trend_bonus).clip(0, 100)

    # Additional bonus for stocks with room to run (not overextended)
    if 'from_high_pct' in df.columns:
        room_bonus = pd.Series(0.0, index=df.index)
        from_high = df['from_high_pct'].fillna(0) # Treat NaN as 0 (no drawdown)

        # If stock is down 15-40% from its high, give 5 points
        room_bonus[(from_high < -15) & (from_high > -40)] = 5
        # If stock is down 20-35% from its high, give 10 points (sweet spot)
        room_bonus[(from_high < -20) & (from_high > -35)] = 10
        df['edge_score'] = (df['edge_score'] + room_bonus).clip(0, 100)

    # Final NaN handling for edge_score
    df['edge_score'] = df['edge_score'].fillna(0)

    # Final classification
    df['edge_category'] = pd.cut(
        df['edge_score'],
        bins=[-0.1, 30, 50, 70, 85, 100.1],
        labels=['NO_EDGE', 'WATCH', 'MODERATE', 'STRONG', 'EXPLOSIVE'],
        right=False # Bins are inclusive on the left, exclusive on the right, except for the last bin
    ).astype(str) # Convert to string to avoid CategoricalDtype issues

    return df

def calculate_position_metrics(df):
    """Calculate position sizing and risk management."""
    # Initialize columns
    df['suggested_position_pct'] = 0.0
    df['stop_loss'] = np.nan
    df['stop_loss_pct'] = np.nan
    df['target_1'] = np.nan
    df['target_2'] = np.nan
    df['target_1_pct'] = np.nan
    df['target_2_pct'] = np.nan

    # Position size based on edge category
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
    if all(col in df.columns for col in ['price', 'low_52w', 'sma_50d']):
        # Ensure price is valid for calculations
        valid_price_mask = (df['price'] > 0)
        df.loc[valid_price_mask, 'stop_loss'] = np.maximum.reduce([
            df.loc[valid_price_mask, 'price'] * 0.93, # Max 7% loss from current price
            df.loc[valid_price_mask, 'sma_50d'] * 0.98.fillna(0), # 2% below 50-day SMA
            df.loc[valid_price_mask, 'low_52w'] * 1.02.fillna(0) # 2% above 52-week low
        ])
        # Ensure stop loss is not higher than price
        df.loc[df['stop_loss'] > df['price'], 'stop_loss'] = df['price'] * 0.93

        # Calculate stop loss percentage
        valid_sl_calc_mask = valid_price_mask & df['stop_loss'].notna()
        df.loc[valid_sl_calc_mask, 'stop_loss_pct'] = \
            ((df.loc[valid_sl_calc_mask, 'stop_loss'] - df.loc[valid_sl_calc_mask, 'price']) / df.loc[valid_sl_calc_mask, 'price'] * 100).round(2)
        df['stop_loss_pct'].fillna(0, inplace=True) # Fill NaNs for display consistency

    # Target calculation
    if 'upside_potential' in df.columns and 'price' in df.columns:
        valid_target_calc_mask = (df['price'] > 0) & df['upside_potential'].notna()
        df.loc[valid_target_calc_mask, 'target_1'] = \
            df.loc[valid_target_calc_mask, 'price'] * (1 + df.loc[valid_target_calc_mask, 'upside_potential'] * 0.25 / 100)
        df.loc[valid_target_calc_mask, 'target_2'] = \
            df.loc[valid_target_calc_mask, 'price'] * (1 + df.loc[valid_target_calc_mask, 'upside_potential'] * 0.5 / 100)

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
    """Visualize edge score distribution."""
    if 'edge_category' in df.columns:
        # Define a consistent order for categories for better visualization
        category_order = ['EXPLOSIVE', 'STRONG', 'MODERATE', 'WATCH', 'NO_EDGE']
        edge_counts = df['edge_category'].value_counts().reindex(category_order, fill_value=0)

        # Filter out categories with zero counts for cleaner display if desired, or keep all
        edge_counts = edge_counts[edge_counts > 0]

        if len(edge_counts) > 0:
            fig = go.Figure(data=[go.Bar(
                x=edge_counts.index,
                y=edge_counts.values,
                text=edge_counts.values,
                textposition='auto',
                marker_color=[
                    '#f5576c',  # Explosive (red-pink)
                    '#ffaa00',  # Strong (orange)
                    '#ffdd44',  # Moderate (yellow)
                    '#888888',  # Watch (gray)
                    '#cccccc'   # No_Edge (light gray)
                ][:len(edge_counts)] # Ensure colors match the number of categories present
            )])

            fig.update_layout(
                title="EDGE Distribution Across Market",
                xaxis_title="EDGE Category",
                yaxis_title="Number of Stocks",
                height=400,
                template="plotly_white"
            )
            return fig
    # Fallback chart if no categories or edge scores
    if 'edge_score' in df.columns and not df['edge_score'].empty:
        fig = go.Figure(data=[go.Histogram(
            x=df['edge_score'],
            nbinsx=20,
            marker_color='lightblue',
            hovertemplate='Score: %{x}<br>Count: %{y}<extra></extra>'
        )])

        fig.update_layout(
            title="EDGE Score Distribution",
            xaxis_title="EDGE Score",
            yaxis_title="Number of Stocks",
            height=400,
            template="plotly_white"
        )
        return fig

    # Return None if no data to plot
    return None

def create_volume_acceleration_scatter(df):
    """The SECRET WEAPON visualization."""
    # Filter for stocks with valid data for the plot, and limit to top stocks for clarity
    valid_df = df[
        (df['edge_score'].notna()) & (df['edge_score'] > 0) &
        df['volume_acceleration'].notna() &
        df['short_momentum'].notna()
    ].nlargest(200, 'edge_score') # Increased to 200 for more visibility

    if len(valid_df) < 5: # Need a minimum number of points for a meaningful scatter
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5,
            text="Insufficient data for volume acceleration map. Need more qualified stocks.",
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
    colors = {
        'EXPLOSIVE': '#f5576c',
        'STRONG': '#ffaa00',
        'MODERATE': '#ffdd44',
        'WATCH': '#888888',
        'NO_EDGE': '#cccccc'
    }

    # Plot each category separately to control legend and colors
    for category in sorted(valid_df['edge_category'].unique(), key=lambda x: list(colors.keys()).index(x) if x in colors else 99):
        cat_stocks = valid_df[valid_df['edge_category'] == category]
        if not cat_stocks.empty:
            fig.add_trace(go.Scatter(
                x=cat_stocks['volume_acceleration'],
                y=cat_stocks['short_momentum'],
                mode='markers+text',
                name=category,
                text=cat_stocks['ticker'],
                textposition="top center",
                textfont=dict(size=8),
                marker=dict(
                    size=(cat_stocks['edge_score'] / 10) + 5, # Scale marker size by edge score, with a min size
                    color=colors.get(category, '#cccccc'), # Use get with default for safety
                    line=dict(width=1, color='black')
                ),
                hovertemplate='<b>%{text}</b><br>Vol Accel: %{x:.1f}%<br>Momentum: %{y:.1f}%<br>Edge Score: %{customdata:.1f}<extra></extra>',
                customdata=cat_stocks['edge_score']
            ))

    # Add quadrant lines
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, annotation_text="Neutral Momentum", annotation_position="bottom right")
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5, annotation_text="Neutral Volume Accel", annotation_position="top left")

    # Add quadrant labels (positioned dynamically or fixed)
    fig.add_annotation(x=valid_df['volume_acceleration'].max() * 0.7, y=valid_df['short_momentum'].max() * 0.7,
                       text="🔥 EXPLOSIVE ZONE", showarrow=False, font=dict(size=14, color="red"))
    fig.add_annotation(x=valid_df['volume_acceleration'].max() * 0.7, y=valid_df['short_momentum'].min() * 0.7 if valid_df['short_momentum'].min() < 0 else -10,
                       text="🏦 STEALTH ACCUMULATION", showarrow=False, font=dict(size=14, color="green"))
    fig.add_annotation(x=valid_df['volume_acceleration'].min() * 0.7 if valid_df['volume_acceleration'].min() < 0 else -20, y=valid_df['short_momentum'].max() * 0.7,
                       text="⚠️ PROFIT TAKING", showarrow=False, font=dict(size=14, color="orange"))
    fig.add_annotation(x=valid_df['volume_acceleration'].min() * 0.7 if valid_df['volume_acceleration'].min() < 0 else -20, y=valid_df['short_momentum'].min() * 0.7 if valid_df['short_momentum'].min() < 0 else -10,
                       text="💀 AVOID", showarrow=False, font=dict(size=14, color="gray"))

    fig.update_layout(
        title="Volume Acceleration Map - Your SECRET EDGE",
        xaxis_title="Volume Acceleration (30d/90d vs 30d/180d)",
        yaxis_title="Short-term Momentum (%)",
        height=600,
        showlegend=True,
        hovermode="closest",
        template="plotly_white"
    )

    return fig

def create_edge_radar(stock_data):
    """Create radar chart for individual stock edge components."""
    # Ensure all required keys are present with default values if missing
    # Default values chosen to be neutral or low for plotting purposes
    categories = ['Volume\nAcceleration', 'Momentum\nDivergence', 'Risk/Reward',
                  'Fundamental\nStrength', 'Trend\nAlignment']

    values = [
        stock_data.get('vol_accel_score', 0),
        stock_data.get('momentum_score', 0),
        stock_data.get('rr_score', 0),
        stock_data.get('fundamental_score', 0),
        # Trend alignment: give 100 if price > SMA200d, else 0
        100 if stock_data.get('price', 0) > stock_data.get('sma_200d', -1) else 0 # -1 to ensure check passes if SMA is 0
    ]

    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='EDGE Components',
        marker_color='#4CAF50', # A pleasant green color
        line_color='#2E8B57'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor='lightgray',
                linecolor='gray'
            ),
            angularaxis=dict(
                linecolor='gray'
            )
        ),
        showlegend=False,
        title=f"<b>{stock_data.get('ticker', 'Stock')}</b> - EDGE Analysis",
        height=400,
        margin=dict(l=50, r=50, t=80, b=50), # Adjust margins for better fit
        template="plotly_white"
    )

    return fig

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Custom CSS for a more vibrant look
    st.markdown("""
    <style>
    .main-header {
        font-size: 3.5em;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
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
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin: 10px 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .explosive-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); /* Pink to Red */
    }
    .stMetric {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
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
        overflow: hidden;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<h1 class="main-header">⚡ EDGE Protocol</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Finding What Others Can\'t See</p>', unsafe_allow_html=True)

    # Load and process data
    with st.spinner("Calculating EDGE across stocks... This might take a moment."):
        df = load_data()

        if df.empty:
            st.error("Failed to load or process data. Please ensure the Google Sheet is correct and accessible.")
            st.stop() # Stop execution if data is not loaded

        # Debug info in sidebar
        with st.sidebar:
            st.markdown("### ⚙️ Debug & Data Insights")
            st.write(f"Total rows loaded: **{len(df)}**")
            st.write(f"Total columns: **{len(df.columns)}**")

            # Check critical columns availability
            critical_cols_check = ['vol_ratio_30d_90d', 'vol_ratio_30d_180d', 'price', 'ticker', 'high_52w', 'low_52w']
            missing_critical = [col for col in critical_cols_check if col not in df.columns]
            if missing_critical:
                st.error(f"🚨 Missing critical columns for full functionality: {', '.join(missing_critical)}")
            else:
                st.success("✅ All critical columns present.")

            # Option to show data sample
            if st.checkbox("Show Raw Data Sample"):
                st.write("First 5 rows with key columns:")
                sample_cols_display = ['ticker', 'price', 'vol_ratio_30d_90d', 'vol_ratio_30d_180d', 'ret_7d', 'edge_score', 'volume_acceleration']
                available_sample_cols = [col for col in sample_cols_display if col in df.columns]
                if available_sample_cols:
                    st.dataframe(df[available_sample_cols].head())
                else:
                    st.info("No relevant sample columns available for display.")

            # Option to test specific stock calculations
            if st.checkbox("Test specific stock calculations (e.g., RELIANCE)"):
                test_ticker = st.text_input("Enter ticker to test:", value="RELIANCE").upper()
                if test_ticker and test_ticker in df['ticker'].values:
                    test_stock_data = df[df['ticker'] == test_ticker].iloc[0]
                    st.write(f"**Detailed Calculation for {test_ticker}:**")
                    st.json({
                        "vol_ratio_30d_90d": test_stock_data.get('vol_ratio_30d_90d', 'N/A'),
                        "vol_ratio_30d_180d": test_stock_data.get('vol_ratio_30d_180d', 'N/A'),
                        "volume_acceleration_calc": test_stock_data.get('volume_acceleration', 'N/A'),
                        "short_momentum": test_stock_data.get('short_momentum', 'N/A'),
                        "long_momentum": test_stock_data.get('long_momentum', 'N/A'),
                        "momentum_divergence": test_stock_data.get('momentum_divergence', 'N/A'),
                        "edge_score": test_stock_data.get('edge_score', 'N/A'),
                        "edge_category": test_stock_data.get('edge_category', 'N/A')
                    })
                else:
                    st.warning(f"Ticker '{test_ticker}' not found in data or invalid input.")

        # Perform all calculations
        df = calculate_volume_acceleration(df)
        df = calculate_momentum_divergence(df)
        df = calculate_risk_reward(df)
        df = calculate_time_arbitrage(df)
        df = calculate_edge_scores(df)
        df = calculate_position_metrics(df)

        # Fallback scoring if edge_score is somehow all zero or missing after calculations
        if 'edge_score' not in df.columns or df['edge_score'].isnull().all() or df['edge_score'].sum() == 0:
            st.warning("⚠️ Edge scoring failed or resulted in all zeros. Applying simplified scoring.")
            df['edge_score'] = 0.0 # Reset to float for calculations

            # Simple momentum score fallback
            if 'ret_7d' in df.columns and 'ret_30d' in df.columns:
                df['simple_momentum'] = (
                    (df['ret_7d'].fillna(0) > 3).astype(int) * 20 +
                    (df['ret_30d'].fillna(0) > 5).astype(int) * 20 +
                    (df['ret_7d'].fillna(0) > df['ret_30d'].fillna(0)/4.3).astype(int) * 20
                )
                df['edge_score'] += df['simple_momentum']

            # Simple value score fallback
            if 'from_high_pct' in df.columns:
                df['simple_value'] = pd.Series(0.0, index=df.index)
                df.loc[(df['from_high_pct'].fillna(0) < -20) & (df['from_high_pct'].fillna(0) > -40), 'simple_value'] = 20
                df['edge_score'] += df['simple_value']

            # Simple trend score fallback
            if all(col in df.columns for col in ['price', 'sma_50d', 'sma_200d']):
                df['simple_trend'] = (
                    (df['price'].fillna(0) > df['sma_50d'].fillna(0)).astype(int) * 10 +
                    (df['price'].fillna(0) > df['sma_200d'].fillna(0)).astype(int) * 10
                )
                df['edge_score'] += df['simple_trend']

            # Re-classify with simplified scoring
            df['edge_category'] = pd.cut(
                df['edge_score'],
                bins=[-0.1, 20, 40, 60, 80, 100.1],
                labels=['NO_EDGE', 'WATCH', 'MODERATE', 'STRONG', 'EXPLOSIVE'],
                right=False
            ).astype(str) # Convert to string to avoid CategoricalDtype issues

            st.info("Using simplified scoring due to potential data incompleteness for advanced calculations. Results may vary.")

        # More debug info in sidebar
        with st.sidebar:
            st.markdown("---")
            st.markdown("### 📊 Calculated Metrics Summary")
            if 'volume_acceleration' in df.columns:
                vol_accel_stats = df['volume_acceleration'].describe(percentiles=[]).round(2)
                st.write("\n**Volume Acceleration Stats:**")
                st.write(f"Mean: {vol_accel_stats.loc['mean']:.2f}%")
                st.write(f"Max: {vol_accel_stats.loc['max']:.2f}%")
                st.write(f"Min: {vol_accel_stats.loc['min']:.2f}%")
                st.write(f"Valid values: {df['volume_acceleration'].notna().sum()}")

            if 'edge_score' in df.columns:
                edge_stats = df['edge_score'].describe(percentiles=[]).round(2)
                st.write("\n**EDGE Score Stats:**")
                st.write(f"Mean: {edge_stats.loc['mean']:.2f}")
                st.write(f"Max: {edge_stats.loc['max']:.2f}")
                st.write(f"Count > 50: {(df['edge_score'] > 50).sum()}")
                st.write(f"Count > 70: {(df['edge_score'] > 70).sum()}")
                st.write(f"Count > 85: {(df['edge_score'] > 85).sum()}")

    # Filter for high edge stocks (ensure edge_category is string)
    df['edge_category'] = df['edge_category'].astype(str)

    explosive_stocks = df[df['edge_category'] == 'EXPLOSIVE'].sort_values('edge_score', ascending=False)
    strong_stocks = df[df['edge_category'] == 'STRONG'].sort_values('edge_score', ascending=False)
    moderate_stocks = df[df['edge_category'] == 'MODERATE'].sort_values('edge_score', ascending=False)

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("🔥 EXPLOSIVE EDGE", len(explosive_stocks),
                 help="Top 1% of opportunities based on EDGE score. Suggested position: 10%.")

    with col2:
        st.metric("💎 STRONG EDGE", len(strong_stocks),
                 help="Top 5% of opportunities based on EDGE score. Suggested position: 5%.")

    with col3:
        st.metric("📈 MODERATE EDGE", len(moderate_stocks),
                 help="Top 10% of opportunities based on EDGE score. Suggested position: 2%.")

    with col4:
        if 'volume_acceleration' in df.columns and not df['volume_acceleration'].empty:
            # Calculate average volume acceleration only for stocks with an edge score > 70
            high_edge_df_for_avg = df[df['edge_score'] > 70].copy()
            if not high_edge_df_for_avg['volume_acceleration'].empty:
                avg_vol_accel = high_edge_df_for_avg['volume_acceleration'].mean()
                st.metric("🔍 Avg Vol Accel (High Edge)", f"{avg_vol_accel:.1f}%",
                         help="Average Volume Acceleration for stocks with EDGE score > 70. This is your SECRET WEAPON metric.")
            else:
                st.metric("🔍 Avg Vol Accel (High Edge)", "N/A", help="No high EDGE stocks to calculate average volume acceleration.")
        else:
            st.metric("🔍 Vol Accel", "N/A", help="Volume acceleration data not available for calculation.")

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔥 Explosive Opportunities",
        "📊 EDGE Analysis",
        "📈 Market Map",
        "📚 How It Works"
    ])

    with tab1:
        st.markdown("### 🔥 Today's EXPLOSIVE EDGE Opportunities")

        if len(explosive_stocks) > 0:
            top_pick = explosive_stocks.iloc[0]

            # Safely get values with defaults
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

            # Detailed analysis
            col1_detail, col2_detail = st.columns([2, 1])

            with col1_detail:
                st.markdown("#### 📊 Why This Has EXPLOSIVE EDGE:")

                # Volume intelligence
                st.success(f"""
                **🔍 Volume Intelligence (YOUR SECRET WEAPON):**
                - **Acceleration**: **{top_pick.get('volume_acceleration', 0):.1f}%** ({top_pick.get('vol_accel_status', 'Unknown')})
                - **30d vs 90d Avg**: {top_pick.get('vol_ratio_30d_90d', 0):.1f}%
                - **30d vs 180d Avg**: {top_pick.get('vol_ratio_30d_180d', 0):.1f}%
                - **Interpretation**: This indicates institutions are **AGGRESSIVELY accumulating** this stock, a strong pre-cursor to significant price moves.
                """)

                # Momentum analysis
                st.info(f"""
                **📈 Momentum Analysis:**
                - **Short-term Momentum (avg 1-7d)**: {top_pick.get('short_momentum', 0):.1f}%
                - **Long-term Momentum (avg 30d-3m)**: {top_pick.get('long_momentum', 0):.1f}%
                - **Divergence Pattern**: {top_pick.get('divergence_pattern', 'Analyzing...')}
                - **Interpretation**: Identifies if momentum is building, breaking out, or if there's **stealth accumulation** (price stagnant, but smart money buying).
                """)

                # Risk/Reward
                st.warning(f"""
                **🎯 Risk/Reward Setup:**
                - **Current Price**: ₹{top_pick.get('price', 0):.2f}
                - **Suggested Stop Loss**: ₹{top_pick.get('stop_loss', 0):.2f} ({top_pick.get('stop_loss_pct', 0):.1f}%)
                - **Target 1**: ₹{top_pick.get('target_1', 0):.2f} (+{top_pick.get('target_1_pct', 0):.1f}%)
                - **Target 2**: ₹{top_pick.get('target_2', 0):.2f} (+{top_pick.get('target_2_pct', 0):.1f}%)
                - **Calculated R:R Ratio**: 1:{top_pick.get('risk_reward_ratio', 0):.1f}
                - **Interpretation**: Ensures trades have a favorable risk-to-reward profile before entry.
                """)

            with col2_detail:
                # Radar chart
                fig_radar = create_edge_radar(top_pick)
                st.plotly_chart(fig_radar, use_container_width=True)

            st.markdown("### 🔥 All EXPLOSIVE EDGE Stocks")

            display_cols_explosive = [
                'ticker', 'company_name', 'edge_score', 'volume_acceleration',
                'short_momentum', 'risk_reward_ratio', 'price',
                'suggested_position_pct', 'stop_loss', 'target_1', 'target_2'
            ]

            available_cols_explosive = [col for col in display_cols_explosive if col in explosive_stocks.columns]

            if len(explosive_stocks) > 0 and len(available_cols_explosive) >= 3:
                display_df_explosive = explosive_stocks[available_cols_explosive].head(20)

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

                st.dataframe(
                    display_df_explosive.style.format(format_dict_explosive).background_gradient(
                        subset=['edge_score'], cmap='Reds'
                    ),
                    use_container_width=True,
                    height=400
                )
            elif len(explosive_stocks) == 0:
                st.info("No EXPLOSIVE EDGE opportunities found today. The market might be consolidating or lacking strong signals. Check the STRONG EDGE category below for next best opportunities.")
            else:
                st.warning("Insufficient data to display EXPLOSIVE EDGE stocks in detail.")


        # Strong edge section
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
                        subset=['edge_score'], cmap='Oranges'
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
            Consider reviewing the 'How It Works' tab to understand the signal criteria.
            """)

    with tab2:
        st.markdown("### 📊 Deep EDGE Analysis")

        # Edge distribution chart
        fig_dist = create_edge_distribution_chart(df)
        if fig_dist:
            st.plotly_chart(fig_dist, use_container_width=True)
        else:
            st.info("No EDGE score distribution data available for plotting.")

        # Top patterns
        st.markdown("### 🎯 Detected Patterns Across Market")

        if 'divergence_pattern' in df.columns:
            # Ensure 'NEUTRAL' is handled correctly or filtered if not relevant for 'detected' patterns
            pattern_counts = df['divergence_pattern'].value_counts()
            # Exclude 'NEUTRAL' and 'NO_DATA' from the display if they are very dominant
            pattern_counts = pattern_counts[~pattern_counts.index.isin(['NEUTRAL', 'NO_DATA'])].sort_values(ascending=False)

            if len(pattern_counts) > 0:
                cols_patterns = st.columns(3)
                for i, (pattern, count) in enumerate(pattern_counts.items()):
                    with cols_patterns[i % 3]:
                        emoji = "🚀" if "EXPLOSIVE" in pattern else ("📈" if "MOMENTUM" in pattern else ("🏦" if "STEALTH" in pattern else "💡"))
                        st.metric(f"{emoji} {pattern}", count)
            else:
                st.info("No significant momentum divergence patterns detected in the market today.")
        else:
            st.warning("Momentum divergence pattern data is not available.")

        # Sector edge analysis
        st.markdown("### 🏭 Edge by Sector")

        if 'sector' in df.columns and 'edge_score' in df.columns:
            # Filter out NaNs in 'sector' and ensure enough data per sector
            sector_edge = df.dropna(subset=['sector', 'edge_score']).groupby('sector').agg(
                edge_score_mean=('edge_score', 'mean'),
                volume_acceleration_mean=('volume_acceleration', lambda x: x.mean() if x.notna().any() else 0),
                stock_count=('ticker', 'count')
            ).sort_values('edge_score_mean', ascending=False)

            # Filter out sectors with very few stocks for meaningful averages
            sector_edge = sector_edge[sector_edge['stock_count'] >= 5].head(15) # Show top 15 sectors

            if not sector_edge.empty:
                fig_sector = go.Figure(data=[go.Bar(
                    x=sector_edge.index,
                    y=sector_edge['edge_score_mean'],
                    text=sector_edge['edge_score_mean'].round(1),
                    textposition='auto',
                    marker_color=sector_edge['edge_score_mean'],
                    marker_colorscale='Viridis'
                )])

                fig_sector.update_layout(
                    title="Average EDGE Score by Sector (Top 15)",
                    xaxis_title="Sector",
                    yaxis_title="Average EDGE Score",
                    height=450,
                    template="plotly_white"
                )
                st.plotly_chart(fig_sector, use_container_width=True)
            else:
                st.info("Not enough data to perform sector-wise EDGE analysis or too few stocks per sector.")
        else:
            st.warning("Sector or Edge Score data is missing, cannot generate Sector Edge Analysis.")

    with tab3: # Market Map Tab
        st.markdown("### 📈 Market EDGE Map")

        # Volume acceleration scatter
        fig_scatter = create_volume_acceleration_scatter(df)
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Market statistics
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
                'edge_score': 'Calculated EDGE Score' # Added to ensure score calculation worked
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
            if 'edge_score' in df.columns:
                stocks_with_edge = (df['edge_score'] > 0).sum()
                st.write(f"**Stocks with any EDGE (>0)**: {stocks_with_edge}")
                if stocks_with_edge > 0:
                    st.write(f"**Average EDGE Score**: {df['edge_score'].mean():.1f}")
            if 'volume_acceleration' in df.columns:
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
            st.markdown("**Next Update in approx.**")
            st.info("5 minutes (data is cached)")

    # Download section
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
        # Export all stocks with an edge score > WATCH threshold
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

    # Footer with disclaimer
    st.markdown("---")
    st.caption("""
    **Disclaimer:** **EDGE Protocol** - Finding What Others Can't See.
    This tool is for informational and educational purposes only. It is not financial advice.
    Your SECRET WEAPON: Volume Acceleration reveals institutional behavior before price moves.
    
    Position sizes are suggestions based on edge strength and are not guarantees.
    Always conduct your own due diligence and consult with a financial professional before making any investment decisions.
    Never risk more capital than you can afford to lose. Past performance and identified patterns do not guarantee future results.
    
    Version: 1.0 FINAL | Data updates every 5 minutes (cached).
    """)

if __name__ == "__main__":
    main()
```
