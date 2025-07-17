"""
EDGE Protocol - Ultimate Trading Intelligence System

Production-ready implementation with all critical fixes and enhancements:
- Corrected/Enhanced volume acceleration calculation
- Comprehensive portfolio risk management
- Dynamic and robust stop losses
- Performance optimization with caching and selective processing
- Robust error handling and data validation
- Improved code clarity, type hinting, and docstrings
- Enhanced UI/UX for better user experience
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import requests
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import time

# Suppress warnings for cleaner output in Streamlit
warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================
# Google Sheet Configuration
SHEET_ID = "1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk"
GID_WATCHLIST = "2026492216"
SHEET_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID_WATCHLIST}"

# Portfolio Risk Management Parameters
MAX_PORTFOLIO_EXPOSURE = 0.80  # 80% max exposure
MAX_POSITIONS = 10             # Maximum concurrent positions
MAX_SECTOR_EXPOSURE = 0.30     # 30% max in one sector
MAX_SUPER_EDGE_POSITIONS = 3   # Max 3 SUPER EDGE at once

# Position Sizing Multipliers (relative to base)
# These are base percentages, adjusted dynamically by risk_adjustment
BASE_POSITION_SIZES = {
    "SUPER_EDGE": 0.15,  # 15% of portfolio for a SUPER_EDGE stock if all conditions met
    "EXPLOSIVE": 0.10,   # 10% for EXPLOSIVE
    "STRONG": 0.05,      # 5% for STRONG
    "MODERATE": 0.02,    # 2% for MODERATE (small exploratory)
    "WATCH": 0.00        # 0% for WATCH (no position)
}

# EDGE Scoring Thresholds
EDGE_THRESHOLDS = {
    "SUPER_EDGE": 92,  # Score needed to be considered for SUPER EDGE
    "EXPLOSIVE": 85,
    "STRONG": 70,
    "MODERATE": 50,
    "WATCH": 0
}

# Performance Settings
MAX_PATTERN_DETECTION_STOCKS = 150  # Limit pattern detection for performance
CACHE_TTL = 300                     # Data cache time-to-live (5 minutes)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def safe_divide(numerator: Union[float, int], denominator: Union[float, int], default: float = 0.0) -> float:
    """
    Safely divides two numbers, handling zero division and non-numeric inputs.

    Args:
        numerator (Union[float, int]): The numerator.
        denominator (Union[float, int]): The denominator.
        default (float): The value to return if division is not possible.

    Returns:
        float: The result of the division or the default value.
    """
    try:
        if pd.isna(numerator) or pd.isna(denominator) or float(denominator) == 0:
            return default
        return float(numerator) / float(denominator)
    except (ValueError, TypeError):
        return default

def parse_market_cap(val: Union[str, float]) -> float:
    """
    Parses market capitalization strings, handling various units and formats
    including Indian Crore (Cr) and Lakh (L).

    Args:
        val (Union[str, float]): The market cap value as a string or float.

    Returns:
        float: The parsed market cap as a numeric value in base units, or NaN if unparseable.
    """
    if pd.isna(val) or val == '':
        return np.nan

    val_str = str(val).strip().lower()

    # Remove common currency symbols and commas
    val_str = val_str.replace("₹", "").replace("$", "").replace("€", "").replace("£", "").replace(",", "").strip()

    # Define multipliers for various suffixes
    multipliers = {
        't': 1e12, 'b': 1e9, 'm': 1e6, 'k': 1e3,  # Standard abbreviations
        'cr': 1e7, 'crore': 1e7,                 # Indian Crore
        'l': 1e5, 'lakh': 1e5, 'lac': 1e5        # Indian Lakh
    }

    for suffix, multiplier in multipliers.items():
        if val_str.endswith(suffix):
            try:
                number_part = float(val_str.replace(suffix, '').strip())
                return number_part * multiplier
            except ValueError:
                return np.nan

    try:
        return float(val_str)
    except ValueError:
        return np.nan

# ============================================================================
# DATA LOADING WITH VALIDATION AND PREPROCESSING
# ============================================================================
@st.cache_data(ttl=CACHE_TTL)
def load_and_validate_data() -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    Loads data from Google Sheets, performs comprehensive validation,
    cleaning, and basic preprocessing.
    """
    diagnostics = {
        "timestamp": datetime.now(),
        "rows_loaded": 0,
        "data_quality_score": 0,
        "critical_columns_missing": [],
        "warnings": [],
        "data_age_hours": 0 # This assumes an external mechanism updates the sheet frequently
    }

    try:
        # Load data
        response = requests.get(SHEET_URL, timeout=30)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        df = pd.read_csv(io.BytesIO(response.content))
        diagnostics["rows_loaded"] = len(df)

        # Standardize column names
        df.columns = (df.columns.str.strip()
                      .str.lower()
                      .str.replace("%", "pct", regex=False)
                      .str.replace(" ", "_", regex=False)
                      .str.replace(r'[^a-zA-Z0-9_]', '', regex=True)) # Remove other special chars

        # Critical columns check for core functionality
        critical_cols = [
            'ticker', 'price', 'volume_1d', 'ret_1d',
            'vol_ratio_7d_90d', 'vol_ratio_30d_90d', 'vol_ratio_30d_180d',
            'low_52w', 'high_52w', 'from_low_pct', 'from_high_pct',
            'sma_20d', 'sma_50d', 'sma_200d', 'rvol', 'sector'
        ]

        missing_critical = [col for col in critical_cols if col not in df.columns]
        diagnostics["critical_columns_missing"] = missing_critical

        if missing_critical:
            diagnostics["warnings"].append(f"Missing critical columns: {', '.join(missing_critical)}. Analysis may be incomplete.")
            # If fundamental columns like 'price' or 'ticker' are missing, return empty DataFrame
            if 'price' in missing_critical or 'ticker' in missing_critical:
                raise ValueError("Core data columns (price/ticker) are missing.")

        # Define all expected numeric columns based on your headers
        numeric_cols = [
            'price', 'ret_1d', 'low_52w', 'high_52w', 'from_low_pct', 'from_high_pct',
            'sma_20d', 'sma_50d', 'sma_200d', 'ret_3d', 'ret_7d', 'ret_30d',
            'ret_3m', 'ret_6m', 'ret_1y', 'ret_3y', 'ret_5y',
            'volume_1d', 'volume_7d', 'volume_30d', 'volume_90d', 'volume_180d',
            'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d',
            'vol_ratio_1d_180d', 'vol_ratio_7d_180d', 'vol_ratio_30d_180d',
            'vol_ratio_90d_180d', 'rvol', 'prev_close', 'pe',
            'eps_current', 'eps_last_qtr', 'eps_change_pct'
        ]

        # Clean and convert numeric columns
        for col in numeric_cols:
            if col in df.columns:
                # Replace non-numeric strings with NaN, then convert
                df[col] = df[col].astype(str).str.replace(r"[₹,$€£%a-zA-Z]", "", regex=True).replace(["", "-", "nan", "NaN", "NA"], np.nan)
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Fill NaN for ratios/volumes to 0 if that makes sense for calculation, otherwise leave as NaN
                if 'vol_ratio' in col or 'volume_' in col or col == 'rvol':
                    df[col] = df[col].fillna(0) # Treat missing volume data as 0 for ratio calculations

        # Specific handling for 'market_cap' - assume it exists from your headers
        if 'market_cap' in df.columns:
            df['market_cap_num'] = df['market_cap'].apply(parse_market_cap)
            # Fill NaN market caps with a small non-zero value or average if needed for filtering
            # For now, let's leave as NaN to indicate missing data for category filtering
        else:
            diagnostics["warnings"].append("Missing 'market_cap' column. Category filtering might be impacted.")
            df['market_cap_num'] = np.nan # Ensure column exists even if missing

        # Map market cap to categories for consistent filtering
        def map_market_category(mc: float) -> str:
            if pd.isna(mc): return 'Unknown'
            if mc >= 200 * 1e9: return 'Large Cap' # 200 Billion USD approx (adjust for INR if needed)
            if mc >= 50 * 1e9: return 'Mid Cap'
            if mc >= 1 * 1e9: return 'Small Cap'
            return 'Micro Cap'

        if 'market_cap_num' in df.columns:
            df['category'] = df['market_cap_num'].apply(map_market_category)
        else:
            df['category'] = 'Unknown' # Default if market_cap_num is missing

        # Data quality score (percentage of critical columns without NaNs)
        valid_data_points = 0
        total_data_points = 0
        for col in critical_cols:
            if col in df.columns:
                valid_data_points += df[col].notna().sum()
                total_data_points += len(df)
        diagnostics["data_quality_score"] = safe_divide(valid_data_points * 100, total_data_points, 0)

        # Remove rows with invalid (e.g., zero or NaN) prices or tickers
        initial_len = len(df)
        df.dropna(subset=['ticker'], inplace=True) # Ticker must not be null
        if 'price' in df.columns:
            df = df[df['price'].notna() & (df['price'] > 0)]
        else:
            diagnostics["warnings"].append("Price column not found after initial load. Cannot filter invalid prices.")

        if len(df) < initial_len:
            diagnostics["warnings"].append(f"Removed {initial_len - len(df)} rows due to missing ticker or invalid price.")

        return df, diagnostics

    except requests.exceptions.RequestException as req_err:
        diagnostics["warnings"].append(f"Network or data source error: {req_err}. Please check your internet connection or Google Sheet URL/permissions.")
        return pd.DataFrame(), diagnostics
    except pd.errors.EmptyDataError:
        diagnostics["warnings"].append("The Google Sheet is empty or contains no parsable data.")
        return pd.DataFrame(), diagnostics
    except Exception as e:
        diagnostics["warnings"].append(f"An unexpected error occurred during data loading: {e}")
        return pd.DataFrame(), diagnostics

# ============================================================================
# ENHANCED VOLUME METRICS CALCULATION
# ============================================================================
def calculate_volume_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates advanced volume acceleration and consistency metrics.
    Ensures that volume ratio columns exist or are handled gracefully.
    """
    df_copy = df.copy() # Work on a copy to avoid SettingWithCopyWarning

    # Ensure necessary volume ratio columns exist, filling with 0 if missing
    for col in ['vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d', 'vol_ratio_7d_30d', 'rvol']:
        if col not in df_copy.columns:
            df_copy[col] = 0.0 # Default to 0 for calculations if missing

    # Volume Acceleration: Recent vs Past momentum
    # True acceleration: Is 7-day volume growing faster than 30-day (compared to 90-day base)?
    # Or, if 7d_30d is available, 7d vs 30d relative to 90d
    vol_accel_base = df_copy['vol_ratio_7d_90d'] - df_copy['vol_ratio_30d_90d']

    if 'vol_ratio_7d_30d' in df_copy.columns:
        # If we have 7d vs 30d ratio, it's a more direct measure of recent acceleration.
        # Use it as a primary indicator, or combine with the 90d-based one.
        # Let's take the max of the two interpretations to capture strong signals.
        vol_accel_v2 = df_copy['vol_ratio_7d_30d'] # Directly 7d momentum vs 30d momentum
        df_copy['volume_acceleration'] = vol_accel_base.combine(vol_accel_v2, max)
    else:
        df_copy['volume_acceleration'] = vol_accel_base

    # Volume consistency score (how many timeframes show positive volume growth)
    vol_ratio_cols_to_check = [col for col in ['vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d'] if col in df_copy.columns]
    if vol_ratio_cols_to_check:
        df_copy['volume_consistency'] = (df_copy[vol_ratio_cols_to_check] > 0).sum(axis=1) / len(vol_ratio_cols_to_check)
    else:
        df_copy['volume_consistency'] = 0.0

    # Volume intensity (RVOL * volume acceleration, scaled)
    # Ensure volume_acceleration is positive for intensity, cap RVOL
    df_copy['volume_intensity'] = df_copy['rvol'].clip(0.5, 5.0) * np.maximum(df_copy['volume_acceleration'], 0) / 100
    df_copy['volume_intensity'] = df_copy['volume_intensity'].fillna(0) # Fill NaN from rvol or volume_acceleration

    # Classification of volume patterns
    conditions = [
        (df_copy['volume_acceleration'] > 60) & (df_copy['rvol'] > 2.5),
        (df_copy['volume_acceleration'] > 45) & (df_copy['rvol'] > 1.8),
        df_copy['volume_acceleration'] > 30,
        df_copy['volume_acceleration'] > 15,
        df_copy['volume_acceleration'] > 0,
        df_copy['volume_acceleration'] > -20
    ]

    choices = [
        "🔥 Explosive Accumulation",
        "🏦 Institutional Loading",
        "📈 Heavy Accumulation",
        "📊 Accumulation",
        "➕ Mild Accumulation",
        "➖ Distribution"
    ]

    df_copy['volume_pattern'] = np.select(conditions, choices, default="💀 Heavy Distribution")

    return df_copy

# ============================================================================
# ENHANCED PATTERN DETECTION (TOP 3 MOST EFFECTIVE)
# ============================================================================
def detect_top_patterns(row: pd.Series) -> Dict:
    """
    Detects only the TOP 3 most effective and distinct patterns.
    Returns the highest-scoring pattern's details.
    """
    patterns = []

    # Helper to get numeric values safely
    def get_val(key, default):
        val = row.get(key)
        return val if pd.notna(val) else default

    price = get_val('price', 0)
    sma_20d = get_val('sma_20d', price)
    sma_50d = get_val('sma_50d', price)
    sma_200d = get_val('sma_200d', price)

    # Pattern 1: Accumulation Under Resistance (High Conviction)
    # Price is consolidating just below 52-week high, with increasing volume.
    vol_ratio_30d_90d = get_val('vol_ratio_30d_90d', 0)
    from_high_pct = get_val('from_high_pct', 0)
    rvol = get_val('rvol', 1)
    ret_30d = get_val('ret_30d', 0)

    if (vol_ratio_30d_90d > 40 and -15 <= from_high_pct <= -1 and # Near resistance
        price > sma_50d and price > sma_20d and # Above short/medium term SMAs
        ret_30d > -5): # Not significantly pulling back over 30 days
        score = 75 + min(vol_ratio_30d_90d / 2, 20) # Max 95 base score
        signals = [f"Volume surge +{vol_ratio_30d_90d:.0f}%", f"Price -{abs(from_high_pct):.0f}% from high"]
        if rvol > 2.0:
            score = min(score * 1.15, 100) # Boost for high RVOL
            signals.append(f"RVOL {rvol:.1f}x confirms strength")
        if get_val('ret_6m', 0) > 30 and get_val('ret_1y', 0) > 50: # Strong longer term trend
            score = min(score * 1.05, 100)
            signals.append("Strong long-term trend")

        if score > 70:
            patterns.append({
                'name': 'Accumulation Under Resistance',
                'score': score,
                'signals': signals
            })

    # Pattern 2: Failed Breakdown Reversal (Value/Turnaround Play)
    # Price bounced strongly from 52-week low, supported by volume.
    from_low_pct = get_val('from_low_pct', 100)
    volume_acceleration = get_val('volume_acceleration', 0)
    ret_7d = get_val('ret_7d', 0)
    ret_30d = get_val('ret_30d', 0)

    if (from_low_pct < 20 and # Close to 52w low (but bounced)
        volume_acceleration > 25 and # Strong recent volume acceleration
        ret_7d > 5 and ret_30d > 0): # Recent positive momentum
        score = 65 + min(volume_acceleration / 1.5, 25) # Max 90 base score
        signals = [f"Reversal from low +{from_low_pct:.0f}%", f"Volume accelerating +{volume_acceleration:.0f}%"]

        if price > sma_20d and price > sma_50d: # Price crossing moving averages
            score = min(score * 1.1, 100)
            signals.append("Crossed short/mid-term SMAs")
        if get_val('ret_3y', 0) > 50: # Check for a history of good returns
            score = min(score * 1.05, 100)
            signals.append("Historical quality stock")

        if score > 60:
            patterns.append({
                'name': 'Failed Breakdown Reversal',
                'score': score,
                'signals': signals
            })

    # Pattern 3: Coiled Spring (Consolidation before Breakout)
    # Tight price range (low volatility) with increasing volume, above key SMAs.
    ret_7d_abs = abs(get_val('ret_7d', 0))
    ret_30d_abs = abs(get_val('ret_30d', 0))
    vol_ratio_30d_90d = get_val('vol_ratio_30d_90d', 0)

    # Volatility Check: price range within a tight band
    low_52w = get_val('low_52w', price * 0.5)
    high_52w = get_val('high_52w', price * 1.5)
    recent_range_pct = safe_divide((high_52w - low_52w), low_52w) * 100

    if (ret_7d_abs < 4 and ret_30d_abs < 8 and # Low recent price volatility
        vol_ratio_30d_90d > 25 and # Volume building up
        price > sma_20d and price > sma_50d and price > sma_200d): # Above all key SMAs
        score = 70 + min(vol_ratio_30d_90d / 1.5, 20) # Max 90 base score
        signals = [f"Tight range ({ret_7d_abs:.1f}% 7d, {ret_30d_abs:.1f}% 30d)", f"Volume building +{vol_ratio_30d_90d:.0f}%"]
        if rvol > 1.5:
            score = min(score * 1.1, 100)
            signals.append(f"RVOL {rvol:.1f}x indicates interest")
        if get_val('from_high_pct', 0) < -5: # Still has room to run
            score = min(score * 1.05, 100)
            signals.append("Room to run to 52w high")

        if score > 65:
            patterns.append({
                'name': 'Coiled Spring',
                'score': score,
                'signals': signals
            })

    # Sort by score and return top pattern
    patterns.sort(key=lambda x: x['score'], reverse=True)

    if patterns:
        top_pattern = patterns[0]
        # Count patterns with high conviction
        high_conviction_patterns = [p for p in patterns if p['score'] >= 75]
        return {
            'pattern_name': top_pattern['name'],
            'pattern_score': top_pattern['score'],
            'pattern_signals': ', '.join(top_pattern['signals']),
            'pattern_count': len(high_conviction_patterns)
        }

    return {
        'pattern_name': '',
        'pattern_score': 0,
        'pattern_signals': '',
        'pattern_count': 0
    }

# ============================================================================
# EDGE SCORING ENGINE WITH STRICTER CRITERIA AND DYNAMIC WEIGHTS
# ============================================================================
def calculate_edge_scores(df: pd.DataFrame, weights: Tuple[float, float, float]) -> pd.DataFrame:
    """
    Calculates the final EDGE score for each stock based on weighted components.
    Applies pattern detection as a bonus.

    Args:
        df (pd.DataFrame): Input DataFrame with all calculated metrics.
        weights (Tuple[float, float, float]): Weights for (volume, momentum, risk/reward) scores.

    Returns:
        pd.DataFrame: DataFrame with 'vol_score', 'mom_score', 'rr_score', 'EDGE',
                      and pattern detection columns added.
    """
    df_copy = df.copy() # Work on a copy

    vol_weight, mom_weight, rr_weight = weights

    # Component 1: Volume Score (weighted by vol_weight)
    df_copy['vol_score'] = 0.0 # Initialize
    if 'volume_acceleration' in df_copy.columns and 'rvol' in df_copy.columns and 'volume_consistency' in df_copy.columns:
        # Base score from acceleration, clipped to prevent extreme values
        base_vol_accel_score = df_copy['volume_acceleration'].clip(-50, 100) * 0.5 + 50 # Scale to 0-100 range roughly
        # RVOL multiplier: significant boost for high RVOL on positive acceleration
        rvol_mult = df_copy['rvol'].clip(0.5, 5.0) # Cap RVOL impact
        # Only boost if volume is actually accelerating
        vol_rvol_impact = np.where(df_copy['volume_acceleration'] > 0, base_vol_accel_score * (rvol_mult / 2.0), base_vol_accel_score)
        
        # Volume consistency bonus: more consistent accumulation is better
        vol_consistency_bonus = df_copy['volume_consistency'] * 20 # Up to 20 points bonus

        df_copy['vol_score'] = (vol_rvol_impact + vol_consistency_bonus).clip(0, 100)
    else:
        # If critical volume columns are missing, default to a neutral/low score
        df_copy['vol_score'] = 20.0

    # Component 2: Momentum Score (weighted by mom_weight)
    df_copy['mom_score'] = 50.0 # Start neutral
    momentum_cols = ['ret_1d', 'ret_3d', 'ret_7d', 'ret_30d', 'ret_3m', 'ret_6m', 'ret_1y']
    available_momentum_cols = [col for col in momentum_cols if col in df_copy.columns]

    if available_momentum_cols:
        # Prioritize recent momentum but consider medium-term
        df_copy['short_term_mom'] = df_copy['ret_1d'].fillna(0) * 0.4 + \
                                    df_copy['ret_3d'].fillna(0) * 0.3 + \
                                    df_copy['ret_7d'].fillna(0) * 0.3
        df_copy['medium_term_mom'] = df_copy['ret_30d'].fillna(0) * 0.6 + \
                                     df_copy['ret_3m'].fillna(0) * 0.4

        df_copy['mom_score'] += (df_copy['short_term_mom'] * 1.5).clip(-30, 30) # High impact for short-term
        df_copy['mom_score'] += (df_copy['medium_term_mom'] * 0.5).clip(-20, 20) # Moderate impact for medium-term

        # Momentum consistency bonus: all recent returns positive
        momentum_aligned = True
        for col in ['ret_1d', 'ret_3d', 'ret_7d']:
            if col in df_copy.columns:
                momentum_aligned = momentum_aligned & (df_copy[col] > 0).fillna(False)
        df_copy.loc[momentum_aligned, 'mom_score'] += 10 # Bonus for aligned short-term momentum
    df_copy['mom_score'] = df_copy['mom_score'].clip(0, 100)

    # Component 3: Risk/Reward Score (weighted by rr_weight)
    df_copy['rr_score'] = 50.0 # Start neutral
    if all(col in df_copy.columns for col in ['from_high_pct', 'from_low_pct', 'ret_3y']):
        # Distance from high (potential upside)
        # Sweet spot: -20% to -5% from 52w high indicates healthy pullback or consolidation
        sweet_spot_mask = (df_copy['from_high_pct'] >= -20) & (df_copy['from_high_pct'] <= -5)
        df_copy.loc[sweet_spot_mask, 'rr_score'] += 25 # Significant bonus

        # Penalize if too close to high or too far from high
        df_copy.loc[df_copy['from_high_pct'] > -5, 'rr_score'] -= 10 # Risk of immediate pullback
        df_copy.loc[df_copy['from_high_pct'] < -40, 'rr_score'] -= 15 # Deeper, potentially broken trend

        # Distance from low (risk perspective) - lower from_low_pct means higher risk of breakdown
        # Higher from_low_pct suggests a bounce, but too high means missed entry or extended
        df_copy['rr_score'] += (df_copy['from_low_pct'] / 3).clip(-10, 15) # Moderate bonus/penalty

        # Quality check (strong long-term performance)
        quality_stocks_mask = df_copy['ret_3y'] > 100 # Good long-term returns indicate quality
        df_copy.loc[quality_stocks_mask, 'rr_score'] = (df_copy.loc[quality_stocks_mask, 'rr_score'] * 1.1).clip(0, 100)
    df_copy['rr_score'] = df_copy['rr_score'].clip(0, 100)

    # Calculate weighted EDGE score
    df_copy['EDGE'] = (
        df_copy['vol_score'] * vol_weight +
        df_copy['mom_score'] * mom_weight +
        df_copy['rr_score'] * rr_weight
    )

    # Initialize pattern columns to avoid KeyError for stocks not in top_edge_stocks
    for col in ['pattern_name', 'pattern_signals']:
        if col not in df_copy.columns:
            df_copy[col] = ''
    for col in ['pattern_score', 'pattern_count']:
        if col not in df_copy.columns:
            df_copy[col] = 0

    # Pattern detection for top stocks only (performance optimization)
    # Apply patterns to stocks above a certain EDGE score or a fixed number of top stocks
    top_edge_candidates = df_copy[df_copy['EDGE'] >= EDGE_THRESHOLDS['MODERATE']].nlargest(MAX_PATTERN_DETECTION_STOCKS, 'EDGE', keep='first')

    for idx, row_data in top_edge_candidates.iterrows():
        pattern_data = detect_top_patterns(row_data)
        for key, value in pattern_data.items():
            df_copy.loc[idx, key] = value

    # Apply pattern bonus to EDGE score only for detected patterns
    df_copy.loc[df_copy['pattern_score'] > 70, 'EDGE'] = (df_copy['EDGE'] * 1.1).clip(0, 100)
    df_copy.loc[df_copy['pattern_count'] > 1, 'EDGE'] = (df_copy['EDGE'] * 1.05).clip(0, 100)
    df_copy['EDGE'] = df_copy['EDGE'].clip(0, 100) # Final clip

    return df_copy

def detect_super_edge_strict(row: pd.Series, sector_ranks: Dict[str, int]) -> bool:
    """
    STRICTER SUPER EDGE detection requiring at least 5 out of 6 conditions.
    """
    conditions_met = 0

    # Helper to get numeric values safely
    def get_val(key, default):
        val = row.get(key)
        return val if pd.notna(val) else default

    # 1. High RVOL
    rvol = get_val('rvol', 0)
    if rvol >= 2.0: # Increased threshold for conviction
        conditions_met += 1

    # 2. Strong volume acceleration
    vol_accel = get_val('volume_acceleration', 0)
    if vol_accel >= 35: # Increased threshold for conviction
        conditions_met += 1

    # 3. EPS acceleration (Quarter-over-Quarter growth)
    eps_current = get_val('eps_current', np.nan)
    eps_last_qtr = get_val('eps_last_qtr', np.nan)
    if pd.notna(eps_current) and pd.notna(eps_last_qtr) and eps_last_qtr != 0:
        eps_growth_pct = safe_divide((eps_current - eps_last_qtr), eps_last_qtr)
        if eps_growth_pct >= 0.20: # 20% QoQ growth
            conditions_met += 1
    elif pd.notna(get_val('eps_change_pct', np.nan)) and get_val('eps_change_pct', 0) >= 20: # Fallback to eps_change_pct if available
         conditions_met += 1


    # 4. Sweet spot zone for risk/reward (price is not too extended, not too broken)
    from_high_pct = get_val('from_high_pct', -100)
    if -20 <= from_high_pct <= -5: # Price is pulled back but near resistance
        conditions_met += 1

    # 5. Momentum alignment (all recent returns positive and accelerating)
    ret_1d = get_val('ret_1d', 0)
    ret_3d = get_val('ret_3d', 0)
    ret_7d = get_val('ret_7d', 0)
    ret_30d = get_val('ret_30d', 0)
    if (ret_1d > 0 and ret_3d > ret_1d and ret_7d > ret_3d and ret_30d > 0): # Accelerating short-term momentum
        conditions_met += 1

    # 6. Sector strength (stock belongs to a top 3 performing sector by avg EDGE)
    sector = row.get('sector')
    if sector and sector in sector_ranks and sector_ranks[sector] <= 3:
        conditions_met += 1

    return conditions_met >= 5 # Require 5 out of 6 conditions for SUPER EDGE

# ============================================================================
# DYNAMIC STOP LOSS CALCULATION
# ============================================================================
def calculate_dynamic_stops(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates intelligent stop losses based on price, volatility, and support levels.
    Also calculates risk adjustment for position sizing.
    """
    df_copy = df.copy()

    # Define default percentages for ATR-like stop based on category
    # These are percentage drops from current price
    category_atr_pct = {
        'micro cap': 0.12, # 12% stop for micro
        'small cap': 0.10, # 10% stop for small
        'mid cap': 0.08,   # 8% stop for mid
        'large cap': 0.07, # 7% stop for large
        'unknown': 0.09    # Default for unknown
    }

    df_copy['stop_loss'] = np.nan
    df_copy['stop_loss_pct'] = np.nan
    df_copy['risk_adjustment'] = 1.0 # Default to no adjustment

    for idx in df_copy.index:
        row = df_copy.loc[idx]
        price = row.get('price')
        if pd.isna(price) or price <= 0:
            continue # Skip invalid prices

        category = str(row.get('category', 'unknown')).lower()
        default_atr_factor = 1.0 - category_atr_pct.get(category, 0.09)
        atr_stop_price = price * default_atr_factor

        # Method 2: Support-based stop levels
        support_levels = []

        # SMA support (slight buffer below SMA)
        sma_20d = row.get('sma_20d')
        sma_50d = row.get('sma_50d')
        sma_200d = row.get('sma_200d')

        if pd.notna(sma_20d) and price > sma_20d: # Only consider SMA as support if price is above it
            support_levels.append(sma_20d * 0.98) # 2% below 20 SMA
        if pd.notna(sma_50d) and price > sma_50d:
            support_levels.append(sma_50d * 0.97) # 3% below 50 SMA
        if pd.notna(sma_200d) and price > sma_200d:
            support_levels.append(sma_200d * 0.96) # 4% below 200 SMA

        # 52-week low as a hard support (with a bounce buffer)
        low_52w = row.get('low_52w')
        if pd.notna(low_52w) and low_52w > 0:
            # If price is close to 52w low, allow a tighter stop to prevent full retrace
            if row.get('from_low_pct', 100) < 15: # If bounced less than 15% from low
                 support_levels.append(low_52w * 1.03) # 3% above 52w low
            else:
                support_levels.append(low_52w * 1.05) # 5% above 52w low for more established bounces

        # Use the highest of the calculated support levels, but not higher than ATR-based
        # This ensures we pick the 'safest' (highest) relevant support level
        if support_levels:
            # Take the max support level, but ensure it's not too tight (at least 5% below current price)
            support_stop_price = max(support_levels)
            support_stop_price = min(support_stop_price, price * 0.95) # Cap at 5% below price
        else:
            support_stop_price = atr_stop_price # Fallback if no valid support levels

        # Final stop loss: Take the higher (closer to current price, but still offering protection)
        # of the ATR-based stop or the calculated support-based stop.
        # This ensures the stop is responsive but also respects structural support.
        final_stop = max(atr_stop_price, support_stop_price)

        # Ensure stop is not above current price and is not excessively tight (min 5% drop)
        final_stop = min(final_stop, price * 0.95) # Max 5% drop from current price for stop (adjust as needed)
        final_stop = max(final_stop, price * 0.70) # Min 30% drop (preventing absurdly wide stops)

        df_copy.loc[idx, 'stop_loss'] = final_stop
        df_copy.loc[idx, 'stop_loss_pct'] = safe_divide((final_stop - price) * 100, price, 0)

        # Risk-based position sizing adjustment: reduce size for higher risk (wider stop)
        stop_distance_pct = abs(df_copy.loc[idx, 'stop_loss_pct'])
        if stop_distance_pct > 10: # Wider than 10%
            df_copy.loc[idx, 'risk_adjustment'] = 0.6 # Significant reduction
        elif stop_distance_pct > 7: # Wider than 7%
            df_copy.loc[idx, 'risk_adjustment'] = 0.8
        else: # 7% or tighter
            df_copy.loc[idx, 'risk_adjustment'] = 1.0

    return df_copy

# ============================================================================
# PORTFOLIO RISK MANAGEMENT AND POSITION SIZING
# ============================================================================
def apply_portfolio_constraints(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies portfolio-level risk constraints, dynamically adjusting position sizes.
    Prioritizes higher EDGE score stocks.
    """
    df_copy = df.copy()

    # Initialize position_size and portfolio_weight columns
    df_copy['position_size'] = 0.0
    df_copy['portfolio_weight'] = 0.0

    # Sort by EDGE score (descending) to prioritize higher conviction trades
    df_copy = df_copy.sort_values('EDGE', ascending=False).reset_index(drop=True)

    total_allocation = 0.0
    sector_allocations: Dict[str, float] = {}
    position_count = 0
    super_edge_count = 0

    for idx in df_copy.index:
        row = df_copy.loc[idx]
        tag = row.get('tag', 'WATCH')
        sector = row.get('sector', 'Unknown')
        risk_adjustment = row.get('risk_adjustment', 1.0) # From dynamic stop loss

        # Get base size for the signal tag, adjust by risk
        base_size = BASE_POSITION_SIZES.get(tag, 0.0)
        adjusted_size = base_size * risk_adjustment

        # Do not allocate if base_size is 0 or adjusted_size becomes very small
        if adjusted_size < 0.005: # Minimum allocation threshold (0.5%)
            continue

        # Check total position count constraint
        if position_count >= MAX_POSITIONS:
            continue

        # Check SUPER EDGE specific limit
        if tag == 'SUPER_EDGE':
            if super_edge_count >= MAX_SUPER_EDGE_POSITIONS:
                # If SUPER EDGE limit reached, downgrade allocation or skip
                adjusted_size = min(adjusted_size, BASE_POSITION_SIZES.get('STRONG', 0.05) * risk_adjustment) # Downgrade to STRONG size
                if adjusted_size < 0.005: # Re-check if it's still worth it
                    continue
            super_edge_count += 1

        # Check portfolio total exposure constraint
        remaining_portfolio_capacity = MAX_PORTFOLIO_EXPOSURE - total_allocation
        if adjusted_size > remaining_portfolio_capacity:
            adjusted_size = remaining_portfolio_capacity

        # Check sector concentration constraint
        current_sector_allocation = sector_allocations.get(sector, 0.0)
        remaining_sector_capacity = MAX_SECTOR_EXPOSURE - current_sector_allocation
        if adjusted_size > remaining_sector_capacity:
            adjusted_size = remaining_sector_capacity

        # If after all checks, adjusted_size is still positive, apply it
        if adjusted_size > 0.005: # Final check for minimum allocation
            df_copy.loc[idx, 'position_size'] = adjusted_size
            df_copy.loc[idx, 'portfolio_weight'] = adjusted_size # Same as position size for now
            total_allocation += adjusted_size
            sector_allocations[sector] = current_sector_allocation + adjusted_size
            position_count += 1
        else:
            df_copy.loc[idx, 'position_size'] = 0.0
            df_copy.loc[idx, 'portfolio_weight'] = 0.0

    # Ensure total portfolio allocation is available (useful for UI)
    df_copy['total_portfolio_allocation'] = total_allocation
    df_copy['current_positions_count'] = position_count

    return df_copy

# ============================================================================
# MAIN SCORING PIPELINE
# ============================================================================
def run_edge_analysis(df: pd.DataFrame, weights: Tuple[float, float, float]) -> pd.DataFrame:
    """
    Executes the complete EDGE analysis pipeline:
    1. Calculates volume metrics.
    2. Determines sector strength.
    3. Calculates main EDGE scores.
    4. Classifies stocks into categories.
    5. Detects and flags SUPER EDGE signals with strict criteria.
    6. Calculates dynamic stop losses and position size adjustments.
    7. Applies portfolio-level risk constraints.
    8. Calculates profit targets.
    9. Assigns final trading decisions.
    """
    if df.empty:
        return pd.DataFrame() # Return empty if no data

    # 1. Calculate volume metrics
    df_processed = calculate_volume_metrics(df.copy())

    # 2. Calculate sector rankings for SUPER EDGE detection
    sector_ranks = {}
    if 'sector' in df_processed.columns and 'volume_acceleration' in df_processed.columns and 'EDGE' in df_processed.columns:
        # Rank sectors by average EDGE score and average volume acceleration
        sector_agg = df_processed.groupby('sector').agg(
            avg_edge=('EDGE', 'mean'),
            avg_vol_accel=('volume_acceleration', 'mean'),
            stock_count=('ticker', 'count')
        ).reset_index()

        # Only rank sectors with at least 3 stocks and reasonable avg EDGE
        sector_agg = sector_agg[(sector_agg['stock_count'] >= 3) & (sector_agg['avg_edge'] > 50)]
        if not sector_agg.empty:
            # Combine rankings: Higher avg_edge and avg_vol_accel are better
            sector_agg['combined_rank'] = sector_agg['avg_edge'] + sector_agg['avg_vol_accel']
            sector_scores_sorted = sector_agg.sort_values('combined_rank', ascending=False)
            for i, sector in enumerate(sector_scores_sorted['sector']):
                sector_ranks[sector] = i + 1
    else:
        st.warning("Sector or volume/EDGE data missing for sector ranking.")

    # 3. Calculate main EDGE scores, including pattern detection
    df_processed = calculate_edge_scores(df_processed, weights)

    # 4. Classify stocks based on EDGE score
    conditions = [
        df_processed['EDGE'] >= EDGE_THRESHOLDS['EXPLOSIVE'],
        df_processed['EDGE'] >= EDGE_THRESHOLDS['STRONG'],
        df_processed['EDGE'] >= EDGE_THRESHOLDS['MODERATE']
    ]
    choices = ['EXPLOSIVE', 'STRONG', 'MODERATE']
    df_processed['tag'] = np.select(conditions, choices, default='WATCH')

    # 5. Detect SUPER EDGE with stricter criteria (only for high EDGE candidates)
    # Ensure 'tag' column is initialized before modification
    if 'tag' not in df_processed.columns:
        df_processed['tag'] = 'WATCH'

    super_edge_candidates_idx = df_processed[df_processed['EDGE'] >= EDGE_THRESHOLDS['SUPER_EDGE']].index
    for idx in super_edge_candidates_idx:
        if detect_super_edge_strict(df_processed.loc[idx], sector_ranks):
            df_processed.loc[idx, 'tag'] = 'SUPER_EDGE'
            # Give a small boost to the EDGE score for confirmed SUPER EDGE
            df_processed.loc[idx, 'EDGE'] = min(df_processed.loc[idx, 'EDGE'] * 1.05, 100) # Small final boost

    # 6. Calculate dynamic stops
    df_processed = calculate_dynamic_stops(df_processed)

    # 7. Apply portfolio constraints (position sizing)
    df_processed = apply_portfolio_constraints(df_processed)

    # 8. Calculate targets (adjust based on stock category, conviction)
    # Conservative targets:
    df_processed['target_1'] = df_processed['price'] * 1.10 # 10% gain as first target
    df_processed['target_2'] = df_processed['price'] * 1.20 # 20% gain as second target

    # Adjust targets for SUPER EDGE (higher conviction, higher potential)
    super_mask = df_processed['tag'] == 'SUPER_EDGE'
    df_processed.loc[super_mask, 'target_1'] = df_processed.loc[super_mask, 'price'] * 1.15 # 15% for SUPER EDGE
    df_processed.loc[super_mask, 'target_2'] = df_processed.loc[super_mask, 'price'] * 1.30 # 30% for SUPER EDGE

    # 9. Add decision column based on 'tag' and 'portfolio_weight'
    df_processed['decision'] = df_processed.apply(
        lambda row: 'BUY NOW' if row['tag'] == 'SUPER_EDGE' and row['portfolio_weight'] > 0 else
                    'BUY' if row['tag'] == 'EXPLOSIVE' and row['portfolio_weight'] > 0 else
                    'ACCUMULATE' if row['tag'] == 'STRONG' and row['portfolio_weight'] > 0 else
                    'WATCH' if row['tag'] == 'MODERATE' else
                    'IGNORE', axis=1
    )

    return df_processed

# ============================================================================
# UI COMPONENTS
# ============================================================================
def render_sidebar_diagnostics(diagnostics: Dict):
    """Renders system health and data diagnostics in the sidebar."""
    with st.sidebar.expander("📊 System Health & Diagnostics", expanded=False):
        # Data quality score with color coding
        quality_score = diagnostics.get('data_quality_score', 0)
        if quality_score > 90:
            st.success(f"Data Quality: {quality_score:.0f}% - Excellent")
        elif quality_score > 70:
            st.warning(f"Data Quality: {quality_score:.0f}% - Good, check warnings")
        else:
            st.error(f"Data Quality: {quality_score:.0f}% - Poor, proceed with caution")

        # Timestamp
        st.write(f"**Last Run:** {diagnostics.get('timestamp', 'Unknown').strftime('%Y-%m-%d %H:%M:%S')}")

        # Data stats
        st.write(f"**Rows Loaded:** {diagnostics.get('rows_loaded', 0):,}")

        # Warnings
        warnings_list = diagnostics.get('warnings', [])
        if warnings_list:
            st.write("**⚠️ Warnings:**")
            for warning in warnings_list[:5]: # Show max 5 warnings
                st.write(f"• {warning}")
            if len(warnings_list) > 5:
                st.write(f"  ... {len(warnings_list) - 5} more warnings.")

        # Critical columns
        missing_critical = diagnostics.get('critical_columns_missing', [])
        if missing_critical:
            st.error(f"Missing critical columns: {', '.join(missing_critical)}. Data may be unreliable.")

        # Download diagnostic report
        diag_data = pd.DataFrame([diagnostics])
        csv = diag_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Diagnostic Report",
            csv,
            "diagnostics.csv",
            "text/csv",
            key="diag_download"
        )

def render_sector_leaderboard(df: pd.DataFrame):
    """Renders a detailed sector leaderboard."""
    st.header("🏆 Sector EDGE Leaderboard")

    if df.empty or 'sector' not in df.columns or 'EDGE' not in df.columns:
        st.info("Sector data or EDGE scores not available for analysis.")
        return

    # Calculate sector metrics
    agg_dict = {
        'ticker': 'count',
        'EDGE': 'mean',
        'volume_acceleration': 'mean',
        'tag': lambda x: (x == 'SUPER_EDGE').sum() # Count SUPER_EDGE stocks
    }

    sector_metrics = df.groupby('sector').agg(agg_dict).reset_index()

    # Rename columns for display
    sector_metrics.rename(columns={
        'ticker': 'Total Stocks',
        'EDGE': 'Avg EDGE Score',
        'volume_acceleration': 'Avg Vol Accel %',
        'tag': 'SUPER EDGE Count'
    }, inplace=True)

    # Filter out sectors with very few stocks for meaningful averages
    sector_metrics = sector_metrics[sector_metrics['Total Stocks'] >= 3]

    # Sort by Avg EDGE Score and then by SUPER EDGE Count
    sector_metrics = sector_metrics.sort_values(
        by=['Avg EDGE Score', 'SUPER EDGE Count'],
        ascending=[False, False]
    ).round(1)

    if sector_metrics.empty:
        st.info("No sectors meet the criteria for a meaningful leaderboard.")
        return

    # Display top N sectors
    display_limit = st.slider("Show Top N Sectors", 5, len(sector_metrics), min(10, len(sector_metrics)))
    display_sectors = sector_metrics.head(display_limit)

    for idx, row in display_sectors.iterrows():
        medal = "🥇" if row.name == 0 else "🥈" if row.name == 1 else "🥉" if row.name == 2 else f"{row.name+1}."
        col1, col2, col3, col4 = st.columns([0.5, 2, 1.5, 1]) # Adjust column widths for better alignment

        with col1:
            st.markdown(f"**{medal}**")
        with col2:
            st.markdown(f"**{row['sector']}**")
        with col3:
            st.metric("Avg EDGE", f"{row['Avg EDGE Score']:.1f}")
        with col4:
            if row['SUPER EDGE Count'] > 0:
                st.metric("Super Edge", f"{int(row['SUPER EDGE Count'])}")
            else:
                st.markdown("N/A")

        # Optional: Mini bar chart for EDGE distribution within the sector (requires more data)
        with st.expander(f"Details for {row['sector']}"):
            sector_stocks = df[df['sector'] == row['sector']].sort_values('EDGE', ascending=False)
            if not sector_stocks.empty:
                cols_to_show = ['ticker', 'company_name', 'EDGE', 'tag', 'price', 'volume_acceleration', 'pattern_name']
                cols_to_show = [c for c in cols_to_show if c in sector_stocks.columns]
                st.dataframe(sector_stocks[cols_to_show].style.format({'EDGE': '{:.1f}', 'price': '₹{:.2f}', 'volume_acceleration': '{:.1f}%'}), use_container_width=True)
            else:
                st.info("No detailed stock data available for this sector.")


def create_excel_report(df_signals: pd.DataFrame, df_all: pd.DataFrame) -> io.BytesIO:
    """
    Creates a multi-sheet Excel report containing various aspects of the analysis.
    """
    output = io.BytesIO()

    # Ensure all dataframes used in the report are not empty
    if df_signals.empty and df_all.empty:
        st.warning("No data to generate Excel report.")
        return output # Return empty BytesIO

    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Sheet 1: Executive Summary
        summary_data = {
            'Metric': [
                'Total Stocks Analyzed',
                'Total Signals Generated',
                'SUPER EDGE Count',
                'EXPLOSIVE Count',
                'Portfolio Allocation (Total)',
                'Avg EDGE Score (Signals)',
                'Top Sector by Avg EDGE',
                'Report Generated On'
            ],
            'Value': [
                len(df_all),
                len(df_signals),
                (df_signals['tag'] == 'SUPER_EDGE').sum() if 'tag' in df_signals.columns else 0,
                (df_signals['tag'] == 'EXPLOSIVE').sum() if 'tag' in df_signals.columns else 0,
                f"{df_signals['portfolio_weight'].sum()*100:.1f}%" if 'portfolio_weight' in df_signals.columns else "0.0%",
                f"{df_signals['EDGE'].mean():.1f}" if 'EDGE' in df_signals.columns and len(df_signals) > 0 else "N/A",
                df_signals.groupby('sector')['EDGE'].mean().idxmax() if 'sector' in df_signals.columns and 'EDGE' in df_signals.columns and len(df_signals) > 0 else 'N/A',
                datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')
            ]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Executive Summary', index=False)

        # Sheet 2: Action Items (SUPER EDGE + EXPLOSIVE) - Most important for trading
        if 'tag' in df_signals.columns:
            action_items = df_signals[df_signals['tag'].isin(['SUPER_EDGE', 'EXPLOSIVE'])].copy()
            if not action_items.empty:
                action_cols = [
                    'ticker', 'company_name', 'tag', 'EDGE', 'decision',
                    'price', 'position_size', 'stop_loss', 'stop_loss_pct', 'target_1', 'target_2',
                    'volume_pattern', 'pattern_name', 'pattern_signals', 'sector', 'rvol', 'volume_acceleration'
                ]
                # Filter to only include columns that actually exist in the dataframe
                action_cols = [col for col in action_cols if col in action_items.columns]
                if action_cols:
                    action_items[action_cols].to_excel(writer, sheet_name='Actionable Signals', index=False)
                else:
                    pd.DataFrame([{"Message": "No relevant columns for Actionable Signals."}]).to_excel(writer, sheet_name='Actionable Signals', index=False)
            else:
                pd.DataFrame([{"Message": "No SUPER EDGE or EXPLOSIVE signals."}]).to_excel(writer, sheet_name='Actionable Signals', index=False)

        # Sheet 3: All Trading Signals (filtered by min EDGE)
        if not df_signals.empty:
            signal_cols = [
                'ticker', 'company_name', 'sector', 'tag', 'EDGE', 'decision',
                'price', 'position_size', 'portfolio_weight',
                'stop_loss', 'stop_loss_pct', 'target_1', 'target_2',
                'vol_score', 'mom_score', 'rr_score',
                'volume_acceleration', 'rvol', 'volume_pattern',
                'pattern_name', 'pattern_score', 'pattern_signals', 'from_high_pct', 'from_low_pct'
            ]
            signal_cols = [col for col in signal_cols if col in df_signals.columns]
            if signal_cols:
                df_signals[signal_cols].to_excel(writer, sheet_name='All Trading Signals', index=False)
            else:
                pd.DataFrame([{"Message": "No relevant columns for All Trading Signals."}]).to_excel(writer, sheet_name='All Trading Signals', index=False)
        else:
            pd.DataFrame([{"Message": "No signals generated based on current filters."}]).to_excel(writer, sheet_name='All Trading Signals', index=False)

        # Sheet 4: Pattern Analysis (all detected patterns)
        if 'pattern_score' in df_all.columns:
            pattern_df = df_all[df_all['pattern_score'] > 0].copy()
            if not pattern_df.empty:
                pattern_cols = ['ticker', 'company_name', 'pattern_name', 'pattern_score', 'pattern_signals', 'EDGE', 'tag', 'price']
                pattern_cols = [col for col in pattern_cols if col in pattern_df.columns]
                if pattern_cols:
                    pattern_df[pattern_cols].sort_values(by='pattern_score', ascending=False).to_excel(writer, sheet_name='Pattern Analysis', index=False)
                else:
                    pd.DataFrame([{"Message": "No relevant columns for Pattern Analysis."}]).to_excel(writer, sheet_name='Pattern Analysis', index=False)
            else:
                pd.DataFrame([{"Message": "No patterns detected in the dataset."}]).to_excel(writer, sheet_name='Pattern Analysis', index=False)

        # Sheet 5: Sector Summary Analysis
        if 'sector' in df_all.columns and not df_all.empty:
            agg_dict_sector = {}
            if 'EDGE' in df_all.columns:
                agg_dict_sector['EDGE'] = ['mean', 'max', 'count']
            if 'volume_acceleration' in df_all.columns:
                agg_dict_sector['volume_acceleration'] = 'mean'
            if 'rvol' in df_all.columns:
                agg_dict_sector['rvol'] = 'mean'
            if 'ret_3m' in df_all.columns:
                agg_dict_sector['ret_3m'] = 'mean'

            if agg_dict_sector:
                sector_analysis = df_all.groupby('sector').agg(agg_dict_sector).round(1)
                # Flatten column names for multi-index aggregation
                sector_analysis.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in sector_analysis.columns]
                sector_analysis.rename(columns={'ticker_count': 'Total Stocks', 'EDGE_mean': 'Avg EDGE', 'EDGE_max': 'Max EDGE',
                                                'volume_acceleration_mean': 'Avg Vol Accel', 'rvol_mean': 'Avg RVOL',
                                                'ret_3m_mean': 'Avg 3m Return'}, inplace=True)
                sector_analysis.sort_values(by='Avg EDGE', ascending=False).to_excel(writer, sheet_name='Sector Analysis')
            else:
                pd.DataFrame([{"Message": "Not enough data for Sector Analysis."}]).to_excel(writer, sheet_name='Sector Analysis', index=False)
        else:
            pd.DataFrame([{"Message": "Sector data missing or DataFrame is empty for Sector Analysis."}]).to_excel(writer, sheet_name='Sector Analysis', index=False)

        # Sheet 6: Raw Data (optional, first 5000 rows to prevent huge files)
        if not df_all.empty:
            df_all.head(5000).to_excel(writer, sheet_name='Raw Data', index=False)
        else:
            pd.DataFrame([{"Message": "Raw data is empty."}]).to_excel(writer, sheet_name='Raw Data', index=False)

    output.seek(0) # Rewind the buffer
    return output

# ============================================================================
# MAIN STREAMLIT UI FUNCTION
# ============================================================================
def render_ui():
    """
    Main Streamlit UI function orchestrating the display of the EDGE Protocol.
    """
    st.set_page_config(
        page_title="EDGE Protocol - Ultimate Trading Intelligence",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for better UI and visual appeal
    st.markdown("""
    <style>
    .super-edge-alert {
        background: linear-gradient(135deg, #FFD700, #FFA500); /* Gold to Orange */
        color: black;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { opacity: 0.8; transform: scale(1); }
        50% { opacity: 1; transform: scale(1.01); }
        100% { opacity: 0.8; transform: scale(1); }
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: rgba(240, 242, 246, 0.5);
        border-radius: 5px 5px 0px 0px;
    }
    /* Style for metric boxes */
    [data-testid="stMetricValue"] {
        font-size: 24px;
    }
    [data-testid="stMetricLabel"] {
        font-size: 16px;
    }
    .dataframe-style {
        font-size: 0.9em; /* Slightly smaller font for dense tables */
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.title("⚡ EDGE Protocol - Ultimate Trading Intelligence")
    st.markdown("**Correct Volume Acceleration + Risk Management + Pattern Recognition = Superior Returns**")

    # Sidebar for Configuration & Filters
    with st.sidebar:
        st.header("⚙️ Protocol Settings")

        st.subheader("📊 Strategy Component Weights")
        st.info("Adjust the influence of each factor on the overall EDGE score.")
        col_vw, col_mw = st.columns(2)
        with col_vw:
            vol_weight = st.slider("Volume %", 30, 70, 50, 5, help="Emphasis on volume acceleration, consistency, and intensity.") / 100
        with col_mw:
            mom_weight = st.slider("Momentum %", 10, 40, 30, 5, help="Emphasis on short to medium term price returns.") / 100
        rr_weight = 1 - vol_weight - mom_weight
        st.markdown(f"**Risk/Reward %:** {rr_weight*100:.0f} (Automatically calculated)")
        weights = (vol_weight, mom_weight, rr_weight)

        st.markdown("---")

        st.subheader("🎯 Signal Filters")
        min_edge = st.slider("Minimum EDGE Score for Signals", 0, 100, 70, 5, help="Only show stocks with an EDGE score above this threshold.")
        exclude_small_caps = st.checkbox("Exclude Micro/Small Cap Stocks", True, help="Filter out stocks generally below $1 Billion market cap, depending on data.")
        max_signals = st.slider("Maximum Number of Signals to Display", 10, 200, 100, 10, help="Limits the total number of signals shown for performance and focus.")

    # Load and validate data
    df_raw, diagnostics = load_and_validate_data()

    # Show diagnostics in sidebar
    render_sidebar_diagnostics(diagnostics)

    if df_raw.empty:
        st.error("❌ Data loading failed or resulted in an empty dataset. Please check the diagnostics in the sidebar.")
        return

    # Apply market cap filter early
    df_filtered_category = df_raw.copy()
    if exclude_small_caps and 'category' in df_filtered_category.columns:
        df_filtered_category = df_filtered_category[
            ~df_filtered_category['category'].str.contains('micro cap|small cap', case=False, na=False)
        ]
        if df_filtered_category.empty:
            st.warning("All stocks filtered out by category. Try unchecking 'Exclude Micro/Small Cap Stocks'.")
            return

    # Run EDGE analysis pipeline
    with st.spinner("🚀 Running EDGE Protocol Analysis... This may take a moment."):
        start_time = time.time()
        df_analyzed = run_edge_analysis(df_filtered_category, weights)
        end_time = time.time()
        st.success(f"Analysis completed in {end_time - start_time:.2f} seconds.")

    # Filter for signals based on user-defined min_edge and max_signals
    df_signals = df_analyzed[df_analyzed['EDGE'] >= min_edge].nlargest(max_signals, 'EDGE', keep='first').copy()

    # SUPER EDGE Alert (only if df_signals is not empty)
    super_edge_count = (df_signals['tag'] == 'SUPER_EDGE').sum() if 'tag' in df_signals.columns else 0
    if super_edge_count > 0:
        st.markdown(f"""
        <div class="super-edge-alert">
            ⭐ {super_edge_count} SUPER EDGE SIGNAL{'S' if super_edge_count > 1 else ''} DETECTED ⭐<br>
            <span style="font-size: 16px;">Maximum conviction trades with strict risk management!</span>
        </div>
        """, unsafe_allow_html=True)

    # Main content tabs
    tabs = st.tabs([
        "🎯 Trading Signals",
        "⭐ SUPER EDGE Focus",
        "🏆 Sector Leaders",
        "📊 Deep Analytics"
    ])

    # Tab 1: Trading Signals
    with tabs[0]:
        st.header("🎯 Today's Trading Signals")

        # Quick metrics for signals
        col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
        with col_s1:
            st.metric("Total Signals", len(df_signals))
        with col_s2:
            st.metric("SUPER EDGE", super_edge_count)
        with col_s3:
            portfolio_used = df_signals['portfolio_weight'].sum() * 100 if 'portfolio_weight' in df_signals.columns else 0
            st.metric("Portfolio Used", f"{portfolio_used:.1f}%")
        with col_s4:
            avg_edge = df_signals['EDGE'].mean() if 'EDGE' in df_signals.columns and len(df_signals) > 0 else 0
            st.metric("Avg EDGE Score", f"{avg_edge:.1f}")
        with col_s5:
            strong_patterns = (df_signals['pattern_score'] >= 75).sum() if 'pattern_score' in df_signals.columns else 0
            st.metric("Strong Patterns", strong_patterns)

        st.markdown("---")

        # Dynamic Filters for the table display
        col_f1, col_f2, col_f3 = st.columns(3)
        current_display_df = df_signals.copy() # Base for tab-specific filters

        with col_f1:
            if 'tag' in current_display_df.columns:
                unique_tags = current_display_df['tag'].dropna().unique().tolist()
                selected_tags = st.multiselect(
                    "Filter by Signal Type",
                    sorted(unique_tags, key=lambda x: ['SUPER_EDGE', 'EXPLOSIVE', 'STRONG', 'MODERATE', 'WATCH', 'IGNORE'].index(x) if x in ['SUPER_EDGE', 'EXPLOSIVE', 'STRONG', 'MODERATE', 'WATCH', 'IGNORE'] else 99),
                    default=unique_tags if len(unique_tags) < 5 else ['SUPER_EDGE', 'EXPLOSIVE', 'STRONG'] # Default to top ones if many
                )
                if selected_tags:
                    current_display_df = current_display_df[current_display_df['tag'].isin(selected_tags)]
        with col_f2:
            if 'sector' in current_display_df.columns:
                unique_sectors = current_display_df['sector'].dropna().unique().tolist()
                selected_sectors = st.multiselect(
                    "Filter by Sector",
                    sorted(unique_sectors),
                    default=unique_sectors if len(unique_sectors) < 10 else [] # No default if too many sectors
                )
                if selected_sectors:
                    current_display_df = current_display_df[current_display_df['sector'].isin(selected_sectors)]
        with col_f3:
            search_ticker = st.text_input("Search by Ticker", "").upper()
            if search_ticker:
                current_display_df = current_display_df[current_display_df['ticker'].str.contains(search_ticker, na=False)]

        # Display main signals table
        if not current_display_df.empty:
            # Define columns to show in the main table, prioritize key info
            display_cols_order = [
                'ticker', 'company_name', 'sector', 'tag', 'EDGE', 'decision',
                'price', 'position_size', 'stop_loss', 'stop_loss_pct',
                'target_1', 'target_2', 'volume_acceleration', 'rvol',
                'volume_pattern', 'pattern_name', 'pattern_score', 'pattern_signals'
            ]
            # Ensure only existing columns are included
            display_cols = [col for col in display_cols_order if col in current_display_df.columns]

            # Custom styling for the DataFrame
            def style_signals_dataframe(df_to_style):
                s = df_to_style.style.format({
                    'EDGE': '{:.1f}',
                    'price': '₹{:.2f}',
                    'position_size': '{:.1%}',
                    'stop_loss': '₹{:.2f}',
                    'stop_loss_pct': '{:.1f}%',
                    'target_1': '₹{:.2f}',
                    'target_2': '₹{:.2f}',
                    'volume_acceleration': '{:.1f}%',
                    'rvol': '{:.1f}x',
                    'pattern_score': '{:.0f}'
                })
                if 'tag' in df_to_style.columns:
                    s = s.applymap(lambda x: 'background-color: gold; font-weight: bold;' if x == 'SUPER_EDGE' else
                                              'background-color: #ffe6e6;' if x == 'EXPLOSIVE' else '', subset=['tag'])
                if 'decision' in df_to_style.columns:
                    s = s.applymap(lambda x: 'color: green; font-weight: bold;' if x == 'BUY NOW' else
                                              'color: #008000;' if x == 'BUY' else
                                              'color: #ff8c00;' if x == 'ACCUMULATE' else
                                              'color: #808080;' if x == 'WATCH' else '', subset=['decision'])
                if 'stop_loss_pct' in df_to_style.columns:
                    s = s.background_gradient(cmap='RdYlGn_r', subset=['stop_loss_pct'], vmin=-15, vmax=0) # Green for tighter, Red for looser
                if 'EDGE' in df_to_style.columns:
                    s = s.background_gradient(cmap='Greens', subset=['EDGE'], vmin=50, vmax=100)
                if 'volume_acceleration' in df_to_style.columns:
                    s = s.background_gradient(cmap='Blues', subset=['volume_acceleration'], vmin=0, vmax=100)

                return s

            st.dataframe(style_signals_dataframe(current_display_df[display_cols]), use_container_width=True, height=600)

            # Export buttons
            col_export_csv, col_export_excel = st.columns(2)
            with col_export_csv:
                csv_export = current_display_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Signals (CSV)",
                    csv_export,
                    f"edge_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    type="primary"
                )
            with col_export_excel:
                # Pass both df_signals (filtered) and df_analyzed (full processed) for comprehensive report
                excel_buffer = create_excel_report(df_signals, df_analyzed)
                st.download_button(
                    "📊 Download Full Report (Excel)",
                    excel_buffer,
                    f"EDGE_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary"
                )
        else:
            st.info("No trading signals match the selected filters. Try adjusting the Min EDGE Score or other filters.")

    # Tab 2: SUPER EDGE Focus
    with tabs[1]:
        st.header("⭐ SUPER EDGE Deep Dive")

        if 'tag' in df_signals.columns:
            super_df = df_signals[df_signals['tag'] == 'SUPER_EDGE'].sort_values('EDGE', ascending=False)
        else:
            super_df = pd.DataFrame()

        if not super_df.empty:
            st.success(f"🎯 Found {len(super_df)} SUPER EDGE opportunities meeting high conviction criteria!")

            for idx, (_, row) in enumerate(super_df.iterrows()):
                # Unique key for expander
                expander_key = f"super_edge_expander_{row['ticker']}_{idx}"
                with st.expander(
                    f"#{idx+1} {row['ticker']} - {row.get('company_name', 'N/A')} | EDGE: {row['EDGE']:.1f} | Decision: {row['decision']}",
                    expanded=(idx == 0), # Expand first one by default
                    key=expander_key
                ):
                    col_det1, col_det2, col_det3 = st.columns(3)

                    with col_det1:
                        st.subheader("📈 Trade Plan")
                        st.metric("Current Price", f"₹{row['price']:.2f}")
                        st.metric("Allocated Size", f"{row['position_size']*100:.1f}% of Portfolio")
                        st.metric("Stop Loss", f"₹{row['stop_loss']:.2f} ({row['stop_loss_pct']:.1f}%)")

                    with col_det2:
                        st.subheader("💰 Potential Returns")
                        st.metric("Target 1", f"₹{row['target_1']:.2f} ({safe_divide((row['target_1']-row['price'])*100, row['price'], 0):.1f}%)")
                        st.metric("Target 2", f"₹{row['target_2']:.2f} ({safe_divide((row['target_2']-row['price'])*100, row['price'], 0):.1f}%)")

                        # Safe risk:reward calculation
                        price = row.get('price', 1)
                        stop_loss = row.get('stop_loss', price * 0.95)
                        target_1 = row.get('target_1', price * 1.10) # Using T1 for R:R
                        risk_amount = abs(price - stop_loss)
                        reward_amount = abs(target_1 - price)

                        if risk_amount > 0:
                            risk_reward_ratio = safe_divide(reward_amount, risk_amount)
                            st.metric("Risk:Reward", f"1:{risk_reward_ratio:.1f}")
                        else:
                            st.metric("Risk:Reward", "N/A (No discernible risk)")


                    with col_det3:
                        st.subheader("🔍 Core Drivers")
                        st.markdown(f"**Sector:** {row.get('sector', 'N/A')}")
                        st.markdown(f"**Volume Pattern:** {row.get('volume_pattern', 'N/A')}")
                        st.markdown(f"**Volume Accel:** {row.get('volume_acceleration', 0):.1f}%")
                        st.markdown(f"**RVOL:** {row.get('rvol', 0):.1f}x")
                        if row.get('pattern_name'):
                            st.markdown(f"**Detected Pattern:** {row['pattern_name']}")
                        if row.get('pattern_signals'):
                            st.caption(f"Signals: {row['pattern_signals']}")

                    st.markdown("---")
                    st.subheader("✅ Why This is a SUPER EDGE Signal:")
                    criteria = []
                    # Dynamically check each SUPER EDGE condition from detect_super_edge_strict
                    # This provides transparent reasoning
                    if row.get('rvol', 0) >= 2.0:
                        criteria.append("✅ **High RVOL (>= 2.0x):** Strong relative volume.")
                    if row.get('volume_acceleration', 0) >= 35:
                        criteria.append(f"✅ **Volume Acceleration (>= 35%):** Recent volume surging significantly.")
                    eps_current = row.get('eps_current', np.nan)
                    eps_last_qtr = row.get('eps_last_qtr', np.nan)
                    eps_change_pct = row.get('eps_change_pct', np.nan)
                    if pd.notna(eps_current) and pd.notna(eps_last_qtr) and eps_last_qtr != 0:
                        eps_growth_pct = safe_divide((eps_current - eps_last_qtr), eps_last_qtr)
                        if eps_growth_pct >= 0.20:
                            criteria.append(f"✅ **EPS Acceleration (QoQ >= 20%):** Strong earnings growth confirms fundamentals.")
                    elif pd.notna(eps_change_pct) and eps_change_pct >= 20:
                        criteria.append(f"✅ **EPS Change % (>= 20%):** Strong earnings growth confirms fundamentals (using change_pct).")
                    if -20 <= row.get('from_high_pct', -100) <= -5:
                        criteria.append(f"✅ **Sweet Spot Zone:** Price is pulled back but near resistance ({row['from_high_pct']:.1f}% from 52w High).")
                    if (row.get('ret_1d', 0) > 0 and row.get('ret_3d', 0) > row.get('ret_1d', 0) and
                        row.get('ret_7d', 0) > row.get('ret_3d', 0) and row.get('ret_30d', 0) > 0):
                        criteria.append("✅ **Momentum Alignment:** Consistent and accelerating positive short-term returns.")
                    sector = row.get('sector')
                    # Recalculate sector ranks for this specific stock to display why it met the criteria
                    temp_df = df_analyzed.copy() # Use full analyzed df to get accurate sector ranks
                    temp_sector_ranks = {}
                    if 'sector' in temp_df.columns and 'EDGE' in temp_df.columns:
                        sector_agg_temp = temp_df.groupby('sector').agg(avg_edge=('EDGE', 'mean'), stock_count=('ticker', 'count')).reset_index()
                        sector_agg_temp = sector_agg_temp[(sector_agg_temp['stock_count'] >= 3) & (sector_agg_temp['avg_edge'] > 50)]
                        if not sector_agg_temp.empty:
                            sector_scores_sorted_temp = sector_agg_temp.sort_values('avg_edge', ascending=False)
                            for i, sec_name in enumerate(sector_scores_sorted_temp['sector']):
                                temp_sector_ranks[sec_name] = i + 1

                    if sector and sector in temp_sector_ranks and temp_sector_ranks[sector] <= 3:
                        criteria.append(f"✅ **Top Sector Strength:** Belongs to a top 3 performing sector ({sector}).")

                    if criteria:
                        for crit in criteria:
                            st.write(crit)
                        st.write(f"**Total Conditions Met:** {len(criteria)}/6 (Requires >=5)")
                    else:
                        st.info("No specific SUPER EDGE conditions met (this should not happen if stock is tagged SUPER EDGE).")

        else:
            st.info("No SUPER EDGE signals detected today. Check the 'Trading Signals' tab for other opportunities.")
            # Show next best opportunities (EXPLOSIVE)
            if 'tag' in df_signals.columns:
                explosive_df = df_signals[df_signals['tag'] == 'EXPLOSIVE'].head(5)
            else:
                explosive_df = pd.DataFrame()

            if not explosive_df.empty:
                st.subheader("🔥 Top EXPLOSIVE Opportunities (Next Best)")
                cols_exp = ['ticker', 'company_name', 'EDGE', 'price', 'volume_pattern', 'decision']
                cols_exp = [col for col in cols_exp if col in explosive_df.columns]
                if cols_exp:
                    st.dataframe(explosive_df[cols_exp].style.format({'EDGE': '{:.1f}', 'price': '₹{:.2f}'}), use_container_width=True)

    # Tab 3: Sector Leaders
    with tabs[2]:
        render_sector_leaderboard(df_analyzed)

    # Tab 4: Deep Analytics
    with tabs[3]:
        st.header("📊 Deep Market Analytics")

        # Volume Acceleration Distribution
        col_g1, col_g2 = st.columns(2)

        with col_g1:
            st.subheader("📈 Volume Acceleration Distribution")
            if 'volume_acceleration' in df_analyzed.columns:
                fig_vol_accel = go.Figure()
                fig_vol_accel.add_trace(go.Histogram(
                    x=df_analyzed['volume_acceleration'],
                    nbinsx=30,
                    name='All Stocks',
                    marker_color='lightblue',
                    opacity=0.7
                ))
                if not df_signals.empty and 'volume_acceleration' in df_signals.columns:
                    fig_vol_accel.add_trace(go.Histogram(
                        x=df_signals['volume_acceleration'],
                        nbinsx=20,
                        name='Signal Stocks',
                        marker_color='gold',
                        opacity=0.8
                    ))

                fig_vol_accel.update_layout(
                    xaxis_title="Volume Acceleration (%)",
                    yaxis_title="Number of Stocks",
                    height=400,
                    barmode='overlay',
                    legend_title_text='Dataset'
                )
                st.plotly_chart(fig_vol_accel, use_container_width=True)
            else:
                st.info("Volume acceleration data not available for charting.")

        with col_g2:
            st.subheader("🎯 EDGE Score Distribution")
            if 'EDGE' in df_analyzed.columns:
                fig_edge_dist = go.Figure()

                # Add colored regions for EDGE thresholds
                fig_edge_dist.add_vrect(x0=0, x1=EDGE_THRESHOLDS['MODERATE'], fillcolor="red", opacity=0.1, annotation_text="Watch", annotation_position="top left", layer="below")
                fig_edge_dist.add_vrect(x0=EDGE_THRESHOLDS['MODERATE'], x1=EDGE_THRESHOLDS['STRONG'], fillcolor="yellow", opacity=0.1, annotation_text="Moderate", annotation_position="top left", layer="below")
                fig_edge_dist.add_vrect(x0=EDGE_THRESHOLDS['STRONG'], x1=EDGE_THRESHOLDS['EXPLOSIVE'], fillcolor="orange", opacity=0.1, annotation_text="Strong", annotation_position="top left", layer="below")
                fig_edge_dist.add_vrect(x0=EDGE_THRESHOLDS['EXPLOSIVE'], x1=EDGE_THRESHOLDS['SUPER_EDGE'], fillcolor="green", opacity=0.1, annotation_text="Explosive", annotation_position="top left", layer="below")
                fig_edge_dist.add_vrect(x0=EDGE_THRESHOLDS['SUPER_EDGE'], x1=100, fillcolor="darkgreen", opacity=0.1, annotation_text="SUPER EDGE", annotation_position="top left", layer="below")


                fig_edge_dist.add_trace(go.Histogram(
                    x=df_analyzed['EDGE'],
                    nbinsx=25,
                    marker_color='darkblue'
                ))

                fig_edge_dist.update_layout(
                    xaxis_title="EDGE Score",
                    yaxis_title="Number of Stocks",
                    height=400
                )
                st.plotly_chart(fig_edge_dist, use_container_width=True)
            else:
                st.info("EDGE score data not available for charting.")

        st.markdown("---")
        # Pattern Analysis
        if 'pattern_name' in df_analyzed.columns:
            st.subheader("🎯 Pattern Detection Summary")

            # Filter out empty pattern names
            pattern_data = df_analyzed[df_analyzed['pattern_name'].notna() & (df_analyzed['pattern_name'] != '')]

            if len(pattern_data) > 0:
                pattern_summary = pattern_data['pattern_name'].value_counts()

                col_pat1, col_pat2 = st.columns([2, 1])

                with col_pat1:
                    fig_patterns = go.Figure(data=[go.Bar(
                        x=pattern_summary.index,
                        y=pattern_summary.values,
                        text=pattern_summary.values,
                        textposition='auto',
                        marker_color=['gold' if 'Accumulation' in p or 'Reversal' in p else 'lightblue' for p in pattern_summary.index]
                    )])

                    fig_patterns.update_layout(
                        xaxis_title="Pattern Type",
                        yaxis_title="Count",
                        height=350,
                        xaxis_tickangle=-45 # Rotate labels if too long
                    )
                    st.plotly_chart(fig_patterns, use_container_width=True)

                with col_pat2:
                    st.markdown("---")
                    st.write("**Pattern Strength & Multiplicity:**")
                    if 'pattern_score' in df_analyzed.columns:
                        strong_patterns = df_analyzed[df_analyzed['pattern_score'] >= 80]
                        st.metric("High Score (>80)", len(strong_patterns))
                    else:
                        st.metric("High Score (>80)", 0)

                    if 'pattern_count' in df_analyzed.columns:
                        multi_patterns = (df_analyzed['pattern_count'] > 1).sum()
                        st.metric("Multiple Patterns", multi_patterns) # Stocks with more than one strong pattern
                    else:
                        st.metric("Multiple Patterns", 0)
            else:
                st.info("No significant patterns detected in current market conditions.")
        st.markdown("---")
        # Market Breadth
        st.subheader("📊 Market Breadth Analysis")

        breadth_metrics = {}
        total_stocks_analyzed = len(df_analyzed)

        if total_stocks_analyzed > 0:
            if 'ret_1d' in df_analyzed.columns:
                breadth_metrics['Stocks Advancing (1D)'] = (df_analyzed['ret_1d'] > 0).sum()
                breadth_metrics['Stocks Declining (1D)'] = (df_analyzed['ret_1d'] < 0).sum()
            if 'price' in df_analyzed.columns and 'sma_50d' in df_analyzed.columns:
                breadth_metrics['Above 50 SMA'] = (df_analyzed['price'] > df_analyzed['sma_50d']).sum()
            if 'price' in df_analyzed.columns and 'sma_200d' in df_analyzed.columns:
                breadth_metrics['Above 200 SMA'] = (df_analyzed['price'] > df_analyzed['sma_200d']).sum()
            if 'rvol' in df_analyzed.columns:
                breadth_metrics['High Volume (RVOL > 1.5)'] = (df_analyzed['rvol'] > 1.5).sum()
            if 'volume_acceleration' in df_analyzed.columns:
                breadth_metrics['Strong Accumulation Patterns'] = (df_analyzed['volume_acceleration'] > 20).sum()

        col_b1, col_b2, col_b3 = st.columns(3)
        for i, (metric, value) in enumerate(breadth_metrics.items()):
            with [col_b1, col_b2, col_b3][i % 3]:
                if total_stocks_analyzed > 0:
                    pct = safe_divide(value * 100, total_stocks_analyzed, 0)
                    st.metric(metric, f"{value} ({pct:.0f}%)")
                else:
                    st.metric(metric, "0 (0%)")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    render_ui()
