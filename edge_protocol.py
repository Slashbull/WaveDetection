import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import requests
import math
import warnings
import re
from functools import lru_cache
from scipy import stats
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

PAGE_TITLE = "EDGE Protocol – Ultimate Trading Intelligence"

# Define different trading profiles with their respective weights for the four components:
# (Volume, Momentum, Risk/Reward, Fundamentals)
PROFILE_PRESETS = {
    "Balanced": (0.35, 0.25, 0.20, 0.20),  # Balanced approach
    "Swing": (0.45, 0.35, 0.15, 0.05),     # Higher emphasis on Volume and Momentum for short-term trades
    "Positional": (0.30, 0.20, 0.30, 0.20), # More weight on Risk/Reward and Fundamentals for longer holds
    "Momentum-only": (0.50, 0.40, 0.10, 0.00), # Pure momentum play, less on long-term factors
    "Breakout": (0.40, 0.40, 0.15, 0.05),  # Strong emphasis on Volume and Momentum for breakouts
    "Long-Term": (0.15, 0.15, 0.35, 0.35), # Focus on Risk/Reward and strong Fundamentals for long-term investments
}

# EDGE Scoring Thresholds for classification
EDGE_THRESHOLDS = {
    "SUPER_EDGE": 92,  # Score needed to be considered for SUPER EDGE
    "EXPLOSIVE": 85,
    "STRONG": 70,
    "MODERATE": 50,
    "WATCH": 0
}

# Portfolio Risk Management Parameters
MAX_PORTFOLIO_EXPOSURE = 0.80  # 80% max exposure
MAX_POSITIONS = 10             # Maximum concurrent positions
MAX_SECTOR_EXPOSURE = 0.30     # 30% max in one sector
MAX_SUPER_EDGE_POSITIONS = 3   # Max 3 SUPER EDGE at once
MIN_STOCKS_PER_SECTOR = 3      # Minimum stocks required for sector to be included in leaderboard

# Position Sizing Multipliers (relative to base)
# These are base percentages, adjusted dynamically by risk_adjustment from stop loss
BASE_POSITION_SIZES = {
    "SUPER_EDGE": 0.15,  # 15% of portfolio for a SUPER_EDGE stock if all conditions met
    "EXPLOSIVE": 0.10,   # 10% for EXPLOSIVE
    "STRONG": 0.05,      # 5% for STRONG
    "MODERATE": 0.02,    # 2% for MODERATE (small exploratory)
    "WATCH": 0.00        # 0% for WATCH (no position)
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
        'cr': 1e7, 'crore': 1e7,                  # Indian Crore
        'l': 1e5, 'lakh': 1e5, 'lac': 1e5         # Indian Lakh
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

def winsorise_series(s: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    """
    Winsorises a pandas Series to cap outliers at specified quantiles.
    Useful for preventing extreme values from skewing metrics.
    """
    if s.empty or not pd.api.types.is_numeric_dtype(s):
        return s
    # Calculate quantiles, handling cases where there aren't enough unique values
    lo = s.quantile(lower_q) if s.nunique() > 1 else s.min()
    hi = s.quantile(upper_q) if s.nunique() > 1 else s.max()
    return s.clip(lo, hi)

def calc_atr20(price_series: pd.Series) -> pd.Series:
    """
    Calculates a proxy for Average True Range (ATR) over 20 periods.
    Uses rolling standard deviation as a volatility measure.
    """
    # Ensure price_series is numeric
    if not pd.api.types.is_numeric_dtype(price_series):
        return pd.Series(np.nan, index=price_series.index)

    # Calculate rolling standard deviation, fill NaNs with the mean of the series
    rolling_std = price_series.rolling(20, min_periods=1).std()
    return rolling_std.fillna(rolling_std.mean()) * math.sqrt(2) # Scale to approximate ATR

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
            'ticker', 'company_name', 'price', 'volume_1d', 'ret_1d',
            'vol_ratio_7d_90d', 'vol_ratio_30d_90d', 'vol_ratio_30d_180d',
            'low_52w', 'high_52w', 'from_low_pct', 'from_high_pct',
            'sma_20d', 'sma_50d', 'sma_200d', 'rvol', 'sector',
            'market_cap', 'eps_current', 'eps_last_qtr', 'eps_change_pct' # Added for fundamentals
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
            'eps_current', 'eps_last_qtr', 'eps_change_pct', 'year'
        ]

        # Clean and convert numeric columns
        for col in numeric_cols:
            if col in df.columns:
                if col == 'market_cap': # Special handling for market_cap parsing
                    df[col] = df[col].astype(str).apply(parse_market_cap)
                else:
                    # Replace non-numeric strings with NaN, then convert
                    df[col] = df[col].astype(str).str.replace(r"[₹,$€£%a-zA-Z]", "", regex=True).replace(["", "-", "nan", "NaN", "NA"], np.nan)
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Fill NaN for ratios/volumes to 0 if that makes sense for calculation, otherwise leave as NaN
                if 'vol_ratio' in col or 'volume_' in col or col == 'rvol':
                    df[col] = df[col].fillna(0) # Treat missing volume data as 0 for ratio calculations

        # Ensure string columns
        for col in ['ticker', 'company_name', 'sector', 'category']:
            if col in df.columns:
                df[col] = df[col].astype(str).fillna("Unknown")
            else:
                # Add missing string columns with 'Unknown' default to prevent errors later
                df[col] = "Unknown"
                diagnostics["warnings"].append(f"Missing non-critical column: '{col}'. Defaulted to 'Unknown'.")

        # Specific handling for 'market_cap' - ensure it exists and is numeric
        if 'market_cap' not in df.columns or not pd.api.types.is_numeric_dtype(df['market_cap']):
            diagnostics["warnings"].append("Missing or invalid 'market_cap' column. Category filtering might be impacted.")
            df['market_cap_num'] = np.nan # Ensure column exists even if missing
        else:
            df['market_cap_num'] = df['market_cap'] # Use the already parsed numeric market_cap

        # Map market cap to categories for consistent filtering
        def map_market_category(mc: float) -> str:
            if pd.isna(mc): return 'Unknown'
            # These thresholds are typical for USD. Adjust if your market_cap data is in INR.
            # For INR: Large Cap > ~20000 Cr, Mid Cap > ~5000 Cr, Small Cap > ~1000 Cr
            if mc >= 200 * 1e9: return 'Large Cap' # 200 Billion USD approx
            if mc >= 50 * 1e9: return 'Mid Cap'
            if mc >= 1 * 1e9: return 'Small Cap'
            return 'Micro Cap'

        if 'market_cap_num' in df.columns:
            df['category'] = df['market_cap_num'].apply(map_market_category)
        else:
            df['category'] = 'Unknown' # Default if market_cap_num is missing

        # Calculate derived columns for immediate use or later scoring
        df["atr_20"] = calc_atr20(df["price"])
        # rs_volume_30d is a rough proxy for 30-day average dollar volume
        df["rs_volume_30d"] = df.get("volume_30d", pd.Series(0, index=df.index)) * df["price"]

        # Apply winsorization to critical numeric columns to handle extreme outliers
        for col in ['ret_1d', 'ret_3d', 'ret_7d', 'ret_30d', 'ret_3m', 'ret_6m', 'ret_1y', 'ret_3y', 'ret_5y',
                    'rvol', 'volume_acceleration', 'from_low_pct', 'from_high_pct', 'eps_change_pct']:
            if col in df.columns:
                df[col] = winsorise_series(df[col], lower_q=0.01, upper_q=0.99)

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
# VOLUME METRICS CALCULATION
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
        # Sum how many of these ratios are positive, then normalize to 0-1
        df_copy['volume_consistency'] = (df_copy[vol_ratio_cols_to_check] > 0).sum(axis=1) / len(vol_ratio_cols_to_check)
    else:
        df_copy['volume_consistency'] = 0.0

    # Volume intensity (RVOL * volume acceleration, scaled)
    # Ensure volume_acceleration is positive for intensity, cap RVOL
    df_copy['volume_intensity'] = df_copy['rvol'].clip(0.5, 5.0) * np.maximum(df_copy['volume_acceleration'], 0) / 100
    df_copy['volume_intensity'] = df_copy['volume_intensity'].fillna(0) # Fill NaN from rvol or volume_acceleration

    # Classification of volume patterns based on acceleration and RVOL
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

    df_copy['volume_classification'] = np.select(conditions, choices, default="💀 Heavy Distribution")

    return df_copy

# ============================================================================
# PATTERN DETECTION FUNCTIONS
# These functions detect specific price-volume patterns and assign a score.
# ============================================================================
def detect_accumulation_under_resistance(row: pd.Series) -> dict:
    """
    Pattern 1: Volume explodes but price stays flat near resistance (52-week high).
    Indicates strong buying interest absorbing supply without a significant price breakout yet.
    """
    score = 0
    signals = []
    
    vol_ratio_30d_90d = row.get('vol_ratio_30d_90d', 0)
    ret_30d = row.get('ret_30d', 0)
    from_high_pct = row.get('from_high_pct', -100) # % from 52w high, negative means below
    rvol = row.get('rvol', 1)
    
    # Volume explosion: Significant increase in 30-day volume relative to 90-day
    if vol_ratio_30d_90d > 50:
        score += 40
        signals.append(f"Volume surge +{vol_ratio_30d_90d:.0f}% (30d/90d)")
    elif vol_ratio_30d_90d > 30:
        score += 25
        signals.append(f"Volume increase +{vol_ratio_30d_90d:.0f}% (30d/90d)")
    
    # Price flat: Price consolidating without large moves in the last 30 days
    if abs(ret_30d) < 5:
        score += 30
        signals.append(f"Price consolidating ({ret_30d:.1f}% in 30d)")
    
    # Near resistance: Price is close to its 52-week high
    if -5 <= from_high_pct <= 0: # Very close to high, possibly testing it
        score += 30
        signals.append("At 52-week high resistance")
    elif -15 <= from_high_pct < -5: # Pulled back slightly but still near resistance
        score += 20
        signals.append("Near 52-week high resistance")
    
    # RVOL bonus: High relative volume confirms strong interest
    if rvol > 2.0:
        score = min(score * 1.2, 100) # Boost score, cap at 100
        signals.append(f"RVOL: {rvol:.1f}x confirms strength")
    
    return {
        'pattern': 'Accumulation Under Resistance',
        'score': score,
        'signals': signals,
        'target': row.get('high_52w', row.get('price', 0)) * 1.05 # Target slightly above 52w high
    }

def detect_coiled_spring(row: pd.Series) -> dict:
    """
    Pattern 2: Volume increases but price remains in a tight range, typically above key moving averages.
    Suggests a period of low volatility compression before a potential explosive move.
    """
    score = 0
    signals = []
    
    vol_ratio_30d_90d = row.get('vol_ratio_30d_90d', 0)
    ret_7d = row.get('ret_7d', 0)
    ret_30d = row.get('ret_30d', 0)
    price = row.get('price', 1)
    sma_20d = row.get('sma_20d', price)
    sma_50d = row.get('sma_50d', price)
    sma_200d = row.get('sma_200d', price)
    
    # Volume increase: Building volume over the medium term
    if vol_ratio_30d_90d > 30:
        score += 35
        signals.append(f"Volume building +{vol_ratio_30d_90d:.0f}% (30d/90d)")
    elif vol_ratio_30d_90d > 15:
        score += 20
        signals.append(f"Mild volume increase +{vol_ratio_30d_90d:.0f}% (30d/90d)")
    
    # Tight range: Low recent price volatility
    if abs(ret_7d) < 4 and abs(ret_30d) < 8: # Small absolute returns over 7 and 30 days
        score += 35
        signals.append(f"Tight price consolidation ({ret_7d:.1f}% 7d, {ret_30d:.1f}% 30d)")
    
    # Above SMAs: Price is trending above key moving averages, indicating underlying strength
    sma_count = 0
    if price > sma_20d: sma_count += 1
    if price > sma_50d: sma_count += 1
    if price > sma_200d: sma_count += 1
    
    if sma_count == 3:
        score += 30
        signals.append("Price above 20d, 50d, and 200d SMAs")
    elif sma_count == 2:
        score += 15
        signals.append("Price above 2 key SMAs")
        
    return {
        'pattern': 'Coiled Spring',
        'score': score,
        'signals': signals,
        'target': price * 1.08 # Moderate target for initial breakout
    }

def detect_absorption_pattern(row: pd.Series) -> dict:
    """
    Pattern 3: High Relative Volume (RVOL) with price stability.
    Suggests that large orders are being "absorbed" by institutional buyers without causing
    a significant price increase, indicating strong underlying demand.
    """
    score = 0
    signals = []
    
    rvol = row.get('rvol', 1)
    ret_7d = row.get('ret_7d', 0)
    vol_ratio_1d_90d = row.get('vol_ratio_1d_90d', 0)
    vol_ratio_7d_90d = row.get('vol_ratio_7d_90d', 0)
    vol_ratio_30d_90d = row.get('vol_ratio_30d_90d', 0)
    
    # High RVOL: Significantly higher volume than average
    if rvol > 2.5:
        score += 50
        signals.append(f"Extreme RVOL: {rvol:.1f}x")
    elif rvol > 1.5:
        score += 30
        signals.append(f"High RVOL: {rvol:.1f}x")
    
    # Consistent volume: Positive volume ratios across multiple timeframes
    positive_ratios = sum([1 for r in [vol_ratio_1d_90d, vol_ratio_7d_90d, vol_ratio_30d_90d] if r > 0])
    if positive_ratios == 3:
        score += 30
        signals.append("Sustained volume increase across timeframes")
    
    # Price absorption: Price remains stable despite high volume
    if abs(ret_7d) < 3: # Very small price movement over 7 days
        score += 20
        signals.append("Price absorbed despite high volume")
        
    return {
        'pattern': 'Absorption Pattern',
        'score': score,
        'signals': signals,
        'target': row.get('price', 0) * 1.10 # Moderate target
    }

def detect_failed_breakdown_reversal(row: pd.Series) -> dict:
    """
    Pattern 4: Price is near its 52-week low but shows strong recent volume acceleration
    and positive short-term momentum, indicating a potential reversal from a failed breakdown.
    """
    score = 0
    signals = []
    
    from_low_pct = row.get('from_low_pct', 100) # % from 52w low, positive means above
    volume_acceleration = row.get('volume_acceleration', 0)
    ret_7d = row.get('ret_7d', 0)
    ret_3y = row.get('ret_3y', 0) # Long-term return as a quality proxy
    
    # Near 52w low: Price has recently bounced from or is close to its 52-week low
    if from_low_pct < 10 and from_low_pct > 0: # Bounced slightly from low
        score += 40
        signals.append(f"Near 52-week low (+{from_low_pct:.0f}% from low)")
    elif from_low_pct < 20 and from_low_pct > 0: # More significant bounce but still low
        score += 25
        signals.append(f"Recent bounce from low (+{from_low_pct:.0f}% from low)")
    
    # Volume acceleration: Strong recent volume increase confirming buying interest
    if volume_acceleration > 20:
        score += 40
        signals.append(f"Volume acceleration: {volume_acceleration:.0f}%")
    elif volume_acceleration > 10:
        score += 25
        signals.append(f"Moderate volume acceleration: {volume_acceleration:.0f}%")
    
    # Momentum reversal: Positive short-term price movement
    if ret_7d > 0:
        score += 20
        signals.append(f"Positive 7-day momentum ({ret_7d:.1f}%)")
    
    # Quality bonus: Historical strong performance suggests a quality company reversing
    if ret_3y > 200: # Very strong 3-year return
        score = min(score * 1.2, 100)
        signals.append("Quality stock with historical strong returns")
        
    return {
        'pattern': 'Failed Breakdown Reversal',
        'score': score,
        'signals': signals,
        'target': row.get('sma_50d', row.get('price', 0) * 1.15) # Target towards 50-day SMA or higher
    }

def detect_stealth_breakout(row: pd.Series) -> dict:
    """
    Pattern 5: Quiet strength building, where price is consistently above key moving averages
    with gradual volume increase and steady price climb, indicating a low-key accumulation.
    """
    score = 0
    signals = []
    
    price = row.get('price', 1)
    sma_20d = row.get('sma_20d', price)
    sma_50d = row.get('sma_50d', price)
    sma_200d = row.get('sma_200d', price)
    vol_ratio_30d_90d = row.get('vol_ratio_30d_90d', 0)
    ret_30d = row.get('ret_30d', 0)
    
    # Above all SMAs: Strong bullish alignment of moving averages
    sma_count = 0
    if price > sma_20d: sma_count += 1
    if price > sma_50d: sma_count += 1
    if price > sma_200d: sma_count += 1
    
    if sma_count == 3:
        score += 40
        signals.append("Price above 20d, 50d, and 200d SMAs (strong trend)")
    elif sma_count == 2:
        score += 25
        signals.append("Price above 2 key SMAs (developing trend)")
    
    # Gradual volume: Consistent, but not explosive, volume increase
    if 10 <= vol_ratio_30d_90d <= 30:
        score += 30
        signals.append(f"Gradual volume increase ({vol_ratio_30d_90d:.0f}% 30d/90d)")
    
    # Steady climb: Consistent positive price movement without large spikes
    if 5 <= ret_30d <= 15:
        score += 30
        signals.append(f"Steady price climb ({ret_30d:.1f}% in 30d)")
        
    return {
        'pattern': 'Stealth Breakout',
        'score': score,
        'signals': signals,
        'target': row.get('high_52w', price * 1.20) # Target towards 52w high or higher
    }

def detect_pre_earnings_accumulation(row: pd.Series) -> dict:
    """
    Pattern 6: Unusual volume activity combined with strong EPS momentum,
    often seen before a positive earnings report.
    """
    score = 0
    signals = []
    
    eps_current = row.get('eps_current', np.nan)
    eps_last_qtr = row.get('eps_last_qtr', np.nan)
    eps_change_pct = row.get('eps_change_pct', np.nan) # Could be YoY or other
    vol_ratio_7d_90d = row.get('vol_ratio_7d_90d', 0)
    from_high_pct = row.get('from_high_pct', 0)
    
    # EPS momentum: Strong recent EPS growth (QoQ preferred, else general change)
    eps_accel = 0.0
    if pd.notna(eps_current) and pd.notna(eps_last_qtr) and eps_last_qtr != 0:
        eps_accel = safe_divide((eps_current - eps_last_qtr), eps_last_qtr) * 100
    elif pd.notna(eps_change_pct): # Fallback if QoQ not available
        eps_accel = eps_change_pct

    if eps_accel > 10: # At least 10% EPS growth
        score += 35
        signals.append(f"Strong EPS acceleration: {eps_accel:.0f}%")
    
    # Recent volume spike: Significant volume increase in the last 7 days
    if vol_ratio_7d_90d > 50:
        score += 35
        signals.append(f"Recent volume spike (+{vol_ratio_7d_90d:.0f}% 7d/90d)")
    elif vol_ratio_7d_90d > 25:
        score += 20
        signals.append(f"Higher recent volume (+{vol_ratio_7d_90d:.0f}% 7d/90d)")
    
    # Accumulation zone: Price is in a healthy pullback zone, not extended
    if -20 <= from_high_pct <= -5: # Pulled back but still in an uptrend
        score += 30
        signals.append(f"Price in accumulation zone ({from_high_pct:.1f}% from high)")
    
    return {
        'pattern': 'Pre-Earnings Accumulation',
        'score': score,
        'signals': signals,
        'target': row.get('price', 0) * 1.12 # Target for earnings-driven pop
    }

def detect_all_patterns(row: pd.Series) -> dict:
    """
    Runs all defined pattern detection functions for a given stock.
    Calculates a confluence score based on the number of high-scoring patterns.
    Also calculates Volume-Price Divergence.
    """
    patterns = [
        detect_accumulation_under_resistance(row),
        detect_coiled_spring(row),
        detect_absorption_pattern(row),
        detect_failed_breakdown_reversal(row),
        detect_stealth_breakout(row),
        detect_pre_earnings_accumulation(row)
    ]
    
    # Sort patterns by score in descending order
    patterns.sort(key=lambda x: x['score'], reverse=True)
    
    # Calculate confluence score based on patterns with a score >= 70
    high_score_patterns = [p for p in patterns if p['score'] >= 70]
    
    confluence_score = 0
    confluence_signals = []
    
    if len(high_score_patterns) >= 3:
        confluence_score = 100 # Ultra high conviction
        confluence_signals.append(f"🔥 {len(high_score_patterns)} PATTERNS ALIGNED! ULTRA HIGH CONVICTION!")
    elif len(high_score_patterns) >= 2:
        confluence_score = 85 # Very strong conviction
        confluence_signals.append(f"📈 {len(high_score_patterns)} patterns converging. Very strong.")
    elif len(high_score_patterns) >= 1:
        confluence_score = 70 # Strong conviction
        confluence_signals.append("🎯 Strongest pattern detected. Good opportunity.")
    
    # Volume-Price Divergence: High volume with relatively flat price suggests accumulation/absorption
    # A high positive value indicates accumulation, a high negative value indicates distribution
    vol_ratio_30d_90d = row.get('vol_ratio_30d_90d', 0)
    ret_30d = row.get('ret_30d', 0)
    
    # Avoid division by zero and scale the divergence
    # If price is flat (ret_30d close to 0), and volume is high, divergence is high.
    # If price is moving strongly with low volume, divergence is low.
    vp_divergence = safe_divide(vol_ratio_30d_90d, abs(ret_30d) + 0.01) * 2
    vp_divergence = np.clip(vp_divergence, -100, 100) # Clip to a reasonable range
    
    return {
        'all_patterns': patterns, # Return all patterns for detailed view
        'top_pattern': patterns[0] if patterns else None,
        'confluence_score': confluence_score,
        'confluence_signals': confluence_signals,
        'vp_divergence': vp_divergence,
        'high_score_patterns': high_score_patterns
    }

# ============================================================================
# CORE EDGE SCORING COMPONENTS
# ============================================================================
def score_vol_accel(row: pd.Series) -> float:
    """
    Scores Volume Acceleration based on recent vs. past volume ratios and RVOL.
    Higher score for stronger and more consistent acceleration.
    """
    vol_accel = row.get("volume_acceleration", 0)
    rvol = row.get("rvol", 1)
    volume_consistency = row.get("volume_consistency", 0)
    
    # Base score from volume acceleration, scaled to roughly 0-100
    base_score = 50 + (vol_accel * 0.5) # More sensitive scaling
    base_score = np.clip(base_score, 0, 100)
    
    # RVOL bonus: Significant boost for high RVOL on positive acceleration
    if rvol > 2.0 and vol_accel > 20:
        base_score = min(base_score * 1.5, 100)
    elif rvol > 1.5 and vol_accel > 10:
        base_score = min(base_score * 1.2, 100)
    
    # Volume consistency bonus: More consistent accumulation is better
    base_score += (volume_consistency * 15) # Up to 15 points bonus
    
    return np.clip(base_score, 0, 100)

def score_momentum(row: pd.Series) -> float:
    """
    Scores Momentum based on short-term and medium-term price returns.
    Includes a bonus for consistent accelerating momentum.
    """
    ret_1d = row.get('ret_1d', 0)
    ret_3d = row.get('ret_3d', 0)
    ret_7d = row.get('ret_7d', 0)
    ret_30d = row.get('ret_30d', 0)
    
    # Simple momentum score: Prioritize recent momentum
    short_term_mom = (ret_1d * 0.4) + (ret_3d * 0.3) + (ret_7d * 0.3)
    mid_term_mom = ret_30d
    
    # Combined momentum, scaled to influence score
    momentum_combined = (short_term_mom * 0.6) + (mid_term_mom * 0.4)
    base_score = 50 + (momentum_combined * 2) # Scale to roughly 0-100 range
    base_score = np.clip(base_score, 0, 100)
    
    # Consistency bonus: All recent returns positive and accelerating
    if (ret_1d > 0 and 
        ret_3d > ret_1d and
        ret_7d > ret_3d and 
        ret_30d > 0):
        base_score = min(base_score * 1.2, 100) # Significant bonus
        
    return np.clip(base_score, 0, 100)

def score_risk_reward(row: pd.Series) -> float:
    """
    Scores Risk/Reward based on proximity to 52-week high/low and long-term quality.
    Higher score for favorable risk/reward profiles (more upside, less downside).
    """
    price = row.get("price", 1)
    high_52w = row.get("high_52w", price)
    low_52w = row.get("low_52w", price)
    from_high_pct = row.get("from_high_pct", 0) # % from 52w high
    from_low_pct = row.get("from_low_pct", 0)   # % from 52w low
    
    # Calculate potential upside and downside based on 52-week range
    # Ensure no division by zero or negative values
    upside_potential = safe_divide(high_52w - price, price) * 100
    downside_risk = safe_divide(price - low_52w, price) * 100
    
    # Initial score based on upside potential vs. downside risk
    # Add 1 to downside_risk to avoid division by zero and give a floor
    rr_ratio = safe_divide(upside_potential, downside_risk + 1)
    base_score = np.clip(rr_ratio * 10, 0, 100) # Scale, higher ratio is better

    # Sweet spot bonus: Price is pulled back but still in a healthy uptrend
    if -20 <= from_high_pct <= -5: # Healthy pullback zone
        base_score = min(base_score * 1.2, 100)
    elif from_high_pct > -5: # Too close to high, potential for immediate pullback
        base_score = base_score * 0.8 # Penalty
    elif from_high_pct < -40: # Too far from high, potentially broken trend
        base_score = base_score * 0.7 # Larger penalty
        
    # Quality bonus: Strong long-term performance indicates a resilient stock
    ret_3y = row.get("ret_3y", 0)
    ret_1y = row.get("ret_1y", 0)
    if ret_3y > 300 and ret_1y < 50: # Very strong 3-year return, but not overextended in 1 year
        base_score = min(base_score * 1.3, 100) # Significant boost for quality on sale
        
    return np.clip(base_score, 0, 100)

def score_fundamentals(row: pd.Series) -> float:
    """
    Scores Fundamentals based on EPS growth, PE ratio, and QoQ EPS acceleration.
    Higher score for strong, improving fundamentals at a reasonable valuation.
    """
    scores = []
    
    # EPS change (general percentage change, could be YoY)
    eps_change_pct = row.get("eps_change_pct", np.nan)
    if pd.notna(eps_change_pct):
        eps_change_score = 50 + (eps_change_pct * 0.5) # Scale around 50
        scores.append(np.clip(eps_change_score, 0, 100))
    
    # PE ratio (Valuation)
    pe = row.get("pe", np.nan)
    if pd.notna(pe) and pe > 0:
        if pe <= 20: # Very attractive PE
            pe_score = 100
        elif pe <= 35: # Attractive PE
            pe_score = 100 - ((pe - 20) * 2)
        elif pe <= 50: # Moderate PE
            pe_score = 70 - ((pe - 35) * 1)
        else: # High PE
            pe_score = 20
        scores.append(np.clip(pe_score, 0, 100))
    
    # EPS acceleration (Quarter-over-Quarter)
    eps_current = row.get("eps_current", np.nan)
    eps_last_qtr = row.get("eps_last_qtr", np.nan)
    if pd.notna(eps_current) and pd.notna(eps_last_qtr) and eps_last_qtr != 0:
        eps_qoq_accel = safe_divide(eps_current - eps_last_qtr, eps_last_qtr) * 100
        if eps_qoq_accel > 15: # Strong QoQ growth
            scores.append(min(eps_qoq_accel * 3, 100)) # Scale for significant impact
    
    # Return average score, default to 50 if no valid fundamental data
    return np.mean(scores) if scores else 50.0

# ============================================================================
# SUPER EDGE DETECTION
# Uses stricter criteria for highest conviction signals.
# ============================================================================
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

    # 3. EPS acceleration (Quarter-over-Quarter growth or general change)
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
    # Use a placeholder if sector_ranks is empty or sector is not found
    sector_rank = sector_ranks.get(sector, 999) if sector else 999
    if sector_rank <= 3:
        conditions_met += 1

    return conditions_met >= 5 # Require 5 out of 6 conditions for SUPER EDGE

# ============================================================================
# DYNAMIC STOP LOSS CALCULATION
# ============================================================================
def calculate_dynamic_stops(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates intelligent stop losses based on price, volatility (ATR proxy), and support levels (SMAs, 52w low).
    Also calculates a risk adjustment factor for position sizing based on stop distance.
    """
    df_copy = df.copy()

    # Define default percentages for ATR-like stop based on category
    # These are percentage drops from current price, serving as a base volatility stop
    category_atr_pct = {
        'micro cap': 0.12, # 12% stop for micro caps (higher volatility)
        'small cap': 0.10, # 10% stop for small caps
        'mid cap': 0.08,   # 8% stop for mid caps
        'large cap': 0.07, # 7% stop for large caps (lower volatility)
        'unknown': 0.09    # Default for unknown category
    }

    df_copy['stop_loss'] = np.nan
    df_copy['stop_loss_pct'] = np.nan
    df_copy['risk_adjustment'] = 1.0 # Default to no adjustment (full size)

    for idx in df_copy.index:
        row = df_copy.loc[idx]
        price = row.get('price')
        if pd.isna(price) or price <= 0:
            continue # Skip invalid prices

        category = str(row.get('category', 'unknown')).lower()
        # Calculate ATR-based stop price using the category-specific factor
        default_atr_factor = 1.0 - category_atr_pct.get(category, 0.09)
        atr_stop_price = price * default_atr_factor

        # Method 2: Support-based stop levels (SMAs and 52-week low)
        support_levels = []

        # SMA support (with a slight buffer below the SMA)
        sma_20d = row.get('sma_20d')
        sma_50d = row.get('sma_50d')
        sma_200d = row.get('sma_200d')

        # Only consider SMAs as support if the current price is above them
        if pd.notna(sma_20d) and price > sma_20d:
            support_levels.append(sma_20d * 0.98) # 2% below 20 SMA
        if pd.notna(sma_50d) and price > sma_50d:
            support_levels.append(sma_50d * 0.97) # 3% below 50 SMA
        if pd.notna(sma_200d) and price > sma_200d:
            support_levels.append(sma_200d * 0.96) # 4% below 200 SMA

        # 52-week low as a hard support (with a bounce buffer)
        low_52w = row.get('low_52w')
        if pd.notna(low_52w) and low_52w > 0:
            # If price has recently bounced from 52w low, allow a tighter stop to protect gains
            if row.get('from_low_pct', 100) < 15: # If bounced less than 15% from low
                 support_levels.append(low_52w * 1.03) # 3% above 52w low
            else:
                support_levels.append(low_52w * 1.05) # 5% above 52w low for more established bounces

        # Determine the support-based stop price: take the highest (closest to current price) of valid support levels
        if support_levels:
            # Cap the support-based stop at 5% below current price to prevent stops that are too tight
            support_stop_price = max(support_levels)
            support_stop_price = min(support_stop_price, price * 0.95)
        else:
            support_stop_price = atr_stop_price # Fallback if no valid support levels found

        # Final stop loss: Take the higher (closer to current price, offering better protection)
        # of the ATR-based stop or the calculated support-based stop.
        # This ensures the stop is responsive to volatility but also respects structural support.
        final_stop = max(atr_stop_price, support_stop_price)

        # Ensure the final stop is not above the current price and is within reasonable bounds
        final_stop = min(final_stop, price * 0.95) # Max 5% drop from current price for stop (adjust as needed)
        final_stop = max(final_stop, price * 0.70) # Min 30% drop (preventing absurdly wide stops on volatile stocks)

        df_copy.loc[idx, 'stop_loss'] = final_stop
        df_copy.loc[idx, 'stop_loss_pct'] = safe_divide((final_stop - price) * 100, price, 0)

        # Risk-based position sizing adjustment: reduce size for higher risk (wider stop)
        stop_distance_pct = abs(df_copy.loc[idx, 'stop_loss_pct'])
        if stop_distance_pct > 10: # Wider than 10%
            df_copy.loc[idx, 'risk_adjustment'] = 0.6 # Significant reduction
        elif stop_distance_pct > 7: # Wider than 7%
            df_copy.loc[idx, 'risk_adjustment'] = 0.8
        else: # 7% or tighter
            df_copy.loc[idx, 'risk_adjustment'] = 1.0 # Full size

    return df_copy

# ============================================================================
# PORTFOLIO RISK MANAGEMENT AND POSITION SIZING
# ============================================================================
def apply_portfolio_constraints(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies portfolio-level risk constraints, dynamically adjusting position sizes
    based on overall exposure, maximum positions, and sector concentration.
    Prioritizes higher EDGE score stocks for allocation.
    """
    df_copy = df.copy()

    # Initialize position_size and portfolio_weight columns
    df_copy['position_size'] = 0.0
    df_copy['portfolio_weight'] = 0.0

    # Sort by EDGE score (descending) to prioritize higher conviction trades for allocation
    df_copy = df_copy.sort_values('EDGE', ascending=False).reset_index(drop=True)

    total_allocation = 0.0
    sector_allocations: Dict[str, float] = {}
    position_count = 0
    super_edge_count = 0

    for idx in df_copy.index:
        row = df_copy.loc[idx]
        tag = row.get('tag', 'WATCH')
        sector = row.get('sector', 'Unknown')
        risk_adjustment = row.get('risk_adjustment', 1.0) # From dynamic stop loss calculation

        # Get base size for the signal tag, adjust by risk factor
        base_size = BASE_POSITION_SIZES.get(tag, 0.0)
        adjusted_size = base_size * risk_adjustment

        # Do not allocate if base_size is 0 or adjusted_size becomes very small after adjustment
        if adjusted_size < 0.005: # Minimum allocation threshold (0.5% of portfolio)
            continue

        # Check total position count constraint
        if position_count >= MAX_POSITIONS:
            continue

        # Check SUPER EDGE specific limit
        if tag == 'SUPER_EDGE':
            if super_edge_count >= MAX_SUPER_EDGE_POSITIONS:
                # If SUPER EDGE limit reached, downgrade allocation to a STRONG size or skip
                adjusted_size = min(adjusted_size, BASE_POSITION_SIZES.get('STRONG', 0.05) * risk_adjustment)
                if adjusted_size < 0.005: # Re-check if it's still worth allocating after downgrade
                    continue
            super_edge_count += 1

        # Check portfolio total exposure constraint
        remaining_portfolio_capacity = MAX_PORTFOLIO_EXPOSURE - total_allocation
        if adjusted_size > remaining_portfolio_capacity:
            adjusted_size = remaining_portfolio_capacity # Limit allocation to remaining capacity

        # Check sector concentration constraint
        current_sector_allocation = sector_allocations.get(sector, 0.0)
        remaining_sector_capacity = MAX_SECTOR_EXPOSURE - current_sector_allocation
        if adjusted_size > remaining_sector_capacity:
            adjusted_size = remaining_sector_capacity # Limit allocation to remaining sector capacity

        # If after all checks, adjusted_size is still positive, apply it
        if adjusted_size > 0.005: # Final check for minimum allocation
            df_copy.loc[idx, 'position_size'] = adjusted_size
            df_copy.loc[idx, 'portfolio_weight'] = adjusted_size # For now, same as position size
            total_allocation += adjusted_size
            sector_allocations[sector] = current_sector_allocation + adjusted_size
            position_count += 1
        else:
            # If allocation is too small or couldn't be made, set to zero
            df_copy.loc[idx, 'position_size'] = 0.0
            df_copy.loc[idx, 'portfolio_weight'] = 0.0

    # Store overall portfolio allocation and position count (useful for UI display)
    df_copy['total_portfolio_allocation'] = total_allocation
    df_copy['current_positions_count'] = position_count

    return df_copy

# ============================================================================
# MAIN SCORING PIPELINE
# ============================================================================
def run_edge_analysis(df: pd.DataFrame, weights: Tuple[float, float, float, float]) -> pd.DataFrame:
    """
    Executes the complete EDGE analysis pipeline:
    1. Calculates volume metrics.
    2. Determines sector strength.
    3. Calculates main EDGE scores (Vol, Mom, RR, Fund).
    4. Classifies stocks into categories (SUPER_EDGE, EXPLOSIVE, etc.).
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
    # Ensure necessary columns exist before grouping
    if 'sector' in df_processed.columns and 'volume_acceleration' in df_processed.columns:
        # Initial aggregation for sector strength
        sector_agg = df_processed.groupby('sector').agg(
            avg_vol_accel=('volume_acceleration', 'mean'),
            stock_count=('ticker', 'count')
        ).reset_index()

        # Only rank sectors with a minimum number of stocks for meaningful averages
        sector_agg = sector_agg[sector_agg['stock_count'] >= MIN_STOCKS_PER_SECTOR]

        if not sector_agg.empty:
            # Sort sectors by average volume acceleration (proxy for sector strength before full EDGE scores)
            sector_scores_sorted = sector_agg.sort_values('avg_vol_accel', ascending=False)
            for i, sector in enumerate(sector_scores_sorted['sector']):
                sector_ranks[sector] = i + 1 # Assign rank (1st, 2nd, 3rd, etc.)
    else:
        st.warning("Sector or volume data missing for initial sector ranking.")

    # 3. Calculate main EDGE scores (Volume, Momentum, Risk/Reward, Fundamentals)
    # The `compute_scores` function handles this, including pattern detection
    df_processed = compute_scores(df_processed, weights)

    # 4. Classify stocks based on EDGE score (already done by compute_scores, but ensuring order)
    # This step is implicitly handled within compute_scores, which assigns the 'tag' column.

    # 5. Detect SUPER EDGE with stricter criteria (only for high EDGE candidates)
    # This is done *after* initial tagging and before portfolio constraints.
    # Ensure 'tag' column is initialized before modification (already done in compute_scores)
    super_edge_candidates_idx = df_processed[df_processed['EDGE'] >= EDGE_THRESHOLDS['SUPER_EDGE']].index
    for idx in super_edge_candidates_idx:
        # Pass the pre-calculated sector_ranks to the strict detection
        if detect_super_edge_strict(df_processed.loc[idx], sector_ranks):
            df_processed.loc[idx, 'tag'] = 'SUPER_EDGE'
            # Give a small boost to the EDGE score for confirmed SUPER EDGE (final polish)
            df_processed.loc[idx, 'EDGE'] = min(df_processed.loc[idx, 'EDGE'] * 1.05, 100)

    # 6. Calculate dynamic stops
    df_processed = calculate_dynamic_stops(df_processed)

    # 7. Apply portfolio constraints (position sizing)
    df_processed = apply_portfolio_constraints(df_processed)

    # 8. Calculate targets (adjust based on stock category, conviction)
    # Conservative targets:
    df_processed['target1'] = df_processed['price'] * 1.10 # 10% gain as first target
    df_processed['target2'] = df_processed['price'] * 1.20 # 20% gain as second target

    # Adjust targets for SUPER EDGE (higher conviction, higher potential)
    super_mask = df_processed['tag'] == 'SUPER_EDGE'
    df_processed.loc[super_mask, 'target1'] = df_processed.loc[super_mask, 'price'] * 1.15 # 15% for SUPER EDGE
    df_processed.loc[super_mask, 'target2'] = df_processed.loc[super_mask, 'price'] * 1.30 # 30% for SUPER EDGE

    # 9. Add decision column based on 'tag' and 'portfolio_weight'
    df_processed['decision'] = df_processed.apply(
        lambda row: 'BUY NOW' if row['tag'] == 'SUPER_EDGE' and row['position_size'] > 0 else
                    'BUY' if row['tag'] == 'EXPLOSIVE' and row['position_size'] > 0 else
                    'ACCUMULATE' if row['tag'] == 'STRONG' and row['position_size'] > 0 else
                    'WATCH' if row['tag'] == 'MODERATE' else
                    'IGNORE', axis=1
    )
    
    # Final cleanup of any NaN in critical display columns
    df_processed['tag'] = df_processed['tag'].fillna('WATCH')
    df_processed['decision'] = df_processed['decision'].fillna('IGNORE')
    df_processed['company_name'] = df_processed['company_name'].fillna('Unknown Company')
    df_processed['sector'] = df_processed['sector'].fillna('Unknown Sector')

    return df_processed

# ============================================================================
# CORE EDGE SCORING ENGINE (Combines individual scores and patterns)
# ============================================================================
def compute_scores(df: pd.DataFrame, weights: Tuple[float, float, float, float]) -> pd.DataFrame:
    """
    Calculates the final EDGE score for each stock based on weighted components
    (Volume Acceleration, Momentum, Risk/Reward, Fundamentals) and applies pattern detection.
    """
    df_copy = df.copy() # Work on a copy to avoid SettingWithCopyWarning
    
    # Calculate individual component scores
    with st.spinner("Calculating component scores (Volume, Momentum, Risk/Reward, Fundamentals)..."):
        df_copy["vol_score"] = df_copy.apply(score_vol_accel, axis=1)
        df_copy["mom_score"] = df_copy.apply(score_momentum, axis=1)
        df_copy["rr_score"] = df_copy.apply(score_risk_reward, axis=1)
        df_copy["fund_score"] = df_copy.apply(score_fundamentals, axis=1)
    
    # Define the columns that hold the individual component scores
    block_cols = ["vol_score", "mom_score", "rr_score", "fund_score"]
    
    # Calculate weighted average for the overall EDGE score
    df_copy["EDGE"] = 0.0 # Initialize EDGE score
    for idx in df_copy.index:
        scores = df_copy.loc[idx, block_cols]
        valid_mask = ~scores.isna() # Identify which scores are not NaN
        
        if valid_mask.sum() == 0:
            # If no valid scores, EDGE remains 0 (or some default low value)
            continue
            
        valid_weights = np.array(weights)[valid_mask]
        valid_scores = scores[valid_mask]
        
        # Normalize weights for only the available valid components
        norm_weights = valid_weights / valid_weights.sum()
        df_copy.loc[idx, "EDGE"] = (valid_scores * norm_weights).sum()
    
    # Detect SUPER EDGE based on specific criteria (before pattern bonus)
    # This is a preliminary flag; final 'SUPER_EDGE' tag is set after full pipeline
    df_copy["is_super_edge"] = df_copy.apply(
        lambda row: detect_super_edge_strict(row, {}), # Pass empty dict for sector_ranks initially, filled later
        axis=1
    )
    
    # Apply pattern detection for top candidates (performance optimization)
    # Only run pattern detection on stocks that have a reasonable base EDGE score
    with st.spinner("Detecting explosive patterns and calculating confluence..."):
        # Initialize pattern columns to avoid KeyError
        df_copy['pattern_analysis'] = None
        df_copy['top_pattern_name'] = ""
        df_copy['top_pattern_score'] = 0.0
        df_copy['pattern_confluence_score'] = 0.0
        df_copy['vp_divergence_score'] = 0.0
        
        # Select candidates for detailed pattern analysis
        high_potential_candidates = df_copy[df_copy['EDGE'] >= EDGE_THRESHOLDS['MODERATE']].nlargest(MAX_PATTERN_DETECTION_STOCKS, 'EDGE', keep='first').copy()
        
        for idx, row_data in high_potential_candidates.iterrows():
            pattern_data = detect_all_patterns(row_data) # Run all patterns for this row
            
            # Store the full pattern analysis dictionary
            df_copy.at[idx, 'pattern_analysis'] = pattern_data
            
            # Extract top pattern details
            if pattern_data['top_pattern']:
                df_copy.at[idx, 'top_pattern_name'] = pattern_data['top_pattern']['pattern']
                df_copy.at[idx, 'top_pattern_score'] = float(pattern_data['top_pattern']['score'])
            
            # Store confluence and divergence scores
            df_copy.at[idx, 'pattern_confluence_score'] = float(pattern_data['confluence_score'])
            df_copy.at[idx, 'vp_divergence_score'] = float(pattern_data['vp_divergence'])
    
    # Apply pattern bonus to EDGE score
    # Stocks with a strong top pattern (score > 70) get a boost
    df_copy.loc[df_copy['top_pattern_score'] > 70, 'EDGE'] = (df_copy['EDGE'] * 1.05).clip(0, 100)
    # Stocks with high pattern confluence (score > 85) get an additional boost
    df_copy.loc[df_copy['pattern_confluence_score'] > 85, 'EDGE'] = (df_copy['EDGE'] * 1.05).clip(0, 100)
    
    # Final clipping of EDGE score to ensure it's within 0-100
    df_copy['EDGE'] = df_copy['EDGE'].clip(0, 100)
    
    # Classify stocks into categories (SUPER_EDGE, EXPLOSIVE, STRONG, MODERATE, WATCH)
    # This uses the final EDGE score and the 'is_super_edge' flag
    conditions = [
        df_copy["is_super_edge"] & (df_copy["EDGE"] >= EDGE_THRESHOLDS["SUPER_EDGE"]),
        df_copy["EDGE"] >= EDGE_THRESHOLDS["EXPLOSIVE"],
        df_copy["EDGE"] >= EDGE_THRESHOLDS["STRONG"],
        df_copy["EDGE"] >= EDGE_THRESHOLDS["MODERATE"],
    ]
    choices = ["SUPER_EDGE", "EXPLOSIVE", "STRONG", "MODERATE"]
    df_copy["tag"] = np.select(conditions, choices, default="WATCH")
    
    # Calculate position sizing based on 'tag' (will be further adjusted by risk_adjustment later)
    position_map = {
        "SUPER_EDGE": BASE_POSITION_SIZES["SUPER_EDGE"],
        "EXPLOSIVE": BASE_POSITION_SIZES["EXPLOSIVE"],
        "STRONG": BASE_POSITION_SIZES["STRONG"],
        "MODERATE": BASE_POSITION_SIZES["MODERATE"],
        "WATCH": BASE_POSITION_SIZES["WATCH"]
    }
    df_copy['position_size_pct'] = df_copy['tag'].map(position_map)
    
    # Add additional derived indicators for deep dive
    df_copy['eps_qoq_acceleration'] = 0.0
    mask_eps = (df_copy['eps_last_qtr'] > 0) & (df_copy['eps_current'].notna()) & (df_copy['eps_last_qtr'].notna())
    df_copy.loc[mask_eps, 'eps_qoq_acceleration'] = (
        (df_copy.loc[mask_eps, 'eps_current'] - df_copy.loc[mask_eps, 'eps_last_qtr']) / 
        df_copy.loc[mask_eps, 'eps_last_qtr'] * 100
    )
    
    # Quality Consolidation: Strong long-term returns but consolidating recently
    df_copy['quality_consolidation'] = (
        (df_copy.get('ret_3y', pd.Series(0, index=df_copy.index)) > 300) & 
        (df_copy.get('ret_1y', pd.Series(0, index=df_copy.index)) < 50) & # Not too hot in the last year
        (df_copy.get('from_high_pct', pd.Series(0, index=df_copy.index)) >= -40) & 
        (df_copy.get('from_high_pct', pd.Series(0, index=df_copy.index)) <= -15) # In a healthy pullback zone
    )
    
    # Momentum Aligned: Consistent accelerating positive short-term momentum
    df_copy['momentum_aligned'] = (
        (df_copy.get('ret_1d', pd.Series(0, index=df_copy.index)) > 0) & 
        (df_copy.get('ret_3d', pd.Series(0, index=df_copy.index)) > df_copy.get('ret_1d', pd.Series(0, index=df_copy.index))) & 
        (df_copy.get('ret_7d', pd.Series(0, index=df_copy.index)) > df_copy.get('ret_3d', pd.Series(0, index=df_copy.index))) & 
        (df_copy.get('ret_30d', pd.Series(0, index=df_copy.index)) > 0)
    )
    
    # Tier classifications for EPS and Price
    df_copy['eps_tier'] = df_copy['eps_current'].apply(get_eps_tier)
    df_copy['price_tier'] = df_copy['price'].apply(get_price_tier)
    
    return df_copy

# ============================================================================
# HELPER FUNCTIONS FOR UI CATEGORIZATION
# ============================================================================
def get_eps_tier(eps: float) -> str:
    """Categorize EPS into performance tiers for display."""
    if pd.isna(eps):
        return "N/A"
    
    if eps >= 0.95: return "95↑"
    if eps >= 0.75: return "75↑"
    if eps >= 0.55: return "55↑"
    if eps >= 0.35: return "35↑"
    if eps >= 0.15: return "15↑"
    if eps >= 0.05: return "5↑"
    return "5↓" # Negative or very low EPS

def get_price_tier(price: float) -> str:
    """Categorize price into tiers for display."""
    if pd.isna(price):
        return "N/A"
    
    if price >= 5000: return "5K↑"
    if price >= 2000: return "2K↑"
    if price >= 1000: return "1K↑"
    if price >= 500: return "500↑"
    if price >= 200: return "200↑"
    if price >= 100: return "100↑"
    return "100↓" # Price below 100

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================
def plot_stock_radar_chart(df_row: pd.Series):
    """
    Creates a radar chart visualizing the four core EDGE components for a single stock.
    Highlights SUPER EDGE stocks.
    """
    categories = ['Volume Accel', 'Momentum', 'Risk/Reward', 'Fundamentals']
    scores = [
        df_row.get('vol_score', 0),
        df_row.get('mom_score', 0),
        df_row.get('rr_score', 0),
        df_row.get('fund_score', 0)
    ]
    # Ensure scores are numeric and handle NaN gracefully for plotting
    scores = [0 if pd.isna(s) else s for s in scores]
    
    line_color = 'gold' if df_row.get('tag') == 'SUPER_EDGE' else 'darkblue'
    fill_color = 'rgba(255, 215, 0, 0.4)' if df_row.get('tag') == 'SUPER_EDGE' else 'rgba(0, 0, 139, 0.4)'
    
    fig = go.Figure(data=go.Scatterpolar(
        r=scores,
        theta=categories,
        fill='toself',
        name=df_row.get('company_name', 'Stock'),
        line_color=line_color,
        fillcolor=fill_color,
        line_width=3
    ))
    
    title = f"EDGE Components - {df_row.get('company_name', 'N/A')} ({df_row.get('ticker', 'N/A')})"
    if df_row.get('tag') == 'SUPER_EDGE':
        title += " ⭐ SUPER EDGE ⭐"
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100], # Scores are normalized to 0-100
                tickvals=[0, 25, 50, 75, 100],
                ticktext=['0', '25', '50', '75', '100']
            )),
        showlegend=False,
        title={
            'text': title,
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        font=dict(size=14, color="#333"),
        margin=dict(l=50, r=50, t=80, b=50) # Adjust margins
    )
    
    return fig

# ============================================================================
# MAIN STREAMLIT UI FUNCTION
# ============================================================================
def render_ui():
    """
    Main Streamlit UI function orchestrating the display of the EDGE Protocol.
    """
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for better UI and visual appeal
    st.markdown("""
    <style>
    .super-edge-banner {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%); /* Gold to Orange */
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
        gap: 8px; /* Spacing between tabs */
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: rgba(240, 242, 246, 0.5);
        border-radius: 5px 5px 0px 0px;
        font-weight: bold;
        color: #333;
    }
    /* Style for metric boxes */
    div[data-testid="metric-container"] {
        background-color: rgba(28, 131, 225, 0.05); /* Light blue background */
        border: 1px solid rgba(28, 131, 225, 0.1); /* Subtle border */
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    [data-testid="stMetricValue"] {
        font-size: 24px;
        color: #007bff; /* Blue color for values */
    }
    [data-testid="stMetricLabel"] {
        font-size: 16px;
        color: #555;
    }
    .dataframe-style {
        font-size: 0.9em; /* Slightly smaller font for dense tables */
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title(PAGE_TITLE)
    st.markdown("**Your edge**: Volume acceleration + Pattern detection + Momentum + Fundamentals = **SUPERIOR RETURNS**")
    
    # Sidebar for Configuration & Filters
    with st.sidebar:
        st.header("⚙️ Protocol Settings")
        
        st.subheader("📊 Strategy Profile")
        # Allow selection of predefined profiles
        profile_name = st.radio("Select Profile", list(PROFILE_PRESETS.keys()), index=0, help="Choose a profile to adjust scoring weights.")
        weights = PROFILE_PRESETS[profile_name] # Get weights based on selected profile
        
        st.markdown("---")
        st.subheader("🎯 Signal Filters")
        min_edge = st.slider("Minimum EDGE Score for Signals", 0, 100, 70, 5, help="Only show stocks with an EDGE score above this threshold.")
        exclude_small_caps = st.checkbox("Exclude Micro/Small Cap Stocks", True, help="Filter out stocks generally below $1 Billion market cap.")
        max_signals = st.slider("Maximum Number of Signals to Display", 10, 200, 100, 10, help="Limits the total number of signals shown for performance and focus.")
        
        st.markdown("---")
        st.markdown("### Current Weights:")
        st.write(f"**Volume:** {weights[0]*100:.0f}%")
        st.write(f"**Momentum:** {weights[1]*100:.0f}%")
        st.write(f"**Risk/Reward:** {weights[2]*100:.0f}%")
        st.write(f"**Fundamentals:** {weights[3]*100:.0f}%")

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
    
    # Remove low liquidity stocks (using 30-day dollar volume)
    if "rs_volume_30d" in df_filtered_category.columns:
        # Filter out stocks with very low average daily dollar volume (e.g., less than $10 million)
        # Adjust threshold as needed based on market and liquidity requirements
        df_filtered_category = df_filtered_category[
            (df_filtered_category["rs_volume_30d"] >= 1e7) | df_filtered_category["rs_volume_30d"].isna()
        ]
        if df_filtered_category.empty:
            st.warning("All stocks filtered out by liquidity. Try adjusting filters or check your data.")
            return

    # Run EDGE analysis pipeline
    with st.spinner("🚀 Running EDGE Protocol Analysis... This may take a moment."):
        start_time = time.time()
        # Pass the 4-element weights tuple to run_edge_analysis
        df_analyzed = run_edge_analysis(df_filtered_category, weights)
        end_time = time.time()
        st.success(f"Analysis completed in {end_time - start_time:.2f} seconds.")

    # Filter for signals based on user-defined min_edge and max_signals
    # Ensure 'EDGE' column exists and is numeric before filtering
    if 'EDGE' not in df_analyzed.columns or not pd.api.types.is_numeric_dtype(df_analyzed['EDGE']):
        st.error("EDGE scores could not be calculated. Cannot display signals.")
        df_signals = pd.DataFrame()
    else:
        df_signals = df_analyzed[df_analyzed['EDGE'] >= min_edge].nlargest(max_signals, 'EDGE', keep='first').copy()

    # SUPER EDGE Alert (only if df_signals is not empty)
    super_edge_count = (df_signals['tag'] == 'SUPER_EDGE').sum() if 'tag' in df_signals.columns else 0
    if super_edge_count > 0:
        st.markdown(f"""
        <div class="super-edge-banner">
            ⭐ {super_edge_count} SUPER EDGE SIGNAL{'S' if super_edge_count > 1 else ''} DETECTED! ⭐<br>
            <span style="font-size: 16px;">Maximum conviction opportunities with strict risk management!</span>
        </div>
        """, unsafe_allow_html=True)

    # Main content tabs
    tabs = st.tabs([
        "📊 Daily Signals",
        "⭐ SUPER EDGE",
        "🎯 Explosive Patterns",
        "📈 Volume Analysis",
        "🔥 Sector Heatmap",
        "🔍 Deep Dive",
        "📋 Raw Data"
    ])

    # Tab 1: Daily Signals
    with tabs[0]:
        st.header("📊 Daily EDGE Signals")
        
        # Quick stats for signals
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
            # Check for 'top_pattern_score' column before summing
            strong_patterns = (df_signals['top_pattern_score'] >= 75).sum() if 'top_pattern_score' in df_signals.columns else 0
            st.metric("Strong Patterns", strong_patterns)

        st.markdown("---")

        # Dynamic Filters for the table display
        col_f1, col_f2, col_f3 = st.columns(3)
        current_display_df = df_signals.copy() # Base for tab-specific filters

        with col_f1:
            if 'tag' in current_display_df.columns:
                unique_tags = current_display_df['tag'].dropna().unique().tolist()
                # Sort tags in a meaningful order
                sorted_tags = sorted(unique_tags, key=lambda x: ['SUPER_EDGE', 'EXPLOSIVE', 'STRONG', 'MODERATE', 'WATCH'].index(x) if x in ['SUPER_EDGE', 'EXPLOSIVE', 'STRONG', 'MODERATE', 'WATCH'] else 99)
                selected_tags = st.multiselect(
                    "Filter by Classification",
                    sorted_tags,
                    default=sorted_tags # Default to all available tags
                )
                if selected_tags:
                    current_display_df = current_display_df[current_display_df['tag'].isin(selected_tags)]
        with col_f2:
            if 'sector' in current_display_df.columns:
                unique_sectors = current_display_df['sector'].dropna().unique().tolist()
                selected_sectors = st.multiselect(
                    "Filter by Sector",
                    sorted(unique_sectors),
                    default=[] # No default if too many sectors, user can select
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
                'target1', 'target2', 'volume_acceleration', 'rvol',
                'volume_classification', 'top_pattern_name', 'top_pattern_score', 'pattern_confluence_score', 'vp_divergence_score',
                'vol_score', 'mom_score', 'rr_score', 'fund_score' # Show individual scores
            ]
            # Ensure only existing columns are included
            display_cols = [col for col in display_cols_order if col in current_display_df.columns]

            # Custom styling for the DataFrame
            def style_signals_dataframe(df_to_style):
                s = df_to_style.style.format({
                    'EDGE': '{:.1f}',
                    'vol_score': '{:.0f}', 'mom_score': '{:.0f}', 'rr_score': '{:.0f}', 'fund_score': '{:.0f}',
                    'price': '₹{:.2f}',
                    'position_size': '{:.1%}',
                    'stop_loss': '₹{:.2f}',
                    'stop_loss_pct': '{:.1f}%',
                    'target1': '₹{:.2f}',
                    'target2': '₹{:.2f}',
                    'volume_acceleration': '{:.1f}%',
                    'rvol': '{:.1f}x',
                    'top_pattern_score': '{:.0f}',
                    'pattern_confluence_score': '{:.0f}',
                    'vp_divergence_score': '{:.1f}'
                })
                
                # Conditional styling for 'tag' column
                if 'tag' in df_to_style.columns:
                    s = s.applymap(lambda x: 'background-color: #FFD700; font-weight: bold; color: black;' if x == 'SUPER_EDGE' else # Gold
                                             'background-color: #FFC0CB; font-weight: bold; color: black;' if x == 'EXPLOSIVE' else # Pink
                                             'background-color: #ADD8E6; color: black;' if x == 'STRONG' else '', # Light Blue
                                             subset=['tag'])
                
                # Conditional styling for 'decision' column
                if 'decision' in df_to_style.columns:
                    s = s.applymap(lambda x: 'color: #28a745; font-weight: bold;' if x == 'BUY NOW' else # Green
                                             'color: #17a2b8;' if x == 'BUY' else # Cyan
                                             'color: #ffc107;' if x == 'ACCUMULATE' else # Yellow
                                             'color: #6c757d;' if x == 'WATCH' else '', # Gray
                                             subset=['decision'])
                
                # Gradient for stop_loss_pct (green for tighter, red for looser)
                if 'stop_loss_pct' in df_to_style.columns:
                    s = s.background_gradient(cmap='RdYlGn_r', subset=['stop_loss_pct'], vmin=-15, vmax=0)
                
                # Gradient for EDGE score (green for higher)
                if 'EDGE' in df_to_style.columns:
                    s = s.background_gradient(cmap='Greens', subset=['EDGE'], vmin=50, vmax=100)
                
                # Gradient for volume_acceleration (blue for higher)
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
                        st.metric("Target 1", f"₹{row['target1']:.2f} ({safe_divide((row['target1']-row['price'])*100, row['price'], 0):.1f}%)")
                        st.metric("Target 2", f"₹{row['target2']:.2f} ({safe_divide((row['target2']-row['price'])*100, row['price'], 0):.1f}%)")

                        # Safe risk:reward calculation
                        price = row.get('price', 1)
                        stop_loss = row.get('stop_loss', price * 0.95)
                        target_1 = row.get('target1', price * 1.10) # Using T1 for R:R
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
                        st.markdown(f"**Volume Pattern:** {row.get('volume_classification', 'N/A')}") # Updated to volume_classification
                        st.markdown(f"**Volume Accel:** {row.get('volume_acceleration', 0):.1f}%")
                        st.markdown(f"**RVOL:** {row.get('rvol', 0):.1f}x")
                        if row.get('top_pattern_name'): # Updated to top_pattern_name
                            st.markdown(f"**Detected Pattern:** {row['top_pattern_name']}")
                        if row.get('pattern_analysis') and row['pattern_analysis'].get('top_pattern'): # Access signals from pattern_analysis
                            top_pattern_signals = row['pattern_analysis']['top_pattern'].get('signals', [])
                            if top_pattern_signals:
                                st.caption(f"Signals: {', '.join(top_pattern_signals)}")

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
                    
                    eps_condition_met = False
                    if pd.notna(eps_current) and pd.notna(eps_last_qtr) and eps_last_qtr != 0:
                        eps_growth_pct = safe_divide((eps_current - eps_last_qtr), eps_last_qtr)
                        if eps_growth_pct >= 0.20:
                            criteria.append(f"✅ **EPS Acceleration (QoQ >= 20%):** Strong earnings growth confirms fundamentals.")
                            eps_condition_met = True
                    if not eps_condition_met and pd.notna(eps_change_pct) and eps_change_pct >= 20:
                        criteria.append(f"✅ **EPS Change % (>= 20%):** Strong earnings growth confirms fundamentals (using change_pct).")
                        eps_condition_met = True

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
                        sector_agg_temp = sector_agg_temp[(sector_agg_temp['stock_count'] >= MIN_STOCKS_PER_SECTOR) & (sector_agg_temp['avg_edge'] > 50)]
                        if not sector_agg_temp.empty:
                            sector_scores_sorted_temp = sector_agg_temp.sort_values('avg_edge', ascending=False)
                            for i, sec_name in enumerate(sector_scores_sorted_temp['sector']):
                                temp_sector_ranks[sec_name] = i + 1

                    sector_rank_for_display = temp_sector_ranks.get(sector, 999) if sector else 999
                    if sector_rank_for_display <= 3:
                        criteria.append(f"✅ **Top Sector Strength:** Belongs to a top {sector_rank_for_display} performing sector ({sector}).")

                    if criteria:
                        for crit in criteria:
                            st.write(crit)
                        st.write(f"**Total Conditions Met:** {len(criteria)}/6 (Requires >=5)")
                    else:
                        st.info("No specific SUPER EDGE conditions met (this should not happen if stock is tagged SUPER EDGE).")

        else:
            st.info("No SUPER EDGE signals detected today. Check the 'Daily Signals' tab for other opportunities.")
            # Show next best opportunities (EXPLOSIVE)
            if 'tag' in df_signals.columns:
                explosive_df = df_signals[df_signals['tag'] == 'EXPLOSIVE'].head(5)
            else:
                explosive_df = pd.DataFrame()

            if not explosive_df.empty:
                st.subheader("🔥 Top EXPLOSIVE Opportunities (Next Best)")
                cols_exp = ['ticker', 'company_name', 'EDGE', 'price', 'volume_classification', 'decision'] # Updated volume_pattern
                cols_exp = [col for col in cols_exp if col in explosive_df.columns]
                if cols_exp:
                    st.dataframe(explosive_df[cols_exp].style.format({'EDGE': '{:.1f}', 'price': '₹{:.2f}'}), use_container_width=True)

    # Tab 3: Explosive Patterns
    with tabs[2]:
        st.header("🎯 Explosive Patterns Discovery")
        
        # Filter for stocks with patterns (top_pattern_score > 0)
        pattern_df = df_analyzed[df_analyzed['top_pattern_score'] > 0].copy()
        
        if not pattern_df.empty:
            # Pattern stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Patterns", len(pattern_df))
            with col2:
                high_confluence = (pattern_df['pattern_confluence_score'] >= 85).sum()
                st.metric("High Confluence", high_confluence)
            with col3:
                avg_score = pattern_df['top_pattern_score'].mean()
                st.metric("Avg Pattern Score", f"{avg_score:.0f}")
            with col4:
                unique_patterns = pattern_df['top_pattern_name'].nunique()
                st.metric("Pattern Types", unique_patterns)
            
            # Pattern distribution chart
            pattern_counts = pattern_df['top_pattern_name'].value_counts()
            
            fig_patterns = go.Figure(data=[go.Bar(
                x=pattern_counts.index,
                y=pattern_counts.values,
                text=pattern_counts.values,
                textposition='auto',
                marker_color='lightblue'
            )])
            
            fig_patterns.update_layout(
                title="Distribution of Top Detected Patterns",
                xaxis_title="Pattern Type",
                yaxis_title="Number of Stocks",
                height=400,
                xaxis_tickangle=-45 # Rotate labels if too long
            )
            
            st.plotly_chart(fig_patterns, use_container_width=True)
            
            # High confluence patterns
            st.subheader("🔥 High Confluence Signals (Top 10)")
            
            high_conf_df = pattern_df[pattern_df['pattern_confluence_score'] >= 70].sort_values(
                'pattern_confluence_score', ascending=False
            ).head(10)
            
            if not high_conf_df.empty:
                display_pattern_cols = [
                    'ticker', 'company_name', 'price', 'EDGE', 'tag',
                    'top_pattern_name', 'top_pattern_score', 
                    'pattern_confluence_score', 'vp_divergence_score'
                ]
                
                st.dataframe(
                    high_conf_df[display_pattern_cols].style.format({
                        'price': '₹{:.2f}',
                        'EDGE': '{:.1f}',
                        'top_pattern_score': '{:.0f}',
                        'pattern_confluence_score': '{:.0f}',
                        'vp_divergence_score': '{:.1f}'
                    }).background_gradient(cmap='YlOrRd', subset=['pattern_confluence_score'], vmin=70, vmax=100), # Heatmap for confluence
                    use_container_width=True
                )
            else:
                st.info("No high confluence patterns detected based on current filters.")
            
            # Pattern guide
            with st.expander("📚 Pattern Guide"):
                st.markdown("""
                **Pattern Types:**
                1.  **Accumulation Under Resistance:** Big volume, flat price near highs. Buyers absorbing supply.
                2.  **Coiled Spring:** Tight price range with increasing volume. Compression before expansion.
                3.  **Absorption Pattern:** High RVOL with price stability. Institutional buying absorbing selling pressure.
                4.  **Failed Breakdown Reversal:** Bounce from 52-week lows with volume. Failed bearish move.
                5.  **Stealth Breakout:** Quiet strength building above key moving averages. Gradual accumulation.
                6.  **Pre-Earnings Accumulation:** Unusual volume and EPS momentum before earnings. Insider anticipation.
                
                **Confluence Score:** Indicates how many strong patterns (score >= 70) are present:
                -   **100:** 3+ patterns aligned (ULTRA HIGH conviction)
                -   **85:** 2 patterns aligned (Very strong conviction)
                -   **70:** 1 strong pattern (Good opportunity)
                
                **VP Divergence Score:** Measures the divergence between Volume and Price movement.
                -   **High Positive:** Strong volume with relatively flat or slightly increasing price (accumulation).
                -   **High Negative:** Strong volume with relatively flat or slightly decreasing price (distribution).
                """)
        else:
            st.info("No significant patterns detected in current market conditions or data. Try adjusting filters.")
    
    # Tab 4: Volume Analysis
    with tabs[3]:
        st.header("📈 Volume Acceleration Analysis")
        
        # Volume acceleration scatter plot
        if "volume_acceleration" in df_analyzed.columns and "from_high_pct" in df_analyzed.columns:
            # Prepare data, drop NaNs for plotting
            plot_df = df_analyzed.dropna(subset=['volume_acceleration', 'from_high_pct', 'rvol', 'tag'])
            
            if not plot_df.empty:
                # Create marker size based on RVOL, scaled for better visualization
                plot_df['marker_size'] = plot_df['rvol'].clip(0.5, 5.0) * 10
                
                fig = px.scatter(
                    plot_df,
                    x="from_high_pct",
                    y="volume_acceleration",
                    color="tag", # Color by classification tag
                    size="marker_size", # Size by RVOL
                    hover_data=['ticker', 'company_name', 'EDGE', 'rvol', 'volume_classification'],
                    title="Volume Acceleration Map (Bubble Size = RVOL)",
                    labels={
                        "from_high_pct": "% From 52-Week High",
                        "volume_acceleration": "Volume Acceleration %"
                    },
                    color_discrete_map={ # Define colors for each tag
                        "SUPER_EDGE": "#FFD700", # Gold
                        "EXPLOSIVE": "#FF4B4B",  # Red
                        "STRONG": "#FFA500",     # Orange
                        "MODERATE": "#90EE90",   # Light Green
                        "WATCH": "#87CEEB"       # Light Blue
                    }
                )
                
                # Add strategic zones/lines
                fig.add_vrect(x0=-30, x1=-15, fillcolor="gold", opacity=0.1, line_width=0,
                              annotation_text="Optimal Accumulation Zone", annotation_position="top left")
                fig.add_hline(y=30, line_dash="dash", line_color="red", 
                              annotation_text="High Volume Acceleration Threshold", 
                              annotation_position="bottom right")
                
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
                
                # Volume insights metrics
                st.subheader("📊 Key Volume Insights")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    high_rvol = (plot_df['rvol'] > 2.0).sum()
                    st.metric("Stocks with High RVOL (>2x)", high_rvol)
                
                with col2:
                    high_accel = (plot_df['volume_acceleration'] > 30).sum()
                    st.metric("Stocks with High Acceleration (>30%)", high_accel)
                
                with col3:
                    inst_loading = (plot_df['volume_classification'] == "Institutional Loading").sum()
                    st.metric("Stocks showing Institutional Loading", inst_loading)
            else:
                st.info("Insufficient data for volume acceleration map. Adjust filters or check data completeness.")
        else:
            st.info("Volume acceleration or price proximity data not available for charting.")
    
    # Tab 5: Sector Heatmap
    with tabs[4]:
        st.header("🔥 Sector Heatmap & Leaderboard")
        
        # Aggregate by sector for treemap and leaderboard
        sector_agg = df_analyzed.groupby('sector').agg(
            avg_edge=('EDGE', 'mean'),
            total_stocks=('ticker', 'count'),
            super_edge_count=('tag', lambda x: (x == 'SUPER_EDGE').sum())
        ).reset_index()
        
        # Filter out sectors with very few stocks for meaningful averages
        sector_agg = sector_agg[sector_agg['total_stocks'] >= MIN_STOCKS_PER_SECTOR]
        
        if not sector_agg.empty:
            # Create treemap
            fig = px.treemap(
                sector_agg,
                path=['sector'],
                values='total_stocks', # Size of box by number of stocks
                color='avg_edge',      # Color by average EDGE score
                hover_data={
                    'avg_edge': ':.1f',
                    'total_stocks': True,
                    'super_edge_count': True
                },
                color_continuous_scale='RdYlGn', # Red-Yellow-Green scale for average EDGE
                range_color=[0, 100], # Ensure color scale covers full range of EDGE scores
                title="Sector EDGE Heatmap (Size by Stocks, Color by Avg EDGE)"
            )
            
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Top sectors table
            st.subheader("🏆 Top Sectors by Average EDGE Score")
            
            top_sectors = sector_agg.sort_values('avg_edge', ascending=False).head(10)
            
            st.dataframe(
                top_sectors.style.format({
                    'avg_edge': '{:.1f}',
                    'total_stocks': '{:.0f}',
                    'super_edge_count': '{:.0f}'
                }).background_gradient(subset=['avg_edge'], cmap='RdYlGn'), # Apply gradient to avg_edge
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("Insufficient data for sector analysis. Ensure at least 3 stocks per sector.")
    
    # Tab 6: Deep Dive (Individual Stock Analysis)
    with tabs[5]:
        st.header("🔍 Stock Deep Dive")
        
        # Stock selector: Only show stocks that have been analyzed (EDGE score is not NaN)
        available_stocks = df_analyzed[df_analyzed['EDGE'].notna()]['ticker'].unique()
        
        if len(available_stocks) > 0:
            # Prioritize high EDGE stocks in the dropdown
            sorted_stocks = (df_analyzed[df_analyzed['ticker'].isin(available_stocks)]
                             .sort_values('EDGE', ascending=False)['ticker'].unique())
            
            selected_ticker = st.selectbox(
                "Select Stock for Deep Dive",
                sorted_stocks,
                # Format function to highlight SUPER EDGE stocks in dropdown
                format_func=lambda x: f"⭐ {x} (SUPER EDGE)" if x in df_analyzed[df_analyzed['tag'] == 'SUPER_EDGE']['ticker'].values else x
            )
            
            # Get data for the selected stock
            stock_data = df_analyzed[df_analyzed['ticker'] == selected_ticker].iloc[0]
            
            # Display header for SUPER EDGE stocks
            if stock_data['tag'] == 'SUPER_EDGE':
                st.markdown("""
                <div style="background: linear-gradient(90deg, #FFD700, #FFA500); padding: 10px; border-radius: 5px; text-align: center;">
                    <h2 style="margin: 0; color: black;">⭐ SUPER EDGE SIGNAL ⭐</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Stock Name and Ticker
            st.subheader(f"{stock_data.get('company_name', 'N/A')} ({stock_data.get('ticker', 'N/A')})")
            
            # Key metrics in columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Current Price", f"₹{stock_data.get('price', 0):.2f}")
                st.metric("EDGE Score", f"{stock_data.get('EDGE', 0):.1f}")
                st.metric("Classification", stock_data.get('tag', 'N/A'))
                st.metric("Sector", stock_data.get('sector', 'N/A'))
            
            with col2:
                st.metric("RVOL", f"{stock_data.get('rvol', 0):.1f}x")
                st.metric("Volume Accel", f"{stock_data.get('volume_acceleration', 0):.1f}%")
                st.metric("Volume Pattern", stock_data.get('volume_classification', 'N/A'))
                st.metric("From 52w High", f"{stock_data.get('from_high_pct', 0):.1f}%")
            
            with col3:
                st.metric("1-Day Return", f"{stock_data.get('ret_1d', 0):.1f}%")
                st.metric("7-Day Return", f"{stock_data.get('ret_7d', 0):.1f}%")
                st.metric("30-Day Return", f"{stock_data.get('ret_30d', 0):.1f}%")
                st.metric("3-Year Return", f"{stock_data.get('ret_3y', 0):.0f}%")
            
            with col4:
                st.metric("Stop Loss", f"₹{stock_data.get('stop_loss', 0):.2f} ({stock_data.get('stop_loss_pct', 0):.1f}%)")
                st.metric("Target 1", f"₹{stock_data.get('target1', 0):.2f}")
                st.metric("Target 2", f"₹{stock_data.get('target2', 0):.2f}")
                st.metric("Position Size", f"{stock_data.get('position_size', 0)*100:.1f}%")
            
            # Radar chart for EDGE components
            fig_radar = plot_stock_radar_chart(stock_data)
            st.plotly_chart(fig_radar, use_container_width=True)
            
            # Detailed Pattern Analysis
            if stock_data.get('pattern_analysis') and isinstance(stock_data['pattern_analysis'], dict):
                st.subheader("🎯 Detailed Pattern Analysis")
                
                pattern_data = stock_data['pattern_analysis']
                all_patterns = pattern_data.get('all_patterns', [])
                
                st.write(f"**Top Pattern:** {stock_data.get('top_pattern_name', 'N/A')} (Score: {stock_data.get('top_pattern_score', 0):.0f})")
                st.write(f"**Pattern Confluence Score:** {stock_data.get('pattern_confluence_score', 0):.0f}")
                st.write(f"**Volume-Price Divergence:** {stock_data.get('vp_divergence_score', 0):.1f}")
                
                # Display all patterns with expanders
                for pattern in all_patterns:
                    if pattern['score'] > 0: # Only show patterns with a non-zero score
                        with st.expander(f"**{pattern['pattern']}** (Score: {pattern['score']:.0f})"):
                            for signal in pattern.get('signals', []):
                                st.write(f"• {signal}")
                            if pattern.get('target'):
                                target_pct = safe_divide((pattern['target'] / stock_data['price'] - 1) * 100, 1, 0)
                                st.success(f"Pattern Specific Target: ₹{pattern['target']:.2f} (+{target_pct:.1f}%)")
            else:
                st.info("No patterns detected for this stock.")
            
            # Special Indicators
            st.subheader("📌 Special Indicators")
            
            indicators = []
            if stock_data.get('quality_consolidation'):
                indicators.append("💎 **Quality Consolidation:** Strong long-term stock consolidating for next move.")
            if stock_data.get('momentum_aligned'):
                indicators.append("📈 **Momentum Aligned:** Consistent and accelerating positive short-term momentum.")
            if stock_data.get('rvol', 0) > 2.0:
                indicators.append("🔥 **High RVOL Activity:** Significant institutional interest.")
            if stock_data.get('eps_qoq_acceleration', 0) > 15:
                indicators.append(f"💰 **EPS Accelerating:** Strong recent earnings growth ({stock_data['eps_qoq_acceleration']:.1f}% QoQ).")
            if stock_data.get('pe', np.nan) is not np.nan and stock_data['pe'] > 0 and stock_data['pe'] <= 30:
                indicators.append(f"✅ **Reasonable Valuation:** PE ratio of {stock_data['pe']:.1f}.")
            
            if indicators:
                for ind in indicators:
                    st.markdown(ind)
            else:
                st.info("No special indicators active for this stock.")

        else:
            st.info("No stocks available for deep dive analysis. Adjust filters or check data.")
    
    # Tab 7: Raw Data & Diagnostics
    with tabs[6]:
        st.header("📋 Raw Data & Diagnostics")
        
        # Summary stats
        st.subheader("📊 Overall Data Statistics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Stocks Loaded", len(df_raw))
            st.metric("Stocks After Filters", len(df_analyzed))
            st.metric("High EDGE (>70)", (df_analyzed['EDGE'] > 70).sum() if 'EDGE' in df_analyzed.columns else 0)
        
        with col2:
            st.metric("Avg EDGE Score", f"{df_analyzed['EDGE'].mean():.1f}" if 'EDGE' in df_analyzed.columns else "N/A")
            st.metric("Avg RVOL", f"{df_analyzed.get('rvol', pd.Series([1])).mean():.2f}")
            st.metric("Patterns Detected", (df_analyzed['top_pattern_score'] > 0).sum() if 'top_pattern_score' in df_analyzed.columns else 0)
        
        with col3:
            st.metric("Data Timestamp", diagnostics.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M'))
            st.metric("Profile", profile_name)
            st.metric("Min EDGE Filter", min_edge)
        
        # Data quality check
        st.subheader("🔍 Data Quality Check (Critical Columns)")
        
        # Re-define critical columns for this display, including newly added ones for fundamentals
        critical_cols_display = [
            'ticker', 'price', 'volume_1d', 'ret_1d', 'rvol', 'sector',
            'market_cap', 'eps_current', 'eps_last_qtr', 'eps_change_pct',
            'vol_ratio_30d_90d', 'vol_ratio_30d_180d', 'from_high_pct', 'from_low_pct',
            'sma_50d', 'sma_200d'
        ]
        quality_data = []
        
        for col in critical_cols_display:
            if col in df_analyzed.columns:
                non_null = df_analyzed[col].notna().sum()
                pct = safe_divide(non_null * 100, len(df_analyzed), 0)
                quality_data.append({
                    'Column': col,
                    'Non-null Count': non_null,
                    'Coverage %': f"{pct:.1f}%",
                    'Status': '✅' if pct > 90 else '⚠️' if pct > 70 else '❌'
                })
        
        quality_df = pd.DataFrame(quality_data)
        st.dataframe(quality_df, use_container_width=True, hide_index=True)
        
        # Sample data
        st.subheader("📄 Sample Data (Top 10 by EDGE Score)")
        
        sample_cols = ['ticker', 'company_name', 'EDGE', 'tag', 'price', 'rvol', 
                       'volume_acceleration', 'top_pattern_name', 'top_pattern_score',
                       'vol_score', 'mom_score', 'rr_score', 'fund_score']
        sample_cols = [col for col in sample_cols if col in df_analyzed.columns] # Ensure columns exist
        
        st.dataframe(
            df_analyzed.nlargest(10, 'EDGE')[sample_cols].style.format({
                'EDGE': '{:.1f}', 'price': '₹{:.2f}', 'rvol': '{:.1f}', 'volume_acceleration': '{:.1f}%',
                'top_pattern_score': '{:.0f}', 'vol_score': '{:.0f}', 'mom_score': '{:.0f}',
                'rr_score': '{:.0f}', 'fund_score': '{:.0f}'
            }),
            use_container_width=True
        )
        
        # Export full data
        st.subheader("💾 Export Options")
        
        col_exp1, col_exp2 = st.columns(2)
        
        with col_exp1:
            csv_full = df_analyzed.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Full Processed Data",
                csv_full,
                f"edge_protocol_full_processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )
        
        with col_exp2:
            # High EDGE only
            high_edge_df = df_analyzed[df_analyzed['EDGE'] >= 70]
            if not high_edge_df.empty:
                csv_high = high_edge_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "🔥 Download High EDGE Signals Only",
                    csv_high,
                    f"edge_high_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv"
                )

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    render_ui()
