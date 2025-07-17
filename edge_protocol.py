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
import re # Import the re module for regular expressions

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================
SHEET_ID = "1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk"
GID_WATCHLIST = "2026492216"
SHEET_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID_WATCHLIST}"

# Portfolio Risk Management
MAX_PORTFOLIO_EXPOSURE = 0.80  # 80% max exposure
MAX_POSITIONS = 10             # Maximum concurrent positions
MAX_SECTOR_EXPOSURE = 0.30     # 30% max in one sector
MAX_SUPER_EDGE_POSITIONS = 3   # Max 3 SUPER EDGE at once

# Position Sizing (will be dynamically adjusted)
BASE_POSITION_SIZES = {
    "SUPER_EDGE": 0.15,
    "EXPLOSIVE": 0.10,
    "STRONG": 0.05,
    "MODERATE": 0.02,
    "WATCH": 0.00
}

# EDGE Thresholds
EDGE_THRESHOLDS = {
    "SUPER_EDGE": 92,  # Raised from 90
    "EXPLOSIVE": 85,
    "STRONG": 70,
    "MODERATE": 50,
    "WATCH": 0
}

# Performance Settings
MAX_PATTERN_DETECTION_STOCKS = 100  # Limit pattern detection for performance
CACHE_TTL = 300  # 5 minutes

# Define the columns that hold the component scores globally
GLOBAL_BLOCK_COLS = ["vol_score", "mom_score", "rr_score", "fund_score"]

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def safe_divide(a: float, b: float, default: float = 0) -> float:
    """Safe division with proper type handling"""
    try:
        if pd.isna(a) or pd.isna(b) or b == 0:
            return default
        return float(a) / float(b)
    except:
        return default

def calculate_atr(prices: pd.Series, period: int = 20) -> float:
    """Calculate Average True Range for dynamic stops"""
    if len(prices) < period:
        return prices.std() * 2.0 if len(prices) > 0 else 0.0
    
    # Simple ATR calculation using price volatility
    high_low_pct = prices.pct_change().abs()
    atr = high_low_pct.rolling(window=period, min_periods=1).mean() * prices.iloc[-1]
    return atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else prices.std() * 2.0

def parse_market_cap(val: Union[str, float]) -> float:
    """Parse market cap with Indian notation and other suffixes"""
    if pd.isna(val) or not isinstance(val, str):
        return np.nan
    
    val_str = val.strip()
    
    # Remove currency symbols and commas
    val_str = val_str.replace("₹", "").replace(",", "").strip()
    
    # Handle multipliers
    multipliers = {
        'cr': 1e7, 'Cr': 1e7, 'crore': 1e7,
        'l': 1e5, 'L': 1e5, 'lakh': 1e5, 'lac': 1e5,
        'k': 1e3, 'K': 1e3, # Thousand
        'm': 1e6, 'M': 1e6, # Million
        'b': 1e9, 'B': 1e9  # Billion
    }
    
    for suffix, multiplier in multipliers.items():
        if val_str.lower().endswith(suffix.lower()):
            try:
                number_part = val_str[:-len(suffix)].strip()
                return float(number_part) * multiplier
            except ValueError:
                return np.nan
    
    try:
        return float(val_str)
    except ValueError:
        return np.nan

# ============================================================================
# DATA LOADING WITH VALIDATION
# ============================================================================
@st.cache_data(ttl=CACHE_TTL)
def load_and_validate_data() -> Tuple[pd.DataFrame, Dict[str, any]]:
    """Load data with comprehensive validation and diagnostics"""
    diagnostics = {
        "timestamp": datetime.now(),
        "rows_loaded": 0,
        "data_quality_score": 0,
        "critical_columns_missing": [],
        "warnings": [],
        "data_age_hours": 0
    }
    
    try:
        # Load data
        response = requests.get(SHEET_URL, timeout=30)
        response.raise_for_status()
        df = pd.read_csv(io.BytesIO(response.content))
        diagnostics["rows_loaded"] = len(df)
        
        # Standardize column names
        df.columns = (df.columns.str.strip()
                      .str.lower()
                      .str.replace("%", "pct")
                      .str.replace(" ", "_"))
        
        # Critical columns check
        critical_cols = [
            'ticker', 'price', 'volume_1d', 
            'vol_ratio_7d_90d', 'vol_ratio_30d_90d', 'vol_ratio_30d_180d'
        ]
        
        missing_critical = [col for col in critical_cols if col not in df.columns]
        diagnostics["critical_columns_missing"] = missing_critical
        
        if missing_critical:
            diagnostics["warnings"].append(f"Missing critical columns: {missing_critical}")
        
        # Define columns that are expected to be numeric
        numeric_cols_to_process = [
            'market_cap', 'volume_1d', 'volume_7d', 'volume_30d', 'volume_90d', 'volume_180d',
            'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d',
            'vol_ratio_1d_180d', 'vol_ratio_7d_180d', 'vol_ratio_30d_180d', 'vol_ratio_90d_180d',
            'price', 'ret_1d', 'low_52w', 'high_52w', 'from_low_pct', 'from_high_pct',
            'sma_20d', 'sma_50d', 'sma_200d', 'ret_3d', 'ret_7d', 'ret_30d', 'ret_3m',
            'ret_6m', 'ret_1y', 'ret_3y', 'ret_5y', 'rvol', 'prev_close', 'pe',
            'eps_current', 'eps_last_qtr', 'eps_change_pct', 'year'
        ]

        # Define columns that are percentages and might be stored as integers (e.g., 50 for 50%)
        percentage_value_cols_to_normalize = [
            'ret_1d', 'from_low_pct', 'from_high_pct', 'ret_3d', 'ret_7d', 'ret_30d', 'ret_3m',
            'ret_6m', 'ret_1y', 'ret_3y', 'ret_5y', 'eps_change_pct',
            'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d',
            'vol_ratio_1d_180d', 'vol_ratio_7d_180d', 'vol_ratio_30d_180d', 'vol_ratio_90d_180d'
        ]
        
        for col in numeric_cols_to_process:
            if col in df.columns:
                # Apply market_cap specific parsing
                if col == 'market_cap':
                    df[col] = df[col].apply(parse_market_cap)
                else:
                    # For other numeric columns, clean and convert
                    # Ensure it's string type before applying .str accessor
                    s_cleaned = df[col].astype(str).str.replace(r"[₹,$€£%,]", "", regex=True)
                    s_cleaned = s_cleaned.replace(["nan", "NaN", "NA", "-", ""], np.nan)
                    df[col] = pd.to_numeric(s_cleaned, errors='coerce')

        # Second pass for percentage columns to normalize values (e.g., 50 -> 0.50)
        for col in percentage_value_cols_to_normalize:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                non_na_values = df[col].dropna()
                # Only divide by 100 if values are likely integer percentages (e.g., 50 for 50%)
                # and not already decimals. Check absolute max value to avoid dividing small decimals.
                if not non_na_values.empty and non_na_values.abs().max() > 1 and non_na_values.abs().max() <= 1000:
                    df[col] = df[col] / 100.0
        
        # Fill NaN for critical numeric columns that are used in calculations to avoid errors
        # Use a sensible default (0 or 1 for ratios) if they are completely missing.
        df['price'] = df['price'].fillna(df['prev_close']).fillna(1.0)
        df['prev_close'] = df['prev_close'].fillna(df['price']).fillna(1.0)
        df['volume_1d'] = df['volume_1d'].fillna(0)
        df['volume_7d'] = df['volume_7d'].fillna(0)
        df['volume_30d'] = df['volume_30d'].fillna(0)
        df['volume_90d'] = df['volume_90d'].fillna(0)
        df['volume_180d'] = df['volume_180d'].fillna(0)
        df['rvol'] = df['rvol'].fillna(1.0) # Default RVOL to 1 (average volume)
        df['eps_current'] = df['eps_current'].fillna(0)
        df['eps_last_qtr'] = df['eps_last_qtr'].fillna(0)
        df['eps_change_pct'] = df['eps_change_pct'].fillna(0) # Fill with 0 for no change
        df['pe'] = df['pe'].fillna(0) # Fill PE with 0 if missing, might indicate no earnings or very high PE

        # Ensure 'sector' and 'category' are strings, fillna with "Unknown" for consistent grouping/filtering
        if 'sector' in df.columns:
            df['sector'] = df['sector'].astype(str).fillna("Unknown")
        if 'category' in df.columns:
            df['category'] = df['category'].astype(str).fillna("Unknown")

        # Calculate ATR after price is cleaned
        # Changed math.sqrt(2) to np.sqrt(2) for consistency with numpy operations
        df["atr_20"] = df['price'].rolling(20, min_periods=1).std().fillna(method="bfill").fillna(0) * np.sqrt(2)
        df["rs_volume_30d"] = df["volume_30d"] * df["price"]

        # Data quality score
        quality_checks = []
        for col in ['price', 'volume_1d', 'ticker']:
            if col in df.columns:
                non_null_pct = df[col].notna().sum() / len(df) * 100
                quality_checks.append(non_null_pct)
        
        diagnostics["data_quality_score"] = np.mean(quality_checks) if quality_checks else 0
        
        # Check data freshness (simplified - would need timestamp column)
        diagnostics["data_age_hours"] = 0  # Assume fresh for now
        
        # Remove invalid rows
        if 'price' in df.columns:
            initial_len = len(df)
            df = df[df['price'] > 0]
            if len(df) < initial_len:
                diagnostics["warnings"].append(f"Removed {initial_len - len(df)} rows with invalid prices")
        
        return df, diagnostics
        
    except Exception as e:
        diagnostics["warnings"].append(f"Data load error: {str(e)}")
        st.exception(e) # Display full traceback in Streamlit
        return pd.DataFrame(), diagnostics

# ============================================================================
# FIXED VOLUME ACCELERATION CALCULATION
# ============================================================================
def calculate_volume_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate CORRECT volume acceleration and metrics"""
    df = df.copy()
    
    # Ensure all required volume columns are numeric and filled before calculations
    volume_cols = ['volume_1d', 'volume_7d', 'volume_30d', 'volume_90d', 'volume_180d']
    for col in volume_cols:
        if col not in df.columns:
            df[col] = 0.0 # Add missing volume columns as 0
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    # Calculate Average Daily Volume for respective periods
    df['avg_vol_30d'] = df['volume_30d'] / 30.0
    df['avg_vol_90d'] = df['volume_90d'] / 90.0
    df['avg_vol_180d'] = df['volume_180d'] / 180.0

    # Calculate Volume Ratios (percentage change)
    # Ensure division by zero is handled for all ratios
    df['vol_ratio_30d_90d_calc'] = np.where(df['avg_vol_90d'] != 0,
                                            (df['avg_vol_30d'] / df['avg_vol_90d'] - 1) * 100, 0)
    df['vol_ratio_30d_180d_calc'] = np.where(df['avg_vol_180d'] != 0,
                                             (df['avg_vol_30d'] / df['avg_vol_180d'] - 1) * 100, 0)
    df['vol_ratio_90d_180d_calc'] = np.where(df['avg_vol_180d'] != 0,
                                             (df['avg_vol_90d'] / df['avg_vol_180d'] - 1) * 100, 0)
    
    # Create vol_ratio_7d_30d if not exists (needed for true acceleration)
    if 'vol_ratio_7d_30d' not in df.columns:
        df['vol_ratio_7d_30d'] = np.where(df['volume_30d'] != 0,
                                         (df['volume_7d'] / df['volume_30d'] - 1) * 100, 0)
    else: # Ensure it's numeric and filled if it exists
        df['vol_ratio_7d_30d'] = pd.to_numeric(df['vol_ratio_7d_30d'], errors='coerce').fillna(0)
    
    # CORRECT Volume Acceleration: Recent vs Past momentum
    # Use the calculated ratios directly
    df['volume_acceleration'] = df['vol_ratio_30d_90d_calc'] - df['vol_ratio_30d_180d_calc']
    
    # Alternative calculation if we have more granular data
    if 'vol_ratio_7d_30d' in df.columns:
        # Even better: Compare very recent to recent
        df['volume_acceleration_v2'] = df['vol_ratio_7d_30d'] - df['vol_ratio_30d_90d_calc']
        # Use the more sensitive metric if it exists and is positive
        df['volume_acceleration'] = df[['volume_acceleration', 'volume_acceleration_v2']].max(axis=1)
    
    # Volume consistency score (all timeframes positive)
    vol_ratio_cols_for_consistency = ['vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d']
    available_vol_cols = [col for col in vol_ratio_cols_for_consistency if col in df.columns]
    
    if available_vol_cols:
        df['volume_consistency'] = (df[available_vol_cols] > 0).sum(axis=1) / len(available_vol_cols)
    else:
        df['volume_consistency'] = 0.0
    
    # Volume intensity (RVOL * volume acceleration)
    if 'rvol' in df.columns:
        df['volume_intensity'] = df['rvol'] * np.maximum(df['volume_acceleration'], 0) / 100
    else:
        df['volume_intensity'] = 0.0
    
    # Classification with better thresholds
    conditions = [
        (df['volume_acceleration'] > 50) & (df['rvol'] > 2.0),
        df['volume_acceleration'] > 40,
        df['volume_acceleration'] > 25,
        df['volume_acceleration'] > 10,
        df['volume_acceleration'] > 0,
        df['volume_acceleration'] > -20
    ]
    
    choices = [
        "🔥 Explosive Accumulation",
        "🏦 Institutional Loading",
        "📈 Heavy Accumulation",
        "📊 Accumulation",
        "➕ Mild Accumulation",
        "➖ Distribution"
    ]
    
    df['volume_pattern'] = np.select(conditions, choices, default="💀 Heavy Distribution")
    
    return df

# ============================================================================
# ENHANCED PATTERN DETECTION (TOP 3 ONLY)
# ============================================================================
def detect_top_patterns(row: pd.Series) -> Dict:
    """Detect only the TOP 3 most effective patterns"""
    patterns = []
    
    # Pattern 1: Accumulation Under Resistance (BEST)
    # Ensure columns exist and are numeric before accessing
    vol_ratio_30d_90d = row.get('vol_ratio_30d_90d', np.nan)
    from_high_pct = row.get('from_high_pct', np.nan)
    rvol = row.get('rvol', np.nan)
    
    if pd.notna(vol_ratio_30d_90d) and pd.notna(from_high_pct) and pd.notna(rvol):
        score = 0
        signals = []
        
        # Volume explosion with price near resistance
        if vol_ratio_30d_90d > 0.40 and -0.20 <= from_high_pct <= -0.05: # Now expecting decimal percentages
            score = 70 + min(vol_ratio_30d_90d * 100 / 2, 20) # Scale vol_ratio back to percentage for scoring
            signals.append(f"Volume +{vol_ratio_30d_90d*100:.0f}% near resistance")
            
            if rvol > 2.0:
                score = min(score * 1.2, 100)
                signals.append(f"RVOL {rvol:.1f}x confirms")
        
        if score > 50:
            patterns.append({
                'name': 'Accumulation Under Resistance',
                'score': score,
                'signals': signals
            })
    
    # Pattern 2: Failed Breakdown Reversal
    from_low_pct = row.get('from_low_pct', np.nan)
    volume_acceleration = row.get('volume_acceleration', np.nan)
    ret_7d = row.get('ret_7d', np.nan)
    
    if pd.notna(from_low_pct) and pd.notna(volume_acceleration) and pd.notna(ret_7d):
        score = 0
        signals = []
        
        # Near 52w low with volume acceleration
        if from_low_pct < 0.15 and volume_acceleration > 20 and ret_7d > 0: # from_low_pct now decimal
            score = 60 + min(volume_acceleration/2, 30)
            signals.append(f"Reversal from low +{from_low_pct*100:.0f}%")
            signals.append(f"Volume accelerating {volume_acceleration:.0f}%")
            
            # Quality check
            ret_3y = row.get('ret_3y', np.nan)
            if pd.notna(ret_3y) and ret_3y > 2.00: # 200% return (decimal)
                score = min(score * 1.15, 100)
                signals.append("Quality stock reversal")
        
        if score > 50:
            patterns.append({
                'name': 'Failed Breakdown Reversal',
                'score': score,
                'signals': signals
            })
    
    # Pattern 3: Coiled Spring
    ret_30d = row.get('ret_30d', np.nan)
    vol_ratio_30d_90d = row.get('vol_ratio_30d_90d', np.nan)
    ret_7d_coiled = row.get('ret_7d', np.nan) # Using a different variable name to avoid conflict
    
    if pd.notna(ret_7d_coiled) and pd.notna(ret_30d) and pd.notna(vol_ratio_30d_90d):
        score = 0
        signals = []
        
        # Tight range with volume building
        if abs(ret_7d_coiled) < 0.05 and abs(ret_30d) < 0.10 and vol_ratio_30d_90d > 0.20: # Now expecting decimal percentages
            score = 50 + min(vol_ratio_30d_90d * 100, 40) # Scale vol_ratio back to percentage for scoring
            signals.append(f"Tight range with vol +{vol_ratio_30d_90d*100:.0f}%")
            
            # Above key SMAs bonus
            price = row.get('price', np.nan)
            sma_200d = row.get('sma_200d', np.nan)
            if pd.notna(price) and pd.notna(sma_200d) and price > sma_200d:
                score += 10
                signals.append("Above 200 SMA")
        
        if score > 50:
            patterns.append({
                'name': 'Coiled Spring',
                'score': score,
                'signals': signals
            })
    
    # Sort by score and return top pattern
    patterns.sort(key=lambda x: x['score'], reverse=True)
    
    if patterns:
        top_pattern = patterns[0]
        return {
            'pattern_name': top_pattern['name'],
            'pattern_score': top_pattern['score'],
            'pattern_signals': ', '.join(top_pattern['signals']),
            'pattern_count': len([p for p in patterns if p['score'] > 70])
        }
    
    return {
        'pattern_name': '',
        'pattern_score': 0,
        'pattern_signals': '',
        'pattern_count': 0
    }

# ============================================================================
# EDGE SCORING ENGINE WITH STRICTER CRITERIA
# ============================================================================
def calculate_edge_scores(df: pd.DataFrame, weights: Tuple[float, float, float]) -> pd.DataFrame:
    """Calculate EDGE scores with enhanced criteria"""
    df = df.copy()
    
    # Component 1: Volume Score (50% weight)
    df['vol_score'] = 0.0
    if 'volume_acceleration' in df.columns:
        # Base score from acceleration (volume_acceleration is already % points)
        df['vol_score'] = 50 + df['volume_acceleration'].clip(-50, 50)
        
        # RVOL multiplier
        if 'rvol' in df.columns:
            rvol_mult = df['rvol'].clip(0.5, 3.0)
            df.loc[df['volume_acceleration'] > 0, 'vol_score'] *= rvol_mult / 1.5
        
        # Volume consistency bonus
        if 'volume_consistency' in df.columns:
            df['vol_score'] += df['volume_consistency'] * 20
        
        df['vol_score'] = df['vol_score'].clip(0, 100)
    
    # Component 2: Momentum Score (30% weight)
    df['mom_score'] = 50.0  # Start neutral
    
    # Short-term momentum (ret_1d, ret_3d, ret_7d are now decimals)
    if all(col in df.columns for col in ['ret_1d', 'ret_3d', 'ret_7d']):
        short_momentum = (df['ret_1d'] * 0.5 + df['ret_3d'] * 0.3 + df['ret_7d'] * 0.2)
        df['mom_score'] += short_momentum * 300 # Scale up since returns are decimals now
        
        # Momentum consistency bonus
        momentum_aligned = (df['ret_1d'] > 0) & (df['ret_3d'] > 0) & (df['ret_7d'] > 0)
        df.loc[momentum_aligned, 'mom_score'] += 10
    
    # Medium-term trend (ret_30d is now decimal)
    if 'ret_30d' in df.columns:
        df['mom_score'] += (df['ret_30d'] * 100).clip(-10, 10) * 1.5 # Convert to % points for clipping
    
    df['mom_score'] = df['mom_score'].clip(0, 100)
    
    # Component 3: Risk/Reward Score (20% weight)
    df['rr_score'] = 50.0
    
    if all(col in df.columns for col in ['from_high_pct', 'from_low_pct']):
        # Distance from high (opportunity) - check each row (from_high_pct is now decimal)
        sweet_spot_mask = (df['from_high_pct'] >= -0.30) & (df['from_high_pct'] <= -0.10)
        df.loc[sweet_spot_mask, 'rr_score'] += 30
        
        # Distance from low (risk) (from_low_pct is now decimal)
        df['rr_score'] += (df['from_low_pct'] * 100 / 2).clip(0, 20) # Convert to % points for scaling/clipping
    
    # Quality discount (ret_3y is now decimal)
    if 'ret_3y' in df.columns:
        quality_stocks = df['ret_3y'] > 2.00 # 200% return (decimal)
        df.loc[quality_stocks, 'rr_score'] *= 1.2
    
    df['rr_score'] = df['rr_score'].clip(0, 100)
    
    # Calculate weighted EDGE score
    df['EDGE'] = (
        df['vol_score'] * weights[0] +
        df['mom_score'] * weights[1] +
        df['rr_score'] * weights[2]
    )
    
    # Pattern detection for top stocks only (performance optimization)
    top_edge_stocks = df.nlargest(MAX_PATTERN_DETECTION_STOCKS, 'EDGE')
    
    for idx in top_edge_stocks.index:
        pattern_data = detect_top_patterns(df.loc[idx])
        for key, value in pattern_data.items():
            df.loc[idx, key] = value
    
    # Fill pattern columns for other stocks
    for col in ['pattern_name', 'pattern_signals']:
        if col not in df.columns:
            df[col] = ''
    for col in ['pattern_score', 'pattern_count']:
        if col not in df.columns:
            df[col] = 0
    
    # Pattern bonus to EDGE score
    df.loc[df['pattern_score'] > 70, 'EDGE'] *= 1.1
    df.loc[df['pattern_count'] > 1, 'EDGE'] *= 1.05
    df['EDGE'] = df['EDGE'].clip(0, 100)
    
    return df

def detect_super_edge_strict(row: pd.Series, sector_ranks: Dict[str, int]) -> bool:
    """STRICTER SUPER EDGE detection (5 out of 6 conditions)"""
    conditions_met = 0
    
    # 1. High RVOL
    rvol = row.get('rvol', 0)
    if pd.notna(rvol) and rvol > 2.0:
        conditions_met += 1
    
    # 2. Strong volume acceleration (already in percentage points)
    vol_accel = row.get('volume_acceleration', 0)
    if pd.notna(vol_accel) and vol_accel > 30:
        conditions_met += 1
    
    # 3. EPS acceleration (eps_current and eps_last_qtr are now raw values, eps_change_pct is decimal)
    eps_current = row.get('eps_current', 0)
    eps_last = row.get('eps_last_qtr', 0)
    eps_change_pct = row.get('eps_change_pct', 0)

    if pd.notna(eps_change_pct) and eps_change_pct > 0.15: # 15% growth (decimal)
        conditions_met += 1
    
    # 4. Sweet spot zone (from_high_pct is now decimal)
    from_high = row.get('from_high_pct', -1.0) # Default to -100%
    if pd.notna(from_high) and -0.30 <= from_high <= -0.10: # -30% to -10% (decimal)
        conditions_met += 1
    
    # 5. Momentum alignment (ret_1d, ret_7d, ret_30d are now decimals)
    ret_1d = row.get('ret_1d', 0)
    ret_7d = row.get('ret_7d', 0)
    ret_30d = row.get('ret_30d', 0)
    if (pd.notna(ret_1d) and pd.notna(ret_7d) and pd.notna(ret_30d) and
        ret_1d > 0 and ret_7d > ret_1d and ret_30d > 0):
        conditions_met += 1
    
    # 6. Sector strength (NEW)
    sector = row.get('sector')
    if sector and sector in sector_ranks and sector_ranks[sector] <= 3:
        conditions_met += 1
    
    return conditions_met >= 5  # Raised from 4

# ============================================================================
# DYNAMIC STOP LOSS CALCULATION
# ============================================================================
def calculate_dynamic_stops(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate intelligent stop losses based on volatility and support"""
    df = df.copy()
    
    for idx in df.index:
        row = df.loc[idx]
        price = row.get('price', 100)
        
        # Method 1: ATR-based stop (if we have price history)
        atr_stop = price * 0.93  # Default 7%
        
        # Adjust for volatility category
        if 'category' in df.columns:
            if 'small' in str(row.get('category', '')).lower():
                atr_stop = price * 0.90  # 10% for small caps
            elif 'mid' in str(row.get('category', '')).lower():
                atr_stop = price * 0.92  # 8% for mid caps
        
        # Method 2: Support-based stop
        support_levels = []
        
        if row.get('sma_50d', 0) > 0:
            support_levels.append(row['sma_50d'] * 0.98)
        
        if row.get('sma_200d', 0) > 0:
            support_levels.append(row['sma_200d'] * 0.97)
        
        if row.get('low_52w', 0) > 0:
            support_levels.append(row['low_52w'] * 1.05)
        
        # Recent swing low approximation (ret_30d is now decimal)
        ret_30d_val = row.get('ret_30d', 0)
        if pd.notna(ret_30d_val) and ret_30d_val < 0:
            recent_low = price * (1 + ret_30d_val * 1.2) # ret_30d_val is already decimal
            support_levels.append(recent_low * 0.98)
        
        # Choose the most appropriate stop
        if support_levels:
            support_stop = max(support_levels)
            # Don't let stop be too close
            support_stop = min(support_stop, price * 0.95)
            
            # Final stop is maximum of ATR and support methods
            df.loc[idx, 'stop_loss'] = max(atr_stop, support_stop)
        else:
            df.loc[idx, 'stop_loss'] = atr_stop
        
        # Calculate stop percentage
        df.loc[idx, 'stop_loss_pct'] = ((df.loc[idx, 'stop_loss'] - price) / price) * 100
        
        # Risk-based position sizing adjustment
        stop_distance = abs(df.loc[idx, 'stop_loss_pct'])
        if stop_distance > 10:
            df.loc[idx, 'risk_adjustment'] = 0.7  # Reduce position size
        elif stop_distance > 7:
            df.loc[idx, 'risk_adjustment'] = 0.85
        else:
            df.loc[idx, 'risk_adjustment'] = 1.0
    
    return df

# ============================================================================
# PORTFOLIO RISK MANAGEMENT
# ============================================================================
def apply_portfolio_constraints(df: pd.DataFrame) -> pd.DataFrame:
    """Apply portfolio-level risk constraints"""
    df = df.copy()
    
    # Sort by EDGE score
    df = df.sort_values('EDGE', ascending=False)
    
    # Track allocations
    total_allocation = 0
    sector_allocations = {}
    position_count = 0
    super_edge_count = 0
    
    # Initialize position_size and portfolio_weight columns
    df['position_size'] = 0.0
    df['portfolio_weight'] = 0.0

    for idx in df.index:
        row = df.loc[idx]
        base_size = BASE_POSITION_SIZES.get(row.get('tag', 'WATCH'), 0)
        
        if base_size > 0:
            # Apply risk adjustment from stop loss calculation
            adjusted_size = base_size * row.get('risk_adjustment', 1.0)
            
            # Check constraints
            if position_count >= MAX_POSITIONS:
                adjusted_size = 0  # No more positions
            
            elif total_allocation + adjusted_size > MAX_PORTFOLIO_EXPOSURE:
                # Partial position to fill remaining
                adjusted_size = max(0, MAX_PORTFOLIO_EXPOSURE - total_allocation)
            
            # Sector concentration check
            sector = row.get('sector', 'Unknown')
            current_sector_allocation = sector_allocations.get(sector, 0)
            
            if current_sector_allocation + adjusted_size > MAX_SECTOR_EXPOSURE:
                adjusted_size = max(0, MAX_SECTOR_EXPOSURE - current_sector_allocation)
            
            # SUPER EDGE limit
            if row.get('tag') == 'SUPER_EDGE':
                if super_edge_count >= MAX_SUPER_EDGE_POSITIONS:
                    adjusted_size = min(adjusted_size, 0.05)  # Downgrade to small position
                else:
                    super_edge_count += 1
            
            # Update tracking
            if adjusted_size > 0:
                total_allocation += adjusted_size
                sector_allocations[sector] = current_sector_allocation + adjusted_size
                position_count += 1
            
            df.loc[idx, 'position_size'] = adjusted_size
            df.loc[idx, 'portfolio_weight'] = adjusted_size
        else:
            df.loc[idx, 'position_size'] = 0.0
            df.loc[idx, 'portfolio_weight'] = 0.0
    
    # Add portfolio metadata
    # This column will be the same for all rows, consider moving to diagnostics if not needed per row
    df['total_portfolio_allocation'] = total_allocation
    
    return df

# ============================================================================
# MAIN SCORING PIPELINE
# ============================================================================
def run_edge_analysis(df: pd.DataFrame, weights: Tuple[float, float, float]) -> pd.DataFrame:
    """Complete EDGE analysis pipeline"""
    # Calculate volume metrics
    df = calculate_volume_metrics(df)
    
    # Calculate sector rankings for SUPER EDGE detection
    sector_ranks = {}
    if 'sector' in df.columns and 'volume_acceleration' in df.columns:
        # Simple sector ranking by average volume acceleration
        sector_df = df.groupby('sector')['volume_acceleration'].agg(['mean', 'count'])
        # Only rank sectors with at least 3 stocks
        sector_df = sector_df[sector_df['count'] >= 3]
        if not sector_df.empty:
            sector_scores = sector_df['mean'].sort_values(ascending=False)
            for i, sector in enumerate(sector_scores.index):
                sector_ranks[sector] = i + 1
    
    # Calculate EDGE scores
    df = calculate_edge_scores(df, weights)
    
    # Classify stocks
    conditions = [
        df['EDGE'] >= EDGE_THRESHOLDS['EXPLOSIVE'],
        df['EDGE'] >= EDGE_THRESHOLDS['STRONG'],
        df['EDGE'] >= EDGE_THRESHOLDS['MODERATE']
    ]
    choices = ['EXPLOSIVE', 'STRONG', 'MODERATE']
    df['tag'] = np.select(conditions, choices, default='WATCH')
    
    # Detect SUPER EDGE with stricter criteria
    for idx in df[df['EDGE'] >= EDGE_THRESHOLDS['SUPER_EDGE']].index:
        if detect_super_edge_strict(df.loc[idx], sector_ranks):
            df.loc[idx, 'tag'] = 'SUPER_EDGE'
            df.loc[idx, 'EDGE'] = min(df.loc[idx, 'EDGE'] * 1.1, 100)  # Boost score
    
    # Calculate dynamic stops
    df = calculate_dynamic_stops(df)
    
    # Apply portfolio constraints
    df = apply_portfolio_constraints(df)
    
    # Calculate targets
    df['target_1'] = df['price'] * 1.07  # Conservative
    df['target_2'] = df['price'] * 1.15  # Aggressive
    
    # Adjust targets for SUPER EDGE
    super_mask = df['tag'] == 'SUPER_EDGE'
    df.loc[super_mask, 'target_1'] = df.loc[super_mask, 'price'] * 1.12
    df.loc[super_mask, 'target_2'] = df.loc[super_mask, 'price'] * 1.25
    
    # Add decision column
    df['decision'] = df['tag'].apply(lambda x: 
        'BUY NOW' if x == 'SUPER_EDGE' else
        'BUY' if x == 'EXPLOSIVE' else
        'ACCUMULATE' if x == 'STRONG' else
        'WATCH' if x == 'MODERATE' else
        'IGNORE'
    )
    
    return df

# ============================================================================
# UI COMPONENTS
# ============================================================================
def render_sidebar_diagnostics(diagnostics: Dict):
    """Render system health in sidebar"""
    with st.sidebar.expander("📊 System Health", expanded=False):
        # Data quality score with color
        quality_score = diagnostics.get('data_quality_score', 0)
        if quality_score > 90:
            st.success(f"Data Quality: {quality_score:.0f}%")
        elif quality_score > 70:
            st.warning(f"Data Quality: {quality_score:.0f}%")
        else:
            st.error(f"Data Quality: {quality_score:.0f}%")
        
        # Timestamp
        st.write(f"**Last Update:** {diagnostics.get('timestamp', 'Unknown').strftime('%H:%M:%S')}")
        
        # Data stats
        st.write(f"**Rows Loaded:** {diagnostics.get('rows_loaded', 0):,}")
        
        # Warnings
        warnings = diagnostics.get('warnings', [])
        if warnings:
            st.write("**⚠️ Warnings:**")
            for warning in warnings[:3]:  # Show max 3
                st.write(f"• {warning}")
        
        # Critical columns
        missing = diagnostics.get('critical_columns_missing', [])
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
        
        # Download diagnostic report
        diag_data = pd.DataFrame([diagnostics])
        csv = diag_data.to_csv(index=False)
        st.download_button(
            "📥 Diagnostic Report",
            csv,
            "diagnostics.csv",
            "text/csv",
            key="diag_download"
        )

def render_sector_leaderboard(df: pd.DataFrame):
    """Render sector leaderboard instead of heatmap"""
    st.header("🏆 Sector EDGE Leaderboard")
    
    if 'sector' not in df.columns:
        st.info("Sector data not available")
        return
    
    # Calculate sector metrics
    agg_dict = {'ticker': 'count'}
    
    if 'EDGE' in df.columns:
        agg_dict['EDGE'] = 'mean'
    
    if 'volume_acceleration' in df.columns:
        agg_dict['volume_acceleration'] = 'mean'
    
    if 'tag' in df.columns:
        agg_dict['tag'] = lambda x: (x == 'SUPER_EDGE').sum()
    
    sector_metrics = df.groupby('sector').agg(agg_dict)
    
    # Rename columns
    rename_dict = {'ticker': 'Total Stocks'}
    if 'EDGE' in sector_metrics.columns:
        rename_dict['EDGE'] = 'Avg EDGE'
    if 'volume_acceleration' in sector_metrics.columns:
        rename_dict['volume_acceleration'] = 'Avg Vol Accel'
    if 'tag' in sector_metrics.columns:
        rename_dict['tag'] = 'Super Edge'
    
    sector_metrics = sector_metrics.rename(columns=rename_dict).round(1)
    
    # Sort by EDGE if available
    if 'Avg EDGE' in sector_metrics.columns:
        sector_metrics = sector_metrics.sort_values('Avg EDGE', ascending=False)
    else:
        sector_metrics = sector_metrics.sort_values('Total Stocks', ascending=False)
    
    # Display top 10 sectors
    for idx, (sector, row) in enumerate(sector_metrics.head(10).iterrows()):
        medal = "🥇" if idx == 0 else "🥈" if idx == 1 else "🥉" if idx == 2 else f"{idx+1}."
        
        col1, col2, col3, col4 = st.columns([1, 3, 2, 2])
        with col1:
            st.write(f"**{medal}**")
        with col2:
            st.write(f"**{sector}**")
        with col3:
            edge_val = row.get('Avg EDGE', 'N/A')
            stocks_val = int(row.get('Total Stocks', 0))
            if edge_val != 'N/A':
                st.write(f"EDGE: {edge_val:.1f} | Stocks: {stocks_val}")
            else:
                st.write(f"Stocks: {stocks_val}")
        with col4:
            super_count = int(row.get('Super Edge', 0))
            if super_count > 0:
                st.write(f"⭐ Super: {super_count}")
            else:
                st.write("")
        
        # Mini bar chart for EDGE distribution (only if EDGE data exists)
        if 'EDGE' in df.columns:
            sector_stocks = df[df['sector'] == sector]['EDGE']
            if len(sector_stocks) > 2:
                fig = go.Figure(data=[go.Histogram(
                    x=sector_stocks,
                    nbinsx=10,
                    marker_color='lightblue',
                    showlegend=False
                )])
                fig.update_layout(
                    height=60,
                    margin=dict(l=0, r=0, t=0, b=0),
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True, key=f"sector_{idx}")

def create_excel_report(df_signals: pd.DataFrame, df_all: pd.DataFrame) -> io.BytesIO:
    """Create multi-sheet Excel report"""
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Sheet 1: Executive Summary
        summary_data = {
            'Metric': [
                'Total Signals',
                'SUPER EDGE Count',
                'EXPLOSIVE Count', 
                'Portfolio Allocation',
                'Top Sector',
                'Avg EDGE Score',
                'Report Date'
            ],
            'Value': [
                len(df_signals),
                (df_signals['tag'] == 'SUPER_EDGE').sum() if 'tag' in df_signals.columns else 0,
                (df_signals['tag'] == 'EXPLOSIVE').sum() if 'tag' in df_signals.columns else 0,
                f"{df_signals['portfolio_weight'].sum()*100:.1f}%" if 'portfolio_weight' in df_signals.columns else "0.0%",
                df_signals.groupby('sector')['EDGE'].mean().idxmax() if 'sector' in df_signals.columns and 'EDGE' in df_signals.columns and len(df_signals) > 0 else 'N/A',
                f"{df_signals['EDGE'].mean():.1f}" if 'EDGE' in df_signals.columns and len(df_signals) > 0 else "0.0",
                datetime.now().strftime('%Y-%m-%d %H:%M')
            ]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Executive Summary', index=False)
        
        # Sheet 2: Action Items (SUPER EDGE + EXPLOSIVE)
        if 'tag' in df_signals.columns:
            action_items = df_signals[df_signals['tag'].isin(['SUPER_EDGE', 'EXPLOSIVE'])].copy()
            if not action_items.empty:
                action_cols = [
                    'ticker', 'company_name', 'tag', 'EDGE', 'decision',
                    'price', 'position_size', 'stop_loss', 'target_1', 'target_2',
                    'volume_pattern', 'pattern_name'
                ]
                action_cols = [col for col in action_cols if col in action_items.columns]
                if action_cols:
                    action_items[action_cols].to_excel(writer, sheet_name='Action Items', index=False)
        
        # Sheet 3: All Signals
        signal_cols = [
            'ticker', 'company_name', 'sector', 'tag', 'EDGE',
            'price', 'volume_acceleration', 'pattern_name', 'decision'
        ]
        signal_cols = [col for col in signal_cols if col in df_signals.columns]
        if signal_cols:
            df_signals[signal_cols].to_excel(writer, sheet_name='All Signals', index=False)
        
        # Sheet 4: Pattern Analysis
        if 'pattern_score' in df_signals.columns:
            pattern_df = df_signals[df_signals['pattern_score'] > 0].copy()
            if not pattern_df.empty:
                pattern_cols = ['ticker', 'pattern_name', 'pattern_score', 'pattern_signals']
                pattern_cols = [col for col in pattern_cols if col in pattern_df.columns]
                if pattern_cols:
                    pattern_df[pattern_cols].to_excel(writer, sheet_name='Pattern Analysis', index=False)
        
        # Sheet 5: Sector Analysis
        if 'sector' in df_all.columns:
            agg_dict = {}
            if 'EDGE' in df_all.columns:
                agg_dict['EDGE'] = ['mean', 'max', 'count']
            if 'volume_acceleration' in df_all.columns:
                agg_dict['volume_acceleration'] = 'mean'
            
            if agg_dict:
                sector_analysis = df_all.groupby('sector').agg(agg_dict).round(1)
                # Flatten column names
                sector_analysis.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in sector_analysis.columns]
                sector_analysis.to_excel(writer, sheet_name='Sector Analysis')
        
        # Sheet 6: Raw Data (optional, first 1000 rows)
        df_all.head(1000).to_excel(writer, sheet_name='Raw Data', index=False)
    
    output.seek(0)
    return output

# ============================================================================
# MAIN UI FUNCTION
# ============================================================================
def render_ui():
    """Main Streamlit UI"""
    st.set_page_config(
        page_title="EDGE Protocol - Ultimate Trading Intelligence",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better UI
    st.markdown("""
    <style>
    .super-edge-alert {
        background: linear-gradient(135deg, #FFD700, #FFA500);
        color: black;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 20px;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { opacity: 0.8; }
        50% { opacity: 1; }
        100% { opacity: 0.8; }
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
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("⚡ EDGE Protocol - Ultimate Trading Intelligence")
    st.markdown("**Correct Volume Acceleration + Risk Management + Pattern Recognition = Superior Returns**")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Weights selection (simplified to 3 components)
        st.subheader("Strategy Weights")
        vol_weight = st.slider("Volume Weight %", 30, 70, 50, 5) / 100
        mom_weight = st.slider("Momentum Weight %", 10, 40, 30, 5) / 100
        rr_weight = 1 - vol_weight - mom_weight
        
        weights = (vol_weight, mom_weight, rr_weight)
        
        st.write(f"Risk/Reward: {rr_weight*100:.0f}%")
        
        st.markdown("---")
        
        # Filters
        st.subheader("🎯 Filters")
        min_edge = st.slider("Min EDGE Score", 0, 100, 50, 5)
        exclude_small_caps = st.checkbox("Exclude Small/Micro Caps", True)
        max_signals = st.slider("Max Signals", 10, 100, 50, 10)
    
    # Load data
    df, diagnostics = load_and_validate_data()
    
    # Show diagnostics in sidebar
    render_sidebar_diagnostics(diagnostics)
    
    if df.empty:
        st.error("❌ Failed to load data. Check connection and data source.")
        return
    
    # Apply basic filters
    if exclude_small_caps and 'category' in df.columns:
        df = df[~df['category'].str.contains('micro|nano|small', case=False, na=False)]
    
    # Run EDGE analysis
    with st.spinner("Running EDGE Protocol Analysis..."):
        df_analyzed = run_edge_analysis(df, weights)
    
    # Filter by minimum EDGE
    df_signals = df_analyzed[df_analyzed['EDGE'] >= min_edge].head(max_signals)
    
    # SUPER EDGE Alert
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
        "📊 Deep Analysis"
    ])
    
    # Tab 1: Trading Signals
    with tabs[0]:
        st.header("🎯 Today's Trading Signals")
        
        # Quick metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Signals", len(df_signals))
        with col2:
            super_edge_count = (df_signals['tag'] == 'SUPER_EDGE').sum() if 'tag' in df_signals.columns else 0
            st.metric("SUPER EDGE", super_edge_count)
        with col3:
            portfolio_used = df_signals['portfolio_weight'].sum() * 100 if 'portfolio_weight' in df_signals.columns else 0
            st.metric("Portfolio Used", f"{portfolio_used:.1f}%")
        with col4:
            avg_edge = df_signals['EDGE'].mean() if 'EDGE' in df_signals.columns and len(df_signals) > 0 else 0
            st.metric("Avg EDGE", f"{avg_edge:.1f}")
        with col5:
            patterns = (df_signals['pattern_score'] > 70).sum() if 'pattern_score' in df_signals.columns else 0
            st.metric("Strong Patterns", patterns)
        
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            # Ensure display_df is defined before accessing its columns
            display_df_for_filter = df_signals.copy() # Use a copy for filtering
            if 'tag' in display_df_for_filter.columns:
                unique_tags = display_df_for_filter['tag'].dropna().unique().tolist()
                signal_types = st.multiselect(
                    "Signal Types",
                    unique_tags,
                    default=unique_tags
                )
            else:
                signal_types = []
        
        with col2:
            if 'sector' in display_df_for_filter.columns:
                unique_sectors = display_df_for_filter['sector'].dropna().unique().tolist()
                sectors = st.multiselect(
                    "Sectors",
                    unique_sectors,
                    default=unique_sectors
                )
            else:
                sectors = []
                
        with col3:
            search = st.text_input("Search Ticker")
        
        # Apply filters
        display_df = df_signals.copy()
        if signal_types:
            display_df = display_df[display_df['tag'].isin(signal_types)]
        if sectors and 'sector' in display_df.columns:
            display_df = display_df[display_df['sector'].isin(sectors)]
        if search:
            display_df = display_df[display_df['ticker'].str.contains(search.upper(), na=False)]
        
        # Display main signals table
        if not display_df.empty:
            # Define columns to show
            display_cols = [
                'ticker', 'company_name', 'sector', 'tag', 'EDGE', 'decision',
                'price', 'position_size', 'stop_loss', 'stop_loss_pct',
                'target_1', 'target_2', 'volume_pattern', 'pattern_name'
            ]
            display_cols = [col for col in display_cols if col in display_df.columns]
            
            # Style the dataframe
            def style_signals(val):
                if val == 'SUPER_EDGE':
                    return 'background-color: gold; font-weight: bold;'
                elif val == 'EXPLOSIVE':
                    return 'background-color: #ffcccc;'
                elif val == 'BUY NOW':
                    return 'color: green; font-weight: bold;'
                return ''
            
            # Build format dict based on available columns
            format_dict = {}
            if 'EDGE' in display_cols:
                format_dict['EDGE'] = '{:.1f}'
            if 'price' in display_cols:
                format_dict['price'] = '₹{:.2f}'
            if 'position_size' in display_cols:
                format_dict['position_size'] = '{:.1%}'
            if 'stop_loss' in display_cols:
                format_dict['stop_loss'] = '₹{:.2f}'
            if 'stop_loss_pct' in display_cols:
                format_dict['stop_loss_pct'] = '{:.1f}%'
            if 'target_1' in display_cols:
                format_dict['target_1'] = '₹{:.2f}'
            if 'target_2' in display_cols:
                format_dict['target_2'] = '₹{:.2f}'
            
            # Apply styling
            styled_df = display_df[display_cols].style.format(format_dict)
            
            # Apply color map only to columns that exist
            if 'tag' in display_cols or 'decision' in display_cols:
                style_subset = []
                if 'tag' in display_cols:
                    style_subset.append('tag')
                if 'decision' in display_cols:
                    style_subset.append('decision')
                styled_df = styled_df.applymap(style_signals, subset=style_subset)
            
            st.dataframe(styled_df, use_container_width=True, height=600)
            
            # Portfolio allocation warning
            if 'portfolio_weight' in display_df.columns:
                total_allocation = display_df['portfolio_weight'].sum()
                if total_allocation > 0.7:
                    st.warning(f"⚠️ High portfolio allocation: {total_allocation*100:.1f}%")
            
            # Export buttons
            col1, col2 = st.columns(2)
            with col1:
                csv = display_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Signals (CSV)",
                    csv,
                    f"edge_signals_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv",
                    type="primary"
                )
            
            with col2:
                if len(df_signals) > 0 and len(df_analyzed) > 0:
                    excel_file = create_excel_report(df_signals, df_analyzed)
                    st.download_button(
                        "📊 Download Full Report (Excel)",
                        excel_file,
                        f"EDGE_Report_{datetime.now().strftime('%Y%m%d')}.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="primary"
                    )
        else:
            st.info("No signals match the selected filters.")
    
    # Tab 2: SUPER EDGE Focus
    with tabs[1]:
        st.header("⭐ SUPER EDGE Deep Dive")
        
        if 'tag' in df_signals.columns:
            super_df = df_signals[df_signals['tag'] == 'SUPER_EDGE']
        else:
            super_df = pd.DataFrame()  # Empty dataframe
        
        if not super_df.empty:
            st.success(f"🎯 {len(super_df)} SUPER EDGE opportunities meeting all 5+ criteria!")
            
            for idx, (_, row) in enumerate(super_df.iterrows()):
                with st.expander(
                    f"#{idx+1} {row['ticker']} - {row.get('company_name', 'N/A')} | EDGE: {row['EDGE']:.1f}",
                    expanded=(idx == 0)
                ):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.subheader("📊 Entry Details")
                        st.metric("Current Price", f"₹{row['price']:.2f}")
                        st.metric("Position Size", f"{row['position_size']*100:.1f}%")
                        st.metric("Stop Loss", f"₹{row['stop_loss']:.2f} ({row['stop_loss_pct']:.1f}%)")
                    
                    with col2:
                        st.subheader("🎯 Targets")
                        st.metric("Target 1 (12%)", f"₹{row['target_1']:.2f}")
                        st.metric("Target 2 (25%)", f"₹{row['target_2']:.2f}")
                        # Safe risk:reward calculation
                        price = row.get('price', 1)
                        stop_loss = row.get('stop_loss', price * 0.95)
                        target_1 = row.get('target_1', price * 1.12)
                        
                        if price > stop_loss:  # Ensure valid stop
                            risk_reward = abs(target_1 - price) / abs(price - stop_loss)
                            st.metric("Risk:Reward", f"1:{risk_reward:.1f}")
                        else:
                            st.metric("Risk:Reward", "N/A")
                    
                    with col3:
                        st.subheader("🔍 Key Signals")
                        st.write(f"**Volume:** {row.get('volume_pattern', 'N/A')}")
                        st.write(f"**Vol Accel:** {row.get('volume_acceleration', 0):.1f}%")
                        st.write(f"**RVOL:** {row.get('rvol', 0):.1f}x")
                        if row.get('pattern_name'):
                            st.write(f"**Pattern:** {row['pattern_name']}")
                    
                    # Why SUPER EDGE?
                    st.write("**Why SUPER EDGE?**")
                    criteria = []
                    if row.get('rvol', 0) > 2:
                        criteria.append("✅ RVOL > 2.0x")
                    if row.get('volume_acceleration', 0) > 30:
                        criteria.append("✅ Volume Acceleration > 30%")
                    if row.get('from_high_pct', -1.0) >= -0.30 and row.get('from_high_pct', 0) <= -0.10: # Now decimal
                        criteria.append("✅ In Sweet Spot Zone (-30% to -10% from High)")
                    if row.get('eps_change_pct', 0) > 0.15: # Now decimal
                        criteria.append("✅ EPS Growth > 15%")
                    
                    # Check if sector_ranks is available and contains the sector
                    sector_rank_val = sector_ranks.get(row.get('sector'), None)
                    if sector_rank_val is not None and sector_rank_val <= 3:
                         criteria.append(f"✅ Top Sector (Rank {sector_rank_val})")

                    st.write(" | ".join(criteria))
        else:
            st.info("No SUPER EDGE signals today. Check EXPLOSIVE category for high-conviction trades.")
            
            # Show next best opportunities
            if 'tag' in df_signals.columns:
                explosive_df = df_signals[df_signals['tag'] == 'EXPLOSIVE'].head(5)
            else:
                explosive_df = pd.DataFrame()
                
            if not explosive_df.empty:
                st.subheader("🔥 Top EXPLOSIVE Opportunities")
                cols = ['ticker', 'company_name', 'EDGE', 'price', 'volume_pattern']
                cols = [col for col in cols if col in explosive_df.columns]
                if cols:
                    st.dataframe(explosive_df[cols], use_container_width=True)
    
    # Tab 3: Sector Leaders
    with tabs[2]:
        render_sector_leaderboard(df_analyzed)
    
    # Tab 4: Deep Analysis
    with tabs[3]:
        st.header("📊 Deep Market Analysis")
        
        # Volume Acceleration Distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Volume Acceleration Distribution")
            
            if 'volume_acceleration' in df_analyzed.columns:
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=df_analyzed['volume_acceleration'],
                    nbinsx=30,
                    name='All Stocks',
                    marker_color='lightblue'
                ))
                
                if len(df_signals) > 0 and 'volume_acceleration' in df_signals.columns:
                    fig.add_trace(go.Histogram(
                        x=df_signals['volume_acceleration'],
                        nbinsx=20,
                        name='Signal Stocks',
                        marker_color='gold',
                        opacity=0.7
                    ))
                
                fig.update_layout(
                    xaxis_title="Volume Acceleration %",
                    yaxis_title="Count",
                    height=400,
                    barmode='overlay'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Volume acceleration data not available")
        
        with col2:
            st.subheader("🎯 EDGE Score Distribution")
            
            if 'EDGE' in df_analyzed.columns:
                fig2 = go.Figure()
                
                # Add regions
                fig2.add_vrect(x0=0, x1=50, fillcolor="red", opacity=0.1, annotation_text="Watch")
                fig2.add_vrect(x0=50, x1=70, fillcolor="yellow", opacity=0.1, annotation_text="Moderate")
                fig2.add_vrect(x0=70, x1=85, fillcolor="orange", opacity=0.1, annotation_text="Strong")
                fig2.add_vrect(x0=85, x1=100, fillcolor="green", opacity=0.1, annotation_text="Explosive")
                
                fig2.add_trace(go.Histogram(
                    x=df_analyzed['EDGE'],
                    nbinsx=25,
                    marker_color='darkblue'
                ))
                
                fig2.update_layout(
                    xaxis_title="EDGE Score",
                    yaxis_title="Count",
                    height=400
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("EDGE score data not available")
        
        # Pattern Analysis
        if 'pattern_name' in df_analyzed.columns:
            st.subheader("🎯 Pattern Detection Summary")
            
            # Filter out empty pattern names
            pattern_data = df_analyzed[df_analyzed['pattern_name'].notna() & (df_analyzed['pattern_name'] != '')]
            
            if len(pattern_data) > 0:
                pattern_summary = pattern_data['pattern_name'].value_counts()
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    fig = go.Figure(data=[go.Bar(
                        x=pattern_summary.index,
                        y=pattern_summary.values,
                        text=pattern_summary.values,
                        textposition='auto',
                        marker_color=['gold' if 'Accumulation' in p else 'lightblue' for p in pattern_summary.index]
                    )])
                    
                    fig.update_layout(
                        xaxis_title="Pattern Type",
                        yaxis_title="Count",
                        height=350
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.write("**Pattern Strength:**")
                    if 'pattern_score' in df_analyzed.columns:
                        strong_patterns = df_analyzed[df_analyzed['pattern_score'] > 80]
                        st.metric("Score > 80", len(strong_patterns))
                    else:
                        st.metric("Score > 80", 0)
                    
                    if 'pattern_count' in df_analyzed.columns:
                        multi_patterns = (df_analyzed['pattern_count'] > 1).sum()
                        st.metric("Multi-Pattern", multi_patterns)
                    else:
                        st.metric("Multi-Pattern", 0)
            else:
                st.info("No patterns detected in current market conditions")
        
        # Market Breadth
        st.subheader("📊 Market Breadth Analysis")
        
        breadth_metrics = {}
        
        if 'ret_1d' in df_analyzed.columns:
            breadth_metrics['Stocks Advancing'] = (df_analyzed['ret_1d'] > 0).sum()
            breadth_metrics['Stocks Declining'] = (df_analyzed['ret_1d'] < 0).sum()
        
        if 'price' in df_analyzed.columns and 'sma_50d' in df_analyzed.columns:
            breadth_metrics['Above 50 SMA'] = (df_analyzed['price'] > df_analyzed['sma_50d']).sum()
        
        if 'price' in df_analyzed.columns and 'sma_200d' in df_analyzed.columns:
            breadth_metrics['Above 200 SMA'] = (df_analyzed['price'] > df_analyzed['sma_200d']).sum()
        
        if 'rvol' in df_analyzed.columns:
            breadth_metrics['High Volume (RVOL>1.5)'] = (df_analyzed['rvol'] > 1.5).sum()
        
        if 'volume_acceleration' in df_analyzed.columns:
            breadth_metrics['Accumulation Patterns'] = (df_analyzed['volume_acceleration'] > 10).sum()
        
        col1, col2, col3 = st.columns(3)
        for i, (metric, value) in enumerate(breadth_metrics.items()):
            with [col1, col2, col3][i % 3]:
                if len(df_analyzed) > 0:
                    pct = value / len(df_analyzed) * 100
                    st.metric(metric, f"{value} ({pct:.0f}%)")
                else:
                    st.metric(metric, "0 (0%)")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    render_ui()
