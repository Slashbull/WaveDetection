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
import datetime # Changed from 'from datetime import datetime, timedelta' to fix NameError

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

def calculate_atr(prices: pd.Series, period: int = 20) -> float:
    """Calculate Average True Range for dynamic stops"""
    if len(prices) < period:
        return prices.std() * 2.0 if len(prices) > 0 else 0.0
    
    # Simple ATR calculation using price volatility
    high_low_pct = prices.pct_change().abs()
    atr = high_low_pct.rolling(window=period, min_periods=1).mean() * prices.iloc[-1]
    return atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else prices.std() * 2.0

# ============================================================================
# DATA LOADING WITH VALIDATION
# ============================================================================
@st.cache_data(ttl=CACHE_TTL)
def load_and_validate_data() -> Tuple[pd.DataFrame, Dict[str, any]]:
    """Load data with comprehensive validation and diagnostics"""
    diagnostics = {
        "timestamp": datetime.datetime.now(), # Corrected to datetime.datetime.now()
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
        
        # Clean numeric columns
        numeric_cols = [
            'price', 'volume_1d', 'volume_7d', 'volume_30d', 'volume_90d', 'volume_180d',
            'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d',
            'vol_ratio_1d_180d', 'vol_ratio_7d_180d', 'vol_ratio_30d_180d',
            'ret_1d', 'ret_3d', 'ret_7d', 'ret_30d', 'ret_3m', 'ret_6m', 'ret_1y', 'ret_3y',
            'low_52w', 'high_52w', 'from_low_pct', 'from_high_pct',
            'sma_20d', 'sma_50d', 'sma_200d', 'rvol', 'pe',
            'eps_current', 'eps_last_qtr', 'eps_change_pct'
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                # Clean and convert
                df[col] = pd.to_numeric(
                    df[col].astype(str)
                    .str.replace(r"[₹,$€£%,]", "", regex=True)
                    .str.replace("cr", "", regex=False)
                    .str.replace("Cr", "", regex=False)
                    .replace(["", "-", "nan", "NaN", "NA"], np.nan),
                    errors='coerce'
                )
        
        # Handle negative volume ratios (they're valid - showing decrease)
        # Just ensure they're numeric
        for col in df.columns:
            if 'vol_ratio' in col and col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Market cap special handling
        if 'market_cap' in df.columns:
            df['market_cap_num'] = df['market_cap'].apply(parse_market_cap)
        
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
        return pd.DataFrame(), diagnostics

def parse_market_cap(val: Union[str, float]) -> float:
    """Parse market cap with Indian notation"""
    if pd.isna(val):
        return np.nan
    
    val_str = str(val).strip()
    
    # Remove currency symbols
    val_str = val_str.replace("₹", "").replace(",", "").strip()
    
    # Handle Indian notation
    multipliers = {
        'cr': 1e7, 'Cr': 1e7, 'crore': 1e7,
        'l': 1e5, 'L': 1e5, 'lakh': 1e5, 'lac': 1e5
    }
    
    for suffix, multiplier in multipliers.items():
        if suffix in val_str:
            try:
                number = float(val_str.replace(suffix, '').strip())
                return number * multiplier
            except:
                return np.nan
    
    try:
        return float(val_str)
    except:
        return np.nan

# ============================================================================
# FIXED VOLUME ACCELERATION CALCULATION
# ============================================================================
def calculate_volume_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate CORRECT volume acceleration and metrics"""
    df_copy = df.copy() # Use a copy to avoid SettingWithCopyWarning
    
    # Create vol_ratio_7d_30d if not exists (needed for true acceleration)
    if 'vol_ratio_7d_30d' not in df_copy.columns:
        if all(col in df_copy.columns for col in ['volume_7d', 'volume_30d']):
            # Calculate 7d vs 30d ratio safely
            df_copy['vol_ratio_7d_30d'] = 0
            mask = df_copy['volume_30d'] > 0
            df_copy.loc[mask, 'vol_ratio_7d_30d'] = (
                (df_copy.loc[mask, 'volume_7d'] - df_copy.loc[mask, 'volume_30d']) / 
                df_copy.loc[mask, 'volume_30d'] * 100
            )
    
    # CORRECT Volume Acceleration: Recent vs Past momentum
    if all(col in df_copy.columns for col in ['vol_ratio_7d_90d', 'vol_ratio_30d_90d']):
        # True acceleration: Is 7-day volume growing faster than 30-day?
        df_copy['volume_acceleration'] = df_copy['vol_ratio_7d_90d'] - df_copy['vol_ratio_30d_90d']
        
        # Alternative calculation if we have more granular data
        if 'vol_ratio_7d_30d' in df_copy.columns:
            # Even better: Compare very recent to recent
            df_copy['volume_acceleration_v2'] = df_copy['vol_ratio_7d_30d'] - df_copy['vol_ratio_30d_90d']
            # Use the more sensitive metric
            df_copy['volume_acceleration'] = df_copy[['volume_acceleration', 'volume_acceleration_v2']].max(axis=1)
    else:
        # Fallback
        df_copy['volume_acceleration'] = 0
    
    # Volume consistency score (all timeframes positive)
    vol_cols = ['vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d']
    available_vol_cols = [col for col in vol_cols if col in df_copy.columns]
    
    if available_vol_cols:
        df_copy['volume_consistency'] = (df_copy[available_vol_cols] > 0).sum(axis=1) / len(available_vol_cols)
    else:
        df_copy['volume_consistency'] = 0
    
    # Volume intensity (RVOL * volume acceleration)
    if 'rvol' in df_copy.columns:
        df_copy['volume_intensity'] = df_copy['rvol'] * np.maximum(df_copy['volume_acceleration'], 0) / 100
    else:
        df_copy['volume_intensity'] = 0
    
    # Classification with better thresholds
    conditions = [
        (df_copy['volume_acceleration'] > 50) & (df_copy['rvol'] > 2.0),
        df_copy['volume_acceleration'] > 40,
        df_copy['volume_acceleration'] > 25,
        df_copy['volume_acceleration'] > 10,
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
# ENHANCED PATTERN DETECTION (TOP 3 ONLY)
# ============================================================================
def detect_top_patterns(row: pd.Series) -> Dict:
    """Detect only the TOP 3 most effective patterns"""
    patterns = []
    
    # Pattern 1: Accumulation Under Resistance (BEST)
    vol_ratio = row.get('vol_ratio_30d_90d', 0)
    from_high = row.get('from_high_pct', 0)
    rvol = row.get('rvol', 1)
    
    if not pd.isna(vol_ratio) and not pd.isna(from_high) and not pd.isna(rvol):
        score = 0
        signals = []
        
        # Volume explosion with price near resistance
        if vol_ratio > 40 and -20 <= from_high <= -5:
            score = 70 + min(vol_ratio/2, 20)
            signals.append(f"Volume +{vol_ratio:.0f}% near resistance")
            
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
    from_low = row.get('from_low_pct', 100)
    vol_accel = row.get('volume_acceleration', 0)
    ret_7d = row.get('ret_7d', 0)
    
    if not pd.isna(from_low) and not pd.isna(vol_accel) and not pd.isna(ret_7d):
        score = 0
        signals = []
        
        # Near 52w low with volume acceleration
        if from_low < 15 and vol_accel > 20 and ret_7d > 0:
            score = 60 + min(vol_accel/2, 30)
            signals.append(f"Reversal from low +{from_low:.0f}%")
            signals.append(f"Volume accelerating {vol_accel:.0f}%")
            
            # Quality check
            ret_3y = row.get('ret_3y', 0)
            if not pd.isna(ret_3y) and ret_3y > 200:
                score = min(score * 1.15, 100)
                signals.append("Quality stock reversal")
        
        if score > 50:
            patterns.append({
                'name': 'Failed Breakdown Reversal',
                'score': score,
                'signals': signals
            })
    
    # Pattern 3: Coiled Spring
    ret_30d = row.get('ret_30d', 0)
    vol_ratio = row.get('vol_ratio_30d_90d', 0)
    
    if not pd.isna(ret_7d) and not pd.isna(ret_30d) and not pd.isna(vol_ratio):
        score = 0
        signals = []
        
        # Tight range with volume building
        if abs(ret_7d) < 5 and abs(ret_30d) < 10 and vol_ratio > 20:
            score = 50 + min(vol_ratio, 40)
            signals.append(f"Tight range with vol +{vol_ratio:.0f}%")
            
            # Above key SMAs bonus
            price = row.get('price', 0)
            sma_200d = row.get('sma_200d', price)
            if not pd.isna(price) and not pd.isna(sma_200d) and price > sma_200d:
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
    df_copy = df.copy() # Use a copy to avoid SettingWithCopyWarning
    
    # Component 1: Volume Score (50% weight)
    df_copy['vol_score'] = 0
    if 'volume_acceleration' in df_copy.columns:
        # Base score from acceleration
        df_copy['vol_score'] = 50 + df_copy['volume_acceleration'].clip(-50, 50)
        
        # RVOL multiplier
        if 'rvol' in df_copy.columns:
            rvol_mult = df_copy['rvol'].clip(0.5, 3.0)
            df_copy.loc[df_copy['volume_acceleration'] > 0, 'vol_score'] *= rvol_mult / 1.5
        
        # Volume consistency bonus
        if 'volume_consistency' in df_copy.columns:
            df_copy['vol_score'] += df_copy['volume_consistency'] * 20
        
        df_copy['vol_score'] = df_copy['vol_score'].clip(0, 100)
    
    # Component 2: Momentum Score (30% weight)
    df_copy['mom_score'] = 50  # Start neutral
    
    # Short-term momentum
    if all(col in df_copy.columns for col in ['ret_1d', 'ret_3d', 'ret_7d']):
        short_momentum = (df_copy['ret_1d'] * 0.5 + df_copy['ret_3d'] * 0.3 + df_copy['ret_7d'] * 0.2)
        df_copy['mom_score'] += short_momentum * 3
        
        # Momentum consistency bonus
        momentum_aligned = (df_copy['ret_1d'] > 0) & (df_copy['ret_3d'] > 0) & (df_copy['ret_7d'] > 0)
        df_copy.loc[momentum_aligned, 'mom_score'] += 10
    
    # Medium-term trend
    if 'ret_30d' in df_copy.columns:
        df_copy['mom_score'] += df_copy['ret_30d'].clip(-10, 10) * 1.5
    
    df_copy['mom_score'] = df_copy['mom_score'].clip(0, 100)
    
    # Component 3: Risk/Reward Score (20% weight)
    df_copy['rr_score'] = 50
    
    if all(col in df_copy.columns for col in ['from_high_pct', 'from_low_pct']):
        # Distance from high (opportunity) - check each row
        sweet_spot_mask = (df_copy['from_high_pct'] >= -30) & (df_copy['from_high_pct'] <= -10)
        df_copy.loc[sweet_spot_mask, 'rr_score'] += 30
        
        # Distance from low (risk)
        df_copy['rr_score'] += (df_copy['from_low_pct'] / 2).clip(0, 20)
    
    # Quality discount
    if 'ret_3y' in df_copy.columns:
        quality_stocks = df_copy['ret_3y'] > 200
        df_copy.loc[quality_stocks, 'rr_score'] *= 1.2
    
    df_copy['rr_score'] = df_copy['rr_score'].clip(0, 100)
    
    # Calculate weighted EDGE score
    df_copy['EDGE'] = (
        df_copy['vol_score'] * weights[0] +
        df_copy['mom_score'] * weights[1] +
        df_copy['rr_score'] * weights[2]
    )
    
    # Pattern detection for top stocks only (performance optimization)
    top_edge_stocks = df_copy.nlargest(MAX_PATTERN_DETECTION_STOCKS, 'EDGE', keep='first') # Use keep='first' for consistency
    
    for idx in top_edge_stocks.index:
        pattern_data = detect_top_patterns(df_copy.loc[idx])
        for key, value in pattern_data.items():
            df_copy.loc[idx, key] = value
    
    # Fill pattern columns for other stocks (ensure they exist even if no patterns detected)
    for col in ['pattern_name', 'pattern_signals']:
        if col not in df_copy.columns:
            df_copy[col] = ''
    for col in ['pattern_score', 'pattern_count']:
        if col not in df_copy.columns:
            df_copy[col] = 0
    
    # Pattern bonus to EDGE score
    df_copy.loc[df_copy['pattern_score'] > 70, 'EDGE'] *= 1.1
    df_copy.loc[df_copy['pattern_count'] > 1, 'EDGE'] *= 1.05
    df_copy['EDGE'] = df_copy['EDGE'].clip(0, 100)
    
    return df_copy

def detect_super_edge_strict(row: pd.Series, sector_ranks: Dict[str, int]) -> bool:
    """STRICTER SUPER EDGE detection (5 out of 6 conditions)"""
    conditions_met = 0
    
    # 1. High RVOL
    rvol = row.get('rvol', 0)
    if not pd.isna(rvol) and rvol > 2.0:
        conditions_met += 1
    
    # 2. Strong volume acceleration
    vol_accel = row.get('volume_acceleration', 0)
    if not pd.isna(vol_accel) and vol_accel > 30:
        conditions_met += 1
    
    # 3. EPS acceleration
    eps_current = row.get('eps_current', 0)
    eps_last = row.get('eps_last_qtr', 0)
    if not pd.isna(eps_current) and not pd.isna(eps_last) and eps_current > 0 and eps_last > 0:
        eps_growth = safe_divide(eps_current - eps_last, eps_last)
        if eps_growth > 0.15:  # 15% QoQ growth
            conditions_met += 1
    
    # 4. Sweet spot zone
    from_high = row.get('from_high_pct', -100)
    if not pd.isna(from_high) and -30 <= from_high <= -10:
        conditions_met += 1
    
    # 5. Momentum alignment
    ret_1d = row.get('ret_1d', 0)
    ret_7d = row.get('ret_7d', 0)
    ret_30d = row.get('ret_30d', 0)
    if (not pd.isna(ret_1d) and not pd.isna(ret_7d) and not pd.isna(ret_30d) and
        ret_1d > 0 and ret_7d > ret_1d and ret_30d > 0):
        conditions_met += 1
    
    # 6. Sector strength (NEW)
    sector = row.get('sector')
    # Ensure sector_ranks is not empty and sector exists in it
    sector_rank = sector_ranks.get(sector, 999) if sector else 999
    if sector_rank <= 3:
        conditions_met += 1
    
    return conditions_met >= 5  # Raised from 4

# ============================================================================
# DYNAMIC STOP LOSS CALCULATION
# ============================================================================
def calculate_dynamic_stops(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate intelligent stop losses based on volatility and support"""
    df_copy = df.copy() # Use a copy to avoid SettingWithCopyWarning
    
    df_copy['stop_loss'] = np.nan
    df_copy['stop_loss_pct'] = np.nan
    df_copy['risk_adjustment'] = 1.0 # Default to no adjustment
    
    for idx in df_copy.index:
        row = df_copy.loc[idx]
        price = row.get('price', 100)
        if pd.isna(price) or price <= 0: # Ensure valid price
            continue
            
        # Method 1: ATR-based stop (if we have price history)
        # Using a fixed percentage based on category as a proxy for ATR
        atr_stop = price * 0.93  # Default 7% below price
        
        # Adjust for volatility category
        category = str(row.get('category', '')).lower()
        if 'small' in category or 'micro' in category:
            atr_stop = price * 0.90  # 10% for small/micro caps (higher volatility)
        elif 'mid' in category:
            atr_stop = price * 0.92  # 8% for mid caps
        
        # Method 2: Support-based stop (using SMAs and 52-week low)
        support_levels = []
        
        # SMAs as dynamic support levels (with a small buffer below them)
        if row.get('sma_50d', 0) > 0 and price > row.get('sma_50d', 0): # Only if price is above SMA
            support_levels.append(row['sma_50d'] * 0.98) # 2% below 50 SMA
        
        if row.get('sma_200d', 0) > 0 and price > row.get('sma_200d', 0): # Only if price is above SMA
            support_levels.append(row['sma_200d'] * 0.97) # 3% below 200 SMA
        
        # 52-week low as a strong support (with a buffer above it)
        if row.get('low_52w', 0) > 0:
            support_levels.append(row['low_52w'] * 1.05) # 5% above 52w low
            
        # Recent swing low approximation (if price has pulled back recently)
        ret_30d_val = row.get('ret_30d', 0)
        if not pd.isna(ret_30d_val) and ret_30d_val < 0: # If 30-day return is negative (pullback)
            # Estimate a recent low based on price and 30-day return, then take a small buffer below it
            recent_low_approx = price * (1 + ret_30d_val/100 * 1.2) # Adjust for deeper pullback
            support_levels.append(recent_low_approx * 0.98)
        
        # Choose the most appropriate stop from support levels
        if support_levels:
            support_stop = max(support_levels) # Take the highest (closest to price, strongest support)
            # Ensure support-based stop is not too tight (at least 5% below current price)
            support_stop = min(support_stop, price * 0.95)
            
            # Final stop is the maximum of ATR-based and support-based stops
            df_copy.loc[idx, 'stop_loss'] = max(atr_stop, support_stop)
        else:
            df_copy.loc[idx, 'stop_loss'] = atr_stop # Fallback to ATR if no valid support levels
            
        # Ensure stop is not above current price and is within reasonable bounds
        df_copy.loc[idx, 'stop_loss'] = min(df_copy.loc[idx, 'stop_loss'], price * 0.95) # Max 5% drop
        df_copy.loc[idx, 'stop_loss'] = max(df_copy.loc[idx, 'stop_loss'], price * 0.70) # Min 30% drop (prevent absurdly wide stops)
            
        # Calculate stop percentage
        df_copy.loc[idx, 'stop_loss_pct'] = safe_divide((df_copy.loc[idx, 'stop_loss'] - price) * 100, price, 0)
        
        # Risk-based position sizing adjustment: reduce size for wider stops
        stop_distance = abs(df_copy.loc[idx, 'stop_loss_pct'])
        if stop_distance > 10:
            df_copy.loc[idx, 'risk_adjustment'] = 0.7  # Reduce position size significantly
        elif stop_distance > 7:
            df_copy.loc[idx, 'risk_adjustment'] = 0.85 # Moderate reduction
        else:
            df_copy.loc[idx, 'risk_adjustment'] = 1.0  # Full position size
    
    return df_copy

# ============================================================================
# PORTFOLIO RISK MANAGEMENT
# ============================================================================
def apply_portfolio_constraints(df: pd.DataFrame) -> pd.DataFrame:
    """Apply portfolio-level risk constraints"""
    df_copy = df.copy() # Use a copy to avoid SettingWithCopyWarning
    
    # Initialize allocation columns
    df_copy['position_size'] = 0.0
    df_copy['portfolio_weight'] = 0.0

    # Sort by EDGE score (descending) to prioritize higher conviction trades
    df_copy = df_copy.sort_values('EDGE', ascending=False)
    
    # Track allocations
    total_allocation = 0.0
    sector_allocations: Dict[str, float] = {}
    position_count = 0
    super_edge_count = 0
    
    for idx in df_copy.index:
        row = df_copy.loc[idx]
        base_size = BASE_POSITION_SIZES.get(row.get('tag', 'WATCH'), 0.0)
        
        if base_size > 0:
            # Apply risk adjustment from stop loss calculation
            adjusted_size = base_size * row.get('risk_adjustment', 1.0)
            
            # Check total position count constraint
            if position_count >= MAX_POSITIONS:
                adjusted_size = 0.0  # No more positions allowed
            
            # Check total portfolio exposure
            if total_allocation + adjusted_size > MAX_PORTFOLIO_EXPOSURE:
                # Only allocate remaining capacity
                adjusted_size = max(0.0, MAX_PORTFOLIO_EXPOSURE - total_allocation)
            
            # Sector concentration check
            sector = row.get('sector', 'Unknown')
            current_sector_allocation = sector_allocations.get(sector, 0.0)
            
            if current_sector_allocation + adjusted_size > MAX_SECTOR_EXPOSURE:
                # Only allocate remaining sector capacity
                adjusted_size = max(0.0, MAX_SECTOR_EXPOSURE - current_sector_allocation)
            
            # SUPER EDGE specific limit
            if row.get('tag') == 'SUPER_EDGE':
                if super_edge_count >= MAX_SUPER_EDGE_POSITIONS:
                    adjusted_size = min(adjusted_size, BASE_POSITION_SIZES.get('STRONG', 0.05))  # Downgrade to a smaller position
                else:
                    super_edge_count += 1
            
            # Update tracking if allocation is still positive
            if adjusted_size > 0.005: # Minimum allocation threshold (0.5%)
                total_allocation += adjusted_size
                sector_allocations[sector] = current_sector_allocation + adjusted_size
                position_count += 1
                
                df_copy.loc[idx, 'position_size'] = adjusted_size
                df_copy.loc[idx, 'portfolio_weight'] = adjusted_size
            else:
                df_copy.loc[idx, 'position_size'] = 0.0
                df_copy.loc[idx, 'portfolio_weight'] = 0.0
        else:
            df_copy.loc[idx, 'position_size'] = 0.0
            df_copy.loc[idx, 'portfolio_weight'] = 0.0
            
    # Add portfolio metadata for UI
    df_copy['total_portfolio_allocation'] = total_allocation
    df_copy['current_positions_count'] = position_count
    
    return df_copy

# ============================================================================
# MAIN SCORING PIPELINE
# ============================================================================
def run_edge_analysis(df: pd.DataFrame, weights: Tuple[float, float, float]) -> pd.DataFrame:
    """Complete EDGE analysis pipeline"""
    if df.empty:
        return pd.DataFrame()

    # Calculate volume metrics
    df_processed = calculate_volume_metrics(df.copy()) # Pass a copy

    # Calculate sector rankings for SUPER EDGE detection
    sector_ranks = {}
    if 'sector' in df_processed.columns and 'volume_acceleration' in df_processed.columns:
        # Simple sector ranking by average volume acceleration
        sector_df = df_processed.groupby('sector')['volume_acceleration'].agg(['mean', 'count'])
        # Only rank sectors with at least 3 stocks
        sector_df = sector_df[sector_df['count'] >= 3]
        if not sector_df.empty:
            sector_scores = sector_df['mean'].sort_values(ascending=False)
            for i, sector in enumerate(sector_scores.index):
                sector_ranks[sector] = i + 1
    
    # Calculate EDGE scores
    df_processed = calculate_edge_scores(df_processed, weights)
    
    # Classify stocks (initial classification based on EDGE score)
    conditions = [
        df_processed['EDGE'] >= EDGE_THRESHOLDS['EXPLOSIVE'],
        df_processed['EDGE'] >= EDGE_THRESHOLDS['STRONG'],
        df_processed['EDGE'] >= EDGE_THRESHOLDS['MODERATE']
    ]
    choices = ['EXPLOSIVE', 'STRONG', 'MODERATE']
    df_processed['tag'] = np.select(conditions, choices, default='WATCH')
    
    # Detect SUPER EDGE with stricter criteria (re-evaluates and upgrades tag)
    # Iterate only over stocks that are already high EDGE candidates
    super_edge_candidates_idx = df_processed[df_processed['EDGE'] >= EDGE_THRESHOLDS['SUPER_EDGE']].index
    for idx in super_edge_candidates_idx:
        if detect_super_edge_strict(df_processed.loc[idx], sector_ranks):
            df_processed.loc[idx, 'tag'] = 'SUPER_EDGE'
            df_processed.loc[idx, 'EDGE'] = min(df_processed.loc[idx, 'EDGE'] * 1.1, 100)  # Boost score for confirmed SUPER EDGE
    
    # Calculate dynamic stops
    df_processed = calculate_dynamic_stops(df_processed)
    
    # Apply portfolio constraints
    df_processed = apply_portfolio_constraints(df_processed)
    
    # Calculate targets
    df_processed['target_1'] = df_processed['price'] * 1.07  # Conservative
    df_processed['target_2'] = df_processed['price'] * 1.15  # Aggressive
    
    # Adjust targets for SUPER EDGE
    super_mask = df_processed['tag'] == 'SUPER_EDGE'
    df_processed.loc[super_mask, 'target_1'] = df_processed.loc[super_mask, 'price'] * 1.12
    df_processed.loc[super_mask, 'target_2'] = df_processed.loc[super_mask, 'price'] * 1.25
    
    # Add decision column
    df_processed['decision'] = df_processed.apply(lambda row: 
        'BUY NOW' if row['tag'] == 'SUPER_EDGE' and row['position_size'] > 0 else # Only BUY NOW if allocated
        'BUY' if row['tag'] == 'EXPLOSIVE' and row['position_size'] > 0 else # Only BUY if allocated
        'ACCUMULATE' if row['tag'] == 'STRONG' and row['position_size'] > 0 else # Only ACCUMULATE if allocated
        'WATCH' if row['tag'] == 'MODERATE' else
        'IGNORE', axis=1
    )
    
    return df_processed

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
        warnings_list = diagnostics.get('warnings', []) # Changed variable name to avoid conflict
        if warnings_list:
            st.write("**⚠️ Warnings:**")
            for warning in warnings_list[:3]:  # Show max 3
                st.write(f"• {warning}")
        
        # Critical columns
        missing = diagnostics.get('critical_columns_missing', [])
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
        
        # Download diagnostic report
        diag_data = pd.DataFrame([diagnostics])
        csv = diag_data.to_csv(index=False).encode('utf-8') # Encode to utf-8
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
    if sector_metrics.empty:
        st.info("No sectors meet the criteria for a meaningful leaderboard.")
        return

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
        
        # Mini bar chart for EDGE distribution (only if EDGE data exists and enough stocks)
        if 'EDGE' in df.columns:
            sector_stocks = df[df['sector'] == sector]['EDGE']
            if len(sector_stocks) > 2: # Need at least 3 data points for a meaningful histogram
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
                datetime.datetime.now().strftime('%Y-%m-%d %H:%M') # Corrected to datetime.datetime.now()
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
        # Ensure display_df is defined before use in filters
        display_df_filtered_for_ui = df_signals.copy() 

        with col1:
            if 'tag' in display_df_filtered_for_ui.columns:
                unique_tags = display_df_filtered_for_ui['tag'].dropna().unique().tolist()
                signal_types = st.multiselect(
                    "Signal Types",
                    unique_tags,
                    default=unique_tags
                )
            else:
                signal_types = []
        
        with col2:
            if 'sector' in display_df_filtered_for_ui.columns:
                unique_sectors = display_df_filtered_for_ui['sector'].dropna().unique().tolist()
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
                csv = display_df.to_csv(index=False).encode('utf-8') # Encode to utf-8
                st.download_button(
                    "📥 Download Signals (CSV)",
                    csv,
                    f"edge_signals_{datetime.datetime.now().strftime('%Y%m%d')}.csv", # Corrected to datetime.datetime.now()
                    "text/csv",
                    type="primary"
                )
            
            with col2:
                if len(df_signals) > 0 and len(df_analyzed) > 0:
                    excel_file = create_excel_report(df_signals, df_analyzed)
                    st.download_button(
                        "📊 Download Full Report (Excel)",
                        excel_file,
                        f"EDGE_Report_{datetime.datetime.now().strftime('%Y%m%d')}.xlsx", # Corrected to datetime.datetime.now()
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
                            risk_reward = safe_divide(abs(target_1 - price), abs(price - stop_loss))
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
                    # Check for EPS acceleration
                    eps_current = row.get('eps_current', np.nan)
                    eps_last_qtr = row.get('eps_last_qtr', np.nan)
                    if pd.notna(eps_current) and pd.notna(eps_last_qtr) and eps_last_qtr != 0:
                        eps_growth = safe_divide(eps_current - eps_last_qtr, eps_last_qtr)
                        if eps_growth > 0.15:
                            criteria.append("✅ EPS Accelerating (QoQ > 15%)")
                    
                    if row.get('from_high_pct', -100) >= -30 and row.get('from_high_pct', 0) <= -10:
                        criteria.append("✅ In Sweet Spot Zone (-30% to -10% from 52w High)")
                    
                    # Momentum alignment check
                    ret_1d = row.get('ret_1d', 0)
                    ret_7d = row.get('ret_7d', 0)
                    ret_30d = row.get('ret_30d', 0)
                    if (not pd.isna(ret_1d) and not pd.isna(ret_7d) and not pd.isna(ret_30d) and
                        ret_1d > 0 and ret_7d > ret_1d and ret_30d > 0):
                        criteria.append("✅ Momentum Aligned (1d, 7d, 30d all positive and accelerating)")
                    
                    # Sector strength check (need to recalculate or pass from main analysis)
                    # For display, we'll assume sector_ranks was available during main analysis
                    # and this stock met the criteria.
                    sector_ranks_temp = {} # Placeholder, ideally pass from outside
                    if 'sector' in df_analyzed.columns and 'volume_acceleration' in df_analyzed.columns:
                        sector_df_temp = df_analyzed.groupby('sector')['volume_acceleration'].agg(['mean', 'count'])
                        sector_df_temp = sector_df_temp[sector_df_temp['count'] >= 3]
                        if not sector_df_temp.empty:
                            sector_scores_temp = sector_df_temp['mean'].sort_values(ascending=False)
                            for i, sec in enumerate(sector_scores_temp.index):
                                sector_ranks_temp[sec] = i + 1
                    
                    sector = row.get('sector')
                    if sector and sector in sector_ranks_temp and sector_ranks_temp[sector] <= 3:
                        criteria.append(f"✅ Top 3 Sector ({sector})")

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
                    pct = safe_divide(value * 100, len(df_analyzed))
                    st.metric(metric, f"{value} ({pct:.0f}%)")
                else:
                    st.metric(metric, "0 (0%)")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    render_ui()
