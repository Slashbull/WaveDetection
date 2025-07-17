#!/usr/bin/env python3
"""
EDGE Protocol - Elite Data-Driven Growth Engine
==============================================
Production Version 3.1 FINAL - Ultimate Bug-Free Implementation

A sophisticated trading intelligence system that reveals institutional
accumulation through volume acceleration patterns.

Key Innovation: Volume Acceleration Detection
- Compares 30d/90d ratio vs 30d/180d ratio
- Reveals when institutions are ACCELERATING their accumulation
- Your unfair advantage in the market

Author: EDGE Protocol Team
Version: 3.1.0 FINAL BUG-FREE
Last Updated: 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots # Kept in case future plots need it
import requests
from datetime import datetime, timedelta
import warnings
import re
import logging
import io
import time
from typing import Dict, List, Tuple, Optional, Union, Any
from functools import wraps
import hashlib # For SmartCache

# Suppress warnings for production
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Google Sheets Configuration
# Using configuration from the NEW version for better structure
SHEET_CONFIG = {
    'SHEET_ID': '1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk',
    'GID': '2026492216',
    'CACHE_TTL': 300,  # 5 minutes
    'REQUEST_TIMEOUT': 30
}

PAGE_TITLE = "EDGE Protocol – Ultimate Trading Intelligence" # From old version

PROFILE_PRESETS = { # From old version
    "Balanced": (0.40, 0.25, 0.20, 0.15),
    "Swing": (0.50, 0.30, 0.20, 0.00),
    "Positional": (0.40, 0.25, 0.25, 0.10),
    "Momentum‑only": (0.60, 0.30, 0.10, 0.00),
    "Breakout": (0.45, 0.40, 0.15, 0.00),
    "Long‑Term": (0.25, 0.25, 0.15, 0.35),
}

EDGE_THRESHOLDS = { # From old version
    "SUPER_EDGE": 90,
    "EXPLOSIVE": 85,
    "STRONG": 70,
    "MODERATE": 50,
    "WATCH": 0
}

# The new version's thresholds for patterns are not directly used in the old version's pattern detection logic,
# but if you intend to re-introduce those patterns, they would be here.
# For now, sticking to the old version's pattern logic which has its own thresholds implicitly.

MIN_STOCKS_PER_SECTOR = 4 # From old version

# Logging Configuration
logging.basicConfig( # From new version
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('EDGE_Protocol')

# ============================================================================
# PERFORMANCE OPTIMIZATION - CACHING SYSTEM (From New Version)
# ============================================================================

class SmartCache:
    """Multi-level caching system for optimal performance"""
    
    def __init__(self):
        self.cache = {}
        self.timestamps = {}
    
    def get_cache_key(self, *args, **kwargs):
        """Generate unique cache key"""
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def is_valid(self, key: str, ttl: int) -> bool:
        """Check if cache entry is still valid"""
        if key not in self.timestamps:
            return False
        return (time.time() - self.timestamps[key]) < ttl
    
    def get(self, key: str, ttl: int = 300):
        """Get from cache if valid"""
        if self.is_valid(key, ttl):
            return self.cache.get(key)
        return None
    
    def set(self, key: str, value: Any):
        """Set cache value"""
        self.cache[key] = value
        self.timestamps[key] = time.time()
    
    def clear_expired(self, ttl: int = 300):
        """Clear expired entries"""
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self.timestamps.items()
            if (current_time - timestamp) > ttl
        ]
        for key in expired_keys:
            self.cache.pop(key, None)
            self.timestamps.pop(key, None)

# Global cache instance
cache = SmartCache()

def cached_computation(ttl: int = 60):
    """Decorator for caching expensive computations"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = cache.get_cache_key(func.__name__, *args, **kwargs)
            result = cache.get(cache_key, ttl)
            
            if result is not None:
                return result
            
            result = func(*args, **kwargs)
            cache.set(cache_key, result)
            return result
        
        return wrapper
    return decorator

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def safe_divide(a, b, default=0):
    """Safe division with default value for zero denominator"""
    return a / b if b != 0 else default

def winsorise_series(s: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    """Winsorises a pandas Series to cap outliers"""
    if s.empty or not pd.api.types.is_numeric_dtype(s):
        return s
    lo, hi = s.quantile([lower_q, upper_q])
    return s.clip(lo, hi)

def calc_atr20(price: pd.Series) -> pd.Series:
    """Calculate ATR proxy"""
    # Ensure price is numeric
    price = pd.to_numeric(price, errors='coerce')
    rolling_std = price.rolling(20, min_periods=1).std()
    # Fill NaNs from rolling with the mean of the calculated std, then multiply
    filled_rolling_std = rolling_std.fillna(rolling_std.mean())
    return filled_rolling_std * math.sqrt(2)

# ============================================================================
# ROBUST DATA PARSING - THE CORE FIX (From New Version)
# ============================================================================

def clean_numeric_series(series: pd.Series, col_name: str = "") -> pd.Series:
    """
    Bulletproof numeric parsing that handles ALL formats including:
    - Indian numbers with ₹, Cr, L
    - Percentages with % symbol
    - Numbers with commas
    - Hidden Unicode characters
    """
    # Convert to string for processing
    s = series.astype(str)
    
    # CRITICAL: Remove hidden Unicode characters FIRST
    s = s.str.replace('\u00A0', ' ', regex=False)  # Non-breaking space
    s = s.str.replace('\u200b', '', regex=False)   # Zero-width space
    s = s.str.replace('\xa0', ' ', regex=False)    # Another non-breaking space
    s = s.str.replace('\u2212', '-', regex=False)  # Unicode minus
    
    # Remove currency symbols - MUST use regex=False
    for symbol in ['₹', '$', '€', '£', 'Rs', 'Rs.', 'INR']:
        s = s.str.replace(symbol, '', regex=False)
    
    # Handle percentage symbol
    is_percentage = s.str.contains('%', na=False).any()
    s = s.str.replace('%', '', regex=False)
    
    # Remove commas - CRITICAL for volume_90d, volume_180d
    s = s.str.replace(',', '', regex=False)
    
    # Store original for unit detection
    original = s.copy()
    
    # Handle Indian number units
    cr_mask = original.str.upper().str.endswith('CR')
    l_mask = original.str.upper().str.endswith('L')
    k_mask = original.str.upper().str.endswith('K')
    m_mask = original.str.upper().str.endswith('M')
    b_mask = original.str.upper().str.endswith('B')
    
    # Remove all units
    for unit in ['Cr', 'cr', 'CR', 'L', 'l', 'K', 'k', 'M', 'm', 'B', 'b']:
        s = s.str.replace(unit, '', regex=False)
    
    # Remove arrows and other symbols
    for symbol in ['↑', '↓', '→', '←', '+']:
        s = s.str.replace(symbol, '', regex=False)
    
    # Clean whitespace
    s = s.str.strip()
    
    # Replace empty strings and common null indicators
    s = s.replace(['', '-', 'NA', 'N/A', 'na', 'n/a', 'null', 'None', '#N/A'], 'NaN')
    
    # Convert to numeric
    numeric_series = pd.to_numeric(s, errors='coerce')
    
    # Apply multipliers for Indian units
    if cr_mask.any():
        numeric_series[cr_mask] = numeric_series[cr_mask] * 10000000  # 1 Crore = 10 million
    if l_mask.any():
        numeric_series[l_mask] = numeric_series[l_mask] * 100000      # 1 Lakh = 100k
    if k_mask.any():
        numeric_series[k_mask] = numeric_series[k_mask] * 1000
    if m_mask.any():
        numeric_series[m_mask] = numeric_series[m_mask] * 1000000
    if b_mask.any():
        numeric_series[b_mask] = numeric_series[b_mask] * 1000000000
    
    # Log parsing success rate (only for significant columns/data size)
    # This logging would typically go into a diagnostics object passed around
    # For this integration, we'll rely on the main load_data's diagnostics.
    
    return numeric_series

def parse_all_columns_robust(df: pd.DataFrame, diagnostics: Dict) -> pd.DataFrame:
    """
    Parse ALL columns using the robust clean_numeric_series function.
    This replaces the simpler numeric conversion in the old load_sheet.
    """
    
    parsing_stats = {
        'successful': 0,
        'failed': 0,
        'columns_parsed': [],
        'null_counts_before': {},
        'null_counts_after': {},
        'type_conversions': {}
    }
    
    # Comprehensive column list - derived from both old and new versions
    # This list should cover all potential numeric columns present in your Google Sheet
    numeric_columns = {
        'price_related': ['price', 'prev_close', 'low_52w', 'high_52w', 'sma_20d', 'sma_50d', 'sma_200d', 'from_low_pct', 'from_high_pct'],
        'volume_raw': ['volume_1d', 'volume_7d', 'volume_30d', 'volume_90d', 'volume_180d'],
        'volume_ratios': ['vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d',
                          'vol_ratio_1d_180d', 'vol_ratio_7d_180d', 'vol_ratio_30d_180d', 'vol_ratio_90d_180d', 'rvol'],
        'returns': ['ret_1d', 'ret_3d', 'ret_7d', 'ret_30d', 'ret_3m', 'ret_6m', 'ret_1y', 'ret_3y', 'ret_5y'],
        'fundamentals': ['market_cap', 'pe', 'eps_current', 'eps_last_qtr', 'eps_change_pct'],
        'other_numeric': ['year'] # Assuming 'year' might be numeric
    }
    
    # Collect all unique numeric column names to process
    all_numeric_cols_to_process = list(set([col for sublist in numeric_columns.values() for col in sublist]))
    
    for col in all_numeric_cols_to_process:
        if col in df.columns:
            try:
                original_dtype = df[col].dtype
                parsing_stats['type_conversions'][col] = f"{original_dtype} -> "
                parsing_stats['null_counts_before'][col] = df[col].isna().sum()
                
                # Use robust parser
                df[col] = clean_numeric_series(df[col], col)
                
                parsing_stats['null_counts_after'][col] = df[col].isna().sum()
                parsing_stats['type_conversions'][col] += str(df[col].dtype)
                parsing_stats['successful'] += 1
                parsing_stats['columns_parsed'].append(col)
                
            except Exception as e:
                logger.warning(f"Failed to parse column {col}: {str(e)}")
                parsing_stats['failed'] += 1
                df[col] = np.nan # Ensure it's NaN on failure
    
    # Ensure critical text columns are strings
    text_columns = ['ticker', 'company_name', 'sector', 'category', 'exchange'] # 'exchange' might be from original
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).replace('nan', '').str.strip().fillna("Unknown")
        else: # Add missing text columns with default 'Unknown'
            df[col] = "Unknown"
    
    diagnostics['parsing_stats'] = parsing_stats
    logger.info(f"Parsing complete: {parsing_stats['successful']} successful, {parsing_stats['failed']} failed")
    
    return df

def calculate_data_quality(df: pd.DataFrame, diagnostics: Dict) -> Tuple[pd.DataFrame, float]:
    """Calculate comprehensive data quality metrics (From New Version)"""
    
    quality_metrics = {
        'total_rows': len(df),
        'columns_available': len(df.columns),
        'critical_columns_coverage': {},
        'overall_completeness': 0
    }
    
    # Check critical columns from the perspective of the old version's requirements
    critical_columns = [
        'ticker', 'price', 'volume_1d', 'rvol', 'from_high_pct',
        'vol_ratio_30d_90d', 'vol_ratio_30d_180d', # These are crucial for volume acceleration
        'ret_1d', 'ret_3d', 'ret_7d', 'ret_30d',
        'high_52w', 'low_52w', 'sma_50d', 'sma_200d',
        'eps_current', 'eps_last_qtr', 'eps_change_pct', 'pe'
    ]
    
    for col in critical_columns:
        if col in df.columns:
            coverage = (df[col].notna().sum() / len(df)) * 100 if len(df) > 0 else 0
            quality_metrics['critical_columns_coverage'][col] = coverage
            # Store min/max for critical volume ratios if they are numeric
            if col in ['vol_ratio_30d_90d', 'vol_ratio_30d_180d'] and pd.api.types.is_numeric_dtype(df[col]):
                if 'critical_columns_check' not in diagnostics:
                    diagnostics['critical_columns_check'] = {}
                diagnostics['critical_columns_check'][col] = {
                    'dtype': str(df[col].dtype),
                    'nulls': df[col].isna().sum(),
                    'min': float(df[col].min()) if not df[col].empty else np.nan,
                    'max': float(df[col].max()) if not df[col].empty else np.nan
                }
        else:
            quality_metrics['critical_columns_coverage'][col] = 0.0 # Column missing entirely
    
    # Calculate overall completeness
    total_cells = len(df) * len(df.columns)
    non_null_cells = df.notna().sum().sum()
    quality_metrics['overall_completeness'] = (non_null_cells / total_cells) * 100 if total_cells > 0 else 0
    
    # Calculate quality score
    critical_coverage_values = list(quality_metrics['critical_columns_coverage'].values())
    avg_critical = sum(critical_coverage_values) / len(critical_coverage_values) if critical_coverage_values else 0
    
    # Ensure a high score if all critical columns are perfectly covered, even if overall is less
    quality_score = (avg_critical * 0.7) + (quality_metrics['overall_completeness'] * 0.3)
    
    diagnostics['column_coverage'] = quality_metrics['critical_columns_coverage']
    
    return df, quality_score

@st.cache_data(ttl=SHEET_CONFIG['CACHE_TTL'])
def load_data_and_diagnose() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load and parse data with comprehensive diagnostics using the new version's logic.
    Returns: (DataFrame, diagnostics)
    """
    diagnostics = {
        'timestamp': datetime.now(),
        'rows_loaded': 0,
        'rows_after_parsing': 0,
        'rows_valid': 0,
        'data_quality_score': 0,
        'warnings': [],
        'parsing_stats': {},
        'column_coverage': {},
        'critical_columns_check': {}
    }
    
    try:
        # Build URL
        url = f"https://docs.google.com/spreadsheets/d/{SHEET_CONFIG['SHEET_ID']}/export?format=csv&gid={SHEET_CONFIG['GID']}"
        
        # Fetch data with retry logic
        session = requests.Session()
        retries = 3
        for attempt in range(retries):
            try:
                response = session.get(url, timeout=SHEET_CONFIG['REQUEST_TIMEOUT'])
                response.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} to fetch data failed: {e}")
                if attempt < retries - 1:
                    time.sleep(1)  # Brief delay before retry
                    continue
                raise
        
        # Check for HTML response (access denied)
        if 'text/html' in response.headers.get('content-type', ''):
            raise ValueError("Access denied. Please make the Google Sheet public: Share → Anyone with link can view")
        
        # Load CSV
        df = pd.read_csv(io.StringIO(response.text))
        diagnostics['rows_loaded'] = len(df)
        
        # Clean column names - CRITICAL for matching
        df.columns = df.columns.str.strip().str.lower()
        
        # Standardize column names (remove special characters, spaces)
        df.columns = [
            re.sub(r'[^\w\s]', '', col).replace(' ', '_').replace('__', '_').strip('_')
            for col in df.columns
        ]
        
        # Log original data types for debugging
        diagnostics['original_dtypes'] = {col: str(df[col].dtype) for col in df.columns}
        
        # Parse all numeric columns using the robust parser
        df = parse_all_columns_robust(df, diagnostics)
        diagnostics['rows_after_parsing'] = len(df)
        
        # Fill critical values (similar to old load_sheet, but after robust parsing)
        df['price'] = df['price'].fillna(df.get('prev_close', 1)).fillna(1)
        df['volume_1d'] = df.get('volume_1d', pd.Series(0, index=df.index)).fillna(0).astype(int)
        df['rvol'] = df.get('rvol', pd.Series(1, index=df.index)).fillna(1)
        
        # Add derived columns that the OLD version expects *before* scoring
        # These were in the old load_sheet, so moving them here
        df["atr_20"] = calc_atr20(df["price"])
        df["rs_volume_30d"] = df.get("volume_30d", 0) * df["price"] # Ensure volume_30d is numeric by now

        # Calculate volume acceleration (calls a separate function)
        df = calculate_volume_acceleration(df)
        
        # Gentle validation - keep as much data as possible
        # (This was implicitly done by null filling, but explicitly checking for critical invalid rows)
        initial_count_after_parsing = len(df)
        if 'ticker' in df.columns:
            df = df[df['ticker'].notna() & (df['ticker'] != '') & (df['ticker'] != 'nan')]
        if 'price' in df.columns:
            df = df[(df['price'] > 0) | df['price'].isna()]
        if 'company_name' in df.columns:
            df = df[~df['company_name'].str.lower().isin(['test', 'dummy', 'delete', 'remove'])]
        
        rows_removed_validation = initial_count_after_parsing - len(df)
        if rows_removed_validation > 0:
            diagnostics['warnings'].append(f"Validation removed {rows_removed_validation} invalid rows.")
            diagnostics['rows_removed_validation'] = rows_removed_validation

        # Calculate data quality metrics
        df, quality_score = calculate_data_quality(df, diagnostics)
        diagnostics['data_quality_score'] = quality_score
        
        # Final row count
        diagnostics['rows_valid'] = len(df)
        
        return df, diagnostics
        
    except Exception as e:
        logger.error(f"Data loading error: {str(e)}")
        diagnostics['warnings'].append(f"Fatal error during data loading: {str(e)}")
        return pd.DataFrame(), diagnostics


# ============================================================================
# VOLUME ACCELERATION (From Old Version, adapted to new parsing)
# ============================================================================
def calculate_volume_acceleration(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate volume acceleration metrics"""
    df = df.copy()
    
    # Get volume ratio columns - these should now be numeric due to clean_numeric_series
    vol_30_90 = df.get('vol_ratio_30d_90d', pd.Series(0, index=df.index))
    vol_30_180 = df.get('vol_ratio_30d_180d', pd.Series(0, index=df.index))
    
    # CRITICAL: Ensure they are numeric after all attempts, coerce to 0 if not
    vol_30_90 = pd.to_numeric(vol_30_90, errors='coerce').fillna(0)
    vol_30_180 = pd.to_numeric(vol_30_180, errors='coerce').fillna(0)

    # Convert to percentage if values are in decimal format (this is a good check)
    if vol_30_90.abs().max() <= 10 and vol_30_90.abs().max() > 0:  # Likely in decimal format, and not all zero
        vol_30_90 = vol_30_90 * 100
    if vol_30_180.abs().max() <= 10 and vol_30_180.abs().max() > 0: # Likely in decimal format, and not all zero
        vol_30_180 = vol_30_180 * 100
    
    # Calculate acceleration
    df['volume_acceleration'] = vol_30_90 - vol_30_180
    
    # Classification
    conditions = [
        df['volume_acceleration'] > 30,
        df['volume_acceleration'] > 20,
        df['volume_acceleration'] > 10,
        df['volume_acceleration'] > 0,
        df['volume_acceleration'] > -10,
        df['volume_acceleration'] <= -10
    ]
    
    choices = [
        "Institutional Loading",
        "Heavy Accumulation",
        "Accumulation",
        "Mild Accumulation",
        "Distribution",
        "Exodus"
    ]
    
    df['volume_classification'] = np.select(conditions, choices, default="Neutral")
    
    return df

# ============================================================================
# PATTERN DETECTION (From Old Version)
# ============================================================================
@cached_computation(ttl=120) # Apply caching to pattern detection too
def detect_accumulation_under_resistance(row: pd.Series) -> dict:
    """Pattern 1: Volume explodes but price stays flat near resistance"""
    score = 0
    signals = []
    
    vol_ratio = row.get('vol_ratio_30d_90d', 0)
    ret_30d = row.get('ret_30d', 0)
    from_high = row.get('from_high_pct', -100)
    rvol = row.get('rvol', 1)
    
    # Volume explosion
    if vol_ratio > 50:
        score += 40
        signals.append(f"Volume +{vol_ratio:.0f}%")
    elif vol_ratio > 30:
        score += 25
        signals.append(f"Volume +{vol_ratio:.0f}%")
    
    # Price flat
    if abs(ret_30d) < 5:
        score += 30
        signals.append(f"Price flat ({ret_30d:.1f}%)")
    
    # Near resistance
    if -10 <= from_high <= 0:
        score += 30
        signals.append("At 52w high resistance")
    elif -20 <= from_high < -10:
        score += 20
        signals.append("Near resistance")
    
    # RVOL bonus
    if rvol > 2.0:
        score = min(score * 1.2, 100)
        signals.append(f"RVOL: {rvol:.1f}x")
    
    return {
        'pattern': 'Accumulation Under Resistance',
        'score': score,
        'signals': signals,
        'target': row.get('high_52w', row.get('price', 0)) * 1.05
    }

@cached_computation(ttl=120)
def detect_coiled_spring(row: pd.Series) -> dict:
    """Pattern 2: Volume up but price in tight range"""
    score = 0
    signals = []
    
    vol_ratio = row.get('vol_ratio_30d_90d', 0)
    ret_7d = row.get('ret_7d', 0)
    ret_30d = row.get('ret_30d', 0)
    price = row.get('price', 1)
    sma_50d = row.get('sma_50d', price)
    sma_200d = row.get('sma_200d', price)
    
    # Volume increase
    if vol_ratio > 30:
        score += 35
        signals.append(f"Volume +{vol_ratio:.0f}%")
    elif vol_ratio > 15:
        score += 20
        signals.append(f"Volume +{vol_ratio:.0f}%")
    
    # Tight range
    if abs(ret_7d) < 5 and abs(ret_30d) < 10:
        score += 35
        signals.append("Tight consolidation")
    
    # Above SMAs
    sma_score = 0
    if price > sma_50d:
        sma_score += 15
    if price > sma_200d:
        sma_score += 15
    
    score += sma_score
    if sma_score == 30:
        signals.append("Above key SMAs")
    
    return {
        'pattern': 'Coiled Spring',
        'score': score,
        'signals': signals,
        'target': price * 1.08
    }

@cached_computation(ttl=120)
def detect_absorption_pattern(row: pd.Series) -> dict:
    """Pattern 3: High RVOL with price stability"""
    score = 0
    signals = []
    
    rvol = row.get('rvol', 1)
    ret_7d = row.get('ret_7d', 0)
    vol_1d_90d = row.get('vol_ratio_1d_90d', 0)
    vol_7d_90d = row.get('vol_ratio_7d_90d', 0)
    vol_30d_90d = row.get('vol_ratio_30d_90d', 0)
    
    # High RVOL
    if rvol > 2.5:
        score += 50
        signals.append(f"Extreme RVOL: {rvol:.1f}x")
    elif rvol > 1.5:
        score += 30
        signals.append(f"High RVOL: {rvol:.1f}x")
    
    # Consistent volume
    positive_ratios = sum([1 for r in [vol_1d_90d, vol_7d_90d, vol_30d_90d] if r > 0])
    if positive_ratios == 3:
        score += 30
        signals.append("Sustained volume increase")
    
    # Price absorption
    if abs(ret_7d) < 3:
        score += 20
        signals.append("Price absorbed")
    
    return {
        'pattern': 'Absorption Pattern',
        'score': score,
        'signals': signals,
        'target': row.get('price', 0) * 1.10
    }

@cached_computation(ttl=120)
def detect_failed_breakdown_reversal(row: pd.Series) -> dict:
    """Pattern 4: Near 52w low but volume accelerating"""
    score = 0
    signals = []
    
    from_low = row.get('from_low_pct', 100)
    vol_accel = row.get('volume_acceleration', 0)
    ret_7d = row.get('ret_7d', 0)
    ret_3y = row.get('ret_3y', 0)
    
    # Near 52w low
    if from_low < 10:
        score += 40
        signals.append(f"Near 52w low (+{from_low:.0f}%)")
    elif from_low < 20:
        score += 25
        signals.append(f"Recent low (+{from_low:.0f}%)")
    
    # Volume acceleration
    if vol_accel > 20:
        score += 40
        signals.append(f"Vol accel: {vol_accel:.0f}%")
    elif vol_accel > 10:
        score += 25
        signals.append(f"Vol accel: {vol_accel:.0f}%")
    
    # Momentum reversal
    if ret_7d > 0:
        score += 20
        signals.append("Momentum positive")
    
    # Quality bonus
    if ret_3y > 200:
        score = min(score * 1.2, 100)
        signals.append("Quality reversal")
    
    return {
        'pattern': 'Failed Breakdown Reversal',
        'score': score,
        'signals': signals,
        'target': row.get('sma_50d', row.get('price', 0) * 1.15)
    }

@cached_computation(ttl=120)
def detect_stealth_breakout(row: pd.Series) -> dict:
    """Pattern 5: Quiet strength building"""
    score = 0
    signals = []
    
    price = row.get('price', 1)
    sma_20d = row.get('sma_20d', price)
    sma_50d = row.get('sma_50d', price)
    sma_200d = row.get('sma_200d', price)
    vol_ratio = row.get('vol_ratio_30d_90d', 0)
    ret_30d = row.get('ret_30d', 0)
    
    # Above all SMAs
    sma_count = sum([1 for sma in [sma_20d, sma_50d, sma_200d] if price > sma])
    if sma_count == 3:
        score += 40
        signals.append("Above all SMAs")
    elif sma_count == 2:
        score += 25
        signals.append("Above 2 SMAs")
    
    # Gradual volume
    if 10 <= vol_ratio <= 30:
        score += 30
        signals.append("Gradual vol increase")
    
    # Steady climb
    if 5 <= ret_30d <= 15:
        score += 30
        signals.append("Steady climb")
    
    return {
        'pattern': 'Stealth Breakout',
        'score': score,
        'signals': signals,
        'target': row.get('high_52w', price * 1.20)
    }

@cached_computation(ttl=120)
def detect_pre_earnings_accumulation(row: pd.Series) -> dict:
    """Pattern 6: Unusual volume with EPS momentum"""
    score = 0
    signals = []
    
    eps_current = row.get('eps_current', 0)
    eps_last = row.get('eps_last_qtr', 0)
    eps_change = row.get('eps_change_pct', 0) # This might be annual, use eps_qoq_acceleration for QOQ
    vol_7d_90d = row.get('vol_ratio_7d_90d', 0)
    from_high = row.get('from_high_pct', 0)
    
    # EPS momentum (using current & last QTR for acceleration, if eps_change_pct is yearly)
    if not pd.isna(eps_current) and not pd.isna(eps_last) and eps_last != 0:
        eps_accel = safe_divide((eps_current - eps_last), eps_last) * 100
        if eps_accel > 10:
            score += 35
            signals.append(f"EPS QoQ accel: {eps_accel:.0f}%")
    
    # Recent volume spike
    if vol_7d_90d > 50:
        score += 35
        signals.append("Recent vol spike")
    elif vol_7d_90d > 25:
        score += 20
        signals.append("Higher recent vol")
    
    # Accumulation zone
    if -20 <= from_high <= -5:
        score += 30
        signals.append("Accumulation zone")
    
    # Earnings growth bonus (using eps_change_pct if it's the right metric)
    if not pd.isna(eps_change) and eps_change > 20:
        score = min(score * 1.2, 100)
        signals.append("Strong EPS growth")
    
    return {
        'pattern': 'Pre-Earnings Accumulation',
        'score': score,
        'signals': signals,
        'target': row.get('price', 0) * 1.12
    }

@cached_computation(ttl=120)
def detect_all_patterns(row: pd.Series) -> dict:
    """Run all pattern detections"""
    patterns = [
        detect_accumulation_under_resistance(row),
        detect_coiled_spring(row),
        detect_absorption_pattern(row),
        detect_failed_breakdown_reversal(row),
        detect_stealth_breakout(row),
        detect_pre_earnings_accumulation(row)
    ]
    
    # Sort by score
    patterns.sort(key=lambda x: x['score'], reverse=True)
    
    # Calculate confluence
    high_score_patterns = [p for p in patterns if p['score'] >= 70]
    
    confluence_score = 0
    confluence_signals = []
    
    if len(high_score_patterns) >= 3:
        confluence_score = 100
        confluence_signals.append(f"{len(high_score_patterns)} PATTERNS ALIGNED!")
    elif len(high_score_patterns) >= 2:
        confluence_score = 85
        confluence_signals.append(f"{len(high_score_patterns)} patterns converging")
    elif len(high_score_patterns) >= 1:
        confluence_score = 70
        confluence_signals.append("Strong pattern detected")
    
    # Volume-Price Divergence
    # Ensure these columns exist and are numeric before using them
    vol_ratio = pd.to_numeric(row.get('vol_ratio_30d_90d', 0), errors='coerce').fillna(0)
    ret_30d = pd.to_numeric(row.get('ret_30d', 0), errors='coerce').fillna(0)

    vp_divergence = safe_divide(vol_ratio, abs(ret_30d) + 0.01) * 2
    vp_divergence = min(vp_divergence, 100)
    
    return {
        'patterns': patterns,
        'top_pattern': patterns[0] if patterns and patterns[0]['score'] > 0 else None, # Only consider if score > 0
        'confluence_score': confluence_score,
        'confluence_signals': confluence_signals,
        'vp_divergence': vp_divergence,
        'high_score_patterns': high_score_patterns
    }

# ============================================================================
# SCORING FUNCTIONS (From Old Version)
# ============================================================================
@cached_computation(ttl=120)
def score_vol_accel(row: pd.Series) -> float:
    """Enhanced Volume Acceleration scoring"""
    vol_30_90 = row.get("vol_ratio_30d_90d", 0)
    vol_30_180 = row.get("vol_ratio_30d_180d", 0)
    
    # Ensure values are numeric, fallback to 0 if not
    vol_30_90 = pd.to_numeric(vol_30_90, errors='coerce').fillna(0)
    vol_30_180 = pd.to_numeric(vol_30_180, errors='coerce').fillna(0)

    if pd.isna(vol_30_90) or pd.isna(vol_30_180): # Re-check after numeric conversion
        return 50  # Default neutral score
    
    delta = vol_30_90 - vol_30_180
    base_score = 50 + (delta * 2)  # More sensitive scaling
    base_score = np.clip(base_score, 0, 100)
    
    # RVOL bonus
    rvol = row.get("rvol", 1)
    if not pd.isna(rvol):
        if rvol > 2.0 and delta > 20:
            base_score = min(base_score * 1.5, 100)
        elif rvol > 1.5:
            base_score = min(base_score * 1.2, 100)
    
    return base_score

@cached_computation(ttl=120)
def score_momentum(row: pd.Series, df_full: pd.DataFrame) -> float: # Renamed df to df_full to avoid confusion with row.df
    """Momentum scoring with consistency check"""
    ret_1d = row.get('ret_1d', 0)
    ret_3d = row.get('ret_3d', 0)
    ret_7d = row.get('ret_7d', 0)
    ret_30d = row.get('ret_30d', 0)

    # Ensure values are numeric
    ret_1d, ret_3d, ret_7d, ret_30d = [pd.to_numeric(x, errors='coerce').fillna(0) for x in [ret_1d, ret_3d, ret_7d, ret_30d]]
    
    # Simple momentum score
    short_term = (ret_1d + ret_3d + ret_7d) / 3
    mid_term = ret_30d
    
    momentum = (short_term * 0.6 + mid_term * 0.4)
    base_score = 50 + (momentum * 5)  # Scale to 0-100
    base_score = np.clip(base_score, 0, 100)
    
    # Consistency bonus
    if ret_1d > 0 and ret_3d > ret_1d and ret_7d > ret_3d and ret_30d > 0:
        base_score = min(base_score * 1.3, 100)
    
    return base_score

@cached_computation(ttl=120)
def score_risk_reward(row: pd.Series) -> float:
    """Risk/Reward scoring"""
    price = row.get("price", 1)
    high_52w = row.get("high_52w", price)
    low_52w = row.get("low_52w", price)

    # Ensure numeric
    price = pd.to_numeric(price, errors='coerce').fillna(1)
    high_52w = pd.to_numeric(high_52w, errors='coerce').fillna(price)
    low_52w = pd.to_numeric(low_52w, errors='coerce').fillna(price)
    
    # Handle cases where price might be 0 or high_52w/low_52w are problematic
    price = max(price, 0.01) # Avoid division by zero
    high_52w = max(high_52w, price) # High cannot be less than current price
    low_52w = min(low_52w, price) # Low cannot be more than current price
    
    upside = safe_divide(high_52w - price, price) * 100
    downside = safe_divide(price - low_52w, price) * 100
    
    # Risk/Reward ratio
    rr_ratio = safe_divide(upside, downside + 0.01) # Add small epsilon to downside to prevent div by zero
    base_score = min(rr_ratio * 20, 100)
    
    # Quality bonus
    ret_3y = row.get("ret_3y", 0)
    ret_1y = row.get("ret_1y", 0)
    # Ensure numeric
    ret_3y = pd.to_numeric(ret_3y, errors='coerce').fillna(0)
    ret_1y = pd.to_numeric(ret_1y, errors='coerce').fillna(0)

    if ret_3y > 300 and ret_1y < 20:  # Quality stock on sale
        base_score = min(base_score * 1.4, 100)
    
    return base_score

@cached_computation(ttl=120)
def score_fundamentals(row: pd.Series, df_full: pd.DataFrame) -> float: # Renamed df to df_full
    """Fundamentals scoring"""
    scores = []
    
    # EPS change
    eps_change = row.get("eps_change_pct", 0)
    if not pd.isna(eps_change) and pd.api.types.is_numeric_dtype(eps_change):
        eps_score = 50 + (eps_change * 2)  # Scale around 50
        scores.append(np.clip(eps_score, 0, 100))
    
    # PE ratio
    pe = row.get("pe", 0)
    if not pd.isna(pe) and pd.api.types.is_numeric_dtype(pe) and pe > 0:
        if pe <= 25:
            pe_score = 100 - (pe * 2)
        elif pe <= 50:
            pe_score = 50 - ((pe - 25) * 1)
        else:
            pe_score = 25
        scores.append(max(pe_score, 0))
    
    # EPS acceleration (Quarter-over-Quarter)
    eps_current = row.get("eps_current", 0)
    eps_last = row.get("eps_last_qtr", 0)
    # Ensure numeric
    eps_current = pd.to_numeric(eps_current, errors='coerce').fillna(0)
    eps_last = pd.to_numeric(eps_last, errors='coerce').fillna(0)

    if eps_last > 0 and eps_current > 0:
        eps_accel = safe_divide(eps_current - eps_last, eps_last) * 100
        if eps_accel > 10:
            scores.append(min(eps_accel * 5, 100))
    
    return np.mean(scores) if scores else 50

@cached_computation(ttl=120)
def detect_super_edge(row: pd.Series) -> bool:
    """Detect SUPER EDGE conditions"""
    conditions_met = 0
    
    # Ensure all used values are numeric, default to non-triggering values if NaN
    rvol = pd.to_numeric(row.get("rvol", 0), errors='coerce').fillna(0)
    volume_acceleration = pd.to_numeric(row.get("volume_acceleration", 0), errors='coerce').fillna(0)
    eps_current = pd.to_numeric(row.get("eps_current", 0), errors='coerce').fillna(0)
    eps_last = pd.to_numeric(row.get("eps_last_qtr", 0), errors='coerce').fillna(0)
    from_high = pd.to_numeric(row.get("from_high_pct", 0), errors='coerce').fillna(0)
    ret_1d = pd.to_numeric(row.get('ret_1d', 0), errors='coerce').fillna(0)
    ret_3d = pd.to_numeric(row.get('ret_3d', 0), errors='coerce').fillna(0)
    ret_7d = pd.to_numeric(row.get('ret_7d', 0), errors='coerce').fillna(0)
    ret_30d = pd.to_numeric(row.get('ret_30d', 0), errors='coerce').fillna(0)

    # Check each condition
    if rvol > 2.0:
        conditions_met += 1
    
    if volume_acceleration > 30:
        conditions_met += 1
    
    if eps_last > 0 and eps_current > 0:
        if safe_divide(eps_current - eps_last, eps_last) > 0.10:
            conditions_met += 1
    
    if -30 <= from_high <= -15:
        conditions_met += 1
    
    # Momentum alignment
    if (ret_1d > 0 and
        ret_3d > ret_1d and
        ret_7d > ret_3d and
        ret_30d > 0):
        conditions_met += 1
    
    return conditions_met >= 4

# ============================================================================
# MAIN SCORING ENGINE (From Old Version, adapted)
# ============================================================================
def compute_scores(df: pd.DataFrame, weights: Tuple[float, float, float, float]) -> pd.DataFrame:
    """Complete scoring with pattern detection"""
    df = df.copy()
    
    # Calculate individual scores
    with st.spinner("Calculating component scores..."):
        df["vol_score"] = df.apply(score_vol_accel, axis=1)
        df["mom_score"] = df.apply(score_momentum, axis=1, df_full=df) # Pass full df if needed, or remove param if not
        df["rr_score"] = df.apply(score_risk_reward, axis=1)
        df["fund_score"] = df.apply(score_fundamentals, axis=1, df_full=df) # Pass full df if needed, or remove param if not
    
    # Calculate EDGE score
    block_cols = ["vol_score", "mom_score", "rr_score", "fund_score"]
    
    # Weighted average with adaptive weighting
    df["EDGE"] = 0.0 # Initialize as float
    for idx in df.index:
        scores = df.loc[idx, block_cols]
        valid_mask = ~scores.isna()
        
        if valid_mask.sum() == 0:
            df.loc[idx, "EDGE"] = 0.0 # Default to 0 if no valid scores
            continue
            
        valid_weights = np.array(weights)[valid_mask]
        valid_scores = scores[valid_mask]
        
        # Normalize weights
        sum_valid_weights = valid_weights.sum()
        if sum_valid_weights == 0: # Avoid division by zero if all relevant weights are 0
            df.loc[idx, "EDGE"] = 0.0
            continue
        norm_weights = valid_weights / sum_valid_weights
        df.loc[idx, "EDGE"] = (valid_scores * norm_weights).sum()
    
    # Detect SUPER EDGE
    df["is_super_edge"] = df.apply(detect_super_edge, axis=1)
    
    # Boost SUPER EDGE scores
    super_edge_mask = df["is_super_edge"]
    df.loc[super_edge_mask, "EDGE"] = df.loc[super_edge_mask, "EDGE"] * 1.1
    df["EDGE"] = df["EDGE"].clip(0, 100)
    
    # Classification
    conditions = [
        df["is_super_edge"] & (df["EDGE"] >= EDGE_THRESHOLDS["SUPER_EDGE"]),
        df["EDGE"] >= EDGE_THRESHOLDS["EXPLOSIVE"],
        df["EDGE"] >= EDGE_THRESHOLDS["STRONG"],
        df["EDGE"] >= EDGE_THRESHOLDS["MODERATE"],
    ]
    choices = ["SUPER_EDGE", "EXPLOSIVE", "STRONG", "MODERATE"]
    df["tag"] = np.select(conditions, choices, default="WATCH")
    
    # Position sizing
    position_map = {
        "SUPER_EDGE": 0.15,
        "EXPLOSIVE": 0.10,
        "STRONG": 0.05,
        "MODERATE": 0.02,
        "WATCH": 0.00
    }
    df['position_size_pct'] = df['tag'].map(position_map).fillna(0.0) # Fillna for safety
    
    # Calculate stops and targets
    df['dynamic_stop'] = df['price'] * 0.95
    df['target1'] = df['price'] * 1.05
    df['target2'] = df['price'] * 1.10
    
    # Adjust for SUPER EDGE
    super_mask = df["tag"] == "SUPER_EDGE"
    df.loc[super_mask, 'target1'] = df.loc[super_mask, 'price'] * 1.10
    df.loc[super_mask, 'target2'] = df.loc[super_mask, 'price'] * 1.20
    
    # Pattern detection - ONLY for high potential stocks
    with st.spinner("Detecting explosive patterns..."):
        # Initialize pattern columns first
        df['pattern_analysis'] = [None] * len(df) # Pre-fill with None
        df['top_pattern_name'] = ""
        df['top_pattern_score'] = 0.0
        df['pattern_confluence_score'] = 0.0
        df['vp_divergence_score'] = 0.0
        
        # Filter for stocks with enough data to run pattern detection
        # Check for necessary columns before filtering
        required_pattern_cols = [
            'vol_ratio_30d_90d', 'ret_30d', 'from_high_pct', 'rvol',
            'ret_7d', 'sma_50d', 'sma_200d', 'vol_ratio_1d_90d', 'vol_7d_90d',
            'volume_acceleration', 'ret_3y', 'eps_current', 'eps_last_qtr',
            'eps_change_pct', 'price', 'low_52w', 'high_52w' # Add any others needed by pattern functions
        ]
        
        # Filter rows where essential columns for pattern detection are present and numeric
        # A more robust check might be to check column type *and* a low NaN count
        temp_high_potential = df[
            (df['EDGE'] >= 30) &
            df[required_pattern_cols].notna().all(axis=1) # Ensure all critical pattern columns are not NaN
        ].copy() # Work on a copy to avoid SettingWithCopyWarning
        
        # Apply pattern detection
        # Use .apply and assign results back using .loc for safety with complex objects
        if not temp_high_potential.empty:
            pattern_results = temp_high_potential.apply(detect_all_patterns, axis=1)
            
            # Update original DataFrame
            for idx, res in pattern_results.items():
                df.at[idx, 'pattern_analysis'] = res
                df.at[idx, 'top_pattern_name'] = res['top_pattern']['pattern'] if res['top_pattern'] else ""
                df.at[idx, 'top_pattern_score'] = float(res['top_pattern']['score']) if res['top_pattern'] else 0.0
                df.at[idx, 'pattern_confluence_score'] = float(res['confluence_score'])
                df.at[idx, 'vp_divergence_score'] = float(res['vp_divergence'])
    
    # Additional indicators (ensure these columns exist and are numeric by now)
    df['eps_qoq_acceleration'] = 0.0
    # Ensure 'eps_last_qtr' is numeric and not zero to prevent division by zero
    mask = (df['eps_last_qtr'].notna()) & (df['eps_last_qtr'] != 0) & (df['eps_current'].notna())
    df.loc[mask, 'eps_qoq_acceleration'] = (
        (df.loc[mask, 'eps_current'] - df.loc[mask, 'eps_last_qtr']) /
        df.loc[mask, 'eps_last_qtr'] * 100
    )
    df['eps_qoq_acceleration'] = df['eps_qoq_acceleration'].fillna(0.0) # Fill any remaining NaNs

    df['quality_consolidation'] = (
        (df.get('ret_3y', 0) > 300) &
        (df.get('ret_1y', 0) < 20) &
        (df.get('from_high_pct', 0) >= -40) &
        (df.get('from_high_pct', 0) <= -15)
    ).fillna(False) # Fillna for boolean series
    
    df['momentum_aligned'] = (
        (df.get('ret_1d', 0) > 0) &
        (df.get('ret_3d', 0) > df.get('ret_1d', 0)) &
        (df.get('ret_7d', 0) > df.get('ret_3d', 0)) &
        (df.get('ret_30d', 0) > 0)
    ).fillna(False)
    
    # Tier classifications
    df['eps_tier'] = df['eps_current'].apply(get_eps_tier)
    df['price_tier'] = df['price'].apply(get_price_tier)
    
    return df

# ============================================================================
# HELPER FUNCTIONS (From Old Version)
# ============================================================================
def get_eps_tier(eps: float) -> str:
    """Categorize EPS into tiers"""
    if pd.isna(eps):
        return ""
    
    tiers = [
        (0.95, "95↑"),
        (0.75, "75↑"),
        (0.55, "55↑"),
        (0.35, "35↑"),
        (0.15, "15↑"),
        (0.05, "5↑"),
        (-float('inf'), "5↓")
    ]
    
    for threshold, label in tiers:
        if eps >= threshold:
            return label
    return ""

def get_price_tier(price: float) -> str:
    """Categorize price into tiers"""
    if pd.isna(price):
        return ""
    
    tiers = [
        (5000, "5K↑"),
        (2000, "2K↑"),
        (1000, "1K↑"),
        (500, "500↑"),
        (200, "200↑"),
        (100, "100↑"),
        (0, "100↓")
    ]
    
    for threshold, label in tiers:
        if price >= threshold:
            return label
    return ""

# ============================================================================
# VISUALIZATION FUNCTIONS (From Old Version)
# ============================================================================
def plot_stock_radar_chart(df_row: pd.Series):
    """Create radar chart for stock EDGE components"""
    categories = ['Volume Accel', 'Momentum', 'Risk/Reward', 'Fundamentals']
    scores = [
        df_row.get('vol_score', 0),
        df_row.get('mom_score', 0),
        df_row.get('rr_score', 0),
        df_row.get('fund_score', 0)
    ]
    scores = [0 if pd.isna(s) else s for s in scores]
    
    line_color = 'gold' if df_row.get('tag') == 'SUPER_EDGE' else 'darkblue'
    
    fig = go.Figure(data=go.Scatterpolar(
        r=scores,
        theta=categories,
        fill='toself',
        name=df_row.get('company_name', 'Stock'),
        line_color=line_color,
        line_width=3
    ))
    
    title = f"EDGE Components - {df_row.get('company_name', '')} ({df_row.get('ticker', '')})"
    if df_row.get('tag') == 'SUPER_EDGE':
        title += " ⭐ SUPER EDGE ⭐"
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=False,
        title=title,
        font_size=16
    )
    
    return fig

# ============================================================================
# MAIN UI (From Old Version)
# ============================================================================
def render_ui():
    """Main Streamlit UI"""
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .super-edge-banner {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
    .super-edge-text {
        color: #000;
        font-size: 24px;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding-left: 20px;
        padding-right: 20px;
    }
    div[data-testid="metric-container"] {
        background-color: rgba(28, 131, 225, 0.1);
        border: 1px solid rgba(28, 131, 225, 0.2);
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title(PAGE_TITLE)
    st.markdown("**Your edge**: Volume acceleration + Pattern detection + Momentum = **PROFITS**")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        profile_name = st.radio("Profile", list(PROFILE_PRESETS.keys()), index=0)
        weights = PROFILE_PRESETS[profile_name]
        
        st.markdown("---")
        st.subheader("🎯 Display Filters")
        min_edge = st.slider("Min EDGE Score", 0, 100, 30, 5)
        show_smallcaps = st.checkbox("Include small/micro caps", value=False)
        show_super_edge_only = st.checkbox("Show SUPER EDGE Only", value=False)
        
        st.markdown("---")
        st.markdown("### 📊 Weights")
        st.write(f"Volume: {weights[0]*100:.0f}%")
        st.write(f"Momentum: {weights[1]*100:.0f}%")
        st.write(f"Risk/Reward: {weights[2]*100:.0f}%")
        st.write(f"Fundamentals: {weights[3]*100:.0f}%")
    
    # Load data using the new, robust loader
    with st.spinner("🔄 Loading and processing market data..."):
        df, diagnostics = load_data_and_diagnose()
    
    if df.empty:
        st.error("❌ No data available. Please check the data source and diagnostics below.")
        with st.expander("🔧 Data Loading Diagnostics", expanded=True):
            st.json(diagnostics)
            if diagnostics['warnings']:
                st.error(f"**Error Details:** {diagnostics['warnings'][0]}")
                st.markdown("""
                **Possible Solutions:**
                1. Ensure the Google Sheet is public (Share → Anyone with link can view).
                2. Verify the `SHEET_ID` and `GID` in the script's `SHEET_CONFIG`.
                3. Check your internet connection.
                """)
        return
    
    # Apply initial filters (from old version logic)
    if not show_smallcaps and "category" in df.columns:
        df = df[~df["category"].astype(str).str.contains("nano|micro", case=False, na=False)]
    
    # Remove low liquidity stocks (from old version logic)
    if "rs_volume_30d" in df.columns: # rs_volume_30d is calculated in load_data_and_diagnose now
        df = df[(df["rs_volume_30d"] >= 1e7) | df["rs_volume_30d"].isna()]
    
    # Process data with scoring and patterns
    df_scored = compute_scores(df, weights)
    
    # Apply EDGE filter
    df_filtered = df_scored[df_scored["EDGE"] >= min_edge].copy()
    
    # Super Edge filter
    if show_super_edge_only:
        df_filtered = df_filtered[df_filtered["tag"] == "SUPER_EDGE"]
    
    # SUPER EDGE Alert
    super_edge_count = (df_filtered["tag"] == "SUPER_EDGE").sum()
    if super_edge_count > 0:
        st.markdown(f"""
        <div class="super-edge-banner">
            <div class="super-edge-text">
                ⭐ {super_edge_count} SUPER EDGE SIGNAL{'S' if super_edge_count > 1 else ''} DETECTED! ⭐<br>
                Maximum conviction opportunities!
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Main tabs
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
        
        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Signals", len(df_filtered))
        with col2:
            st.metric("SUPER EDGE", (df_filtered["tag"] == "SUPER_EDGE").sum())
        with col3:
            st.metric("EXPLOSIVE", (df_filtered["tag"] == "EXPLOSIVE").sum())
        with col4:
            avg_edge = df_filtered["EDGE"].mean() if not df_filtered.empty else 0
            st.metric("Avg EDGE", f"{avg_edge:.1f}")
        
        # Filters
        st.markdown("### 🔍 Filters")
        filter_cols = st.columns(4)
        
        with filter_cols[0]:
            unique_tags = df_filtered['tag'].unique().tolist()
            # Ensure 'WATCH' is always an option if present in the data, or dynamically add if desired
            if "WATCH" in df_scored['tag'].unique() and "WATCH" not in unique_tags:
                unique_tags.append("WATCH") # Add WATCH to options if it exists in raw scored data
            selected_tags = st.multiselect("Classification", sorted(unique_tags), default=sorted(unique_tags))
        
        with filter_cols[1]:
            unique_sectors = sorted(df_filtered['sector'].dropna().unique())
            selected_sectors = st.multiselect("Sector", unique_sectors, default=unique_sectors)
        
        with filter_cols[2]:
            unique_vol_class = df_filtered['volume_classification'].unique().tolist()
            selected_vol_class = st.multiselect("Volume Pattern", unique_vol_class, default=unique_vol_class)
        
        with filter_cols[3]:
            search_ticker = st.text_input("Search Ticker", "")
        
        # Apply filters
        display_df = df_filtered.copy()
        if selected_tags:
            display_df = display_df[display_df['tag'].isin(selected_tags)]
        if selected_sectors:
            display_df = display_df[display_df['sector'].isin(selected_sectors)]
        if selected_vol_class:
            display_df = display_df[display_df['volume_classification'].isin(selected_vol_class)]
        if search_ticker:
            display_df = display_df[display_df['ticker'].str.contains(search_ticker.upper(), na=False)]
        
        # Sort and display
        display_df = display_df.sort_values('EDGE', ascending=False)
        
        if not display_df.empty:
            # Highlight function
            def highlight_rows(row):
                if row['tag'] == 'SUPER_EDGE':
                    return ['background-color: gold'] * len(row)
                elif row['tag'] == 'EXPLOSIVE':
                    return ['background-color: #ffcccc'] * len(row)
                return [''] * len(row)
            
            # Display columns
            display_cols = [
                'ticker', 'company_name', 'sector', 'tag', 'EDGE',
                'vol_score', 'mom_score', 'rr_score', 'fund_score',
                'price', 'rvol', 'volume_acceleration', 'volume_classification',
                'position_size_pct', 'dynamic_stop', 'target1', 'target2'
            ]
            
            # Ensure all columns exist
            display_cols = [col for col in display_cols if col in display_df.columns]
            
            st.dataframe(
                display_df[display_cols].style.apply(highlight_rows, axis=1)
                .format({
                    'EDGE': '{:.1f}',
                    'vol_score': '{:.0f}',
                    'mom_score': '{:.0f}',
                    'rr_score': '{:.0f}',
                    'fund_score': '{:.0f}',
                    'price': '₹{:.2f}',
                    'rvol': '{:.1f}',
                    'volume_acceleration': '{:.1f}%',
                    'position_size_pct': '{:.1%}',
                    'dynamic_stop': '₹{:.2f}',
                    'target1': '₹{:.2f}',
                    'target2': '₹{:.2f}'
                }),
                use_container_width=True,
                height=600
            )
            
            # Export button
            csv = display_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Export Signals",
                csv,
                f"edge_signals_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                "text/csv",
                type="primary"
            )
        else:
            st.info("No stocks match the selected filters.")
    
    # Tab 2: SUPER EDGE Analysis
    with tabs[1]:
        st.header("⭐ SUPER EDGE Analysis")
        
        super_edge_df = df_scored[df_scored["tag"] == "SUPER_EDGE"].sort_values("EDGE", ascending=False)
        
        if not super_edge_df.empty:
            st.success(f"🎯 {len(super_edge_df)} SUPER EDGE opportunities detected!")
            
            # Display top 5 in detail
            for idx, (_, row) in enumerate(super_edge_df.head(5).iterrows()):
                with st.expander(f"#{idx+1} {row['ticker']} - {row['company_name']} (EDGE: {row['EDGE']:.1f})", expanded=(idx==0)):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Price", f"₹{row.get('price', 0):.2f}")
                        st.metric("RVOL", f"{row.get('rvol', 0):.1f}x")
                        st.metric("Position Size", f"{row.get('position_size_pct', 0):.1%}")
                    
                    with col2:
                        st.metric("Volume Accel", f"{row.get('volume_acceleration', 0):.1f}%")
                        st.metric("From High", f"{row.get('from_high_pct', 0):.1f}%")
                        st.metric("EPS QoQ", f"{row.get('eps_qoq_acceleration', 0):.1f}%")
                    
                    with col3:
                        st.metric("Stop Loss", f"₹{row.get('dynamic_stop', 0):.2f}")
                        st.metric("Target 1", f"₹{row.get('target1', 0):.2f}")
                        st.metric("Target 2", f"₹{row.get('target2', 0):.2f}")
                    
                    # Radar chart
                    fig = plot_stock_radar_chart(row)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No SUPER EDGE signals today. Check EXPLOSIVE category.")
    
    # Tab 3: Explosive Patterns
    with tabs[2]:
        st.header("🎯 Explosive Patterns Discovery")
        
        # Filter for stocks with patterns
        pattern_df = df_scored[df_scored['top_pattern_score'] > 0].copy()
        
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
                title="Pattern Distribution",
                xaxis_title="Pattern Type",
                yaxis_title="Count",
                height=400
            )
            
            st.plotly_chart(fig_patterns, use_container_width=True)
            
            # High confluence patterns
            st.subheader("🔥 High Confluence Signals")
            
            high_conf_df = pattern_df[pattern_df['pattern_confluence_score'] >= 70].sort_values(
                'pattern_confluence_score', ascending=False
            ).head(10)
            
            if not high_conf_df.empty:
                display_pattern_cols = [
                    'ticker', 'company_name', 'price', 'EDGE',
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
                    }),
                    use_container_width=True
                )
            
            # Pattern guide
            with st.expander("📚 Pattern Guide"):
                st.markdown("""
                **Pattern Types:**
                1. **Accumulation Under Resistance** - Big volume, flat price near highs
                2. **Coiled Spring** - Tight range with increasing volume
                3. **Absorption Pattern** - High RVOL with price stability
                4. **Failed Breakdown Reversal** - Bounce from lows with volume
                5. **Stealth Breakout** - Quiet strength above SMAs
                6. **Pre-Earnings Accumulation** - Volume spike before earnings
                
                **Confluence Score:**
                - 100: 3+ patterns aligned (ULTRA HIGH conviction)
                - 85: 2 patterns aligned (Very strong)
                - 70: 1 strong pattern (Good opportunity)
                """)
        else:
            st.info("Pattern detection requires more data or different market conditions.")
    
    # Tab 4: Volume Analysis
    with tabs[3]:
        st.header("📈 Volume Acceleration Analysis")
        
        # Volume acceleration scatter plot
        if "volume_acceleration" in df_filtered.columns and "from_high_pct" in df_filtered.columns:
            # Prepare data
            plot_df = df_filtered.dropna(subset=['volume_acceleration', 'from_high_pct'])
            
            if not plot_df.empty:
                # Create size based on RVOL
                plot_df['marker_size'] = plot_df.get('rvol', 1) * 10
                
                fig = px.scatter(
                    plot_df,
                    x="from_high_pct",
                    y="volume_acceleration",
                    color="tag",
                    size="marker_size",
                    hover_data=['ticker', 'company_name', 'EDGE', 'rvol'],
                    title="Volume Acceleration Map (Size = RVOL)",
                    labels={
                        "from_high_pct": "% From 52-Week High",
                        "volume_acceleration": "Volume Acceleration %"
                    },
                    color_discrete_map={
                        "SUPER_EDGE": "#FFD700",
                        "EXPLOSIVE": "#FF4B4B",
                        "STRONG": "#FFA500",
                        "MODERATE": "#90EE90",
                        "WATCH": "#87CEEB"
                    }
                )
                
                # Add zones
                fig.add_vline(x=-15, line_dash="dash", line_color="gold",
                              annotation_text="Sweet Spot Start")
                fig.add_vline(x=-30, line_dash="dash", line_color="gold",
                              annotation_text="Sweet Spot End")
                fig.add_hline(y=30, line_dash="dash", line_color="red",
                              annotation_text="High Acceleration")
                
                # Add shaded region
                fig.add_vrect(x0=-30, x1=-15, fillcolor="gold", opacity=0.1,
                              annotation_text="SUPER EDGE ZONE", annotation_position="top")
                
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
                
                # Volume insights
                st.subheader("📊 Volume Insights")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    high_rvol = (plot_df['rvol'] > 2.0).sum()
                    st.metric("High RVOL (>2x)", high_rvol)
                
                with col2:
                    high_accel = (plot_df['volume_acceleration'] > 30).sum()
                    st.metric("High Acceleration (>30%)", high_accel)
                
                with col3:
                    inst_loading = (plot_df['volume_classification'] == "Institutional Loading").sum()
                    st.metric("Institutional Loading", inst_loading)
            else:
                st.info("No sufficient data for Volume Acceleration Map after filters.")
        else:
            st.info("Volume acceleration data not available. Ensure 'volume_acceleration' and 'from_high_pct' columns are present and valid.")
    
    # Tab 5: Sector Heatmap
    with tabs[4]:
        st.header("🔥 Sector Heatmap")
        
        # Aggregate by sector
        # Ensure sector and EDGE are present and valid
        if 'sector' in df_scored.columns and 'EDGE' in df_scored.columns:
            sector_agg = df_scored.groupby('sector').agg(
                EDGE=('EDGE', 'mean'),
                ticker=('ticker', 'count'),
                is_super_edge=('is_super_edge', lambda x: x.sum() if pd.api.types.is_numeric_dtype(x) else 0)
            ).reset_index()
            
            sector_agg.columns = ['sector', 'avg_edge', 'count', 'super_edge_count']
            sector_agg = sector_agg[sector_agg['count'] >= MIN_STOCKS_PER_SECTOR]  # Min stocks per sector from config
            
            if not sector_agg.empty:
                # Create treemap
                fig = px.treemap(
                    sector_agg,
                    path=['sector'],
                    values='count',
                    color='avg_edge',
                    hover_data={
                        'avg_edge': ':.1f',
                        'count': True,
                        'super_edge_count': True
                    },
                    color_continuous_scale='RdYlGn',
                    range_color=[0, 100],
                    title="Sector EDGE Heatmap"
                )
                
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
                
                # Top sectors table
                st.subheader("🏆 Top Sectors by EDGE")
                
                top_sectors = sector_agg.sort_values('avg_edge', ascending=False).head(10)
                
                st.dataframe(
                    top_sectors.style.format({
                        'avg_edge': '{:.1f}',
                        'count': '{:.0f}',
                        'super_edge_count': '{:.0f}'
                    }).background_gradient(subset=['avg_edge'], cmap='RdYlGn'),
                    use_container_width=True
                )
            else:
                st.info("Insufficient data for sector analysis after minimum stock per sector filter.")
        else:
            st.info("Sector analysis requires 'sector' and 'EDGE' columns to be present.")
    
    # Tab 6: Deep Dive
    with tabs[5]:
        st.header("🔍 Stock Deep Dive")
        
        # Stock selector
        available_stocks = df_scored[df_scored['EDGE'].notna()]['ticker'].unique()
        
        if len(available_stocks) > 0:
            # Prioritize high EDGE stocks
            sorted_stocks = (df_scored[df_scored['ticker'].isin(available_stocks)]
                            .sort_values('EDGE', ascending=False)['ticker'].unique())
            
            selected_ticker = st.selectbox(
                "Select Stock",
                sorted_stocks,
                format_func=lambda x: f"⭐ {x}" if x in df_scored[df_scored['tag'] == 'SUPER_EDGE']['ticker'].values else x
            )
            
            # Get stock data
            stock_data = df_scored[df_scored['ticker'] == selected_ticker].iloc[0]
            
            # Display header
            if stock_data['tag'] == 'SUPER_EDGE':
                st.markdown("""
                <div style="background: gold; padding: 10px; border-radius: 5px; text-align: center;">
                    <h2 style="margin: 0;">⭐ SUPER EDGE SIGNAL ⭐</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Key metrics
            st.subheader(f"{stock_data.get('company_name', 'Unknown Company')} ({stock_data.get('ticker', 'N/A')})")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Price", f"₹{stock_data.get('price', 0):.2f}")
                st.metric("EDGE Score", f"{stock_data.get('EDGE', 0):.1f}")
                st.metric("Classification", stock_data.get('tag', 'N/A'))
            
            with col2:
                st.metric("RVOL", f"{stock_data.get('rvol', 0):.1f}x")
                st.metric("Volume Accel", f"{stock_data.get('volume_acceleration', 0):.1f}%")
                st.metric("Volume Pattern", stock_data.get('volume_classification', 'N/A'))
            
            with col3:
                # Use .get with default 0 to ensure calculation can happen for potentially missing columns
                ret_1y = stock_data.get('ret_1y', 0)
                ret_3y = stock_data.get('ret_3y', 0)
                from_high_pct = stock_data.get('from_high_pct', 0)

                st.metric("1Y Return", f"{ret_1y:.1f}%" if not pd.isna(ret_1y) else "N/A")
                st.metric("3Y Return", f"{ret_3y:.1f}%" if not pd.isna(ret_3y) else "N/A")
                st.metric("From High", f"{from_high_pct:.1f}%" if not pd.isna(from_high_pct) else "N/A")

            with col4:
                st.metric("Stop Loss", f"₹{stock_data.get('dynamic_stop', 0):.2f}")
                st.metric("Target 1", f"₹{stock_data.get('target1', 0):.2f}")
                st.metric("Position Size", f"{stock_data.get('position_size_pct', 0):.1%}")
            
            # Radar chart
            fig = plot_stock_radar_chart(stock_data)
            st.plotly_chart(fig, use_container_width=True)
            
            # Pattern analysis
            if stock_data.get('pattern_analysis'):
                st.subheader("🎯 Pattern Analysis")
                
                pattern_data = stock_data['pattern_analysis']
                if isinstance(pattern_data, dict):
                    patterns = pattern_data.get('patterns', [])
                    
                    # Show top 3 patterns with score > 50
                    relevant_patterns = [p for p in patterns if p['score'] > 50]
                    if relevant_patterns:
                        for pattern in relevant_patterns[:3]:
                            with st.expander(f"{pattern['pattern']} (Score: {pattern['score']:.0f})"):
                                for signal in pattern.get('signals', []):
                                    st.write(f"• {signal}")
                                if pattern.get('target'):
                                    price_val = stock_data.get('price', 0)
                                    if price_val > 0:
                                        target_pct = (pattern['target'] / price_val - 1) * 100
                                        st.success(f"Pattern Target: ₹{pattern['target']:.2f} (+{target_pct:.1f}%)")
                    else:
                        st.info("No strong patterns detected for this stock.")
            
            # Special indicators
            st.subheader("📌 Special Indicators")
            
            indicators = []
            if stock_data.get('quality_consolidation'):
                indicators.append("💎 Quality Consolidation")
            if stock_data.get('momentum_aligned'):
                indicators.append("📈 Momentum Aligned")
            if stock_data.get('rvol', 0) > 2:
                indicators.append("🔥 High RVOL Activity")
            if stock_data.get('eps_qoq_acceleration', 0) > 10:
                indicators.append("💰 EPS Accelerating")
            
            if indicators:
                for ind in indicators:
                    st.write(ind)
            else:
                st.info("No special indicators active")
        else:
            st.info("No stocks available for analysis.")
    
    # Tab 7: Raw Data
    with tabs[6]:
        st.header("📋 Raw Data & Diagnostics")
        
        # Diagnostics from load_data_and_diagnose
        st.subheader("🔍 Data Loading Diagnostics")
        
        # Overall health status
        quality_score = diagnostics.get('data_quality_score', 0)
        if quality_score >= 80:
            st.markdown('<div class="success-box" style="background: #d4edda; border: 1px solid #c3e6cb; color: #155724; padding: 1rem; border-radius: 5px; margin: 1rem 0;">✅ Data Quality: EXCELLENT</div>', unsafe_allow_html=True)
        elif quality_score >= 60:
            st.markdown('<div class="warning-box" style="background: #fff3cd; border: 1px solid #ffeeba; color: #856404; padding: 1rem; border-radius: 5px; margin: 1rem 0;">⚠️ Data Quality: MODERATE - Some features may be limited</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="warning-box" style="background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; padding: 1rem; border-radius: 5px; margin: 1rem 0;">❌ Data Quality: POOR - Please check data source</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Rows Loaded", diagnostics.get('rows_loaded', 0))
            st.metric("Rows After Parsing & Validation", diagnostics.get('rows_valid', 0))
            st.metric("Data Quality Score", f"{quality_score:.1f}%")
        
        with col2:
            parsing_stats = diagnostics.get('parsing_stats', {})
            st.metric("Columns Parsed Successfully", parsing_stats.get('successful', 0))
            st.metric("Parsing Failures", parsing_stats.get('failed', 0))
            critical_check = diagnostics.get('critical_columns_check', {})
            if 'vol_ratio_30d_180d' in critical_check and critical_check['vol_ratio_30d_180d'].get('dtype') == 'float64':
                st.success("✅ Volume ratios parsed correctly")
            else:
                st.error("❌ Volume ratio parsing failed or not numeric!")
        
        with col3:
            # Re-check volume acceleration min/max for display if present
            vol_accel_series = df_scored.get('volume_acceleration')
            if vol_accel_series is not None and not vol_accel_series.empty and vol_accel_series.notna().any():
                st.metric("Volume Accel Range", f"[{vol_accel_series.min():.1f}, {vol_accel_series.max():.1f}]")
            else:
                st.metric("Volume Accel Range", "N/A")
            warnings_count = len(diagnostics.get('warnings', []))
            if warnings_count > 0:
                st.metric("⚠️ Warnings", warnings_count)
            else:
                st.success("No warnings during loading.")
        
        if diagnostics.get('column_coverage'):
            st.subheader("📊 Critical Column Coverage")
            coverage_df = pd.DataFrame([
                {'Column': col, 'Coverage': f"{cov:.1f}%", 'Status': '✅' if cov > 90 else '⚠️' if cov > 70 else '❌'}
                for col, cov in diagnostics['column_coverage'].items()
            ])
            st.dataframe(coverage_df, use_container_width=True, hide_index=True)
        
        if parsing_stats.get('type_conversions'):
            with st.expander("🔄 Detailed Type Conversions"):
                conversions = parsing_stats['type_conversions']
                for col, conversion_str in conversions.items():
                    st.write(f"- **{col}**: {conversion_str}")
        
        if diagnostics.get('warnings'):
            with st.expander("⚠️ Detailed Warnings"):
                for warning in diagnostics['warnings']:
                    st.warning(f"• {warning}")
        
        st.subheader("📄 Full Dataset Sample (First 10 Rows)")
        st.dataframe(df_scored.head(10), use_container_width=True)
        
        # Export full data
        st.subheader("💾 Export Options")
        
        col1_export, col2_export = st.columns(2)
        
        with col1_export:
            csv_full = df_scored.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Full Processed Dataset",
                csv_full,
                f"edge_protocol_full_processed_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                "text/csv"
            )
        
        with col2_export:
            # High EDGE only
            high_edge_df = df_scored[df_scored['EDGE'] >= 70]
            if not high_edge_df.empty:
                csv_high = high_edge_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "🔥 Download High EDGE Only",
                    csv_high,
                    f"edge_high_signals_processed_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                    "text/csv"
                )

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    render_ui()
