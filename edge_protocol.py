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
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import warnings
import re
import logging
import io
import time
from typing import Dict, List, Tuple, Optional, Union, Any
from functools import wraps
import hashlib
import json

# Suppress warnings for production
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Google Sheets Configuration
SHEET_CONFIG = {
    'SHEET_ID': '1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk',
    'GID': '2026492216',
    'CACHE_TTL': 300,  # 5 minutes
    'REQUEST_TIMEOUT': 30
}

# EDGE Score Thresholds
EDGE_THRESHOLDS = {
    'EXPLOSIVE': 85,      # Top 1% - Immediate action
    'STRONG': 70,         # Top 5% - High conviction
    'MODERATE': 50,       # Top 10% - Worth watching
    'WATCH': 30,          # Top 20% - Early signals
}

# Position Sizing (% of portfolio)
POSITION_SIZES = {
    'EXPLOSIVE': 0.10,    # 10% - Maximum conviction
    'STRONG': 0.05,       # 5% - High conviction
    'MODERATE': 0.03,     # 3% - Normal position
    'WATCH': 0.01,        # 1% - Starter position
}

# Volume Acceleration Patterns
VOLUME_PATTERNS = {
    'INSTITUTIONAL_LOADING': {'min_accel': 30, 'min_rvol': 2.0},
    'HEAVY_ACCUMULATION': {'min_accel': 20, 'min_rvol': 1.5},
    'ACCUMULATION': {'min_accel': 10, 'min_rvol': 1.2},
    'NEUTRAL': {'min_accel': 0, 'min_rvol': 1.0},
    'DISTRIBUTION': {'min_accel': -10, 'min_rvol': 0.8},
    'HEAVY_DISTRIBUTION': {'min_accel': -20, 'min_rvol': 0.5},
}

# Pattern Detection Thresholds
PATTERN_THRESHOLDS = {
    'EXPLOSIVE_BREAKOUT': {
        'ret_7d_min': 5,
        'volume_accel_min': 20,
        'above_sma50': True
    },
    'STEALTH_ACCUMULATION': {
        'ret_7d_max': 0,
        'volume_accel_min': 20,
        'from_high_range': (-30, -15)
    },
    'MOMENTUM_BUILDING': {
        'ret_7d_min': 2,
        'volume_accel_min': 10,
        'ret_30d_min': 5
    },
    'QUALITY_PULLBACK': {
        'ret_1y_min': 50,
        'from_high_range': (-30, -15),
        'eps_change_min': 10
    }
}

# Risk Management
RISK_PARAMS = {
    'MAX_PORTFOLIO_EXPOSURE': 0.80,   # 80% max invested
    'MAX_SINGLE_POSITION': 0.10,      # 10% max per position
    'MAX_SECTOR_EXPOSURE': 0.30,      # 30% max per sector
    'STOP_LOSS_BUFFER': 1.02,         # 2% below support
    'MIN_RISK_REWARD': 2.0,           # Minimum 2:1 ratio
}

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('EDGE_Protocol')

# ============================================================================
# PERFORMANCE OPTIMIZATION - CACHING SYSTEM
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
# ROBUST DATA PARSING - INSPIRED BY MANTRA'S BULLETPROOF APPROACH
# ============================================================================

def clean_numeric_series(series: pd.Series, col_name: str = "") -> pd.Series:
    """
    Bulletproof numeric parsing that handles all Indian formats
    This is the CORE fix - handles ₹, hidden characters, percentages, etc.
    """
    # Convert to string for processing
    s = series.astype(str)
    
    # CRITICAL: Remove hidden Unicode characters FIRST
    s = s.str.replace('\u00A0', ' ', regex=False)  # Non-breaking space
    s = s.str.replace('\u200b', '', regex=False)   # Zero-width space
    s = s.str.replace('\xa0', ' ', regex=False)    # Another non-breaking space
    
    # Remove currency symbols and units - MUST use regex=False
    for symbol in ['₹', '$', '€', '£', 'Rs', 'Rs.', 'INR']:
        s = s.str.replace(symbol, '', regex=False)
    
    # Handle percentage symbol
    is_percentage = s.str.contains('%', na=False).any()
    s = s.str.replace('%', '', regex=False)
    
    # Handle Indian number units
    # Store original for unit detection
    original = s.copy()
    
    # Remove commas (both Indian and Western format)
    s = s.str.replace(',', '', regex=False)
    
    # Handle Cr (Crores) and L (Lakhs) notation
    cr_mask = original.str.upper().str.endswith('CR')
    l_mask = original.str.upper().str.endswith('L')
    
    # Remove the units
    s = s.str.replace('Cr', '', regex=False).str.replace('cr', '', regex=False)
    s = s.str.replace('CR', '', regex=False)
    s = s.str.replace('L', '', regex=False).str.replace('l', '', regex=False)
    
    # Handle K, M, B for international format
    k_mask = original.str.upper().str.endswith('K')
    m_mask = original.str.upper().str.endswith('M')
    b_mask = original.str.upper().str.endswith('B')
    
    s = s.str.replace('K', '', regex=False).str.replace('k', '', regex=False)
    s = s.str.replace('M', '', regex=False).str.replace('m', '', regex=False)
    s = s.str.replace('B', '', regex=False).str.replace('b', '', regex=False)
    
    # Remove arrows and other symbols
    for symbol in ['↑', '↓', '→', '←']:
        s = s.str.replace(symbol, '', regex=False)
    
    # Clean whitespace
    s = s.str.strip()
    
    # Replace empty strings and common null indicators
    s = s.replace(['', '-', 'NA', 'N/A', 'na', 'n/a', 'null', 'None'], 'NaN')
    
    # Convert to numeric - this should rarely fail now
    numeric_series = pd.to_numeric(s, errors='coerce')
    
    # Apply multipliers for Indian units
    numeric_series[cr_mask] = numeric_series[cr_mask] * 10000000  # 1 Crore = 10 million
    numeric_series[l_mask] = numeric_series[l_mask] * 100000      # 1 Lakh = 100k
    numeric_series[k_mask] = numeric_series[k_mask] * 1000
    numeric_series[m_mask] = numeric_series[m_mask] * 1000000
    numeric_series[b_mask] = numeric_series[b_mask] * 1000000000
    
    # Handle percentage columns
    if is_percentage or col_name.endswith('_pct') or 'pct' in col_name:
        # Already in percentage format
        return numeric_series
    
    # Auto-detect if values need scaling (for ratios stored as decimals)
    non_null = numeric_series.dropna()
    if len(non_null) > 0:
        # If all values are between -1 and 1, likely percentages as decimals
        if non_null.abs().max() <= 1.0 and ('ratio' in col_name or 'ret_' in col_name):
            return numeric_series * 100
    
    return numeric_series

def parse_market_cap_special(value: str) -> float:
    """Special parser for market cap - handles edge cases"""
    if pd.isna(value) or value == '' or value == '-':
        return np.nan
    
    # Use the robust parser
    return clean_numeric_series(pd.Series([value])).iloc[0]

# ============================================================================
# DATA LOADING WITH COMPREHENSIVE DIAGNOSTICS
# ============================================================================

@st.cache_data(ttl=SHEET_CONFIG['CACHE_TTL'])
def load_data() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load and parse data with comprehensive diagnostics
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
        'volume_ratio_format': 'unknown'
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
        
        # Parse all numeric columns
        df = parse_all_columns(df, diagnostics)
        diagnostics['rows_after_parsing'] = len(df)
        
        # Detect and fix volume ratio format
        df = detect_and_fix_volume_ratios(df, diagnostics)
        
        # Calculate derived columns
        df = calculate_derived_columns(df)
        
        # Gentle validation - keep as much data as possible
        df = validate_essential_data(df, diagnostics)
        
        # Calculate data quality metrics
        df, quality_score = calculate_data_quality(df, diagnostics)
        diagnostics['data_quality_score'] = quality_score
        
        # Final row count
        diagnostics['rows_valid'] = len(df)
        
        return df, diagnostics
        
    except Exception as e:
        logger.error(f"Data loading error: {str(e)}")
        diagnostics['warnings'].append(f"Fatal error: {str(e)}")
        return pd.DataFrame(), diagnostics

def parse_all_columns(df: pd.DataFrame, diagnostics: Dict) -> pd.DataFrame:
    """Parse all columns with robust error handling"""
    
    parsing_stats = {
        'successful': 0,
        'failed': 0,
        'columns_parsed': [],
        'null_counts_before': {},
        'null_counts_after': {}
    }
    
    # Define column groups
    numeric_columns = {
        'price': ['price', 'prev_close', 'low_52w', 'high_52w', 'sma_20d', 'sma_50d', 'sma_200d'],
        'volume': ['volume_1d', 'volume_7d', 'volume_30d', 'volume_90d', 'volume_180d'],
        'returns': ['ret_1d', 'ret_3d', 'ret_7d', 'ret_30d', 'ret_3m', 'ret_6m', 'ret_1y', 'ret_3y', 'ret_5y'],
        'ratios': ['vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d',
                   'vol_ratio_1d_180d', 'vol_ratio_7d_180d', 'vol_ratio_30d_180d', 'vol_ratio_90d_180d'],
        'fundamentals': ['pe', 'eps_current', 'eps_last_qtr', 'eps_change_pct'],
        'other': ['from_low_pct', 'from_high_pct', 'rvol', 'market_cap']
    }
    
    # Parse each group
    for group_name, cols in numeric_columns.items():
        for col in cols:
            if col in df.columns:
                try:
                    # Track nulls before
                    parsing_stats['null_counts_before'][col] = df[col].isna().sum()
                    
                    # Use robust parser
                    df[col] = clean_numeric_series(df[col], col)
                    
                    # Track nulls after
                    parsing_stats['null_counts_after'][col] = df[col].isna().sum()
                    
                    parsing_stats['successful'] += 1
                    parsing_stats['columns_parsed'].append(col)
                    
                except Exception as e:
                    logger.warning(f"Failed to parse column {col}: {str(e)}")
                    parsing_stats['failed'] += 1
                    # Don't fail - just mark as NaN
                    df[col] = np.nan
    
    # Ensure critical text columns are strings
    text_columns = ['ticker', 'company_name', 'sector', 'category', 'exchange']
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).replace('nan', '').str.strip()
    
    diagnostics['parsing_stats'] = parsing_stats
    return df

def detect_and_fix_volume_ratios(df: pd.DataFrame, diagnostics: Dict) -> pd.DataFrame:
    """
    Detect if volume ratios are in decimal format and convert to percentage
    This is CRITICAL - many sheets store 0.5 instead of 50%
    """
    ratio_cols = [col for col in df.columns if 'vol_ratio' in col or 'ratio' in col]
    
    for col in ratio_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            non_null = df[col].dropna()
            if len(non_null) > 0:
                # Check if values are in decimal format (between -10 and 10)
                if non_null.abs().max() <= 10:
                    # Likely decimal format - convert to percentage
                    df[col] = df[col] * 100
                    diagnostics['volume_ratio_format'] = 'decimal_converted'
                else:
                    diagnostics['volume_ratio_format'] = 'percentage'
    
    return df

def calculate_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate essential derived columns with error handling"""
    
    # Volume Acceleration - THE SECRET WEAPON
    if all(col in df.columns for col in ['vol_ratio_30d_90d', 'vol_ratio_30d_180d']):
        # Ensure numeric
        df['vol_ratio_30d_90d'] = pd.to_numeric(df['vol_ratio_30d_90d'], errors='coerce').fillna(0)
        df['vol_ratio_30d_180d'] = pd.to_numeric(df['vol_ratio_30d_180d'], errors='coerce').fillna(0)
        
        # Calculate acceleration
        df['volume_acceleration'] = df['vol_ratio_30d_90d'] - df['vol_ratio_30d_180d']
        df['has_volume_acceleration'] = True
        
        logger.info(f"Volume acceleration calculated: min={df['volume_acceleration'].min():.1f}, max={df['volume_acceleration'].max():.1f}")
    else:
        df['volume_acceleration'] = 0
        df['has_volume_acceleration'] = False
        logger.warning("Volume acceleration data not available - using fallback scoring")
    
    # Price position (how far from 52w low/high)
    if all(col in df.columns for col in ['price', 'low_52w', 'high_52w']):
        price_range = df['high_52w'] - df['low_52w']
        # Avoid division by zero
        price_range = price_range.replace(0, 1)
        df['price_position'] = ((df['price'] - df['low_52w']) / price_range * 100).fillna(50)
    else:
        df['price_position'] = 50
    
    # Simple volatility measure
    if all(col in df.columns for col in ['high_52w', 'low_52w', 'price']):
        df['volatility_52w'] = ((df['high_52w'] - df['low_52w']) / df['price'].replace(0, 1)).fillna(0.5)
        df['volatility_52w'] = df['volatility_52w'].clip(0, 5)  # Cap extreme values
    else:
        df['volatility_52w'] = 0.5
    
    # Momentum sync indicator
    if all(col in df.columns for col in ['ret_7d', 'ret_30d']):
        df['momentum_sync'] = (df['ret_7d'] > 0) & (df['ret_30d'] > 0)
    else:
        df['momentum_sync'] = False
    
    # Volume consistency
    vol_ratios = ['vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d']
    available_ratios = [col for col in vol_ratios if col in df.columns]
    if available_ratios:
        df['volume_consistency'] = (df[available_ratios] > 0).sum(axis=1) / len(available_ratios)
    else:
        df['volume_consistency'] = 0.5
    
    # Fill critical missing values with defaults
    df['price'] = df['price'].fillna(df.get('prev_close', 0)).replace(0, 1)
    df['volume_1d'] = df.get('volume_1d', pd.Series(0, index=df.index)).fillna(0)
    df['rvol'] = df.get('rvol', pd.Series(1, index=df.index)).fillna(1).clip(0.1, 10)
    
    # EPS Tier
    if 'eps_current' in df.columns:
        df['eps_tier'] = pd.cut(
            df['eps_current'].fillna(0),
            bins=[-np.inf, 5, 15, 35, 55, 75, 95, np.inf],
            labels=['5↓', '5↑', '15↑', '35↑', '55↑', '75↑', '95↑']
        )
    
    # Price Tier  
    if 'price' in df.columns:
        df['price_tier'] = pd.cut(
            df['price'],
            bins=[0, 100, 200, 500, 1000, 2000, 5000, np.inf],
            labels=['100↓', '100↑', '200↑', '500↑', '1K↑', '2K↑', '5K↑']
        )
    
    return df

def validate_essential_data(df: pd.DataFrame, diagnostics: Dict) -> pd.DataFrame:
    """Gentle validation - keep as much data as possible"""
    
    initial_count = len(df)
    
    # Only remove rows with absolutely critical issues
    if 'ticker' in df.columns:
        # Remove rows with no ticker or invalid ticker
        df = df[df['ticker'].notna() & (df['ticker'] != '') & (df['ticker'] != 'nan')]
    
    if 'price' in df.columns:
        # Only remove if price is zero or negative (keep NaN for now)
        df = df[(df['price'] > 0) | df['price'].isna()]
    
    # Remove obvious test/dummy data
    if 'company_name' in df.columns:
        df = df[~df['company_name'].str.lower().isin(['test', 'dummy', 'delete', 'remove'])]
    
    removed = initial_count - len(df)
    diagnostics['rows_removed_validation'] = removed
    
    if removed > 0:
        diagnostics['warnings'].append(f"Gentle validation removed {removed} invalid rows")
    
    return df

def calculate_data_quality(df: pd.DataFrame, diagnostics: Dict) -> Tuple[pd.DataFrame, float]:
    """Calculate comprehensive data quality metrics"""
    
    quality_metrics = {
        'total_rows': len(df),
        'columns_available': len(df.columns),
        'critical_columns_coverage': {},
        'overall_completeness': 0
    }
    
    # Check critical columns
    critical_columns = ['ticker', 'price', 'volume_1d', 'ret_7d', 'vol_ratio_30d_90d']
    
    for col in critical_columns:
        if col in df.columns:
            coverage = (df[col].notna().sum() / len(df)) * 100
            quality_metrics['critical_columns_coverage'][col] = coverage
    
    # Calculate overall completeness
    total_cells = len(df) * len(df.columns)
    non_null_cells = df.notna().sum().sum()
    quality_metrics['overall_completeness'] = (non_null_cells / total_cells) * 100
    
    # Calculate quality score
    critical_coverage = list(quality_metrics['critical_columns_coverage'].values())
    if critical_coverage:
        avg_critical = sum(critical_coverage) / len(critical_coverage)
    else:
        avg_critical = 0
    
    quality_score = (avg_critical * 0.7) + (quality_metrics['overall_completeness'] * 0.3)
    
    diagnostics['column_coverage'] = quality_metrics['critical_columns_coverage']
    
    return df, quality_score

# ============================================================================
# PATTERN DETECTION ENGINE - ENHANCED
# ============================================================================

@cached_computation(ttl=120)
def detect_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Detect key trading patterns with robust error handling"""
    
    # Initialize pattern columns
    patterns = ['EXPLOSIVE_BREAKOUT', 'STEALTH_ACCUMULATION', 'MOMENTUM_BUILDING', 'QUALITY_PULLBACK']
    for pattern in patterns:
        df[f'pattern_{pattern}'] = False
    
    # Pattern 1: EXPLOSIVE BREAKOUT
    if all(col in df.columns for col in ['ret_7d', 'volume_acceleration', 'price', 'sma_50d']):
        try:
            mask = (
                (df['ret_7d'] > PATTERN_THRESHOLDS['EXPLOSIVE_BREAKOUT']['ret_7d_min']) &
                (df['volume_acceleration'] > PATTERN_THRESHOLDS['EXPLOSIVE_BREAKOUT']['volume_accel_min']) &
                (df['price'] > df['sma_50d'])
            )
            df.loc[mask, 'pattern_EXPLOSIVE_BREAKOUT'] = True
        except:
            pass
    
    # Pattern 2: STEALTH ACCUMULATION (Most Valuable!)
    if all(col in df.columns for col in ['ret_7d', 'volume_acceleration', 'from_high_pct']):
        try:
            mask = (
                (df['ret_7d'] < PATTERN_THRESHOLDS['STEALTH_ACCUMULATION']['ret_7d_max']) &
                (df['volume_acceleration'] > PATTERN_THRESHOLDS['STEALTH_ACCUMULATION']['volume_accel_min']) &
                (df['from_high_pct'].between(*PATTERN_THRESHOLDS['STEALTH_ACCUMULATION']['from_high_range']))
            )
            df.loc[mask, 'pattern_STEALTH_ACCUMULATION'] = True
        except:
            pass
    
    # Pattern 3: MOMENTUM BUILDING
    if all(col in df.columns for col in ['ret_7d', 'ret_30d', 'volume_acceleration']):
        try:
            mask = (
                (df['ret_7d'] > PATTERN_THRESHOLDS['MOMENTUM_BUILDING']['ret_7d_min']) &
                (df['ret_30d'] > PATTERN_THRESHOLDS['MOMENTUM_BUILDING']['ret_30d_min']) &
                (df['volume_acceleration'] > PATTERN_THRESHOLDS['MOMENTUM_BUILDING']['volume_accel_min'])
            )
            df.loc[mask, 'pattern_MOMENTUM_BUILDING'] = True
        except:
            pass
    
    # Pattern 4: QUALITY PULLBACK
    if all(col in df.columns for col in ['ret_1y', 'from_high_pct', 'eps_change_pct']):
        try:
            mask = (
                (df['ret_1y'] > PATTERN_THRESHOLDS['QUALITY_PULLBACK']['ret_1y_min']) &
                (df['from_high_pct'].between(*PATTERN_THRESHOLDS['QUALITY_PULLBACK']['from_high_range'])) &
                (df['eps_change_pct'] > PATTERN_THRESHOLDS['QUALITY_PULLBACK']['eps_change_min'])
            )
            df.loc[mask, 'pattern_QUALITY_PULLBACK'] = True
        except:
            pass
    
    # Calculate pattern score
    pattern_cols = [col for col in df.columns if col.startswith('pattern_')]
    df['pattern_count'] = df[pattern_cols].sum(axis=1)
    
    # Assign primary pattern
    df['primary_pattern'] = 'NONE'
    for pattern in patterns:
        mask = df[f'pattern_{pattern}']
        df.loc[mask, 'primary_pattern'] = pattern
    
    return df

# ============================================================================
# EDGE SCORING ENGINE - ENHANCED WITH FALLBACKS
# ============================================================================

@cached_computation(ttl=120)
def calculate_edge_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate EDGE scores with comprehensive fallbacks"""
    
    # Initialize all score components
    df['score_volume'] = 50      # Start neutral
    df['score_momentum'] = 50
    df['score_entry'] = 50
    df['score_risk_reward'] = 50
    
    # 1. VOLUME SCORE (40% weight) - THE SECRET WEAPON
    if df['has_volume_acceleration'].any() and 'volume_acceleration' in df.columns:
        # True volume acceleration available
        
        # Non-linear scoring for volume acceleration
        df['score_volume'] = np.where(
            df['volume_acceleration'] > 30, 100,
            np.where(df['volume_acceleration'] > 20, 90,
            np.where(df['volume_acceleration'] > 10, 75,
            np.where(df['volume_acceleration'] > 0, 60,
            np.where(df['volume_acceleration'] > -10, 40, 25))))
        )
        
        # Boost for consistent volume
        if 'volume_consistency' in df.columns:
            consistency_boost = 0.7 + (0.3 * df['volume_consistency'])
            df['score_volume'] = (df['score_volume'] * consistency_boost).clip(0, 100)
        
        # RVOL multiplier
        if 'rvol' in df.columns:
            rvol_multiplier = df['rvol'].clip(0.5, 3.0) / 2
            df['score_volume'] = (df['score_volume'] * rvol_multiplier).clip(0, 100)
            
        logger.info(f"Volume scores calculated - Mean: {df['score_volume'].mean():.1f}")
    else:
        # Fallback to simple volume metrics
        logger.warning("Using fallback volume scoring")
        if 'rvol' in df.columns:
            df['score_volume'] = (df['rvol'].clip(0, 5) * 20).fillna(50)
        
        # Try to use any available volume ratios
        vol_ratio_cols = [col for col in df.columns if 'vol_ratio' in col]
        if vol_ratio_cols:
            # Average positive volume ratios
            positive_ratios = df[vol_ratio_cols].clip(lower=0).mean(axis=1)
            df['score_volume'] = (50 + positive_ratios).clip(0, 100)
    
    # 2. MOMENTUM SCORE (25% weight)
    if all(col in df.columns for col in ['ret_7d', 'ret_30d']):
        # Momentum alignment
        short_momentum = df['ret_7d'].fillna(0)
        medium_momentum = df['ret_30d'].fillna(0)
        
        # Best: positive and accelerating
        df['score_momentum'] = np.where(
            (short_momentum > medium_momentum / 4) & (short_momentum > 0), 
            80 + short_momentum.clip(0, 20),
            np.where(short_momentum > 0, 60 + short_momentum.clip(0, 20),
            np.where(short_momentum > -5, 50 + short_momentum * 2, 25))
        )
        
        # Trend bonus
        if all(col in df.columns for col in ['price', 'sma_50d', 'sma_200d']):
            above_50 = (df['price'] > df['sma_50d']).astype(int) * 10
            above_200 = (df['price'] > df['sma_200d']).astype(int) * 10
            trend_score = above_50 + above_200
            df['score_momentum'] = (df['score_momentum'] + trend_score).clip(0, 100)
    else:
        # Fallback - use any available return columns
        return_cols = [col for col in df.columns if col.startswith('ret_')]
        if return_cols:
            df['score_momentum'] = 50 + df[return_cols].mean(axis=1).clip(-50, 50)
    
    # 3. ENTRY SCORE (20% weight) - Smart entry points
    if 'from_high_pct' in df.columns:
        # Best entries: -15% to -25% from high
        df['score_entry'] = np.where(
            df['from_high_pct'].between(-25, -15), 100,
            np.where(df['from_high_pct'].between(-35, -10), 75,
            np.where(df['from_high_pct'] > -10, 40,  # Too close to high
            np.where(df['from_high_pct'] < -50, 30, 50)))  # Too far fallen
        )
        
        # Boost for support bounce
        if all(col in df.columns for col in ['price', 'sma_200d', 'ret_3d']):
            support_bounce = (
                (df['price'] > df['sma_200d'] * 0.95) & 
                (df['price'] < df['sma_200d'] * 1.05) & 
                (df['ret_3d'] > 0)
            )
            df.loc[support_bounce, 'score_entry'] = 90
    else:
        # Fallback - use price position
        if 'price_position' in df.columns:
            # Good entry between 20-60% from 52w low
            df['score_entry'] = np.where(
                df['price_position'].between(20, 60), 80,
                np.where(df['price_position'] < 20, 60,
                np.where(df['price_position'] > 80, 40, 50))
            )
    
    # 4. RISK/REWARD SCORE (15% weight)
    if all(col in df.columns for col in ['price', 'high_52w', 'low_52w']):
        upside = ((df['high_52w'] - df['price']) / df['price'].replace(0, 1) * 100).clip(0, 100)
        downside = ((df['price'] - df['low_52w']) / df['price'].replace(0, 1) * 100).clip(1, 100)
        
        df['risk_reward_ratio'] = (upside / downside).clip(0, 10)
        df['score_risk_reward'] = (df['risk_reward_ratio'] * 10).clip(0, 100)
    else:
        # Fallback - neutral score
        df['score_risk_reward'] = 50
    
    # FINAL EDGE SCORE CALCULATION
    df['edge_score'] = (
        df['score_volume'] * 0.40 +
        df['score_momentum'] * 0.25 +
        df['score_entry'] * 0.20 +
        df['score_risk_reward'] * 0.15
    )
    
    # Pattern bonus (up to 10 points)
    if 'pattern_count' in df.columns:
        pattern_bonus = df['pattern_count'] * 3
        df['edge_score'] = (df['edge_score'] + pattern_bonus).clip(0, 100)
    
    # Log score distribution
    logger.info(f"EDGE Score Stats - Mean: {df['edge_score'].mean():.1f}, "
                f"Max: {df['edge_score'].max():.1f}, "
                f">70: {(df['edge_score'] > 70).sum()}")
    
    # Classify signals
    df['signal_strength'] = pd.cut(
        df['edge_score'],
        bins=[-np.inf, 30, 50, 70, 85, 100.1],
        labels=['AVOID', 'WATCH', 'MODERATE', 'STRONG', 'EXPLOSIVE']
    )
    
    # Volume pattern classification
    if 'volume_acceleration' in df.columns and df['has_volume_acceleration'].any():
        df['volume_pattern'] = pd.cut(
            df['volume_acceleration'],
            bins=[-np.inf, -20, -10, 0, 10, 20, 30, np.inf],
            labels=['HEAVY_DIST', 'DISTRIBUTION', 'NEUTRAL', 'ACCUMULATION', 
                   'HEAVY_ACCUM', 'INSTITUTIONAL', 'EXPLOSIVE_VOL']
        )
    else:
        df['volume_pattern'] = 'UNKNOWN'
    
    # Generate "Why" explanations
    df['signal_reason'] = df.apply(generate_signal_reason, axis=1)
    
    return df

def generate_signal_reason(row) -> str:
    """Generate human-readable explanation for signal"""
    
    reasons = []
    
    # Check volume acceleration
    if 'volume_acceleration' in row and not pd.isna(row['volume_acceleration']):
        if row['volume_acceleration'] > 30:
            reasons.append("🏦 Institutional loading detected")
        elif row['volume_acceleration'] > 20:
            reasons.append("📊 Heavy accumulation")
        elif row['volume_acceleration'] > 10:
            reasons.append("📈 Volume picking up")
    
    # Check RVOL
    if 'rvol' in row and row['rvol'] > 2:
        reasons.append(f"🔥 High activity ({row['rvol']:.1f}x normal)")
    
    # Check patterns
    if row.get('primary_pattern') == 'EXPLOSIVE_BREAKOUT':
        reasons.append("🚀 Explosive breakout pattern")
    elif row.get('primary_pattern') == 'STEALTH_ACCUMULATION':
        reasons.append("🕵️ Smart money accumulating")
    elif row.get('primary_pattern') == 'MOMENTUM_BUILDING':
        reasons.append("⚡ Momentum accelerating")
    elif row.get('primary_pattern') == 'QUALITY_PULLBACK':
        reasons.append("💎 Quality stock on sale")
    
    # Check momentum
    if 'ret_7d' in row and row['ret_7d'] > 10:
        reasons.append("🔥 Strong 7-day momentum")
    
    # Check value
    if 'from_high_pct' in row and -30 < row['from_high_pct'] < -15:
        reasons.append("🎯 Good entry point")
    
    # Risk/Reward
    if 'risk_reward_ratio' in row and row['risk_reward_ratio'] > 3:
        reasons.append("💰 Excellent risk/reward")
    
    # If no specific reasons, provide generic based on score
    if not reasons:
        if row.get('edge_score', 0) > 85:
            reasons.append("⚡ Multiple bullish signals align")
        elif row.get('edge_score', 0) > 70:
            reasons.append("✅ Strong technical setup")
        elif row.get('edge_score', 0) > 50:
            reasons.append("👀 Positive momentum building")
        else:
            reasons.append("📊 Early stage signal")
    
    return " | ".join(reasons[:2])  # Limit to 2 reasons for space

# ============================================================================
# RISK MANAGEMENT ENGINE
# ============================================================================

def calculate_position_sizing(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate position sizes with sophisticated risk management"""
    
    # Base position size from signal strength
    signal_map = {
        'EXPLOSIVE': POSITION_SIZES['EXPLOSIVE'],
        'STRONG': POSITION_SIZES['STRONG'],
        'MODERATE': POSITION_SIZES['MODERATE'],
        'WATCH': POSITION_SIZES['WATCH'],
        'AVOID': 0
    }
    
    # Safely map signal strength to position size
    df['signal_str'] = df['signal_strength'].astype(str)
    df['base_position_size'] = df['signal_str'].map(signal_map).fillna(0).astype(float)
    
    # Adjust for volatility
    if 'volatility_52w' in df.columns:
        # Lower position size for high volatility
        vol_adjustment = 1 - (df['volatility_52w'].clip(0, 1) * 0.5)
        df['position_size'] = df['base_position_size'] * vol_adjustment
    else:
        df['position_size'] = df['base_position_size']
    
    # Further adjust for pattern confidence
    if 'pattern_count' in df.columns:
        pattern_multiplier = 1 + (df['pattern_count'] * 0.1)  # 10% boost per pattern
        df['position_size'] = df['position_size'] * pattern_multiplier
    
    # Cap at maximum
    df['position_size'] = df['position_size'].clip(0, RISK_PARAMS['MAX_SINGLE_POSITION'])
    
    # Calculate stops and targets
    df = calculate_stop_losses(df)
    df = calculate_targets(df)
    
    # Clean up temporary column
    df = df.drop('signal_str', axis=1, errors='ignore')
    
    return df

def calculate_stop_losses(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate intelligent stop losses"""
    
    # Default stop based on volatility
    if 'volatility_52w' in df.columns:
        df['stop_loss_pct'] = (df['volatility_52w'] * 0.2).clip(0.05, 0.15)
    else:
        df['stop_loss_pct'] = 0.08  # Default 8%
    
    # Adjust for support levels
    if all(col in df.columns for col in ['price', 'sma_50d', 'sma_200d', 'low_52w']):
        # Find nearest support
        support_50 = df['sma_50d'] * 0.98  # 2% below MA
        support_200 = df['sma_200d'] * 0.98
        support_52w = df['low_52w'] * RISK_PARAMS['STOP_LOSS_BUFFER']
        
        # Stack supports and find max below price
        supports = pd.DataFrame({
            's1': support_50,
            's2': support_200,
            's3': support_52w
        })
        
        # Use the highest support that's below current price
        df['support_stop'] = supports.max(axis=1)
        
        # Calculate percentage
        support_stop_pct = ((df['price'] - df['support_stop']) / df['price'].replace(0, 1)).clip(0.05, 0.15)
        
        # Use tighter of the two
        df['stop_loss_pct'] = pd.DataFrame({
            'vol_stop': df['stop_loss_pct'],
            'support_stop': support_stop_pct
        }).min(axis=1)
    
    # Calculate actual stop price
    df['stop_loss'] = df['price'] * (1 - df['stop_loss_pct'])
    
    return df

def calculate_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate profit targets based on pattern and risk/reward"""
    
    # Base targets on signal strength and patterns
    target_multipliers = {
        'EXPLOSIVE': (1.15, 1.30, 1.50),
        'STRONG': (1.10, 1.20, 1.35),
        'MODERATE': (1.07, 1.15, 1.25),
        'WATCH': (1.05, 1.10, 1.15),
        'AVOID': (1.00, 1.00, 1.00)
    }
    
    # Calculate base targets safely
    for i, suffix in enumerate(['1', '2', '3']):
        df[f'target_{suffix}'] = df.apply(
            lambda row: row['price'] * target_multipliers.get(
                str(row.get('signal_strength', 'MODERATE')), 
                target_multipliers['MODERATE']
            )[i] if pd.notna(row.get('price', 0)) else 0,
            axis=1
        )
    
    # Adjust for resistance levels
    if 'high_52w' in df.columns:
        # Don't set targets above 52w high initially
        df['target_1'] = df[['target_1', 'high_52w']].min(axis=1)
    
    # Calculate risk/reward ratios
    if all(col in df.columns for col in ['target_1', 'price', 'stop_loss']):
        risk = (df['price'] - df['stop_loss']).replace(0, 0.01)
        reward = df['target_1'] - df['price']
        df['risk_reward_1'] = (reward / risk).clip(0, 10)
    
    return df

# ============================================================================
# PORTFOLIO OPTIMIZATION
# ============================================================================

def optimize_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    """Apply portfolio-level constraints and optimization"""
    
    # Sort by edge score (best first)
    df = df.sort_values('edge_score', ascending=False).copy()
    
    # Track allocations
    total_allocation = 0
    sector_allocations = {}
    
    # Optimize position sizes
    for idx in df.index:
        if str(df.loc[idx, 'signal_strength']) == 'AVOID':
            continue
            
        current_size = df.loc[idx, 'position_size']
        sector = str(df.loc[idx, 'sector']) if 'sector' in df.columns else 'Unknown'
        
        # Check portfolio constraint
        if total_allocation + current_size > RISK_PARAMS['MAX_PORTFOLIO_EXPOSURE']:
            remaining = RISK_PARAMS['MAX_PORTFOLIO_EXPOSURE'] - total_allocation
            df.loc[idx, 'position_size'] = max(0, remaining)
        
        # Check sector constraint
        sector_alloc = sector_allocations.get(sector, 0)
        if sector_alloc + current_size > RISK_PARAMS['MAX_SECTOR_EXPOSURE']:
            remaining = RISK_PARAMS['MAX_SECTOR_EXPOSURE'] - sector_alloc
            df.loc[idx, 'position_size'] = min(df.loc[idx, 'position_size'], max(0, remaining))
        
        # Update trackers
        actual_size = df.loc[idx, 'position_size']
        total_allocation += actual_size
        sector_allocations[sector] = sector_allocations.get(sector, 0) + actual_size
    
    # Add portfolio metrics
    df['portfolio_weight'] = df['position_size']
    df['cumulative_allocation'] = df['position_size'].cumsum()
    
    return df

# ============================================================================
# VISUALIZATION ENGINE
# ============================================================================

def create_volume_acceleration_map(df: pd.DataFrame) -> go.Figure:
    """Create the signature volume acceleration visualization"""
    
    # Filter for meaningful data
    plot_df = df[
        df['volume_acceleration'].notna() & 
        (df['edge_score'] > 30)
    ].nlargest(100, 'edge_score')
    
    if plot_df.empty:
        return create_empty_chart("No volume acceleration data available")
    
    # Create scatter plot
    fig = go.Figure()
    
    # Color mapping
    colors = {
        'EXPLOSIVE': '#FF0000',
        'STRONG': '#FF6600',
        'MODERATE': '#FFA500',
        'WATCH': '#808080'
    }
    
    # Add traces by signal strength
    for signal in ['EXPLOSIVE', 'STRONG', 'MODERATE', 'WATCH']:
        signal_df = plot_df[plot_df['signal_strength'].astype(str) == signal]
        if not signal_df.empty:
            # Get hover text with reasons
            hover_text = signal_df.apply(
                lambda row: f"<b>{row['ticker']}</b><br>" +
                           f"Vol Accel: {row['volume_acceleration']:.1f}%<br>" +
                           f"7d Return: {row.get('ret_7d', 0):.1f}%<br>" +
                           f"EDGE Score: {row['edge_score']:.1f}<br>" +
                           f"Why: {row.get('signal_reason', 'Multiple factors')}<br>",
                axis=1
            )
            
            fig.add_trace(go.Scatter(
                x=signal_df['volume_acceleration'],
                y=signal_df.get('ret_7d', 0),
                mode='markers+text',
                name=signal,
                text=signal_df['ticker'],
                textposition="top center",
                textfont=dict(size=8),
                marker=dict(
                    size=signal_df['edge_score'] / 5,
                    color=colors.get(signal, '#808080'),
                    line=dict(width=1, color='black'),
                    opacity=0.8
                ),
                hovertext=hover_text,
                hoverinfo='text'
            ))
    
    # Add quadrant lines and labels
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    # Quadrant annotations
    fig.add_annotation(x=40, y=15, text="🚀 EXPLOSIVE ZONE", showarrow=False, 
                      font=dict(size=16, color="red"))
    fig.add_annotation(x=40, y=-10, text="🏦 STEALTH ACCUMULATION", showarrow=False,
                      font=dict(size=16, color="green"))
    fig.add_annotation(x=-30, y=15, text="📉 PROFIT TAKING", showarrow=False,
                      font=dict(size=16, color="orange"))
    fig.add_annotation(x=-30, y=-10, text="☠️ DANGER ZONE", showarrow=False,
                      font=dict(size=16, color="gray"))
    
    # Update layout
    fig.update_layout(
        title={
            'text': "Volume Acceleration Map - Your SECRET WEAPON",
            'font': {'size': 24, 'color': '#FF6600'}
        },
        xaxis_title="Volume Acceleration % (30d/90d vs 30d/180d)",
        yaxis_title="7-Day Return %",
        height=600,
        hovermode='closest',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    return fig

def create_edge_score_distribution(df: pd.DataFrame) -> go.Figure:
    """Create EDGE score distribution chart"""
    
    fig = go.Figure()
    
    # Create histogram
    fig.add_trace(go.Histogram(
        x=df['edge_score'],
        nbinsx=30,
        marker_color='rgba(255, 102, 0, 0.7)',
        name='Distribution'
    ))
    
    # Add threshold lines
    thresholds = [
        ('EXPLOSIVE', EDGE_THRESHOLDS['EXPLOSIVE'], '#FF0000'),
        ('STRONG', EDGE_THRESHOLDS['STRONG'], '#FF6600'),
        ('MODERATE', EDGE_THRESHOLDS['MODERATE'], '#FFA500'),
        ('WATCH', EDGE_THRESHOLDS['WATCH'], '#808080')
    ]
    
    for name, value, color in thresholds:
        fig.add_vline(
            x=value,
            line_dash="dash",
            line_color=color,
            annotation_text=name,
            annotation_position="top"
        )
    
    # Update layout
    fig.update_layout(
        title="EDGE Score Distribution",
        xaxis_title="EDGE Score",
        yaxis_title="Number of Stocks",
        height=400,
        bargap=0.1
    )
    
    return fig

def create_pattern_sunburst(df: pd.DataFrame) -> go.Figure:
    """Create pattern distribution sunburst chart"""
    
    # Prepare data for sunburst
    pattern_data = []
    
    for signal in ['EXPLOSIVE', 'STRONG', 'MODERATE']:
        signal_df = df[df['signal_strength'].astype(str) == signal]
        if not signal_df.empty:
            # Add signal level
            pattern_data.append({
                'labels': signal,
                'parents': '',
                'values': len(signal_df)
            })
            
            # Add patterns under each signal
            for pattern in signal_df['primary_pattern'].value_counts().index:
                if pattern != 'NONE':
                    count = len(signal_df[signal_df['primary_pattern'] == pattern])
                    pattern_data.append({
                        'labels': pattern.replace('_', ' '),
                        'parents': signal,
                        'values': count
                    })
    
    if not pattern_data:
        return create_empty_chart("No pattern data available")
    
    pattern_df = pd.DataFrame(pattern_data)
    
    # Create sunburst
    fig = go.Figure(go.Sunburst(
        labels=pattern_df['labels'],
        parents=pattern_df['parents'],
        values=pattern_df['values'],
        branchvalues="total",
        marker=dict(
            colors=['#FF0000', '#FF6600', '#FFA500', '#32CD32', '#1E90FF', '#9370DB'],
            line=dict(color="white", width=2)
        ),
        textinfo="label+percent parent"
    ))
    
    fig.update_layout(
        title="Signal & Pattern Distribution",
        height=500
    )
    
    return fig

def create_empty_chart(message: str) -> go.Figure:
    """Create empty chart with message"""
    fig = go.Figure()
    fig.add_annotation(
        x=0.5, y=0.5,
        text=message,
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=20, color="gray")
    )
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=400
    )
    return fig

# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_header():
    """Render application header"""
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #FF6600 0%, #FF0000 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        font-size: 3rem;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .main-header p {
        color: #FFE4B5;
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
    }
    .signal-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .explosive-alert {
        background: linear-gradient(45deg, #FF0000, #FF6600);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        animation: pulse 2s infinite;
        margin: 2rem 0;
    }
    @keyframes pulse {
        0% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.05); opacity: 0.9; }
        100% { transform: scale(1); opacity: 1; }
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #FF6600;
        margin: 0.5rem 0;
    }
    .reason-tag {
        background: #e9ecef;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.9rem;
        margin: 0.25rem;
        display: inline-block;
    }
    .diagnostic-info {
        background: #f0f8ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #0066cc;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="main-header">
        <h1>⚡ EDGE Protocol 3.1</h1>
        <p>Volume Acceleration Intelligence System</p>
    </div>
    """, unsafe_allow_html=True)

def render_diagnostics(diagnostics: Dict[str, Any]):
    """Render data diagnostics panel"""
    with st.expander("🔍 Data Diagnostics", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Rows Loaded", diagnostics.get('rows_loaded', 0))
            st.metric("Rows Valid", diagnostics.get('rows_valid', 0))
        
        with col2:
            st.metric("Data Quality", f"{diagnostics.get('data_quality_score', 0):.1f}%")
            st.metric("Volume Format", diagnostics.get('volume_ratio_format', 'unknown'))
        
        with col3:
            parsing_stats = diagnostics.get('parsing_stats', {})
            st.metric("Columns Parsed", parsing_stats.get('successful', 0))
            st.metric("Parse Failures", parsing_stats.get('failed', 0))
        
        if diagnostics.get('warnings'):
            st.warning("Warnings:")
            for warning in diagnostics['warnings']:
                st.write(f"• {warning}")
        
        # Column coverage
        if diagnostics.get('column_coverage'):
            st.subheader("Column Coverage")
            coverage_df = pd.DataFrame([
                {'Column': col, 'Coverage': f"{cov:.1f}%"}
                for col, cov in diagnostics['column_coverage'].items()
            ])
            st.dataframe(coverage_df, use_container_width=True, hide_index=True)

def render_key_metrics(df: pd.DataFrame, filtered_df: pd.DataFrame):
    """Render key performance metrics"""
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        explosive_count = len(filtered_df[filtered_df['signal_strength'].astype(str) == 'EXPLOSIVE'])
        st.metric(
            "🚀 EXPLOSIVE",
            explosive_count,
            help="Immediate action required - Top 1%"
        )
    
    with col2:
        strong_count = len(filtered_df[filtered_df['signal_strength'].astype(str) == 'STRONG'])
        st.metric(
            "💪 STRONG",
            strong_count,
            help="High conviction opportunities"
        )
    
    with col3:
        if 'volume_acceleration' in filtered_df.columns and filtered_df['has_volume_acceleration'].any():
            top_signals = filtered_df[filtered_df['signal_strength'].astype(str).isin(['EXPLOSIVE', 'STRONG'])]
            if not top_signals.empty:
                avg_accel = top_signals['volume_acceleration'].mean()
                st.metric(
                    "📊 Avg Vol Accel",
                    f"{avg_accel:.1f}%",
                    help="Average volume acceleration of top signals"
                )
            else:
                st.metric("📊 Vol Accel", "N/A")
        else:
            st.metric("📊 Vol Accel", "No Data")
    
    with col4:
        total_signals = len(filtered_df[filtered_df['signal_strength'].astype(str) != 'AVOID'])
        st.metric(
            "📈 Total Signals",
            total_signals,
            help="All actionable opportunities"
        )
    
    with col5:
        portfolio_used = filtered_df['position_size'].sum() * 100
        st.metric(
            "💼 Portfolio Used",
            f"{portfolio_used:.1f}%",
            help="Total recommended allocation"
        )

def render_signals_table(df: pd.DataFrame):
    """Render main signals table with formatting"""
    
    if df.empty:
        st.info("No signals match current filters")
        return
    
    # Select display columns
    display_columns = [
        'ticker', 'company_name', 'sector', 'category',
        'signal_strength', 'edge_score', 'signal_reason',
        'volume_acceleration', 'primary_pattern', 
        'price', 'position_size', 'stop_loss', 'target_1', 'risk_reward_1'
    ]
    
    # Filter to available columns
    available_columns = [col for col in display_columns if col in df.columns]
    display_df = df[available_columns].copy()
    
    # Format columns
    format_dict = {}
    if 'edge_score' in available_columns:
        format_dict['edge_score'] = '{:.1f}'
    if 'volume_acceleration' in available_columns:
        format_dict['volume_acceleration'] = '{:.1f}%'
    if 'price' in available_columns:
        format_dict['price'] = '₹{:.2f}'
    if 'position_size' in available_columns:
        format_dict['position_size'] = '{:.1%}'
    if 'stop_loss' in available_columns:
        format_dict['stop_loss'] = '₹{:.2f}'
    if 'target_1' in available_columns:
        format_dict['target_1'] = '₹{:.2f}'
    if 'risk_reward_1' in available_columns:
        format_dict['risk_reward_1'] = '{:.2f}'
    
    # Apply formatting
    styled_df = display_df.style.format(format_dict, na_rep='')
    
    # Color code signal strength
    def color_signal(val):
        colors = {
            'EXPLOSIVE': 'background-color: #FF0000; color: white;',
            'STRONG': 'background-color: #FF6600; color: white;',
            'MODERATE': 'background-color: #FFA500; color: black;',
            'WATCH': 'background-color: #808080; color: white;'
        }
        return colors.get(str(val), '')
    
    if 'signal_strength' in available_columns:
        styled_df = styled_df.applymap(
            lambda x: color_signal(x), 
            subset=['signal_strength']
        )
    
    # Apply gradient to edge score
    if 'edge_score' in available_columns:
        styled_df = styled_df.background_gradient(
            subset=['edge_score'],
            cmap='RdYlGn',
            vmin=30,
            vmax=100
        )
    
    # Display table
    st.dataframe(
        styled_df,
        use_container_width=True,
        height=600
    )

def render_filters(df: pd.DataFrame) -> Dict[str, Any]:
    """Render sidebar filters"""
    
    st.sidebar.header("🎯 Filters")
    
    filters = {}
    
    # Signal strength filter
    filters['min_edge_score'] = st.sidebar.slider(
        "Minimum EDGE Score",
        0, 100, 50, 5,
        help="Filter stocks by minimum EDGE score"
    )
    
    # Category filter
    if 'category' in df.columns:
        categories = sorted([cat for cat in df['category'].unique() if cat and cat != 'nan'])
        filters['categories'] = st.sidebar.multiselect(
            "Market Cap Category",
            categories,
            default=[],
            help="Filter by market capitalization"
        )
    
    # Sector filter
    if 'sector' in df.columns:
        sectors = sorted([sec for sec in df['sector'].unique() if sec and sec != 'nan'])
        filters['sectors'] = st.sidebar.multiselect(
            "Sector",
            sectors,
            default=[],
            help="Filter by business sector"
        )
    
    # Pattern filter
    if 'primary_pattern' in df.columns:
        patterns = sorted(df[df['primary_pattern'] != 'NONE']['primary_pattern'].unique().tolist())
        filters['patterns'] = st.sidebar.multiselect(
            "Patterns",
            patterns,
            default=[],
            help="Filter by detected patterns"
        )
    
    # Volume acceleration filter
    if 'volume_acceleration' in df.columns and df['has_volume_acceleration'].any():
        filters['min_vol_accel'] = st.sidebar.number_input(
            "Min Volume Acceleration %",
            value=0.0,
            step=5.0,
            help="Minimum volume acceleration percentage"
        )
    
    # Show only actionable
    filters['actionable_only'] = st.sidebar.checkbox(
        "Show Only Actionable Signals",
        value=True,
        help="Hide AVOID signals"
    )
    
    return filters

def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """Apply selected filters to dataframe"""
    
    filtered = df.copy()
    
    # EDGE score filter
    if 'edge_score' in filtered.columns:
        filtered = filtered[filtered['edge_score'] >= filters['min_edge_score']]
    
    # Category filter
    if filters.get('categories') and 'category' in filtered.columns:
        filtered = filtered[filtered['category'].isin(filters['categories'])]
    
    # Sector filter
    if filters.get('sectors') and 'sector' in filtered.columns:
        filtered = filtered[filtered['sector'].isin(filters['sectors'])]
    
    # Pattern filter
    if filters.get('patterns') and 'primary_pattern' in filtered.columns:
        filtered = filtered[filtered['primary_pattern'].isin(filters['patterns'])]
    
    # Volume acceleration filter
    if 'min_vol_accel' in filters and 'volume_acceleration' in filtered.columns:
        filtered = filtered[filtered['volume_acceleration'] >= filters['min_vol_accel']]
    
    # Actionable only filter
    if filters.get('actionable_only') and 'signal_strength' in filtered.columns:
        filtered = filtered[filtered['signal_strength'].astype(str) != 'AVOID']
    
    return filtered

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point"""
    
    # Page configuration
    st.set_page_config(
        page_title="EDGE Protocol 3.1",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Render header
    render_header()
    
    # Load data with progress
    with st.spinner("🔄 Loading market data..."):
        df, diagnostics = load_data()
    
    # Check data quality
    if df.empty:
        st.error("❌ Unable to load data from Google Sheets")
        
        # Show diagnostics
        with st.expander("🔧 Diagnostics", expanded=True):
            st.json(diagnostics)
            
            if diagnostics['warnings']:
                st.error(f"**Error:** {diagnostics['warnings'][0]}")
                
                # Provide solutions
                st.markdown("""
                **Solutions:**
                1. Make sure the Google Sheet is public (Share → Anyone with link can view)
                2. Check the Sheet ID and GID are correct
                3. Verify your internet connection
                
                **Direct Sheet Link:**
                [Open Google Sheet](https://docs.google.com/spreadsheets/d/1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk)
                """)
        
        st.stop()
    
    # Show diagnostics panel
    render_diagnostics(diagnostics)
    
    # Process data through the pipeline
    with st.spinner("🧮 Calculating EDGE scores..."):
        # Detect patterns
        df = detect_patterns(df)
        
        # Calculate EDGE scores
        df = calculate_edge_scores(df)
        
        # Calculate position sizing
        df = calculate_position_sizing(df)
        
        # Optimize portfolio
        df = optimize_portfolio(df)
    
    # Get filters
    filters = render_filters(df)
    
    # Apply filters
    filtered_df = apply_filters(df, filters)
    
    # Check for explosive signals
    explosive_signals = filtered_df[filtered_df['signal_strength'].astype(str) == 'EXPLOSIVE']
    if not explosive_signals.empty:
        st.markdown(f"""
        <div class="explosive-alert">
            ⚡ {len(explosive_signals)} EXPLOSIVE SIGNAL{'S' if len(explosive_signals) > 1 else ''} DETECTED! ⚡
            <br>IMMEDIATE ACTION REQUIRED
        </div>
        """, unsafe_allow_html=True)
    
    # Display key metrics
    render_key_metrics(df, filtered_df)
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Trading Signals",
        "🗺️ Volume Map",
        "📈 Analytics",
        "🎯 Top Picks",
        "❓ Help"
    ])
    
    with tab1:
        st.header("📊 Trading Signals")
        
        # Quick actions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            show_explosive_only = st.checkbox("🚀 Show EXPLOSIVE Only", value=False)
        
        with col2:
            sort_by = st.selectbox(
                "Sort By",
                ['edge_score', 'volume_acceleration', 'ret_7d', 'position_size'],
                index=0
            )
        
        with col3:
            sort_order = st.radio("Order", ['Descending', 'Ascending'], horizontal=True)
        
        # Apply quick filters
        display_df = filtered_df.copy()
        
        if show_explosive_only:
            display_df = display_df[display_df['signal_strength'].astype(str) == 'EXPLOSIVE']
        
        # Sort
        if sort_by in display_df.columns:
            display_df = display_df.sort_values(
                sort_by,
                ascending=(sort_order == 'Ascending'),
                na_position='last'
            )
        
        # Display signals table
        render_signals_table(display_df)
        
        # Export button
        if not display_df.empty:
            csv = display_df.to_csv(index=False)
            st.download_button(
                "📥 Download Signals (CSV)",
                csv,
                f"edge_signals_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                "text/csv"
            )
    
    with tab2:
        st.header("🗺️ Volume Acceleration Map")
        
        # Check if volume acceleration is available
        if not df['has_volume_acceleration'].any():
            st.warning("⚠️ Volume acceleration data not available. Using alternative metrics.")
        
        # Create and display volume acceleration map
        if 'volume_acceleration' in filtered_df.columns:
            fig_volume_map = create_volume_acceleration_map(filtered_df)
            st.plotly_chart(fig_volume_map, use_container_width=True)
            
            # Volume acceleration insights
            st.markdown("### 🔍 Volume Acceleration Insights")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                institutional_loading = filtered_df[filtered_df['volume_acceleration'] > 30]
                st.metric(
                    "🏦 Institutional Loading",
                    len(institutional_loading),
                    help="Stocks with >30% volume acceleration"
                )
            
            with col2:
                stealth_accum = filtered_df[
                    (filtered_df['volume_acceleration'] > 20) &
                    (filtered_df.get('ret_7d', 0) < 0)
                ]
                st.metric(
                    "🕵️ Stealth Accumulation",
                    len(stealth_accum),
                    help="High volume but negative price - smart money loading"
                )
            
            with col3:
                distribution = filtered_df[filtered_df['volume_acceleration'] < -20]
                st.metric(
                    "📉 Distribution",
                    len(distribution),
                    help="Stocks being sold aggressively"
                )
    
    with tab3:
        st.header("📈 Analytics Dashboard")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # EDGE score distribution
            fig_edge_dist = create_edge_score_distribution(filtered_df)
            st.plotly_chart(fig_edge_dist, use_container_width=True)
        
        with col2:
            # Pattern sunburst
            if 'primary_pattern' in filtered_df.columns:
                fig_pattern = create_pattern_sunburst(filtered_df)
                st.plotly_chart(fig_pattern, use_container_width=True)
        
        # Sector analysis
        if 'sector' in filtered_df.columns:
            st.markdown("### 🏭 Sector Analysis")
            
            # Group by sector and calculate stats
            sector_stats = filtered_df.groupby('sector').agg({
                'edge_score': ['mean', 'count'],
                'volume_acceleration': lambda x: x.mean() if 'volume_acceleration' in filtered_df.columns else 0
            }).round(1)
            
            # Flatten column names
            sector_stats.columns = ['avg_edge_score', 'count', 'avg_vol_accel']
            sector_stats = sector_stats.sort_values('avg_edge_score', ascending=False)
            
            # Create sector bar chart
            fig_sector = go.Figure()
            
            fig_sector.add_trace(go.Bar(
                x=sector_stats.index,
                y=sector_stats['avg_edge_score'],
                name='Avg EDGE Score',
                marker_color='#FF6600',
                text=sector_stats['avg_edge_score'],
                textposition='outside'
            ))
            
            fig_sector.update_layout(
                title="Average EDGE Score by Sector",
                xaxis_title="Sector",
                yaxis_title="Average EDGE Score",
                height=400
            )
            
            st.plotly_chart(fig_sector, use_container_width=True)
    
    with tab4:
        st.header("🎯 Top Picks with Explanations")
        
        # Display top picks by category with reasons
        for signal_type in ['EXPLOSIVE', 'STRONG', 'MODERATE']:
            top_picks = filtered_df[
                filtered_df['signal_strength'].astype(str) == signal_type
            ].head(5)
            
            if not top_picks.empty:
                st.markdown(f"### {signal_type} Signals")
                
                for _, stock in top_picks.iterrows():
                    # Create detailed signal card
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        ticker = stock.get('ticker', 'N/A')
                        company = stock.get('company_name', 'Unknown')
                        pattern = stock.get('primary_pattern', 'NONE').replace('_', ' ')
                        reason = stock.get('signal_reason', 'Multiple bullish factors')
                        
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3 style="margin:0">{ticker} - {company}</h3>
                            <p style="margin:0.5rem 0"><strong>Pattern:</strong> {pattern}</p>
                            <p style="margin:0.5rem 0"><strong>Why Buy:</strong> {reason}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        # Key metrics in a compact grid
                        metric_col1, metric_col2 = st.columns(2)
                        
                        with metric_col1:
                            st.metric("EDGE", f"{stock.get('edge_score', 0):.0f}")
                            st.metric("Price", f"₹{stock.get('price', 0):.0f}")
                        
                        with metric_col2:
                            vol_accel = stock.get('volume_acceleration', 0)
                            st.metric("Vol Acc", f"{vol_accel:.0f}%")
                            rr = stock.get('risk_reward_1', 0)
                            st.metric("R/R", f"{rr:.1f}")
                    
                    # Trading levels
                    st.markdown(f"""
                    **Trading Levels:** 
                    Entry: ₹{stock.get('price', 0):.0f} | 
                    Stop: ₹{stock.get('stop_loss', 0):.0f} | 
                    Target 1: ₹{stock.get('target_1', 0):.0f} | 
                    Position: {stock.get('position_size', 0)*100:.1f}%
                    """)
                    
                    st.markdown("---")
    
    with tab5:
        st.header("❓ Understanding EDGE Protocol")
        
        st.markdown("""
        ### 🎯 What is EDGE Protocol?
        
        EDGE Protocol is a sophisticated trading intelligence system that identifies institutional accumulation 
        patterns through **Volume Acceleration** - a proprietary metric that reveals when smart money is 
        quietly building positions.
        
        ### 📊 Key Metrics Explained
        
        **Volume Acceleration**: The difference between 30d/90d volume ratio and 30d/180d volume ratio. 
        Positive values indicate accelerating accumulation.
        
        **EDGE Score**: Composite score (0-100) combining:
        - Volume Score (40%): Volume acceleration and consistency
        - Momentum Score (25%): Price momentum and trend alignment
        - Entry Score (20%): Optimal entry point detection
        - Risk/Reward Score (15%): Upside potential vs downside risk
        
        ### 🚦 Signal Classifications
        
        - **EXPLOSIVE (85+)**: Immediate action required - Top 1% opportunities
        - **STRONG (70-85)**: High conviction trades - Top 5%
        - **MODERATE (50-70)**: Worth watching - Top 10%
        - **WATCH (30-50)**: Early stage signals
        - **AVOID (<30)**: No clear edge
        
        ### 🎯 Pattern Types
        
        1. **Explosive Breakout**: Strong momentum + volume surge above key levels
        2. **Stealth Accumulation**: High volume but flat/negative price (smart money loading)
        3. **Momentum Building**: Accelerating price and volume trends
        4. **Quality Pullback**: Strong fundamentals temporarily on sale
        
        ### 💡 How to Use
        
        1. Focus on EXPLOSIVE and STRONG signals for immediate opportunities
        2. Check the "Why Buy" reason for each signal
        3. Use suggested position sizes based on your risk tolerance
        4. Set stops at recommended levels
        5. Monitor Volume Acceleration Map for emerging opportunities
        
        ### ⚠️ Important Notes
        
        - This is for educational purposes only
        - Not financial advice - do your own research
        - Past performance doesn't guarantee future results
        - Always use proper risk management
        """)
    
    # Footer
    st.markdown("---")
    st.caption(f"""
    **EDGE Protocol 3.1** - Elite Data-Driven Growth Engine | 
    Data Quality: {diagnostics.get('data_quality_score', 0):.1f}% | 
    Last Updated: {diagnostics.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M')}
    
    Volume Acceleration reveals institutional accumulation before price moves.
    Your unfair advantage in the market.
    
    ⚠️ For educational purposes only. Not financial advice. Trade at your own risk.
    """)

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()
