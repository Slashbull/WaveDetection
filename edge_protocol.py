#!/usr/bin/env python3
"""
EDGE Protocol - Elite Data-Driven Growth Engine
==============================================
Production Version 3.0 - The Ultimate Implementation

A sophisticated trading intelligence system that reveals institutional 
accumulation through volume acceleration patterns.

Key Innovation: Volume Acceleration Detection
- Compares 30d/90d ratio vs 30d/180d ratio
- Reveals when institutions are ACCELERATING their accumulation
- Your unfair advantage in the market

Author: EDGE Protocol Team
Version: 3.0.0 FINAL
Last Updated: 2024
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
# DATA LOADING & PARSING - INDIAN FORMAT AWARE
# ============================================================================

def parse_indian_number(value: Union[str, float, int]) -> float:
    """
    Parse Indian number format to float
    Handles: ₹1,23,45,678.90 | 1,23,45,678 | -12.34% | 5.5Cr | 2.3L
    """
    if pd.isna(value) or value == '' or value == '-' or value == 'NA':
        return np.nan
    
    # If already numeric, return as is
    if isinstance(value, (int, float)):
        return float(value)
    
    # Convert to string and clean
    val_str = str(value).strip()
    
    # Remove currency symbols
    val_str = val_str.replace('₹', '').replace('$', '').replace('Rs', '')
    
    # Handle percentage
    is_percentage = '%' in val_str
    val_str = val_str.replace('%', '')
    
    # Handle negative signs (including unicode minus)
    val_str = val_str.replace('−', '-')
    
    # Handle Cr (Crores) and L (Lakhs)
    multiplier = 1
    if val_str.upper().endswith('CR'):
        multiplier = 10000000  # 1 Crore
        val_str = val_str[:-2]
    elif val_str.upper().endswith('L'):
        multiplier = 100000    # 1 Lakh
        val_str = val_str[:-1]
    
    # Remove commas (works for both Indian and Western formats)
    val_str = val_str.replace(',', '')
    
    # Clean any remaining spaces
    val_str = val_str.strip()
    
    try:
        result = float(val_str) * multiplier
        return result
    except (ValueError, TypeError):
        return np.nan

@st.cache_data(ttl=SHEET_CONFIG['CACHE_TTL'])
def load_data() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load and parse data with Indian format awareness
    Returns: (DataFrame, diagnostics)
    """
    diagnostics = {
        'timestamp': datetime.now(),
        'rows_loaded': 0,
        'rows_valid': 0,
        'data_quality_score': 0,
        'warnings': [],
        'parsing_stats': {}
    }
    
    try:
        # Build URL
        url = f"https://docs.google.com/spreadsheets/d/{SHEET_CONFIG['SHEET_ID']}/export?format=csv&gid={SHEET_CONFIG['GID']}"
        
        # Fetch data
        response = requests.get(url, timeout=SHEET_CONFIG['REQUEST_TIMEOUT'])
        response.raise_for_status()
        
        # Check for HTML response (access denied)
        if 'text/html' in response.headers.get('content-type', ''):
            raise ValueError("Access denied. Please make the Google Sheet public: Share → Anyone with link can view")
        
        # Load CSV
        df = pd.read_csv(io.StringIO(response.text))
        diagnostics['rows_loaded'] = len(df)
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Parse numeric columns with Indian format support
        df = parse_all_columns(df, diagnostics)
        
        # Gentle validation - keep as much data as possible
        df = validate_essential_data(df, diagnostics)
        
        # Calculate derived columns
        df = calculate_derived_columns(df)
        
        # Calculate data quality score
        essential_cols = ['ticker', 'price', 'volume_1d', 'ret_7d']
        available_cols = [col for col in essential_cols if col in df.columns]
        valid_cells = sum(df[col].notna().sum() for col in available_cols)
        total_cells = len(df) * len(available_cols)
        diagnostics['data_quality_score'] = (valid_cells / total_cells * 100) if total_cells > 0 else 0
        
        return df, diagnostics
        
    except Exception as e:
        logger.error(f"Data loading error: {str(e)}")
        diagnostics['warnings'].append(str(e))
        return pd.DataFrame(), diagnostics

def parse_all_columns(df: pd.DataFrame, diagnostics: Dict) -> pd.DataFrame:
    """Parse all columns with Indian format awareness"""
    
    parsing_stats = {
        'successful': 0,
        'failed': 0,
        'columns_parsed': []
    }
    
    # Define column types
    price_cols = ['price', 'low_52w', 'high_52w', 'sma_20d', 'sma_50d', 'sma_200d', 'prev_close']
    volume_cols = ['volume_1d', 'volume_7d', 'volume_30d', 'volume_90d', 'volume_180d']
    return_cols = ['ret_1d', 'ret_3d', 'ret_7d', 'ret_30d', 'ret_3m', 'ret_6m', 'ret_1y', 'ret_3y', 'ret_5y']
    ratio_cols = ['vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d',
                  'vol_ratio_1d_180d', 'vol_ratio_7d_180d', 'vol_ratio_30d_180d', 'vol_ratio_90d_180d']
    fundamental_cols = ['pe', 'eps_current', 'eps_last_qtr', 'eps_change_pct']
    other_numeric = ['from_low_pct', 'from_high_pct', 'rvol']
    
    # Parse each group
    all_numeric_cols = price_cols + volume_cols + return_cols + ratio_cols + fundamental_cols + other_numeric
    
    for col in all_numeric_cols:
        if col in df.columns:
            try:
                df[col] = df[col].apply(parse_indian_number)
                parsing_stats['successful'] += 1
                parsing_stats['columns_parsed'].append(col)
            except Exception as e:
                logger.warning(f"Failed to parse column {col}: {str(e)}")
                parsing_stats['failed'] += 1
    
    # Special handling for market_cap
    if 'market_cap' in df.columns:
        df['market_cap_value'] = df['market_cap'].apply(parse_market_cap_special)
    
    diagnostics['parsing_stats'] = parsing_stats
    return df

def parse_market_cap_special(value: str) -> float:
    """Special parser for market cap with Cr notation"""
    if pd.isna(value) or value == '' or value == '-':
        return np.nan
    
    val_str = str(value).strip()
    
    # Remove currency symbol
    val_str = val_str.replace('₹', '').replace(',', '')
    
    # Check for Cr at the end
    if val_str.upper().endswith('CR'):
        val_str = val_str[:-2].strip()
        try:
            return float(val_str)
        except:
            return np.nan
    
    # Try parsing as regular number
    return parse_indian_number(val_str)

def validate_essential_data(df: pd.DataFrame, diagnostics: Dict) -> pd.DataFrame:
    """Gentle validation - keep as much data as possible"""
    
    initial_count = len(df)
    
    # Only remove rows with no ticker or invalid price
    if 'ticker' in df.columns:
        df = df[df['ticker'].notna()]
    
    if 'price' in df.columns:
        df = df[(df['price'] > 0) | df['price'].isna()]  # Keep nulls, only remove zero/negative
    
    removed = initial_count - len(df)
    diagnostics['rows_valid'] = len(df)
    
    if removed > 0:
        diagnostics['warnings'].append(f"Removed {removed} rows with invalid ticker or price")
    
    return df

def calculate_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate essential derived columns"""
    
    # Volume Acceleration - THE SECRET WEAPON
    if all(col in df.columns for col in ['vol_ratio_30d_90d', 'vol_ratio_30d_180d']):
        df['volume_acceleration'] = df['vol_ratio_30d_90d'] - df['vol_ratio_30d_180d']
        df['has_volume_acceleration'] = True
    else:
        df['volume_acceleration'] = 0
        df['has_volume_acceleration'] = False
        logger.warning("Volume acceleration data not available - using fallback scoring")
    
    # Price position
    if all(col in df.columns for col in ['price', 'low_52w', 'high_52w']):
        price_range = df['high_52w'] - df['low_52w']
        df['price_position'] = ((df['price'] - df['low_52w']) / price_range * 100).fillna(50)
    
    # Simple volatility
    if all(col in df.columns for col in ['high_52w', 'low_52w', 'price']):
        df['volatility_52w'] = ((df['high_52w'] - df['low_52w']) / df['price']).fillna(0.5)
    
    # Momentum sync
    if all(col in df.columns for col in ['ret_7d', 'ret_30d']):
        df['momentum_sync'] = (df['ret_7d'] > 0) & (df['ret_30d'] > 0)
    
    # Volume consistency
    vol_ratios = ['vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d']
    available_ratios = [col for col in vol_ratios if col in df.columns]
    if available_ratios:
        df['volume_consistency'] = (df[available_ratios] > 0).sum(axis=1) / len(available_ratios)
    
    # EPS Tier
    if 'eps_current' in df.columns:
        df['eps_tier'] = pd.cut(
            df['eps_current'],
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

# ============================================================================
# PATTERN DETECTION ENGINE
# ============================================================================

@cached_computation(ttl=120)
def detect_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Detect key trading patterns"""
    
    # Initialize pattern columns
    patterns = ['EXPLOSIVE_BREAKOUT', 'STEALTH_ACCUMULATION', 'MOMENTUM_BUILDING', 'QUALITY_PULLBACK']
    for pattern in patterns:
        df[f'pattern_{pattern}'] = False
    
    # Pattern 1: EXPLOSIVE BREAKOUT
    if all(col in df.columns for col in ['ret_7d', 'volume_acceleration', 'price', 'sma_50d']):
        mask = (
            (df['ret_7d'] > PATTERN_THRESHOLDS['EXPLOSIVE_BREAKOUT']['ret_7d_min']) &
            (df['volume_acceleration'] > PATTERN_THRESHOLDS['EXPLOSIVE_BREAKOUT']['volume_accel_min']) &
            (df['price'] > df['sma_50d'])
        )
        df.loc[mask, 'pattern_EXPLOSIVE_BREAKOUT'] = True
    
    # Pattern 2: STEALTH ACCUMULATION (Most Valuable!)
    if all(col in df.columns for col in ['ret_7d', 'volume_acceleration', 'from_high_pct']):
        mask = (
            (df['ret_7d'] < PATTERN_THRESHOLDS['STEALTH_ACCUMULATION']['ret_7d_max']) &
            (df['volume_acceleration'] > PATTERN_THRESHOLDS['STEALTH_ACCUMULATION']['volume_accel_min']) &
            (df['from_high_pct'].between(*PATTERN_THRESHOLDS['STEALTH_ACCUMULATION']['from_high_range']))
        )
        df.loc[mask, 'pattern_STEALTH_ACCUMULATION'] = True
    
    # Pattern 3: MOMENTUM BUILDING
    if all(col in df.columns for col in ['ret_7d', 'ret_30d', 'volume_acceleration']):
        mask = (
            (df['ret_7d'] > PATTERN_THRESHOLDS['MOMENTUM_BUILDING']['ret_7d_min']) &
            (df['ret_30d'] > PATTERN_THRESHOLDS['MOMENTUM_BUILDING']['ret_30d_min']) &
            (df['volume_acceleration'] > PATTERN_THRESHOLDS['MOMENTUM_BUILDING']['volume_accel_min'])
        )
        df.loc[mask, 'pattern_MOMENTUM_BUILDING'] = True
    
    # Pattern 4: QUALITY PULLBACK
    if all(col in df.columns for col in ['ret_1y', 'from_high_pct', 'eps_change_pct']):
        mask = (
            (df['ret_1y'] > PATTERN_THRESHOLDS['QUALITY_PULLBACK']['ret_1y_min']) &
            (df['from_high_pct'].between(*PATTERN_THRESHOLDS['QUALITY_PULLBACK']['from_high_range'])) &
            (df['eps_change_pct'] > PATTERN_THRESHOLDS['QUALITY_PULLBACK']['eps_change_min'])
        )
        df.loc[mask, 'pattern_QUALITY_PULLBACK'] = True
    
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
# EDGE SCORING ENGINE - THE CORE
# ============================================================================

@cached_computation(ttl=120)
def calculate_edge_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate EDGE scores with volume acceleration as the core"""
    
    # Initialize all score components
    df['score_volume'] = 50      # Start neutral
    df['score_momentum'] = 50
    df['score_entry'] = 50
    df['score_risk_reward'] = 50
    
    # 1. VOLUME SCORE (40% weight) - THE SECRET WEAPON
    if df['has_volume_acceleration'].any():
        # True volume acceleration available
        accel_percentile = df['volume_acceleration'].rank(pct=True) * 100
        
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
            df['score_volume'] = df['score_volume'] * (0.7 + 0.3 * df['volume_consistency'])
        
        # RVOL multiplier
        if 'rvol' in df.columns:
            rvol_multiplier = df['rvol'].clip(0.5, 3.0) / 2
            df['score_volume'] = (df['score_volume'] * rvol_multiplier).clip(0, 100)
    else:
        # Fallback to simple volume metrics
        if 'rvol' in df.columns:
            df['score_volume'] = (df['rvol'].clip(0, 5) * 20).fillna(50)
    
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
            trend_score = (
                (df['price'] > df['sma_50d']).astype(int) * 10 +
                (df['price'] > df['sma_200d']).astype(int) * 10
            )
            df['score_momentum'] = (df['score_momentum'] + trend_score).clip(0, 100)
    
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
    
    # 4. RISK/REWARD SCORE (15% weight)
    if all(col in df.columns for col in ['price', 'high_52w', 'low_52w']):
        upside = ((df['high_52w'] - df['price']) / df['price'] * 100).clip(0, 100)
        downside = ((df['price'] - df['low_52w']) / df['price'] * 100).clip(1, 100)
        
        df['risk_reward_ratio'] = (upside / downside).clip(0, 10)
        df['score_risk_reward'] = (df['risk_reward_ratio'] * 10).clip(0, 100)
    
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
    
    # Classify signals
    df['signal_strength'] = pd.cut(
        df['edge_score'],
        bins=[-np.inf, 30, 50, 70, 85, 100.1],
        labels=['AVOID', 'WATCH', 'MODERATE', 'STRONG', 'EXPLOSIVE']
    )
    
    # Volume pattern classification
    if 'volume_acceleration' in df.columns:
        df['volume_pattern'] = pd.cut(
            df['volume_acceleration'],
            bins=[-np.inf, -20, -10, 0, 10, 20, 30, np.inf],
            labels=['HEAVY_DIST', 'DISTRIBUTION', 'NEUTRAL', 'ACCUMULATION', 
                   'HEAVY_ACCUM', 'INSTITUTIONAL', 'EXPLOSIVE_VOL']
        )
    
    return df

# ============================================================================
# RISK MANAGEMENT ENGINE
# ============================================================================

def calculate_position_sizing(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate position sizes with sophisticated risk management"""
    
    # Base position size from signal strength
    df['base_position_size'] = df['signal_strength'].map({
        'EXPLOSIVE': POSITION_SIZES['EXPLOSIVE'],
        'STRONG': POSITION_SIZES['STRONG'],
        'MODERATE': POSITION_SIZES['MODERATE'],
        'WATCH': POSITION_SIZES['WATCH'],
        'AVOID': 0
    }).fillna(0)
    
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
        
        # Use the highest support that's below current price
        df['support_stop'] = pd.DataFrame({
            's1': support_50,
            's2': support_200,
            's3': support_52w
        }).max(axis=1)
        
        # Calculate percentage
        support_stop_pct = ((df['price'] - df['support_stop']) / df['price']).clip(0.05, 0.15)
        
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
    
    # Calculate base targets
    for i, suffix in enumerate(['1', '2', '3']):
        df[f'target_{suffix}'] = df.apply(
            lambda row: row['price'] * target_multipliers.get(
                row.get('signal_strength', 'MODERATE'), 
                (1.10, 1.20, 1.30)
            )[i],
            axis=1
        )
    
    # Adjust for resistance levels
    if 'high_52w' in df.columns:
        # Don't set targets above 52w high initially
        df['target_1'] = df[['target_1', 'high_52w']].min(axis=1)
    
    # Calculate risk/reward ratios
    if all(col in df.columns for col in ['target_1', 'price', 'stop_loss']):
        risk = df['price'] - df['stop_loss']
        reward = df['target_1'] - df['price']
        df['risk_reward_1'] = (reward / risk.replace(0, 0.01)).clip(0, 10)
    
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
        if df.loc[idx, 'signal_strength'] == 'AVOID':
            continue
            
        current_size = df.loc[idx, 'position_size']
        sector = df.loc[idx, 'sector'] if 'sector' in df.columns else 'Unknown'
        
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
        signal_df = plot_df[plot_df['signal_strength'] == signal]
        if not signal_df.empty:
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
                hovertemplate='<b>%{text}</b><br>' +
                             'Vol Accel: %{x:.1f}%<br>' +
                             '7d Return: %{y:.1f}%<br>' +
                             'EDGE Score: %{customdata:.1f}<br>' +
                             '<extra></extra>',
                customdata=signal_df['edge_score']
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
        signal_df = df[df['signal_strength'] == signal]
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
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="main-header">
        <h1>⚡ EDGE Protocol 3.0</h1>
        <p>Volume Acceleration Intelligence System</p>
    </div>
    """, unsafe_allow_html=True)

def render_key_metrics(df: pd.DataFrame, filtered_df: pd.DataFrame):
    """Render key performance metrics"""
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        explosive_count = len(filtered_df[filtered_df['signal_strength'] == 'EXPLOSIVE'])
        st.metric(
            "🚀 EXPLOSIVE",
            explosive_count,
            help="Immediate action required - Top 1%"
        )
    
    with col2:
        strong_count = len(filtered_df[filtered_df['signal_strength'] == 'STRONG'])
        st.metric(
            "💪 STRONG",
            strong_count,
            help="High conviction opportunities"
        )
    
    with col3:
        if 'volume_acceleration' in filtered_df.columns:
            avg_accel = filtered_df[filtered_df['signal_strength'].isin(['EXPLOSIVE', 'STRONG'])]['volume_acceleration'].mean()
            st.metric(
                "📊 Avg Vol Accel",
                f"{avg_accel:.1f}%" if not pd.isna(avg_accel) else "N/A",
                help="Average volume acceleration of top signals"
            )
        else:
            st.metric("📊 Vol Accel", "N/A")
    
    with col4:
        total_signals = len(filtered_df[filtered_df['signal_strength'] != 'AVOID'])
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
        'signal_strength', 'edge_score', 'volume_acceleration',
        'primary_pattern', 'price', 'position_size',
        'stop_loss', 'target_1', 'risk_reward_1'
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
    styled_df = display_df.style.format(format_dict)
    
    # Color code signal strength
    def color_signal(val):
        colors = {
            'EXPLOSIVE': 'background-color: #FF0000; color: white;',
            'STRONG': 'background-color: #FF6600; color: white;',
            'MODERATE': 'background-color: #FFA500; color: black;',
            'WATCH': 'background-color: #808080; color: white;'
        }
        return colors.get(val, '')
    
    if 'signal_strength' in available_columns:
        styled_df = styled_df.applymap(color_signal, subset=['signal_strength'])
    
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
        categories = sorted(df['category'].dropna().unique().tolist())
        filters['categories'] = st.sidebar.multiselect(
            "Market Cap Category",
            categories,
            default=[],
            help="Filter by market capitalization"
        )
    
    # Sector filter
    if 'sector' in df.columns:
        sectors = sorted(df['sector'].dropna().unique().tolist())
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
    if 'volume_acceleration' in df.columns:
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
        filtered = filtered[filtered['signal_strength'] != 'AVOID']
    
    return filtered

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point"""
    
    # Page configuration
    st.set_page_config(
        page_title="EDGE Protocol 3.0",
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
    
    # Display data quality metrics
    if diagnostics['data_quality_score'] < 50:
        st.warning(f"⚠️ Data quality is low ({diagnostics['data_quality_score']:.1f}%). Some features may be limited.")
    
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
    explosive_signals = filtered_df[filtered_df['signal_strength'] == 'EXPLOSIVE']
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
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Trading Signals",
        "🗺️ Volume Map",
        "📈 Analytics",
        "🎯 Top Picks"
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
            display_df = display_df[display_df['signal_strength'] == 'EXPLOSIVE']
        
        # Sort
        if sort_by in display_df.columns:
            display_df = display_df.sort_values(
                sort_by,
                ascending=(sort_order == 'Ascending')
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
        else:
            st.warning("Volume acceleration data not available")
    
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
            
            sector_stats = filtered_df.groupby('sector').agg({
                'edge_score': ['mean', 'count'],
                'volume_acceleration': 'mean' if 'volume_acceleration' in filtered_df.columns else lambda x: 0
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
        st.header("🎯 Top Picks")
        
        # Display top picks by category
        for signal_type in ['EXPLOSIVE', 'STRONG', 'MODERATE']:
            top_picks = filtered_df[
                filtered_df['signal_strength'] == signal_type
            ].head(5)
            
            if not top_picks.empty:
                st.markdown(f"### {signal_type} Signals")
                
                for _, stock in top_picks.iterrows():
                    # Create signal card
                    col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 1])
                    
                    with col1:
                        ticker = stock.get('ticker', 'N/A')
                        company = stock.get('company_name', 'Unknown')
                        pattern = stock.get('primary_pattern', 'NONE').replace('_', ' ')
                        
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4 style="margin:0">{ticker} - {company}</h4>
                            <p style="margin:0">Pattern: {pattern}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.metric("EDGE", f"{stock.get('edge_score', 0):.1f}")
                    
                    with col3:
                        st.metric("Price", f"₹{stock.get('price', 0):.0f}")
                    
                    with col4:
                        vol_accel = stock.get('volume_acceleration', 0)
                        st.metric("Vol Accel", f"{vol_accel:.0f}%")
                    
                    with col5:
                        rr = stock.get('risk_reward_1', 0)
                        st.metric("R/R", f"{rr:.1f}")
    
    # Footer
    st.markdown("---")
    st.caption("""
    **EDGE Protocol 3.0** - Elite Data-Driven Growth Engine
    
    Volume Acceleration reveals institutional accumulation before price moves.
    Your unfair advantage in the market.
    
    ⚠️ For educational purposes only. Not financial advice. Trade at your own risk.
    """)

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()
