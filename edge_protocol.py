#!/usr/bin/env python3
"""
EDGE Protocol - Elite Data-Driven Growth Engine
==============================================
Production Version 3.2 ULTIMATE - All Bugs Fixed

The most robust implementation that handles all data formats and edge cases.
Built from learnings across our entire conversation.

Key Innovation: Volume Acceleration Detection
- Compares 30d/90d ratio vs 30d/180d ratio
- Reveals when institutions are ACCELERATING their accumulation
- Your unfair advantage in the market

Author: EDGE Protocol Team
Version: 3.2.0 ULTIMATE FINAL
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
# ROBUST DATA PARSING - HANDLES ALL FORMATS
# ============================================================================

def universal_number_parser(value: Union[str, float, int]) -> float:
    """
    Ultimate number parser that handles ALL formats intelligently
    Based on patterns seen throughout our conversation
    """
    # Handle already numeric
    if isinstance(value, (int, float)):
        return float(value)
    
    # Handle null-like values
    if pd.isna(value) or value in ['', '-', 'NA', 'N/A', '#N/A', '#DIV/0!', 'null', 'None']:
        return np.nan
    
    # Convert to string for processing
    val_str = str(value).strip()
    
    # Handle empty after stripping
    if not val_str:
        return np.nan
    
    # Remove currency symbols (multiple possible formats)
    currency_symbols = ['₹', '$', 'Rs', 'Rs.', 'INR', '€', '£']
    for symbol in currency_symbols:
        val_str = val_str.replace(symbol, '')
    
    # Handle percentage (may or may not have % symbol)
    is_percentage = '%' in val_str
    val_str = val_str.replace('%', '')
    
    # Handle negative signs (including unicode variants)
    val_str = val_str.replace('−', '-').replace('–', '-')
    
    # Handle Indian notation (Cr, L, K, M, B)
    multiplier = 1
    multiplier_map = {
        'CR': 10000000,    # Crore
        'CRORE': 10000000,
        'CRORES': 10000000,
        'L': 100000,       # Lakh
        'LAC': 100000,
        'LAKH': 100000,
        'LAKHS': 100000,
        'K': 1000,         # Thousand
        'M': 1000000,      # Million
        'B': 1000000000,   # Billion
        'T': 1000000000000 # Trillion
    }
    
    # Check for multiplier at the end
    val_upper = val_str.upper()
    for suffix, mult in multiplier_map.items():
        if val_upper.endswith(suffix):
            multiplier = mult
            # Remove the suffix
            val_str = val_str[:-(len(suffix))]
            break
    
    # Remove all types of commas (Indian and Western)
    # Indian: 1,23,45,678  Western: 1,234,567
    val_str = val_str.replace(',', '')
    
    # Remove any remaining spaces
    val_str = val_str.strip()
    
    # Handle parentheses for negative numbers (accounting format)
    if val_str.startswith('(') and val_str.endswith(')'):
        val_str = '-' + val_str[1:-1]
    
    # Try to convert
    try:
        result = float(val_str) * multiplier
        
        # Sanity checks
        if abs(result) > 1e15:  # Extremely large number
            logger.warning(f"Extremely large number parsed: {result} from {value}")
        
        return result
    except (ValueError, TypeError) as e:
        logger.debug(f"Failed to parse '{value}': {str(e)}")
        return np.nan

@st.cache_data(ttl=SHEET_CONFIG['CACHE_TTL'])
def load_data() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load and parse data with maximum robustness
    """
    diagnostics = {
        'timestamp': datetime.now(),
        'rows_loaded': 0,
        'rows_valid': 0,
        'data_quality_score': 0,
        'warnings': [],
        'parsing_stats': {},
        'debug_info': {}
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
        
        # Clean column names (preserve original for debugging)
        original_columns = df.columns.tolist()
        df.columns = df.columns.str.strip()
        
        # Store debug info
        diagnostics['debug_info']['original_columns'] = original_columns
        diagnostics['debug_info']['dtypes_before'] = df.dtypes.to_dict()
        
        # CRITICAL: Parse numeric columns BEFORE any validation
        df = parse_all_columns_robust(df, diagnostics)
        
        # Store debug info after parsing
        diagnostics['debug_info']['dtypes_after'] = df.dtypes.to_dict()
        
        # NOW do validation with numeric data
        df = validate_data_lenient(df, diagnostics)
        
        # Calculate derived columns
        df = calculate_derived_columns_safe(df, diagnostics)
        
        # Filter obvious dead stocks
        df = filter_dead_stocks_smart(df, diagnostics)
        
        # Calculate data quality score
        df, quality_score = assess_data_quality(df, diagnostics)
        diagnostics['data_quality_score'] = quality_score
        
        return df, diagnostics
        
    except Exception as e:
        logger.error(f"Data loading error: {str(e)}")
        diagnostics['warnings'].append(str(e))
        return pd.DataFrame(), diagnostics

def parse_all_columns_robust(df: pd.DataFrame, diagnostics: Dict) -> pd.DataFrame:
    """
    Parse all columns with maximum robustness
    """
    parsing_stats = {
        'successful': 0,
        'failed': 0,
        'columns_parsed': [],
        'sample_values': {}
    }
    
    # Define column groups
    numeric_columns = {
        'prices': ['price', 'low_52w', 'high_52w', 'sma_20d', 'sma_50d', 'sma_200d', 'prev_close'],
        'volumes': ['volume_1d', 'volume_7d', 'volume_30d', 'volume_3m', 'volume_90d', 'volume_180d'],
        'returns': ['ret_1d', 'ret_3d', 'ret_7d', 'ret_30d', 'ret_3m', 'ret_6m', 'ret_1y', 'ret_3y', 'ret_5y'],
        'ratios': ['vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d',
                   'vol_ratio_1d_180d', 'vol_ratio_7d_180d', 'vol_ratio_30d_180d', 'vol_ratio_90d_180d'],
        'fundamentals': ['pe', 'eps_current', 'eps_last_qtr', 'eps_change_pct', 'eps_duplicate'],
        'percentages': ['from_low_pct', 'from_high_pct'],
        'other': ['rvol']
    }
    
    # Flatten all numeric columns
    all_numeric_cols = []
    for group, cols in numeric_columns.items():
        all_numeric_cols.extend(cols)
    
    # Parse each column
    for col in all_numeric_cols:
        if col in df.columns:
            try:
                # Store sample values for debugging
                sample_values = df[col].head(5).tolist()
                parsing_stats['sample_values'][col] = sample_values
                
                # Apply universal parser
                df[col] = df[col].apply(universal_number_parser)
                
                # Count non-null values
                non_null_count = df[col].notna().sum()
                if non_null_count > 0:
                    parsing_stats['successful'] += 1
                    parsing_stats['columns_parsed'].append(col)
                else:
                    parsing_stats['failed'] += 1
                    logger.warning(f"Column {col} has no valid values after parsing")
                    
            except Exception as e:
                parsing_stats['failed'] += 1
                logger.error(f"Failed to parse column {col}: {str(e)}")
    
    # Special handling for market_cap
    if 'market_cap' in df.columns:
        df['market_cap_value'] = df['market_cap'].apply(universal_number_parser)
    
    diagnostics['parsing_stats'] = parsing_stats
    return df

def validate_data_lenient(df: pd.DataFrame, diagnostics: Dict) -> pd.DataFrame:
    """
    Lenient validation - keep as much data as possible
    """
    initial_count = len(df)
    
    # Only remove rows with critical issues
    if 'ticker' in df.columns:
        # Remove rows with no ticker
        df = df[df['ticker'].notna()]
        df = df[df['ticker'].str.strip() != '']
    
    if 'price' in df.columns:
        # Only remove rows with invalid price (keep NaN for now)
        df = df[(df['price'] > 0) | df['price'].isna()]
    
    removed = initial_count - len(df)
    diagnostics['rows_valid'] = len(df)
    
    if removed > 0:
        diagnostics['warnings'].append(f"Removed {removed} rows with critical issues")
    
    # Fill NaN in price with a method that preserves data
    if 'price' in df.columns and 'prev_close' in df.columns:
        df['price'] = df['price'].fillna(df['prev_close'])
    
    return df

def calculate_derived_columns_safe(df: pd.DataFrame, diagnostics: Dict) -> pd.DataFrame:
    """
    Calculate derived columns with safety checks
    """
    try:
        # Volume Acceleration - THE SECRET WEAPON
        if all(col in df.columns for col in ['vol_ratio_30d_90d', 'vol_ratio_30d_180d']):
            # Ensure numeric
            df['vol_ratio_30d_90d'] = pd.to_numeric(df['vol_ratio_30d_90d'], errors='coerce')
            df['vol_ratio_30d_180d'] = pd.to_numeric(df['vol_ratio_30d_180d'], errors='coerce')
            
            # Calculate acceleration
            df['volume_acceleration'] = df['vol_ratio_30d_90d'] - df['vol_ratio_30d_180d']
            
            # Count valid calculations
            valid_accel = df['volume_acceleration'].notna().sum()
            diagnostics['debug_info']['volume_acceleration_valid'] = valid_accel
            
            df['has_volume_acceleration'] = df['volume_acceleration'].notna()
        else:
            df['volume_acceleration'] = 0
            df['has_volume_acceleration'] = False
            logger.warning("Volume acceleration columns not available")
    except Exception as e:
        logger.error(f"Error calculating volume acceleration: {str(e)}")
        df['volume_acceleration'] = 0
        df['has_volume_acceleration'] = False
    
    # Price position (safe calculation)
    try:
        if all(col in df.columns for col in ['price', 'low_52w', 'high_52w']):
            price_range = df['high_52w'] - df['low_52w']
            # Avoid division by zero
            price_range = price_range.replace(0, np.nan)
            df['price_position'] = ((df['price'] - df['low_52w']) / price_range * 100).fillna(50)
    except:
        df['price_position'] = 50
    
    # Simple volatility (safe)
    try:
        if all(col in df.columns for col in ['high_52w', 'low_52w', 'price']):
            # Avoid division by zero
            safe_price = df['price'].replace(0, np.nan)
            df['volatility_52w'] = ((df['high_52w'] - df['low_52w']) / safe_price).fillna(0.5).clip(0, 2)
    except:
        df['volatility_52w'] = 0.5
    
    # Momentum sync
    try:
        if all(col in df.columns for col in ['ret_7d', 'ret_30d']):
            df['momentum_sync'] = (
                (df['ret_7d'] > 0) & 
                (df['ret_30d'] > 0)
            ).astype(int)
    except:
        df['momentum_sync'] = 0
    
    # Volume consistency
    try:
        vol_ratios = ['vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d']
        available_ratios = [col for col in vol_ratios if col in df.columns]
        if available_ratios:
            df['volume_consistency'] = (df[available_ratios] > 0).sum(axis=1) / len(available_ratios)
        else:
            df['volume_consistency'] = 0.5
    except:
        df['volume_consistency'] = 0.5
    
    # Safe tier calculations
    df = calculate_tiers_safe(df)
    
    return df

def calculate_tiers_safe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate tiers safely with proper error handling
    """
    # EPS Tier
    if 'eps_current' in df.columns:
        try:
            # Ensure numeric
            eps_values = pd.to_numeric(df['eps_current'], errors='coerce')
            
            # Define bins with explicit handling of infinity
            bins = [-1000000, 5, 15, 35, 55, 75, 95, 1000000]
            labels = ['5↓', '5↑', '15↑', '35↑', '55↑', '75↑', '95↑']
            
            df['eps_tier'] = pd.cut(eps_values, bins=bins, labels=labels, include_lowest=True)
            
            # Fill NaN with default
            df['eps_tier'] = df['eps_tier'].cat.add_categories(['Unknown']).fillna('Unknown')
        except Exception as e:
            logger.error(f"Error calculating EPS tier: {str(e)}")
            df['eps_tier'] = 'Unknown'
    else:
        df['eps_tier'] = 'Unknown'
    
    # Price Tier  
    if 'price' in df.columns:
        try:
            # Ensure numeric
            price_values = pd.to_numeric(df['price'], errors='coerce')
            
            # Define bins
            bins = [0, 100, 200, 500, 1000, 2000, 5000, 1000000]
            labels = ['100↓', '100↑', '200↑', '500↑', '1K↑', '2K↑', '5K↑']
            
            df['price_tier'] = pd.cut(price_values, bins=bins, labels=labels, include_lowest=True)
            
            # Fill NaN with default
            df['price_tier'] = df['price_tier'].cat.add_categories(['Unknown']).fillna('Unknown')
        except Exception as e:
            logger.error(f"Error calculating price tier: {str(e)}")
            df['price_tier'] = 'Unknown'
    else:
        df['price_tier'] = 'Unknown'
    
    return df

def filter_dead_stocks_smart(df: pd.DataFrame, diagnostics: Dict) -> pd.DataFrame:
    """
    Smart filtering of dead stocks - not too aggressive
    """
    initial_count = len(df)
    
    try:
        # Only filter truly dead stocks
        dead_criteria = []
        
        # Extreme negative performance across multiple timeframes
        if all(col in df.columns for col in ['ret_7d', 'ret_30d', 'ret_3m']):
            extreme_negative = (
                (df['ret_7d'] < -20) & 
                (df['ret_30d'] < -30) & 
                (df['ret_3m'] < -40)
            )
            dead_criteria.append(extreme_negative)
        
        # No volume for extended period
        if 'volume_1d' in df.columns and 'rvol' in df.columns:
            no_volume = (df['volume_1d'] == 0) | ((df['volume_1d'] < 1000) & (df['rvol'] < 0.1))
            dead_criteria.append(no_volume)
        
        # Apply filters if criteria exist
        if dead_criteria:
            is_dead = pd.concat(dead_criteria, axis=1).any(axis=1)
            df = df[~is_dead]
            
            removed = initial_count - len(df)
            if removed > 0:
                diagnostics['warnings'].append(f"Filtered {removed} dead/zombie stocks")
    except Exception as e:
        logger.error(f"Error in dead stock filter: {str(e)}")
        # If filtering fails, return original data
        return df
    
    return df

def assess_data_quality(df: pd.DataFrame, diagnostics: Dict) -> Tuple[pd.DataFrame, float]:
    """
    Assess data quality and provide score
    """
    quality_metrics = {}
    
    # Essential columns for EDGE calculation
    essential_cols = {
        'price': 0.20,
        'volume_1d': 0.10,
        'ret_7d': 0.15,
        'ret_30d': 0.10,
        'vol_ratio_30d_90d': 0.20,
        'vol_ratio_30d_180d': 0.15,
        'rvol': 0.10
    }
    
    total_score = 0
    
    for col, weight in essential_cols.items():
        if col in df.columns:
            # Calculate percentage of non-null values
            non_null_pct = df[col].notna().sum() / len(df)
            quality_metrics[col] = non_null_pct
            total_score += non_null_pct * weight
        else:
            quality_metrics[col] = 0
    
    # Store in diagnostics
    diagnostics['debug_info']['quality_metrics'] = quality_metrics
    
    return df, total_score * 100

# ============================================================================
# PATTERN DETECTION ENGINE - ROBUST VERSION
# ============================================================================

def detect_patterns_safe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect patterns with safety checks
    """
    # Initialize pattern columns
    patterns = ['EXPLOSIVE_BREAKOUT', 'STEALTH_ACCUMULATION', 'MOMENTUM_BUILDING', 'QUALITY_PULLBACK']
    for pattern in patterns:
        df[f'pattern_{pattern}'] = False
    
    try:
        # Pattern 1: EXPLOSIVE BREAKOUT
        if all(col in df.columns for col in ['ret_7d', 'volume_acceleration', 'price', 'sma_50d']):
            mask = (
                (df['ret_7d'] > PATTERN_THRESHOLDS['EXPLOSIVE_BREAKOUT']['ret_7d_min']) &
                (df['volume_acceleration'] > PATTERN_THRESHOLDS['EXPLOSIVE_BREAKOUT']['volume_accel_min']) &
                (df['price'] > df['sma_50d'])
            )
            df.loc[mask, 'pattern_EXPLOSIVE_BREAKOUT'] = True
    except Exception as e:
        logger.debug(f"Failed to detect EXPLOSIVE_BREAKOUT: {str(e)}")
    
    try:
        # Pattern 2: STEALTH ACCUMULATION
        if all(col in df.columns for col in ['ret_7d', 'volume_acceleration', 'from_high_pct']):
            mask = (
                (df['ret_7d'] < PATTERN_THRESHOLDS['STEALTH_ACCUMULATION']['ret_7d_max']) &
                (df['volume_acceleration'] > PATTERN_THRESHOLDS['STEALTH_ACCUMULATION']['volume_accel_min']) &
                (df['from_high_pct'].between(*PATTERN_THRESHOLDS['STEALTH_ACCUMULATION']['from_high_range']))
            )
            df.loc[mask, 'pattern_STEALTH_ACCUMULATION'] = True
    except Exception as e:
        logger.debug(f"Failed to detect STEALTH_ACCUMULATION: {str(e)}")
    
    try:
        # Pattern 3: MOMENTUM BUILDING
        if all(col in df.columns for col in ['ret_7d', 'ret_30d', 'volume_acceleration']):
            mask = (
                (df['ret_7d'] > PATTERN_THRESHOLDS['MOMENTUM_BUILDING']['ret_7d_min']) &
                (df['ret_30d'] > PATTERN_THRESHOLDS['MOMENTUM_BUILDING']['ret_30d_min']) &
                (df['volume_acceleration'] > PATTERN_THRESHOLDS['MOMENTUM_BUILDING']['volume_accel_min'])
            )
            df.loc[mask, 'pattern_MOMENTUM_BUILDING'] = True
    except Exception as e:
        logger.debug(f"Failed to detect MOMENTUM_BUILDING: {str(e)}")
    
    try:
        # Pattern 4: QUALITY PULLBACK
        if all(col in df.columns for col in ['ret_1y', 'from_high_pct', 'eps_change_pct']):
            mask = (
                (df['ret_1y'] > PATTERN_THRESHOLDS['QUALITY_PULLBACK']['ret_1y_min']) &
                (df['from_high_pct'].between(*PATTERN_THRESHOLDS['QUALITY_PULLBACK']['from_high_range'])) &
                (df['eps_change_pct'] > PATTERN_THRESHOLDS['QUALITY_PULLBACK']['eps_change_min'])
            )
            df.loc[mask, 'pattern_QUALITY_PULLBACK'] = True
    except Exception as e:
        logger.debug(f"Failed to detect QUALITY_PULLBACK: {str(e)}")
    
    # Calculate pattern score safely
    pattern_cols = [col for col in df.columns if col.startswith('pattern_')]
    if pattern_cols:
        df['pattern_count'] = df[pattern_cols].sum(axis=1)
    else:
        df['pattern_count'] = 0
    
    # Assign primary pattern
    df['primary_pattern'] = 'NONE'
    for pattern in patterns:
        pattern_col = f'pattern_{pattern}'
        if pattern_col in df.columns:
            mask = df[pattern_col] == True
            df.loc[mask, 'primary_pattern'] = pattern
    
    return df

# ============================================================================
# EDGE SCORING ENGINE - BULLETPROOF VERSION
# ============================================================================

def calculate_edge_scores_robust(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate EDGE scores with maximum robustness
    """
    # Initialize all components with safe defaults
    df['score_volume'] = 30       # Start low, not neutral
    df['score_momentum'] = 30
    df['score_entry'] = 30
    df['score_risk_reward'] = 30
    
    # 1. VOLUME SCORE (40% weight) - With fallbacks
    try:
        if 'volume_acceleration' in df.columns and df['volume_acceleration'].notna().any():
            # Primary: Use true acceleration
            accel_values = df['volume_acceleration'].fillna(0)
            
            # Non-linear scoring
            df['score_volume'] = np.where(
                accel_values > 30, 100,
                np.where(accel_values > 20, 85,
                np.where(accel_values > 10, 70,
                np.where(accel_values > 0, 55,
                np.where(accel_values > -10, 40, 20))))
            )
            
            # Boost for RVOL if available
            if 'rvol' in df.columns:
                rvol_boost = df['rvol'].fillna(1).clip(0.5, 3) / 3 * 20
                df['score_volume'] = (df['score_volume'] + rvol_boost).clip(0, 100)
        else:
            # Fallback: Use simple volume metrics
            if 'rvol' in df.columns:
                df['score_volume'] = (df['rvol'].fillna(1).clip(0, 5) * 15 + 25).clip(0, 100)
            
            if 'vol_ratio_30d_90d' in df.columns:
                vol_ratio_score = df['vol_ratio_30d_90d'].fillna(0).clip(-50, 100) / 2 + 50
                df['score_volume'] = (df['score_volume'] * 0.5 + vol_ratio_score * 0.5)
    except Exception as e:
        logger.error(f"Error calculating volume score: {str(e)}")
        df['score_volume'] = 40  # Default
    
    # 2. MOMENTUM SCORE (25% weight) - Safe calculation
    try:
        if all(col in df.columns for col in ['ret_7d', 'ret_30d']):
            short_momentum = df['ret_7d'].fillna(0)
            medium_momentum = df['ret_30d'].fillna(0)
            
            # Momentum alignment scoring
            df['score_momentum'] = np.where(
                (short_momentum > 5) & (medium_momentum > 10), 85,
                np.where((short_momentum > 0) & (medium_momentum > 0), 65,
                np.where(short_momentum > -5, 45, 25))
            )
            
            # Trend bonus if available
            if all(col in df.columns for col in ['price', 'sma_50d']):
                try:
                    trend_bonus = ((df['price'] > df['sma_50d']) * 15).fillna(0)
                    df['score_momentum'] = (df['score_momentum'] + trend_bonus).clip(0, 100)
                except:
                    pass
        else:
            # Fallback to simple momentum
            if 'ret_7d' in df.columns:
                df['score_momentum'] = (df['ret_7d'].fillna(0).clip(-20, 20) * 2 + 50).clip(0, 100)
    except Exception as e:
        logger.error(f"Error calculating momentum score: {str(e)}")
        df['score_momentum'] = 40
    
    # 3. ENTRY SCORE (20% weight) - Smart entry points
    try:
        if 'from_high_pct' in df.columns:
            from_high = df['from_high_pct'].fillna(-10)
            
            # Optimal entry zones
            df['score_entry'] = np.where(
                from_high.between(-25, -15), 90,
                np.where(from_high.between(-35, -10), 70,
                np.where(from_high > -5, 30,
                np.where(from_high < -50, 25, 50)))
            )
        else:
            df['score_entry'] = 50  # Neutral if no data
    except Exception as e:
        logger.error(f"Error calculating entry score: {str(e)}")
        df['score_entry'] = 50
    
    # 4. RISK/REWARD SCORE (15% weight)
    try:
        if all(col in df.columns for col in ['price', 'high_52w', 'low_52w']):
            # Safe calculation avoiding division by zero
            price_safe = df['price'].replace(0, np.nan).fillna(1)
            upside = ((df['high_52w'] - df['price']) / price_safe * 100).fillna(0).clip(0, 100)
            downside = ((df['price'] - df['low_52w']) / price_safe * 100).fillna(50).clip(1, 100)
            
            df['risk_reward_ratio'] = (upside / downside).fillna(1).clip(0, 10)
            df['score_risk_reward'] = (df['risk_reward_ratio'] * 10).clip(0, 100)
        else:
            df['score_risk_reward'] = 50
            df['risk_reward_ratio'] = 2.0
    except Exception as e:
        logger.error(f"Error calculating risk/reward score: {str(e)}")
        df['score_risk_reward'] = 50
    
    # FINAL EDGE SCORE CALCULATION
    df['edge_score'] = (
        df['score_volume'] * 0.40 +
        df['score_momentum'] * 0.25 +
        df['score_entry'] * 0.20 +
        df['score_risk_reward'] * 0.15
    )
    
    # Pattern bonus if available
    if 'pattern_count' in df.columns:
        pattern_bonus = df['pattern_count'] * 3
        df['edge_score'] = (df['edge_score'] + pattern_bonus).clip(0, 100)
    
    # Add noise to break ties
    df['edge_score'] = df['edge_score'] + np.random.uniform(-0.1, 0.1, size=len(df))
    
    # Ensure we have reasonable distribution
    # If all scores are too low, apply curve adjustment
    if df['edge_score'].max() < 60:
        # Apply sqrt curve to spread scores
        df['edge_score'] = (np.sqrt(df['edge_score'] / 100) * 100).clip(0, 100)
    
    # Classify signals - ensure proper handling
    try:
        df['signal_strength'] = pd.cut(
            df['edge_score'],
            bins=[-0.01, 30, 50, 70, 85, 100.01],
            labels=['AVOID', 'WATCH', 'MODERATE', 'STRONG', 'EXPLOSIVE'],
            include_lowest=True
        )
    except Exception as e:
        logger.error(f"Error in signal classification: {str(e)}")
        # Fallback classification
        df['signal_strength'] = 'MODERATE'
    
    # Volume pattern classification if possible
    if 'volume_acceleration' in df.columns:
        try:
            df['volume_pattern'] = pd.cut(
                df['volume_acceleration'].fillna(0),
                bins=[-1000, -20, -10, 0, 10, 20, 30, 1000],
                labels=['HEAVY_DIST', 'DISTRIBUTION', 'NEUTRAL', 'ACCUMULATION', 
                       'HEAVY_ACCUM', 'INSTITUTIONAL', 'EXPLOSIVE_VOL']
            )
        except:
            df['volume_pattern'] = 'NEUTRAL'
    
    # Generate signal reasons
    df['signal_reason'] = df.apply(generate_signal_reason_safe, axis=1)
    
    return df

def generate_signal_reason_safe(row) -> str:
    """
    Generate explanation with error handling
    """
    try:
        reasons = []
        
        # Volume signals
        if 'volume_acceleration' in row.index and not pd.isna(row.get('volume_acceleration')):
            vol_accel = row['volume_acceleration']
            if vol_accel > 30:
                reasons.append("🏦 Institutional loading detected")
            elif vol_accel > 20:
                reasons.append("📊 Heavy accumulation")
            elif vol_accel > 10:
                reasons.append("📈 Volume picking up")
        
        # Pattern signals
        primary_pattern = row.get('primary_pattern', 'NONE')
        if primary_pattern != 'NONE':
            pattern_map = {
                'EXPLOSIVE_BREAKOUT': "🚀 Explosive breakout",
                'STEALTH_ACCUMULATION': "🕵️ Smart money accumulating",
                'MOMENTUM_BUILDING': "⚡ Momentum accelerating",
                'QUALITY_PULLBACK': "💎 Quality stock on sale"
            }
            if primary_pattern in pattern_map:
                reasons.append(pattern_map[primary_pattern])
        
        # Price action
        if 'ret_7d' in row.index and row.get('ret_7d', 0) > 10:
            reasons.append("🔥 Strong momentum")
        
        # Risk/Reward
        if 'risk_reward_ratio' in row.index and row.get('risk_reward_ratio', 0) > 3:
            reasons.append("💰 Excellent R/R")
        
        # Default reason based on score
        if not reasons:
            edge_score = row.get('edge_score', 0)
            if edge_score > 85:
                reasons.append("⚡ Multiple bullish signals")
            elif edge_score > 70:
                reasons.append("✅ Strong technical setup")
            elif edge_score > 50:
                reasons.append("📊 Positive indicators")
            else:
                reasons.append("👀 Early stage signal")
        
        return " | ".join(reasons[:2])
    except:
        return "📊 Technical signal detected"

# ============================================================================
# RISK MANAGEMENT ENGINE - FAILSAFE VERSION
# ============================================================================

def calculate_position_sizing_safe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate position sizes with complete safety
    """
    # Initialize position size
    df['position_size'] = 0.0
    
    try:
        # Map signal strength to position size
        signal_map = {
            'EXPLOSIVE': POSITION_SIZES['EXPLOSIVE'],
            'STRONG': POSITION_SIZES['STRONG'],
            'MODERATE': POSITION_SIZES['MODERATE'],
            'WATCH': POSITION_SIZES['WATCH'],
            'AVOID': 0
        }
        
        # Safe conversion and mapping
        if 'signal_strength' in df.columns:
            # Convert to string to ensure proper mapping
            signal_str = df['signal_strength'].astype(str)
            df['base_position_size'] = signal_str.map(signal_map).fillna(0).astype(float)
        else:
            df['base_position_size'] = 0.02  # Default 2%
        
        # Volatility adjustment if available
        if 'volatility_52w' in df.columns:
            vol_factor = 1 - (df['volatility_52w'].fillna(0.5).clip(0, 1) * 0.3)
            df['position_size'] = df['base_position_size'] * vol_factor
        else:
            df['position_size'] = df['base_position_size']
        
        # Pattern confidence boost
        if 'pattern_count' in df.columns:
            pattern_boost = 1 + (df['pattern_count'].fillna(0) * 0.05)
            df['position_size'] = df['position_size'] * pattern_boost
        
        # Final safety cap
        df['position_size'] = df['position_size'].clip(0, RISK_PARAMS['MAX_SINGLE_POSITION'])
        
    except Exception as e:
        logger.error(f"Error in position sizing: {str(e)}")
        # Fallback position sizing
        df['position_size'] = 0.02  # 2% default
    
    # Calculate stops and targets
    df = calculate_stop_losses_safe(df)
    df = calculate_targets_safe(df)
    
    return df

def calculate_stop_losses_safe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate stop losses with safety
    """
    try:
        # Default stop loss percentage
        df['stop_loss_pct'] = 0.08  # 8% default
        
        # Adjust based on volatility if available
        if 'volatility_52w' in df.columns:
            df['stop_loss_pct'] = (df['volatility_52w'].fillna(0.4) * 0.15).clip(0.05, 0.15)
        
        # Calculate stop price
        if 'price' in df.columns:
            df['stop_loss'] = df['price'] * (1 - df['stop_loss_pct'])
        else:
            df['stop_loss'] = 0
            
    except Exception as e:
        logger.error(f"Error calculating stops: {str(e)}")
        df['stop_loss_pct'] = 0.08
        df['stop_loss'] = df.get('price', 100) * 0.92
    
    return df

def calculate_targets_safe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate targets with safety
    """
    try:
        # Default targets based on signal strength
        target_map = {
            'EXPLOSIVE': (1.15, 1.30, 1.50),
            'STRONG': (1.10, 1.20, 1.35),
            'MODERATE': (1.07, 1.15, 1.25),
            'WATCH': (1.05, 1.10, 1.15),
            'AVOID': (1.00, 1.00, 1.00)
        }
        
        # Safe target calculation
        if 'signal_strength' in df.columns and 'price' in df.columns:
            for i, suffix in enumerate(['1', '2', '3']):
                df[f'target_{suffix}'] = df.apply(
                    lambda row: row['price'] * target_map.get(
                        str(row.get('signal_strength', 'MODERATE')), 
                        (1.10, 1.20, 1.30)
                    )[i] if row.get('price', 0) > 0 else 0,
                    axis=1
                )
        else:
            # Fallback targets
            price = df.get('price', 100)
            df['target_1'] = price * 1.10
            df['target_2'] = price * 1.20
            df['target_3'] = price * 1.30
        
        # Calculate risk/reward
        if all(col in df.columns for col in ['target_1', 'price', 'stop_loss']):
            risk = (df['price'] - df['stop_loss']).replace(0, 1)
            reward = df['target_1'] - df['price']
            df['risk_reward_1'] = (reward / risk).fillna(2).clip(0, 10)
        else:
            df['risk_reward_1'] = 2.0
            
    except Exception as e:
        logger.error(f"Error calculating targets: {str(e)}")
        df['target_1'] = df.get('price', 100) * 1.10
        df['risk_reward_1'] = 2.0
    
    return df

# ============================================================================
# MAIN APPLICATION WITH DIAGNOSTICS
# ============================================================================

def main():
    """Main application with comprehensive diagnostics"""
    
    # Page configuration
    st.set_page_config(
        page_title="EDGE Protocol 3.2 ULTIMATE",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
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
    .diagnostic-box {
        background: #f0f0f0;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-family: monospace;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>⚡ EDGE Protocol 3.2 ULTIMATE</h1>
        <p>The Most Robust Implementation Ever Created</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data with detailed diagnostics
    with st.spinner("🔄 Loading and parsing market data..."):
        df, diagnostics = load_data()
    
    # Show diagnostics in sidebar
    with st.sidebar:
        st.header("🔍 System Diagnostics")
        
        if st.checkbox("Show Data Loading Details", value=False):
            st.json(diagnostics)
        
        # Key metrics
        st.metric("Rows Loaded", diagnostics.get('rows_loaded', 0))
        st.metric("Rows Valid", diagnostics.get('rows_valid', 0))
        st.metric("Data Quality", f"{diagnostics.get('data_quality_score', 0):.1f}%")
        
        if diagnostics.get('warnings'):
            st.warning(f"⚠️ {len(diagnostics['warnings'])} warnings")
            for warning in diagnostics['warnings'][:3]:
                st.caption(warning)
    
    # Check if data loaded successfully
    if df.empty:
        st.error("❌ Unable to load data")
        
        # Detailed error display
        if diagnostics.get('warnings'):
            st.error(f"Error: {diagnostics['warnings'][0]}")
        
        st.info("""
        **Troubleshooting Steps:**
        1. Ensure Google Sheet is public (Share → Anyone with link can view)
        2. Check Sheet ID: `1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk`
        3. Check GID: `2026492216`
        """)
        st.stop()
    
    # Process data with comprehensive error handling
    with st.spinner("🧮 Running EDGE analysis..."):
        try:
            # Pattern detection
            df = detect_patterns_safe(df)
            
            # EDGE scoring
            df = calculate_edge_scores_robust(df)
            
            # Position sizing
            df = calculate_position_sizing_safe(df)
            
            # Show processing stats
            if st.checkbox("Show Processing Statistics", value=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Avg EDGE Score", f"{df['edge_score'].mean():.1f}")
                    st.metric("Max EDGE Score", f"{df['edge_score'].max():.1f}")
                with col2:
                    if 'volume_acceleration' in df.columns:
                        valid_accel = df['volume_acceleration'].notna().sum()
                        st.metric("Valid Vol Accel", valid_accel)
                with col3:
                    signal_counts = df['signal_strength'].value_counts()
                    st.metric("Total Signals", len(df[df['signal_strength'] != 'AVOID']))
            
        except Exception as e:
            st.error(f"Processing error: {str(e)}")
            st.stop()
    
    # Filters
    st.sidebar.header("🎯 Filters")
    
    # Edge score filter - START AT 0 to see all
    min_edge_score = st.sidebar.slider(
        "Minimum EDGE Score",
        0, 100, 0, 5,  # Default to 0 to show everything
        help="Set to 0 to see all stocks"
    )
    
    # Show actionable only - Default FALSE to see everything
    show_actionable_only = st.sidebar.checkbox(
        "Show Only Actionable Signals",
        value=False,  # Changed to False
        help="Uncheck to see all stocks including AVOID"
    )
    
    # Apply filters
    filtered_df = df.copy()
    
    if min_edge_score > 0:
        filtered_df = filtered_df[filtered_df['edge_score'] >= min_edge_score]
    
    if show_actionable_only:
        filtered_df = filtered_df[filtered_df['signal_strength'].astype(str) != 'AVOID']
    
    # Show current stats
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Current View")
    st.sidebar.metric("Stocks Shown", len(filtered_df))
    st.sidebar.metric("Original Stocks", len(df))
    
    # Main metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        explosive = len(filtered_df[filtered_df['signal_strength'].astype(str) == 'EXPLOSIVE'])
        st.metric("🚀 EXPLOSIVE", explosive)
    
    with col2:
        strong = len(filtered_df[filtered_df['signal_strength'].astype(str) == 'STRONG'])
        st.metric("💪 STRONG", strong)
    
    with col3:
        moderate = len(filtered_df[filtered_df['signal_strength'].astype(str) == 'MODERATE'])
        st.metric("📈 MODERATE", moderate)
    
    with col4:
        watch = len(filtered_df[filtered_df['signal_strength'].astype(str) == 'WATCH'])
        st.metric("👀 WATCH", watch)
    
    with col5:
        avoid = len(filtered_df[filtered_df['signal_strength'].astype(str) == 'AVOID'])
        st.metric("❌ AVOID", avoid)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📊 Signals", "🔍 Diagnostics", "📈 Analysis"])
    
    with tab1:
        st.header("📊 Trading Signals")
        
        if filtered_df.empty:
            st.warning("No signals found. Try adjusting filters.")
        else:
            # Display the signals table
            display_cols = [
                'ticker', 'company_name', 'signal_strength', 'edge_score',
                'signal_reason', 'price', 'volume_acceleration', 
                'ret_7d', 'ret_30d', 'position_size', 'stop_loss', 'target_1'
            ]
            
            available_cols = [col for col in display_cols if col in filtered_df.columns]
            
            # Sort by edge score
            display_df = filtered_df[available_cols].sort_values('edge_score', ascending=False)
            
            # Format the dataframe
            st.dataframe(
                display_df.style.format({
                    'edge_score': '{:.1f}',
                    'price': '₹{:.2f}',
                    'volume_acceleration': '{:.1f}%',
                    'ret_7d': '{:.1f}%',
                    'ret_30d': '{:.1f}%',
                    'position_size': '{:.1%}',
                    'stop_loss': '₹{:.2f}',
                    'target_1': '₹{:.2f}'
                }).background_gradient(subset=['edge_score'], cmap='RdYlGn'),
                use_container_width=True,
                height=600
            )
            
            # Export button
            csv = display_df.to_csv(index=False)
            st.download_button(
                "📥 Download Signals",
                csv,
                f"edge_signals_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                "text/csv"
            )
    
    with tab2:
        st.header("🔍 System Diagnostics")
        
        # Data quality breakdown
        st.subheader("Data Quality Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Column availability
            st.markdown("**Key Columns Status:**")
            key_cols = {
                'price': '💰 Price',
                'volume_acceleration': '📊 Volume Acceleration',
                'ret_7d': '📈 7-Day Return',
                'vol_ratio_30d_90d': '🔄 Vol Ratio 30d/90d',
                'vol_ratio_30d_180d': '🔄 Vol Ratio 30d/180d'
            }
            
            for col, name in key_cols.items():
                if col in df.columns:
                    valid_pct = (df[col].notna().sum() / len(df) * 100)
                    if valid_pct > 90:
                        st.success(f"✅ {name}: {valid_pct:.0f}%")
                    elif valid_pct > 50:
                        st.warning(f"⚠️ {name}: {valid_pct:.0f}%")
                    else:
                        st.error(f"❌ {name}: {valid_pct:.0f}%")
                else:
                    st.error(f"❌ {name}: Missing")
        
        with col2:
            # Score distribution
            st.markdown("**EDGE Score Distribution:**")
            
            if 'edge_score' in df.columns:
                score_ranges = {
                    '90-100': len(df[df['edge_score'] >= 90]),
                    '70-90': len(df[(df['edge_score'] >= 70) & (df['edge_score'] < 90)]),
                    '50-70': len(df[(df['edge_score'] >= 50) & (df['edge_score'] < 70)]),
                    '30-50': len(df[(df['edge_score'] >= 30) & (df['edge_score'] < 50)]),
                    '0-30': len(df[df['edge_score'] < 30])
                }
                
                for range_name, count in score_ranges.items():
                    st.metric(f"Score {range_name}", count)
        
        # Raw data sample
        if st.checkbox("Show Raw Data Sample"):
            st.dataframe(df.head(20))
    
    with tab3:
        st.header("📈 Market Analysis")
        
        # EDGE score distribution
        if 'edge_score' in df.columns:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=df['edge_score'],
                nbinsx=30,
                marker_color='orange'
            ))
            fig.update_layout(
                title="EDGE Score Distribution",
                xaxis_title="EDGE Score",
                yaxis_title="Count"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Volume acceleration analysis
        if 'volume_acceleration' in df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                # Stats
                st.markdown("**Volume Acceleration Stats:**")
                accel_stats = df['volume_acceleration'].describe()
                st.dataframe(accel_stats)
            
            with col2:
                # Distribution
                fig2 = go.Figure()
                fig2.add_trace(go.Histogram(
                    x=df['volume_acceleration'],
                    nbinsx=30,
                    marker_color='blue'
                ))
                fig2.update_layout(
                    title="Volume Acceleration Distribution",
                    xaxis_title="Volume Acceleration %",
                    yaxis_title="Count"
                )
                st.plotly_chart(fig2, use_container_width=True)

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()
