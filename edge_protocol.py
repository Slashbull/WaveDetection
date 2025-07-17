#!/usr/bin/env python3
"""
EDGE Protocol - Elite Data-Driven Growth Engine
==============================================
Final Production Version 2.0

A sophisticated trading intelligence system that combines:
- Volume acceleration analysis
- Pattern recognition
- Risk-adjusted position sizing
- Multi-factor scoring

Author: EDGE Protocol Team
Version: 2.0.0 FINAL
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
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

# Google Sheets Configuration
SHEET_CONFIG = {
    'SHEET_ID': '1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk',
    'GID': '2026492216',
    'CACHE_TTL': 300,  # 5 minutes
    'REQUEST_TIMEOUT': 30
}

# Market Cap Classification (in Crores)
MARKET_CAP_TIERS = {
    'Mega Cap': 200000,
    'Large Cap': 20000,
    'Mid Cap': 5000,
    'Small Cap': 500,
    'Micro Cap': 100,
    'Nano Cap': 0
}

# EPS Tiers
EPS_TIERS = {
    '95↑': (95, float('inf')),
    '75↑': (75, 95),
    '55↑': (55, 75),
    '35↑': (35, 55),
    '15↑': (15, 35),
    '5↑': (5, 15),
    '5↓': (float('-inf'), 5)
}

# Price Tiers (in INR)
PRICE_TIERS = {
    '5K↑': (5000, float('inf')),
    '2K↑': (2000, 5000),
    '1K↑': (1000, 2000),
    '500↑': (500, 1000),
    '200↑': (200, 500),
    '100↑': (100, 200),
    '100↓': (0, 100)
}

# EDGE Score Thresholds
EDGE_THRESHOLDS = {
    'SUPER_EDGE': 90,
    'EXPLOSIVE': 80,
    'STRONG': 70,
    'MODERATE': 50,
    'WATCH': 30
}

# Position Sizing (% of portfolio)
POSITION_SIZES = {
    'SUPER_EDGE': 0.10,  # 10%
    'EXPLOSIVE': 0.07,   # 7%
    'STRONG': 0.05,      # 5%
    'MODERATE': 0.03,    # 3%
    'WATCH': 0.01        # 1%
}

# Risk Management Parameters
RISK_PARAMS = {
    'MAX_PORTFOLIO_EXPOSURE': 0.75,  # 75% max
    'MAX_SINGLE_POSITION': 0.10,     # 10% max
    'MAX_SECTOR_EXPOSURE': 0.25,     # 25% max per sector
    'MIN_LIQUIDITY_FILTER': 10000000, # 1 Cr daily volume minimum
    'STOP_LOSS_ATR_MULTIPLIER': 2.0  # 2x ATR for stop loss
}

# Scoring Weights for different strategies
STRATEGY_WEIGHTS = {
    'Balanced': {'volume': 0.30, 'momentum': 0.25, 'quality': 0.25, 'value': 0.20},
    'Momentum': {'volume': 0.35, 'momentum': 0.40, 'quality': 0.15, 'value': 0.10},
    'Value': {'volume': 0.25, 'momentum': 0.15, 'quality': 0.30, 'value': 0.30},
    'Quality': {'volume': 0.20, 'momentum': 0.20, 'quality': 0.40, 'value': 0.20}
}

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class StockSignal:
    """Data structure for a trading signal"""
    ticker: str
    company_name: str
    edge_score: float
    category: str
    decision: str
    price: float
    stop_loss: float
    target_1: float
    target_2: float
    position_size: float
    risk_reward: float
    volume_pattern: str
    sector: str = 'Unknown'
    
class SignalStrength(Enum):
    """Signal strength classification"""
    SUPER_EDGE = "SUPER_EDGE"
    EXPLOSIVE = "EXPLOSIVE"
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WATCH = "WATCH"
    AVOID = "AVOID"

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('EDGE_Protocol')

# ============================================================================
# DATA LOADING & VALIDATION
# ============================================================================

@st.cache_data(ttl=SHEET_CONFIG['CACHE_TTL'])
def load_data() -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    Load data from Google Sheets with comprehensive validation
    
    Returns:
        Tuple of (DataFrame, diagnostics dict)
    """
    diagnostics = {
        'timestamp': datetime.now(),
        'rows_loaded': 0,
        'data_quality_score': 0,
        'warnings': [],
        'critical_columns_missing': []
    }
    
    try:
        # Construct URL
        url = f"https://docs.google.com/spreadsheets/d/{SHEET_CONFIG['SHEET_ID']}/export?format=csv&gid={SHEET_CONFIG['GID']}"
        
        # Fetch data
        response = requests.get(url, timeout=SHEET_CONFIG['REQUEST_TIMEOUT'])
        response.raise_for_status()
        
        # Load into DataFrame
        df = pd.read_csv(pd.io.common.StringIO(response.text))
        diagnostics['rows_loaded'] = len(df)
        
        # Clean column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Define critical columns
        critical_cols = [
            'ticker', 'company_name', 'price', 'market_cap',
            'volume_1d', 'rvol', 'sector', 'category',
            'vol_ratio_30d_90d', 'vol_ratio_30d_180d',
            'ret_1d', 'ret_7d', 'ret_30d',
            'from_high_pct', 'from_low_pct',
            'eps_current', 'pe', 'sma_50d', 'sma_200d'
        ]
        
        # Check missing columns
        missing = [col for col in critical_cols if col not in df.columns]
        diagnostics['critical_columns_missing'] = missing
        
        if missing:
            diagnostics['warnings'].append(f"Missing columns: {', '.join(missing)}")
        
        # Data type conversions
        df = clean_and_convert_data(df)
        
        # Validate data quality
        df, quality_score = validate_data_quality(df)
        diagnostics['data_quality_score'] = quality_score
        
        # Add derived columns
        df = add_derived_columns(df)
        
        return df, diagnostics
        
    except Exception as e:
        logger.error(f"Data loading failed: {str(e)}")
        diagnostics['warnings'].append(f"Critical error: {str(e)}")
        return pd.DataFrame(), diagnostics

def clean_and_convert_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and convert data types properly"""
    
    # Price and numeric columns
    numeric_cols = [
        'price', 'low_52w', 'high_52w', 'prev_close',
        'sma_20d', 'sma_50d', 'sma_200d',
        'volume_1d', 'volume_7d', 'volume_30d', 'volume_90d',
        'rvol', 'pe', 'eps_current', 'eps_last_qtr', 'eps_change_pct'
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            # Remove currency symbols and convert
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace('[₹,$,]', '', regex=True),
                errors='coerce'
            )
    
    # Percentage columns
    pct_cols = [
        'ret_1d', 'ret_3d', 'ret_7d', 'ret_30d', 'ret_3m', 'ret_6m', 'ret_1y',
        'from_high_pct', 'from_low_pct', 'eps_change_pct',
        'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d',
        'vol_ratio_30d_180d', 'vol_ratio_90d_180d'
    ]
    
    for col in pct_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace('%', ''),
                errors='coerce'
            )
    
    # Market cap special handling
    if 'market_cap' in df.columns:
        df['market_cap_cr'] = df['market_cap'].apply(parse_market_cap)
    
    # Category assignment based on market cap
    if 'market_cap_cr' in df.columns:
        df['category'] = df['market_cap_cr'].apply(assign_market_category)
    
    return df

def parse_market_cap(val: Union[str, float]) -> float:
    """Parse market cap strings to numeric values in Crores"""
    if pd.isna(val):
        return np.nan
    
    val_str = str(val).strip().upper()
    val_str = val_str.replace('₹', '').replace(',', '').strip()
    
    # Handle Cr suffix
    if 'CR' in val_str:
        val_str = val_str.replace('CR', '').strip()
        try:
            return float(val_str)
        except:
            return np.nan
    
    # If no suffix, assume it's already in Crores
    try:
        return float(val_str)
    except:
        return np.nan

def assign_market_category(market_cap_cr: float) -> str:
    """Assign category based on market cap in Crores"""
    if pd.isna(market_cap_cr):
        return 'Unknown'
    
    for category, min_cap in MARKET_CAP_TIERS.items():
        if market_cap_cr >= min_cap:
            return category
    
    return 'Nano Cap'

def validate_data_quality(df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    """Validate data quality and calculate quality score"""
    
    # Remove invalid rows
    initial_len = len(df)
    
    # Must have valid ticker and price
    df = df.dropna(subset=['ticker'])
    df = df[df['price'] > 0]
    
    # Calculate quality score
    essential_cols = ['price', 'volume_1d', 'rvol', 'ret_7d', 'vol_ratio_30d_90d']
    valid_data_points = sum(df[col].notna().sum() for col in essential_cols if col in df.columns)
    total_possible = len(df) * len(essential_cols)
    
    quality_score = (valid_data_points / total_possible * 100) if total_possible > 0 else 0
    
    logger.info(f"Data quality score: {quality_score:.1f}%")
    logger.info(f"Removed {initial_len - len(df)} invalid rows")
    
    return df, quality_score

def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns for analysis"""
    
    # EPS Tier assignment
    if 'eps_current' in df.columns:
        df['eps_tier'] = df['eps_current'].apply(assign_eps_tier)
    
    # Price Tier assignment
    if 'price' in df.columns:
        df['price_tier'] = df['price'].apply(assign_price_tier)
    
    # Volume in Rupees (for liquidity filter)
    if all(col in df.columns for col in ['volume_1d', 'price']):
        df['volume_rupees'] = df['volume_1d'] * df['price']
    
    # True Volume Acceleration (if both ratios available)
    if all(col in df.columns for col in ['vol_ratio_30d_90d', 'vol_ratio_30d_180d']):
        # This is the TRUE acceleration: comparing recent vs longer-term momentum
        df['volume_acceleration'] = df['vol_ratio_30d_90d'] - df['vol_ratio_30d_180d']
        df['has_valid_acceleration'] = True
    else:
        # Flag that we don't have valid acceleration
        df['volume_acceleration'] = 0
        df['has_valid_acceleration'] = False
        logger.warning("Missing vol_ratio_30d_180d - volume acceleration unavailable")
    
    # Simple volatility proxy
    if all(col in df.columns for col in ['high_52w', 'low_52w', 'price']):
        df['volatility_52w'] = (df['high_52w'] - df['low_52w']) / df['price']
    
    return df

def assign_eps_tier(eps: float) -> str:
    """Assign EPS tier based on value"""
    if pd.isna(eps):
        return 'Unknown'
    
    for tier, (min_val, max_val) in EPS_TIERS.items():
        if min_val <= eps < max_val:
            return tier
    
    return 'Unknown'

def assign_price_tier(price: float) -> str:
    """Assign price tier based on value"""
    if pd.isna(price):
        return 'Unknown'
    
    for tier, (min_val, max_val) in PRICE_TIERS.items():
        if min_val <= price < max_val:
            return tier
    
    return 'Unknown'

# ============================================================================
# SCORING ENGINE
# ============================================================================

def calculate_edge_scores(df: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
    """
    Calculate EDGE scores using multi-factor model
    
    Args:
        df: Input DataFrame
        weights: Dictionary of scoring weights
    
    Returns:
        DataFrame with EDGE scores and components
    """
    
    # Initialize score components
    df['volume_score'] = 0
    df['momentum_score'] = 0
    df['quality_score'] = 0
    df['value_score'] = 0
    
    # 1. Volume Score (includes acceleration and consistency)
    if 'has_valid_acceleration' in df.columns and df['has_valid_acceleration'].any():
        # Use true acceleration if available
        df['volume_score'] = calculate_volume_score(df)
    else:
        # Fallback to simple volume metrics
        df['volume_score'] = calculate_simple_volume_score(df)
    
    # 2. Momentum Score
    df['momentum_score'] = calculate_momentum_score(df)
    
    # 3. Quality Score
    df['quality_score'] = calculate_quality_score(df)
    
    # 4. Value Score
    df['value_score'] = calculate_value_score(df)
    
    # Calculate weighted EDGE score
    df['edge_score'] = (
        df['volume_score'] * weights['volume'] +
        df['momentum_score'] * weights['momentum'] +
        df['quality_score'] * weights['quality'] +
        df['value_score'] * weights['value']
    )
    
    # Classify based on score
    df['signal_strength'] = df['edge_score'].apply(classify_signal_strength)
    
    return df

def calculate_volume_score(df: pd.DataFrame) -> pd.Series:
    """Calculate volume score with acceleration"""
    score = pd.Series(0, index=df.index)
    
    # Volume acceleration component (40%)
    if 'volume_acceleration' in df.columns:
        accel_normalized = df['volume_acceleration'].clip(-100, 100) / 100
        score += (50 + accel_normalized * 50) * 0.4
    
    # RVOL component (30%)
    if 'rvol' in df.columns:
        rvol_score = df['rvol'].clip(0, 5) * 20
        score += rvol_score * 0.3
    
    # Volume consistency (30%)
    vol_cols = ['vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d']
    available_cols = [col for col in vol_cols if col in df.columns]
    if available_cols:
        positive_count = (df[available_cols] > 0).sum(axis=1)
        consistency_score = (positive_count / len(available_cols)) * 100
        score += consistency_score * 0.3
    
    return score.clip(0, 100)

def calculate_simple_volume_score(df: pd.DataFrame) -> pd.Series:
    """Fallback volume score without acceleration"""
    score = pd.Series(50, index=df.index)  # Start at neutral
    
    # RVOL component (50%)
    if 'rvol' in df.columns:
        rvol_score = df['rvol'].clip(0, 5) * 20
        score = score * 0.5 + rvol_score * 0.5
    
    # Volume ratios (50%)
    if 'vol_ratio_30d_90d' in df.columns:
        ratio_score = df['vol_ratio_30d_90d'].clip(-50, 150) / 1.5
        score = score * 0.5 + ratio_score * 0.5
    
    return score.clip(0, 100)

def calculate_momentum_score(df: pd.DataFrame) -> pd.Series:
    """Calculate momentum score based on returns"""
    score = pd.Series(50, index=df.index)  # Start at neutral
    
    # Short-term momentum (40%)
    if all(col in df.columns for col in ['ret_1d', 'ret_7d']):
        short_momentum = (df['ret_1d'] + df['ret_7d'] * 2) / 3
        short_score = 50 + short_momentum * 2
        score = score * 0.6 + short_score.clip(0, 100) * 0.4
    
    # Medium-term momentum (40%)
    if 'ret_30d' in df.columns:
        medium_score = 50 + df['ret_30d']
        score = score * 0.6 + medium_score.clip(0, 100) * 0.4
    
    # Trend alignment bonus (20%)
    if all(col in df.columns for col in ['price', 'sma_50d', 'sma_200d']):
        above_50 = (df['price'] > df['sma_50d']).astype(int) * 10
        above_200 = (df['price'] > df['sma_200d']).astype(int) * 10
        score += above_50 + above_200
    
    return score.clip(0, 100)

def calculate_quality_score(df: pd.DataFrame) -> pd.Series:
    """Calculate quality score based on fundamentals"""
    score = pd.Series(50, index=df.index)  # Start at neutral
    
    # EPS growth (50%)
    if 'eps_change_pct' in df.columns:
        eps_score = 50 + df['eps_change_pct'].clip(-50, 100) / 2
        score = score * 0.5 + eps_score * 0.5
    
    # PE ratio (30%)
    if 'pe' in df.columns:
        # Lower PE is better (within reason)
        pe_score = pd.Series(50, index=df.index)
        pe_score[df['pe'].between(5, 15)] = 100
        pe_score[df['pe'].between(15, 25)] = 75
        pe_score[df['pe'].between(25, 35)] = 50
        pe_score[df['pe'] > 35] = 25
        score = score * 0.7 + pe_score * 0.3
    
    # Long-term performance (20%)
    if 'ret_1y' in df.columns:
        long_term_score = 50 + df['ret_1y'].clip(-50, 100) / 2
        score = score * 0.8 + long_term_score * 0.2
    
    return score.clip(0, 100)

def calculate_value_score(df: pd.DataFrame) -> pd.Series:
    """Calculate value score based on risk/reward"""
    score = pd.Series(50, index=df.index)  # Start at neutral
    
    # Distance from 52w high (40%)
    if 'from_high_pct' in df.columns:
        # Sweet spot: -20% to -10% from high
        pullback_score = pd.Series(50, index=df.index)
        pullback_score[df['from_high_pct'].between(-25, -15)] = 100
        pullback_score[df['from_high_pct'].between(-35, -25)] = 75
        pullback_score[df['from_high_pct'] > -10] = 25  # Too close to high
        score = score * 0.6 + pullback_score * 0.4
    
    # Risk/Reward potential (60%)
    if all(col in df.columns for col in ['price', 'high_52w', 'low_52w']):
        upside = ((df['high_52w'] - df['price']) / df['price'] * 100).clip(0, 100)
        downside = ((df['price'] - df['low_52w']) / df['price'] * 100).clip(0, 100)
        rr_ratio = (upside / (downside + 1)).clip(0, 5) * 20
        score = score * 0.4 + rr_ratio * 0.6
    
    return score.clip(0, 100)

def classify_signal_strength(edge_score: float) -> str:
    """Classify signal based on EDGE score"""
    if edge_score >= EDGE_THRESHOLDS['SUPER_EDGE']:
        return SignalStrength.SUPER_EDGE.value
    elif edge_score >= EDGE_THRESHOLDS['EXPLOSIVE']:
        return SignalStrength.EXPLOSIVE.value
    elif edge_score >= EDGE_THRESHOLDS['STRONG']:
        return SignalStrength.STRONG.value
    elif edge_score >= EDGE_THRESHOLDS['MODERATE']:
        return SignalStrength.MODERATE.value
    elif edge_score >= EDGE_THRESHOLDS['WATCH']:
        return SignalStrength.WATCH.value
    else:
        return SignalStrength.AVOID.value

# ============================================================================
# PATTERN DETECTION (Simplified)
# ============================================================================

def detect_key_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Detect key actionable patterns"""
    
    # Pattern 1: Volume Surge with Price Stability
    df['pattern_accumulation'] = False
    if all(col in df.columns for col in ['rvol', 'ret_7d']):
        mask = (df['rvol'] > 2.0) & (df['ret_7d'].abs() < 5)
        df.loc[mask, 'pattern_accumulation'] = True
    
    # Pattern 2: Momentum Breakout
    df['pattern_breakout'] = False
    if all(col in df.columns for col in ['ret_7d', 'volume_acceleration', 'price', 'sma_50d']):
        mask = (df['ret_7d'] > 5) & (df['volume_acceleration'] > 20) & (df['price'] > df['sma_50d'])
        df.loc[mask, 'pattern_breakout'] = True
    
    # Pattern 3: Quality Pullback
    df['pattern_quality_pullback'] = False
    if all(col in df.columns for col in ['ret_1y', 'from_high_pct', 'eps_change_pct']):
        mask = (df['ret_1y'] > 50) & (df['from_high_pct'].between(-30, -15)) & (df['eps_change_pct'] > 10)
        df.loc[mask, 'pattern_quality_pullback'] = True
    
    # Combined pattern score
    df['pattern_score'] = (
        df['pattern_accumulation'].astype(int) * 30 +
        df['pattern_breakout'].astype(int) * 40 +
        df['pattern_quality_pullback'].astype(int) * 30
    )
    
    return df

# ============================================================================
# RISK MANAGEMENT
# ============================================================================

def calculate_position_sizing(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate position sizes with risk management"""
    
    # Base position size from signal strength
    df['base_position_size'] = df['signal_strength'].map(POSITION_SIZES)
    
    # Risk adjustment based on volatility
    if 'volatility_52w' in df.columns:
        volatility_adjustment = 1 - (df['volatility_52w'].clip(0, 1) * 0.5)
        df['position_size'] = df['base_position_size'] * volatility_adjustment
    else:
        df['position_size'] = df['base_position_size']
    
    # Cap at maximum single position
    df['position_size'] = df['position_size'].clip(0, RISK_PARAMS['MAX_SINGLE_POSITION'])
    
    # Calculate stop loss
    df = calculate_dynamic_stops(df)
    
    # Calculate targets
    df = calculate_targets(df)
    
    return df

def calculate_dynamic_stops(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate dynamic stop losses"""
    
    # Default stop based on category volatility
    category_stops = {
        'Mega Cap': 0.07,
        'Large Cap': 0.08,
        'Mid Cap': 0.10,
        'Small Cap': 0.12,
        'Micro Cap': 0.15,
        'Nano Cap': 0.20,
        'Unknown': 0.10
    }
    
    df['stop_loss_pct'] = df['category'].map(category_stops).fillna(0.10)
    
    # Adjust for support levels
    if all(col in df.columns for col in ['price', 'sma_50d', 'low_52w']):
        # Use max of category stop or support-based stop
        support_stop_50 = (df['price'] - df['sma_50d']) / df['price']
        support_stop_52w = (df['price'] - df['low_52w'] * 1.05) / df['price']
        
        df['stop_loss_pct'] = df[['stop_loss_pct', support_stop_50.abs(), support_stop_52w.abs()]].min(axis=1)
    
    # Calculate stop price
    df['stop_loss'] = df['price'] * (1 - df['stop_loss_pct'])
    
    return df

def calculate_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate profit targets"""
    
    # Base targets on signal strength
    target_multipliers = {
        'SUPER_EDGE': (1.15, 1.30),
        'EXPLOSIVE': (1.12, 1.25),
        'STRONG': (1.10, 1.20),
        'MODERATE': (1.07, 1.15),
        'WATCH': (1.05, 1.10),
        'AVOID': (1.00, 1.00)
    }
    
    df['target_1'] = df.apply(
        lambda row: row['price'] * target_multipliers.get(row['signal_strength'], (1.10, 1.20))[0],
        axis=1
    )
    
    df['target_2'] = df.apply(
        lambda row: row['price'] * target_multipliers.get(row['signal_strength'], (1.10, 1.20))[1],
        axis=1
    )
    
    # Calculate risk/reward
    df['risk_reward'] = (df['target_1'] - df['price']) / (df['price'] - df['stop_loss'])
    
    return df

def apply_portfolio_constraints(df: pd.DataFrame) -> pd.DataFrame:
    """Apply portfolio-level risk constraints"""
    
    # Sort by edge score (highest first)
    df = df.sort_values('edge_score', ascending=False).copy()
    
    # Track allocations
    total_allocation = 0
    sector_allocations = {}
    
    # Apply constraints
    for idx in df.index:
        current_position = df.loc[idx, 'position_size']
        sector = df.loc[idx, 'sector']
        
        # Check total portfolio constraint
        if total_allocation + current_position > RISK_PARAMS['MAX_PORTFOLIO_EXPOSURE']:
            remaining = RISK_PARAMS['MAX_PORTFOLIO_EXPOSURE'] - total_allocation
            df.loc[idx, 'position_size'] = max(0, remaining)
        
        # Check sector constraint
        current_sector_allocation = sector_allocations.get(sector, 0)
        if current_sector_allocation + current_position > RISK_PARAMS['MAX_SECTOR_EXPOSURE']:
            remaining = RISK_PARAMS['MAX_SECTOR_EXPOSURE'] - current_sector_allocation
            df.loc[idx, 'position_size'] = min(df.loc[idx, 'position_size'], max(0, remaining))
        
        # Update trackers
        actual_position = df.loc[idx, 'position_size']
        total_allocation += actual_position
        sector_allocations[sector] = sector_allocations.get(sector, 0) + actual_position
    
    return df

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_edge_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Create EDGE score distribution chart"""
    
    fig = go.Figure()
    
    # Add histogram
    fig.add_trace(go.Histogram(
        x=df['edge_score'],
        nbinsx=20,
        name='Distribution',
        marker_color='lightblue',
        opacity=0.7
    ))
    
    # Add threshold lines
    for name, threshold in EDGE_THRESHOLDS.items():
        fig.add_vline(
            x=threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=name,
            annotation_position="top"
        )
    
    fig.update_layout(
        title="EDGE Score Distribution",
        xaxis_title="EDGE Score",
        yaxis_title="Number of Stocks",
        height=400
    )
    
    return fig

def create_sector_performance_chart(df: pd.DataFrame) -> go.Figure:
    """Create sector performance chart"""
    
    # Calculate sector metrics
    sector_metrics = df.groupby('sector').agg({
        'edge_score': 'mean',
        'ticker': 'count',
        'volume_acceleration': 'mean'
    }).round(1)
    
    sector_metrics = sector_metrics.sort_values('edge_score', ascending=True)
    
    # Create horizontal bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=sector_metrics['edge_score'],
        y=sector_metrics.index,
        orientation='h',
        text=sector_metrics['edge_score'],
        textposition='outside',
        marker_color=sector_metrics['edge_score'],
        marker_colorscale='RdYlGn',
        marker_cmin=30,
        marker_cmax=70
    ))
    
    fig.update_layout(
        title="Average EDGE Score by Sector",
        xaxis_title="Average EDGE Score",
        yaxis_title="Sector",
        height=600,
        margin=dict(l=150)
    )
    
    return fig

def create_signal_scatter(df: pd.DataFrame) -> go.Figure:
    """Create signal strength scatter plot"""
    
    # Filter for signals only
    signals_df = df[df['signal_strength'] != 'AVOID'].copy()
    
    if signals_df.empty:
        return go.Figure().add_annotation(
            text="No signals to display",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    # Create scatter plot
    fig = px.scatter(
        signals_df,
        x='risk_reward',
        y='edge_score',
        size='position_size',
        color='signal_strength',
        hover_data=['ticker', 'company_name', 'sector', 'price'],
        title="Signal Analysis: Risk/Reward vs EDGE Score",
        labels={
            'risk_reward': 'Risk/Reward Ratio',
            'edge_score': 'EDGE Score'
        },
        color_discrete_map={
            'SUPER_EDGE': '#FFD700',
            'EXPLOSIVE': '#FF4500',
            'STRONG': '#32CD32',
            'MODERATE': '#1E90FF',
            'WATCH': '#808080'
        }
    )
    
    # Add quadrant lines
    fig.add_hline(y=70, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=2, line_dash="dash", line_color="gray", opacity=0.5)
    
    # Add annotations
    fig.add_annotation(x=3, y=85, text="🎯 Sweet Spot", showarrow=False, font=dict(size=16))
    
    fig.update_layout(height=600)
    
    return fig

# ============================================================================
# MAIN UI COMPONENTS
# ============================================================================

def render_filters_sidebar(df: pd.DataFrame) -> Dict[str, any]:
    """Render sidebar filters and return selected values"""
    
    st.sidebar.header("🎯 Filters")
    
    filters = {}
    
    # Strategy selection
    filters['strategy'] = st.sidebar.selectbox(
        "Strategy Profile",
        list(STRATEGY_WEIGHTS.keys()),
        index=0
    )
    
    # Category filter
    if 'category' in df.columns:
        categories = ['All'] + sorted(df['category'].unique().tolist())
        filters['category'] = st.sidebar.multiselect(
            "Market Cap Category",
            categories[1:],  # Exclude 'All'
            default=[]
        )
    
    # Sector filter
    if 'sector' in df.columns:
        sectors = ['All'] + sorted(df['sector'].unique().tolist())
        filters['sector'] = st.sidebar.multiselect(
            "Sector",
            sectors[1:],  # Exclude 'All'
            default=[]
        )
    
    # EPS Tier filter
    if 'eps_tier' in df.columns:
        eps_tiers = ['All'] + sorted(df['eps_tier'].unique().tolist())
        filters['eps_tier'] = st.sidebar.multiselect(
            "EPS Tier",
            eps_tiers[1:],  # Exclude 'All'
            default=[]
        )
    
    # Price Tier filter
    if 'price_tier' in df.columns:
        price_tiers = ['All'] + sorted(df['price_tier'].unique().tolist())
        filters['price_tier'] = st.sidebar.multiselect(
            "Price Tier",
            price_tiers[1:],  # Exclude 'All'
            default=[]
        )
    
    # Signal strength filter
    filters['min_edge_score'] = st.sidebar.slider(
        "Minimum EDGE Score",
        0, 100, 50, 5
    )
    
    # Liquidity filter
    filters['min_liquidity'] = st.sidebar.checkbox(
        "Exclude Low Liquidity Stocks",
        value=True
    )
    
    return filters

def apply_filters(df: pd.DataFrame, filters: Dict[str, any]) -> pd.DataFrame:
    """Apply selected filters to dataframe"""
    
    filtered_df = df.copy()
    
    # Category filter
    if filters.get('category'):
        filtered_df = filtered_df[filtered_df['category'].isin(filters['category'])]
    
    # Sector filter
    if filters.get('sector'):
        filtered_df = filtered_df[filtered_df['sector'].isin(filters['sector'])]
    
    # EPS Tier filter
    if filters.get('eps_tier'):
        filtered_df = filtered_df[filtered_df['eps_tier'].isin(filters['eps_tier'])]
    
    # Price Tier filter
    if filters.get('price_tier'):
        filtered_df = filtered_df[filtered_df['price_tier'].isin(filters['price_tier'])]
    
    # EDGE score filter
    filtered_df = filtered_df[filtered_df['edge_score'] >= filters['min_edge_score']]
    
    # Liquidity filter
    if filters.get('min_liquidity') and 'volume_rupees' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['volume_rupees'] >= RISK_PARAMS['MIN_LIQUIDITY_FILTER']]
    
    return filtered_df

def display_key_metrics(df: pd.DataFrame):
    """Display key metrics in columns"""
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_signals = len(df[df['signal_strength'] != 'AVOID'])
        st.metric("Total Signals", total_signals)
    
    with col2:
        super_edge = len(df[df['signal_strength'] == 'SUPER_EDGE'])
        st.metric("SUPER EDGE", super_edge)
    
    with col3:
        avg_edge = df[df['signal_strength'] != 'AVOID']['edge_score'].mean()
        st.metric("Avg EDGE Score", f"{avg_edge:.1f}")
    
    with col4:
        portfolio_used = df['position_size'].sum() * 100
        st.metric("Portfolio Used", f"{portfolio_used:.1f}%")
    
    with col5:
        high_rr = len(df[df['risk_reward'] >= 2])
        st.metric("High R/R (≥2)", high_rr)

def display_signals_table(df: pd.DataFrame):
    """Display main signals table"""
    
    # Filter for actionable signals
    signals_df = df[df['signal_strength'] != 'AVOID'].copy()
    
    if signals_df.empty:
        st.info("No signals match current filters")
        return
    
    # Select display columns
    display_cols = [
        'ticker', 'company_name', 'sector', 'category', 'signal_strength',
        'edge_score', 'price', 'position_size', 'stop_loss', 'target_1',
        'risk_reward', 'volume_acceleration', 'pattern_score'
    ]
    
    # Filter to available columns
    display_cols = [col for col in display_cols if col in signals_df.columns]
    
    # Format for display
    format_dict = {
        'edge_score': '{:.1f}',
        'price': '₹{:.2f}',
        'position_size': '{:.1%}',
        'stop_loss': '₹{:.2f}',
        'target_1': '₹{:.2f}',
        'risk_reward': '{:.2f}',
        'volume_acceleration': '{:.1f}%',
        'pattern_score': '{:.0f}'
    }
    
    # Apply conditional formatting
    styled_df = signals_df[display_cols].style.format(format_dict)
    
    # Color code signal strength
    def color_signal_strength(val):
        colors = {
            'SUPER_EDGE': 'background-color: #FFD700; color: black;',
            'EXPLOSIVE': 'background-color: #FF4500; color: white;',
            'STRONG': 'background-color: #32CD32; color: black;',
            'MODERATE': 'background-color: #1E90FF; color: white;',
            'WATCH': 'background-color: #808080; color: white;'
        }
        return colors.get(val, '')
    
    if 'signal_strength' in display_cols:
        styled_df = styled_df.applymap(color_signal_strength, subset=['signal_strength'])
    
    # Display table
    st.dataframe(styled_df, use_container_width=True, height=600)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point"""
    
    # Page configuration
    st.set_page_config(
        page_title="EDGE Protocol 2.0",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 5px;
    }
    .super-edge-alert {
        background: linear-gradient(45deg, #FFD700, #FFA500);
        color: black;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
        margin: 20px 0;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("⚡ EDGE Protocol 2.0")
    st.markdown("**Elite Data-Driven Growth Engine** - Your Systematic Path to Market Outperformance")
    
    # Disclaimer
    with st.expander("⚠️ Important Disclaimer", expanded=False):
        st.warning("""
        **FINANCIAL DISCLAIMER:**
        - This tool is for educational and research purposes only
        - Not financial advice or recommendations
        - Past performance does not guarantee future results
        - Always consult qualified financial advisors
        - Trading involves risk of loss
        - The developers assume no liability for trading decisions
        """)
    
    # Load data
    with st.spinner("Loading market data..."):
        df, diagnostics = load_data()
    
    if df.empty:
        st.error("❌ Unable to load data. Please check your connection and try again.")
        st.stop()
    
    # Display diagnostics in sidebar
    with st.sidebar:
        with st.expander("📊 Data Quality", expanded=False):
            st.metric("Quality Score", f"{diagnostics['data_quality_score']:.1f}%")
            st.metric("Rows Loaded", diagnostics['rows_loaded'])
            if diagnostics['warnings']:
                st.warning(f"{len(diagnostics['warnings'])} warnings")
                for warning in diagnostics['warnings'][:3]:
                    st.caption(warning)
    
    # Get filters
    filters = render_filters_sidebar(df)
    
    # Process data
    with st.spinner("Analyzing market dynamics..."):
        # Apply strategy weights
        weights = STRATEGY_WEIGHTS[filters['strategy']]
        
        # Calculate EDGE scores
        df = calculate_edge_scores(df, weights)
        
        # Detect patterns
        df = detect_key_patterns(df)
        
        # Calculate position sizing and risk metrics
        df = calculate_position_sizing(df)
        
        # Apply portfolio constraints
        df = apply_portfolio_constraints(df)
        
        # Apply filters
        filtered_df = apply_filters(df, filters)
    
    # Check for SUPER EDGE signals
    super_edge_count = len(filtered_df[filtered_df['signal_strength'] == 'SUPER_EDGE'])
    if super_edge_count > 0:
        st.markdown(f"""
        <div class="super-edge-alert">
            ⭐ {super_edge_count} SUPER EDGE SIGNAL{'S' if super_edge_count > 1 else ''} DETECTED! ⭐
        </div>
        """, unsafe_allow_html=True)
    
    # Display metrics
    display_key_metrics(filtered_df)
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Trading Signals",
        "📈 Analytics",
        "🏆 Top Picks",
        "📉 Risk Analysis",
        "📚 Documentation"
    ])
    
    with tab1:
        st.header("📊 Trading Signals")
        
        # Quick filters
        col1, col2, col3 = st.columns(3)
        with col1:
            show_only = st.selectbox(
                "Show Only",
                ["All Signals", "SUPER EDGE", "EXPLOSIVE", "STRONG"],
                index=0
            )
        
        # Apply quick filter
        display_df = filtered_df.copy()
        if show_only != "All Signals":
            display_df = display_df[display_df['signal_strength'] == show_only]
        
        # Display signals table
        display_signals_table(display_df)
        
        # Export buttons
        col1, col2 = st.columns(2)
        with col1:
            if not display_df.empty:
                csv = display_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Signals (CSV)",
                    csv,
                    f"edge_signals_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv"
                )
    
    with tab2:
        st.header("📈 Analytics Dashboard")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # EDGE distribution
            fig_dist = create_edge_distribution_chart(filtered_df)
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            # Sector performance
            fig_sector = create_sector_performance_chart(filtered_df)
            st.plotly_chart(fig_sector, use_container_width=True)
        
        # Signal scatter
        st.subheader("Signal Quality Analysis")
        fig_scatter = create_signal_scatter(filtered_df)
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with tab3:
        st.header("🏆 Top Picks by Category")
        
        # Display top picks for each signal strength
        for strength in ['SUPER_EDGE', 'EXPLOSIVE', 'STRONG']:
            top_picks = filtered_df[filtered_df['signal_strength'] == strength].head(5)
            
            if not top_picks.empty:
                st.subheader(f"{strength} Signals")
                
                for _, stock in top_picks.iterrows():
                    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                    
                    with col1:
                        st.write(f"**{stock['ticker']} - {stock['company_name']}**")
                        st.caption(f"{stock['sector']} | {stock['category']}")
                    
                    with col2:
                        st.metric("EDGE Score", f"{stock['edge_score']:.1f}")
                    
                    with col3:
                        st.metric("Price", f"₹{stock['price']:.2f}")
                    
                    with col4:
                        st.metric("R/R Ratio", f"{stock['risk_reward']:.2f}")
    
    with tab4:
        st.header("📉 Risk Analysis")
        
        # Portfolio allocation summary
        st.subheader("Portfolio Allocation")
        
        allocation_df = filtered_df[filtered_df['position_size'] > 0].copy()
        
        if not allocation_df.empty:
            # Sector allocation pie chart
            sector_allocation = allocation_df.groupby('sector')['position_size'].sum()
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=sector_allocation.index,
                values=sector_allocation.values,
                hole=0.4
            )])
            
            fig_pie.update_layout(
                title="Sector Allocation",
                height=400
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Risk metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_stop = allocation_df['stop_loss_pct'].mean() * 100
                st.metric("Avg Stop Loss", f"{avg_stop:.1f}%")
            
            with col2:
                avg_rr = allocation_df['risk_reward'].mean()
                st.metric("Avg Risk/Reward", f"{avg_rr:.2f}")
            
            with col3:
                max_drawdown = (allocation_df['position_size'] * allocation_df['stop_loss_pct']).sum() * 100
                st.metric("Max Portfolio Drawdown", f"{max_drawdown:.1f}%")
    
    with tab5:
        st.header("📚 Documentation")
        
        st.markdown("""
        ### How EDGE Protocol Works
        
        The EDGE Protocol uses a multi-factor scoring system to identify high-probability trading opportunities:
        
        #### 1. **Volume Analysis**
        - Tracks volume acceleration (30d vs 90d vs 180d)
        - Monitors relative volume (RVOL)
        - Identifies accumulation patterns
        
        #### 2. **Momentum Scoring**
        - Short-term momentum (1-7 days)
        - Medium-term momentum (30 days)
        - Trend alignment with moving averages
        
        #### 3. **Quality Assessment**
        - EPS growth and acceleration
        - Valuation metrics (PE ratio)
        - Long-term performance
        
        #### 4. **Value Analysis**
        - Distance from 52-week high/low
        - Risk/reward potential
        - Entry timing optimization
        
        ### Signal Classifications
        
        - **SUPER EDGE (90+)**: Highest conviction trades with multiple confirmations
        - **EXPLOSIVE (80-90)**: Strong momentum with volume confirmation
        - **STRONG (70-80)**: Solid opportunities with good risk/reward
        - **MODERATE (50-70)**: Developing opportunities to watch
        - **WATCH (30-50)**: Early stage signals requiring confirmation
        
        ### Risk Management
        
        - Dynamic position sizing based on signal strength
        - Volatility-adjusted stop losses
        - Sector diversification limits
        - Maximum portfolio exposure controls
        
        ### Best Practices
        
        1. **Never risk more than 2% per trade**
        2. **Diversify across sectors**
        3. **Honor stop losses**
        4. **Take partial profits at Target 1**
        5. **Review positions daily**
        """)

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()
