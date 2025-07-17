#!/usr/bin/env python3
"""
EDGE Protocol Ultimate - Best of All Worlds Edition
==================================================
Combines the best features from all implementations:
- Bulletproof data parsing from Final version
- Advanced pattern detection from Alternative version  
- Superior UI/UX from Alternative version
- Comprehensive diagnostics from Final version
- Optimized caching from Alternative version

This is the definitive production version.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
from datetime import datetime
import warnings
import re
import io
import time
from typing import Dict, List, Tuple, Optional, Union, Any
from functools import lru_cache
import math
from scipy import stats

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Google Sheets Configuration
SHEET_CONFIG = {
    'SHEET_ID': '1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk',
    'GID': '2026492216',
    'CACHE_TTL': 300,
    'REQUEST_TIMEOUT': 30
}

# Trading Profiles (from Alternative)
PROFILE_PRESETS = {
    "Balanced": (0.40, 0.25, 0.20, 0.15),
    "Swing": (0.50, 0.30, 0.20, 0.00),
    "Positional": (0.40, 0.25, 0.25, 0.10),
    "Momentum-only": (0.60, 0.30, 0.10, 0.00),
    "Breakout": (0.45, 0.40, 0.15, 0.00),
    "Long-Term": (0.25, 0.25, 0.15, 0.35),
}

# EDGE Score Thresholds (Enhanced from Alternative)
EDGE_THRESHOLDS = {
    'SUPER_EDGE': 90,     # New tier from Alternative
    'EXPLOSIVE': 85,
    'STRONG': 70,
    'MODERATE': 50,
    'WATCH': 30,
}

# Position Sizing
POSITION_SIZES = {
    'SUPER_EDGE': 0.15,   # New tier
    'EXPLOSIVE': 0.10,
    'STRONG': 0.05,
    'MODERATE': 0.03,
    'WATCH': 0.01,
}

# ============================================================================
# UTILITY FUNCTIONS (from Alternative)
# ============================================================================

def safe_divide(a, b, default=0):
    """Safe division with default value for zero denominator"""
    return a / b if b != 0 else default

def winsorize_series(s: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    """Winsorize a pandas Series to cap outliers"""
    if s.empty or not pd.api.types.is_numeric_dtype(s):
        return s
    lo, hi = s.quantile([lower_q, upper_q])
    return s.clip(lo, hi)

# ============================================================================
# BULLETPROOF DATA PARSING (from Final)
# ============================================================================

def clean_numeric_series(series: pd.Series, col_name: str = "") -> pd.Series:
    """
    Bulletproof numeric parsing that handles ALL formats including:
    - Indian numbers with ₹, Cr, L
    - Percentages with % symbol
    - Numbers with commas
    - Hidden Unicode characters (THE KEY FIX)
    """
    s = series.astype(str)
    
    # CRITICAL: Remove hidden Unicode characters FIRST
    s = s.str.replace('\u00A0', ' ', regex=False)  # Non-breaking space
    s = s.str.replace('\u200b', '', regex=False)   # Zero-width space
    s = s.str.replace('\xa0', ' ', regex=False)    # Another non-breaking space
    s = s.str.replace('\u2212', '-', regex=False)  # Unicode minus
    
    # Remove currency symbols
    for symbol in ['₹', '$', '€', '£', 'Rs', 'Rs.', 'INR']:
        s = s.str.replace(symbol, '', regex=False)
    
    # Handle percentage
    is_percentage = s.str.contains('%', na=False).any()
    s = s.str.replace('%', '', regex=False)
    
    # Remove commas
    s = s.str.replace(',', '', regex=False)
    
    # Store original for unit detection
    original = s.copy()
    
    # Handle Indian number units
    cr_mask = original.str.upper().str.endswith('CR')
    l_mask = original.str.upper().str.endswith('L')
    k_mask = original.str.upper().str.endswith('K')
    
    # Remove units
    for unit in ['Cr', 'cr', 'CR', 'L', 'l', 'K', 'k', 'M', 'm', 'B', 'b']:
        s = s.str.replace(unit, '', regex=False)
    
    # Clean remaining
    s = s.str.strip()
    s = s.replace(['', '-', 'NA', 'N/A', 'na', 'n/a', 'null', 'None', '#N/A'], 'NaN')
    
    # Convert to numeric
    numeric_series = pd.to_numeric(s, errors='coerce')
    
    # Apply multipliers
    if cr_mask.any():
        numeric_series[cr_mask] = numeric_series[cr_mask] * 10000000
    if l_mask.any():
        numeric_series[l_mask] = numeric_series[l_mask] * 100000
    if k_mask.any():
        numeric_series[k_mask] = numeric_series[k_mask] * 1000
    
    return numeric_series

# ============================================================================
# DATA LOADING (Hybrid approach)
# ============================================================================

@st.cache_data(ttl=SHEET_CONFIG['CACHE_TTL'])
def load_data() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load and parse data with comprehensive diagnostics"""
    
    # Initialize diagnostics (from Final)
    diagnostics = {
        'timestamp': datetime.now(),
        'rows_loaded': 0,
        'rows_valid': 0,
        'data_quality_score': 0,
        'warnings': [],
        'parsing_stats': {},
        'column_coverage': {},
        'critical_columns_check': {}
    }
    
    try:
        # Fetch data
        url = f"https://docs.google.com/spreadsheets/d/{SHEET_CONFIG['SHEET_ID']}/export?format=csv&gid={SHEET_CONFIG['GID']}"
        response = requests.get(url, timeout=SHEET_CONFIG['REQUEST_TIMEOUT'])
        response.raise_for_status()
        
        # Check for HTML response
        if 'text/html' in response.headers.get('content-type', ''):
            raise ValueError("Access denied. Please make the Google Sheet public")
        
        # Load CSV
        df = pd.read_csv(io.StringIO(response.text))
        diagnostics['rows_loaded'] = len(df)
        
        # Clean column names
        df.columns = df.columns.str.strip().str.lower()
        df.columns = [re.sub(r'[^\w\s]', '', col).replace(' ', '_').strip('_') for col in df.columns]
        
        # Parse all numeric columns using bulletproof parser
        numeric_columns = [
            'price', 'prev_close', 'low_52w', 'high_52w', 'sma_20d', 'sma_50d', 'sma_200d',
            'volume_1d', 'volume_7d', 'volume_30d', 'volume_90d', 'volume_180d',
            'ret_1d', 'ret_3d', 'ret_7d', 'ret_30d', 'ret_3m', 'ret_6m', 'ret_1y', 'ret_3y', 'ret_5y',
            'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d',
            'vol_ratio_1d_180d', 'vol_ratio_7d_180d', 'vol_ratio_30d_180d', 'vol_ratio_90d_180d',
            'pe', 'eps_current', 'eps_last_qtr', 'eps_change_pct',
            'from_low_pct', 'from_high_pct', 'rvol', 'market_cap', 'year'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = clean_numeric_series(df[col], col)
                diagnostics['parsing_stats'][col] = df[col].notna().sum()
        
        # Ensure text columns
        for col in ['ticker', 'company_name', 'sector', 'category']:
            if col in df.columns:
                df[col] = df[col].astype(str).replace('nan', '').str.strip()
        
        # Calculate derived columns
        df = calculate_derived_columns(df, diagnostics)
        
        # Basic validation
        if 'ticker' in df.columns:
            df = df[df['ticker'].notna() & (df['ticker'] != '')]
        if 'price' in df.columns:
            df = df[(df['price'] > 0) | df['price'].isna()]
        
        diagnostics['rows_valid'] = len(df)
        
        # Calculate data quality
        essential_cols = ['ticker', 'price', 'volume_1d', 'ret_7d', 'vol_ratio_30d_90d', 'vol_ratio_30d_180d']
        for col in essential_cols:
            if col in df.columns:
                coverage = (df[col].notna().sum() / len(df)) * 100
                diagnostics['column_coverage'][col] = coverage
        
        avg_coverage = np.mean(list(diagnostics['column_coverage'].values()))
        diagnostics['data_quality_score'] = avg_coverage
        
        return df, diagnostics
        
    except Exception as e:
        diagnostics['warnings'].append(str(e))
        return pd.DataFrame(), diagnostics

def calculate_derived_columns(df: pd.DataFrame, diagnostics: Dict) -> pd.DataFrame:
    """Calculate essential derived columns"""
    
    # Volume Acceleration (with validation from Final)
    if all(col in df.columns for col in ['vol_ratio_30d_90d', 'vol_ratio_30d_180d']):
        if pd.api.types.is_numeric_dtype(df['vol_ratio_30d_90d']) and pd.api.types.is_numeric_dtype(df['vol_ratio_30d_180d']):
            df['volume_acceleration'] = df['vol_ratio_30d_90d'] - df['vol_ratio_30d_180d']
            df['volume_acceleration'] = df['volume_acceleration'].clip(-500, 500)
            df['has_volume_acceleration'] = True
            
            # Volume classification (from Alternative)
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
        else:
            df['volume_acceleration'] = 0
            df['has_volume_acceleration'] = False
            df['volume_classification'] = "Unknown"
    else:
        df['volume_acceleration'] = 0
        df['has_volume_acceleration'] = False
        df['volume_classification'] = "Unknown"
    
    # Additional calculations (from Alternative)
    df["atr_20"] = df["price"].rolling(20, min_periods=1).std().fillna(df["price"].std()) * math.sqrt(2)
    df["rs_volume_30d"] = df.get("volume_30d", 0) * df["price"]
    
    # Price position
    if all(col in df.columns for col in ['price', 'low_52w', 'high_52w']):
        price_range = (df['high_52w'] - df['low_52w']).replace(0, 1)
        df['price_position'] = ((df['price'] - df['low_52w']) / price_range * 100).fillna(50)
    
    return df

# ============================================================================
# ADVANCED PATTERN DETECTION (from Alternative)
# ============================================================================

def detect_accumulation_under_resistance(row: pd.Series) -> dict:
    """Pattern 1: Volume explodes but price stays flat near resistance"""
    score = 0
    signals = []
    
    vol_ratio = row.get('vol_ratio_30d_90d', 0)
    ret_30d = row.get('ret_30d', 0)
    from_high = row.get('from_high_pct', -100)
    rvol = row.get('rvol', 1)
    
    if vol_ratio > 50:
        score += 40
        signals.append(f"Volume +{vol_ratio:.0f}%")
    elif vol_ratio > 30:
        score += 25
        signals.append(f"Volume +{vol_ratio:.0f}%")
    
    if abs(ret_30d) < 5:
        score += 30
        signals.append(f"Price flat ({ret_30d:.1f}%)")
    
    if -10 <= from_high <= 0:
        score += 30
        signals.append("At 52w high resistance")
    elif -20 <= from_high < -10:
        score += 20
        signals.append("Near resistance")
    
    if rvol > 2.0:
        score = min(score * 1.2, 100)
        signals.append(f"RVOL: {rvol:.1f}x")
    
    return {
        'pattern': 'Accumulation Under Resistance',
        'score': score,
        'signals': signals,
        'target': row.get('high_52w', row.get('price', 0)) * 1.05
    }

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
    
    if vol_ratio > 30:
        score += 35
        signals.append(f"Volume +{vol_ratio:.0f}%")
    elif vol_ratio > 15:
        score += 20
        signals.append(f"Volume +{vol_ratio:.0f}%")
    
    if abs(ret_7d) < 5 and abs(ret_30d) < 10:
        score += 35
        signals.append("Tight consolidation")
    
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

def detect_absorption_pattern(row: pd.Series) -> dict:
    """Pattern 3: High RVOL with price stability"""
    score = 0
    signals = []
    
    rvol = row.get('rvol', 1)
    ret_7d = row.get('ret_7d', 0)
    vol_1d_90d = row.get('vol_ratio_1d_90d', 0)
    vol_7d_90d = row.get('vol_ratio_7d_90d', 0)
    vol_30d_90d = row.get('vol_ratio_30d_90d', 0)
    
    if rvol > 2.5:
        score += 50
        signals.append(f"Extreme RVOL: {rvol:.1f}x")
    elif rvol > 1.5:
        score += 30
        signals.append(f"High RVOL: {rvol:.1f}x")
    
    positive_ratios = sum([1 for r in [vol_1d_90d, vol_7d_90d, vol_30d_90d] if r > 0])
    if positive_ratios == 3:
        score += 30
        signals.append("Sustained volume increase")
    
    if abs(ret_7d) < 3:
        score += 20
        signals.append("Price absorbed")
    
    return {
        'pattern': 'Absorption Pattern',
        'score': score,
        'signals': signals,
        'target': row.get('price', 0) * 1.10
    }

def detect_failed_breakdown_reversal(row: pd.Series) -> dict:
    """Pattern 4: Near 52w low but volume accelerating"""
    score = 0
    signals = []
    
    from_low = row.get('from_low_pct', 100)
    vol_accel = row.get('volume_acceleration', 0)
    ret_7d = row.get('ret_7d', 0)
    ret_3y = row.get('ret_3y', 0)
    
    if from_low < 10:
        score += 40
        signals.append(f"Near 52w low (+{from_low:.0f}%)")
    elif from_low < 20:
        score += 25
        signals.append(f"Recent low (+{from_low:.0f}%)")
    
    if vol_accel > 20:
        score += 40
        signals.append(f"Vol accel: {vol_accel:.0f}%")
    elif vol_accel > 10:
        score += 25
        signals.append(f"Vol accel: {vol_accel:.0f}%")
    
    if ret_7d > 0:
        score += 20
        signals.append("Momentum positive")
    
    if ret_3y > 200:
        score = min(score * 1.2, 100)
        signals.append("Quality reversal")
    
    return {
        'pattern': 'Failed Breakdown Reversal',
        'score': score,
        'signals': signals,
        'target': row.get('sma_50d', row.get('price', 0) * 1.15)
    }

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
    
    sma_count = sum([1 for sma in [sma_20d, sma_50d, sma_200d] if price > sma])
    if sma_count == 3:
        score += 40
        signals.append("Above all SMAs")
    elif sma_count == 2:
        score += 25
        signals.append("Above 2 SMAs")
    
    if 10 <= vol_ratio <= 30:
        score += 30
        signals.append("Gradual vol increase")
    
    if 5 <= ret_30d <= 15:
        score += 30
        signals.append("Steady climb")
    
    return {
        'pattern': 'Stealth Breakout',
        'score': score,
        'signals': signals,
        'target': row.get('high_52w', price * 1.20)
    }

def detect_pre_earnings_accumulation(row: pd.Series) -> dict:
    """Pattern 6: Unusual volume with EPS momentum"""
    score = 0
    signals = []
    
    eps_current = row.get('eps_current', 0)
    eps_last = row.get('eps_last_qtr', 0)
    eps_change = row.get('eps_change_pct', 0)
    vol_7d_90d = row.get('vol_ratio_7d_90d', 0)
    from_high = row.get('from_high_pct', 0)
    
    if eps_last > 0 and eps_current > 0:
        eps_accel = safe_divide((eps_current - eps_last), eps_last) * 100
        if eps_accel > 10:
            score += 35
            signals.append(f"EPS accel: {eps_accel:.0f}%")
    
    if vol_7d_90d > 50:
        score += 35
        signals.append("Recent vol spike")
    elif vol_7d_90d > 25:
        score += 20
        signals.append("Higher recent vol")
    
    if -20 <= from_high <= -5:
        score += 30
        signals.append("Accumulation zone")
    
    if eps_change > 20:
        score = min(score * 1.2, 100)
        signals.append("Strong EPS growth")
    
    return {
        'pattern': 'Pre-Earnings Accumulation',
        'score': score,
        'signals': signals,
        'target': row.get('price', 0) * 1.12
    }

@lru_cache(maxsize=1000)
def detect_all_patterns(ticker: str, vol_ratio: float, ret_7d: float, ret_30d: float, 
                        from_high: float, from_low: float, rvol: float, price: float,
                        eps_current: float, eps_last: float) -> dict:
    """Run all pattern detections with caching"""
    
    # Create row dict for pattern functions
    row_data = {
        'ticker': ticker,
        'vol_ratio_30d_90d': vol_ratio,
        'ret_7d': ret_7d,
        'ret_30d': ret_30d,
        'from_high_pct': from_high,
        'from_low_pct': from_low,
        'rvol': rvol,
        'price': price,
        'eps_current': eps_current,
        'eps_last_qtr': eps_last
    }
    row = pd.Series(row_data)
    
    patterns = [
        detect_accumulation_under_resistance(row),
        detect_coiled_spring(row),
        detect_absorption_pattern(row),
        detect_failed_breakdown_reversal(row),
        detect_stealth_breakout(row),
        detect_pre_earnings_accumulation(row)
    ]
    
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
    vp_divergence = safe_divide(vol_ratio, abs(ret_30d) + 0.01) * 2
    vp_divergence = min(vp_divergence, 100)
    
    return {
        'patterns': patterns,
        'top_pattern': patterns[0] if patterns else None,
        'confluence_score': confluence_score,
        'confluence_signals': confluence_signals,
        'vp_divergence': vp_divergence,
        'high_score_patterns': high_score_patterns
    }

# ============================================================================
# SUPER EDGE DETECTION (from Alternative)
# ============================================================================

def detect_super_edge(row: pd.Series) -> bool:
    """Detect SUPER EDGE conditions"""
    conditions_met = 0
    
    # Check each condition
    if row.get("rvol", 1) > 2.0:
        conditions_met += 1
    
    if row.get("volume_acceleration", 0) > 30:
        conditions_met += 1
    
    eps_current = row.get("eps_current", 0)
    eps_last = row.get("eps_last_qtr", 0)
    if eps_last > 0 and eps_current > 0:
        if safe_divide(eps_current - eps_last, eps_last) > 0.10:
            conditions_met += 1
    
    from_high = row.get("from_high_pct", 0)
    if -30 <= from_high <= -15:
        conditions_met += 1
    
    # Momentum alignment
    if (row.get('ret_1d', 0) > 0 and 
        row.get('ret_3d', 0) > row.get('ret_1d', 0) and
        row.get('ret_7d', 0) > row.get('ret_3d', 0) and 
        row.get('ret_30d', 0) > 0):
        conditions_met += 1
    
    return conditions_met >= 4

# ============================================================================
# EDGE SCORING ENGINE (Hybrid)
# ============================================================================

def compute_edge_scores(df: pd.DataFrame, weights: Tuple[float, float, float, float]) -> pd.DataFrame:
    """Calculate EDGE scores with profile-based weights"""
    df = df.copy()
    
    # Individual component scores
    df['vol_score'] = df.apply(lambda row: score_volume_component(row), axis=1)
    df['mom_score'] = df.apply(lambda row: score_momentum_component(row), axis=1)
    df['rr_score'] = df.apply(lambda row: score_risk_reward_component(row), axis=1)
    df['fund_score'] = df.apply(lambda row: score_fundamentals_component(row), axis=1)
    
    # Calculate weighted EDGE score
    df['edge_score'] = (
        df['vol_score'] * weights[0] +
        df['mom_score'] * weights[1] +
        df['rr_score'] * weights[2] +
        df['fund_score'] * weights[3]
    )
    
    # Detect SUPER EDGE
    df['is_super_edge'] = df.apply(detect_super_edge, axis=1)
    
    # Boost SUPER EDGE scores
    super_edge_mask = df['is_super_edge']
    df.loc[super_edge_mask, 'edge_score'] = df.loc[super_edge_mask, 'edge_score'] * 1.1
    df['edge_score'] = df['edge_score'].clip(0, 100)
    
    # Classification
    conditions = [
        df['is_super_edge'] & (df['edge_score'] >= EDGE_THRESHOLDS['SUPER_EDGE']),
        df['edge_score'] >= EDGE_THRESHOLDS['EXPLOSIVE'],
        df['edge_score'] >= EDGE_THRESHOLDS['STRONG'],
        df['edge_score'] >= EDGE_THRESHOLDS['MODERATE'],
    ]
    choices = ['SUPER_EDGE', 'EXPLOSIVE', 'STRONG', 'MODERATE']
    df['signal_strength'] = np.select(conditions, choices, default='WATCH')
    
    # Position sizing
    df['position_size'] = df['signal_strength'].map(POSITION_SIZES).fillna(0)
    
    # Pattern detection for high-score stocks
    high_potential = df[df['edge_score'] >= 50].copy()
    
    # Initialize pattern columns
    df['pattern_data'] = None
    df['top_pattern_name'] = ""
    df['top_pattern_score'] = 0.0
    df['pattern_confluence_score'] = 0.0
    df['vp_divergence_score'] = 0.0
    
    for idx, row in high_potential.iterrows():
        pattern_data = detect_all_patterns(
            row['ticker'],
            row.get('vol_ratio_30d_90d', 0),
            row.get('ret_7d', 0),
            row.get('ret_30d', 0),
            row.get('from_high_pct', 0),
            row.get('from_low_pct', 0),
            row.get('rvol', 1),
            row.get('price', 0),
            row.get('eps_current', 0),
            row.get('eps_last_qtr', 0)
        )
        
        df.at[idx, 'pattern_data'] = pattern_data
        df.at[idx, 'top_pattern_name'] = pattern_data['top_pattern']['pattern'] if pattern_data['top_pattern'] else ""
        df.at[idx, 'top_pattern_score'] = float(pattern_data['top_pattern']['score']) if pattern_data['top_pattern'] else 0.0
        df.at[idx, 'pattern_confluence_score'] = float(pattern_data['confluence_score'])
        df.at[idx, 'vp_divergence_score'] = float(pattern_data['vp_divergence'])
    
    # Generate signal reasons
    df['signal_reason'] = df.apply(generate_signal_reason, axis=1)
    
    return df

def score_volume_component(row: pd.Series) -> float:
    """Enhanced volume scoring with acceleration focus"""
    vol_accel = row.get('volume_acceleration', 0)
    
    if row.get('has_volume_acceleration', False):
        base_score = 50 + (vol_accel * 2)
        base_score = np.clip(base_score, 0, 100)
        
        # RVOL bonus
        rvol = row.get('rvol', 1)
        if rvol > 2.0 and vol_accel > 20:
            base_score = min(base_score * 1.5, 100)
        elif rvol > 1.5:
            base_score = min(base_score * 1.2, 100)
    else:
        # Fallback
        base_score = 50
        if 'rvol' in row:
            base_score = (row['rvol'] * 20).clip(0, 100)
    
    return base_score

def score_momentum_component(row: pd.Series) -> float:
    """Momentum scoring with consistency check"""
    ret_1d = row.get('ret_1d', 0)
    ret_3d = row.get('ret_3d', 0)
    ret_7d = row.get('ret_7d', 0)
    ret_30d = row.get('ret_30d', 0)
    
    short_term = (ret_1d + ret_3d + ret_7d) / 3
    mid_term = ret_30d
    
    momentum = (short_term * 0.6 + mid_term * 0.4)
    base_score = 50 + (momentum * 5)
    base_score = np.clip(base_score, 0, 100)
    
    # Consistency bonus
    if ret_1d > 0 and ret_3d > ret_1d and ret_7d > ret_3d and ret_30d > 0:
        base_score = min(base_score * 1.3, 100)
    
    return base_score

def score_risk_reward_component(row: pd.Series) -> float:
    """Risk/Reward scoring"""
    price = row.get('price', 1)
    high_52w = row.get('high_52w', price)
    low_52w = row.get('low_52w', price)
    
    upside = safe_divide(high_52w - price, price) * 100
    downside = safe_divide(price - low_52w, price) * 100
    
    rr_ratio = safe_divide(upside, downside + 1)
    base_score = min(rr_ratio * 20, 100)
    
    # Quality bonus
    ret_3y = row.get('ret_3y', 0)
    ret_1y = row.get('ret_1y', 0)
    if ret_3y > 300 and ret_1y < 20:
        base_score = min(base_score * 1.4, 100)
    
    return base_score

def score_fundamentals_component(row: pd.Series) -> float:
    """Fundamentals scoring"""
    scores = []
    
    eps_change = row.get('eps_change_pct', 0)
    if not pd.isna(eps_change):
        eps_score = 50 + (eps_change * 2)
        scores.append(np.clip(eps_score, 0, 100))
    
    pe = row.get('pe', 0)
    if not pd.isna(pe) and pe > 0:
        if pe <= 25:
            pe_score = 100 - (pe * 2)
        elif pe <= 50:
            pe_score = 50 - ((pe - 25) * 1)
        else:
            pe_score = 25
        scores.append(max(pe_score, 0))
    
    return np.mean(scores) if scores else 50

def generate_signal_reason(row) -> str:
    """Generate human-readable signal explanation"""
    reasons = []
    
    # Volume signals
    if row.get('volume_acceleration', 0) > 30:
        reasons.append("🏦 Institutional loading")
    elif row.get('volume_acceleration', 0) > 20:
        reasons.append("📊 Heavy accumulation")
    elif row.get('volume_acceleration', 0) > 10:
        reasons.append("📈 Volume accelerating")
    
    # RVOL
    if row.get('rvol', 1) > 2:
        reasons.append(f"🔥 RVOL {row['rvol']:.1f}x")
    
    # Pattern
    if row.get('top_pattern_name'):
        pattern_map = {
            'Accumulation Under Resistance': "🎯 Accumulation pattern",
            'Coiled Spring': "🌀 Coiled spring ready",
            'Absorption Pattern': "🧲 Volume absorption",
            'Failed Breakdown Reversal': "🔄 Reversal pattern",
            'Stealth Breakout': "🚀 Stealth breakout",
            'Pre-Earnings Accumulation': "💰 Pre-earnings action"
        }
        if row['top_pattern_name'] in pattern_map:
            reasons.append(pattern_map[row['top_pattern_name']])
    
    # Momentum
    if row.get('ret_7d', 0) > 10:
        reasons.append(f"📈 +{row['ret_7d']:.0f}% momentum")
    
    # Confluence
    if row.get('pattern_confluence_score', 0) >= 85:
        reasons.append("⚡ Multiple patterns align")
    
    # SUPER EDGE
    if row.get('is_super_edge', False):
        reasons = ["⭐ SUPER EDGE SIGNAL ⭐"] + reasons
    
    # Default
    if not reasons:
        if row.get('edge_score', 0) > 70:
            reasons.append("✅ Strong setup")
        else:
            reasons.append("📊 Building momentum")
    
    return " | ".join(reasons[:2])

# ============================================================================
# VISUALIZATIONS (Best of both)
# ============================================================================

def create_volume_acceleration_map(df: pd.DataFrame) -> go.Figure:
    """Enhanced volume acceleration scatter plot"""
    if 'volume_acceleration' not in df.columns or df['volume_acceleration'].isna().all():
        return create_empty_chart("Volume acceleration data not available")
    
    plot_df = df[df['volume_acceleration'].notna() & (df['edge_score'] > 30)].nlargest(100, 'edge_score')
    
    if plot_df.empty:
        return create_empty_chart("No data for volume map")
    
    fig = go.Figure()
    
    # Color mapping
    colors = {
        'SUPER_EDGE': '#FFD700',
        'EXPLOSIVE': '#FF0000',
        'STRONG': '#FF6600',
        'MODERATE': '#FFA500',
        'WATCH': '#808080'
    }
    
    for signal in ['SUPER_EDGE', 'EXPLOSIVE', 'STRONG', 'MODERATE', 'WATCH']:
        signal_df = plot_df[plot_df['signal_strength'] == signal]
        if not signal_df.empty:
            hover_text = signal_df.apply(
                lambda row: f"<b>{row['ticker']}</b><br>" +
                           f"Vol Accel: {row['volume_acceleration']:.1f}%<br>" +
                           f"7d Return: {row.get('ret_7d', 0):.1f}%<br>" +
                           f"EDGE Score: {row['edge_score']:.1f}<br>" +
                           f"Signal: {row.get('signal_reason', '')}<br>" +
                           f"Pattern: {row.get('top_pattern_name', 'None')}",
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
    
    # Add zones
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    # Annotations
    fig.add_annotation(x=40, y=15, text="🚀 EXPLOSIVE ZONE", showarrow=False, font=dict(size=14, color="red"))
    fig.add_annotation(x=40, y=-10, text="🏦 STEALTH ACCUMULATION", showarrow=False, font=dict(size=14, color="green"))
    fig.add_annotation(x=-30, y=15, text="📉 PROFIT TAKING", showarrow=False, font=dict(size=14, color="orange"))
    fig.add_annotation(x=-30, y=-10, text="☠️ DANGER ZONE", showarrow=False, font=dict(size=14, color="gray"))
    
    fig.update_layout(
        title="Volume Acceleration Map - The EDGE",
        xaxis_title="Volume Acceleration %",
        yaxis_title="7-Day Return %",
        height=600,
        hovermode='closest'
    )
    
    return fig

def create_stock_radar_chart(row: pd.Series) -> go.Figure:
    """Create radar chart for individual stock analysis"""
    categories = ['Volume Accel', 'Momentum', 'Risk/Reward', 'Fundamentals']
    scores = [
        row.get('vol_score', 0),
        row.get('mom_score', 0),
        row.get('rr_score', 0),
        row.get('fund_score', 0)
    ]
    
    line_color = 'gold' if row.get('signal_strength') == 'SUPER_EDGE' else 'darkblue'
    
    fig = go.Figure(data=go.Scatterpolar(
        r=scores,
        theta=categories,
        fill='toself',
        name=row.get('company_name', 'Stock'),
        line_color=line_color,
        line_width=3
    ))
    
    title = f"EDGE Components - {row.get('company_name', '')} ({row.get('ticker', '')})"
    if row.get('signal_strength') == 'SUPER_EDGE':
        title += " ⭐ SUPER EDGE ⭐"
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        title=title,
        font_size=16
    )
    
    return fig

def create_sector_heatmap(df: pd.DataFrame) -> go.Figure:
    """Create sector heatmap using treemap"""
    if 'sector' not in df.columns or 'edge_score' not in df.columns:
        return create_empty_chart("Sector data not available")
    
    sector_agg = df.groupby('sector').agg({
        'edge_score': 'mean',
        'ticker': 'count',
        'is_super_edge': 'sum' if 'is_super_edge' in df.columns else lambda x: 0
    }).reset_index()
    
    sector_agg.columns = ['sector', 'avg_edge', 'count', 'super_edge_count']
    sector_agg = sector_agg[sector_agg['count'] >= 3]
    
    if sector_agg.empty:
        return create_empty_chart("Insufficient sector data")
    
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
# MAIN UI (Best of all versions)
# ============================================================================

def render_ui():
    """Main Streamlit UI combining best features"""
    st.set_page_config(
        page_title="EDGE Protocol Ultimate",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS (from Alternative)
    st.markdown("""
    <style>
    .super-edge-banner {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    .super-edge-text {
        color: #000;
        font-size: 24px;
        font-weight: bold;
    }
    .diagnostic-info {
        background: #f0f8ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #0066cc;
        margin: 1rem 0;
    }
    div[data-testid="metric-container"] {
        background-color: rgba(28, 131, 225, 0.1);
        border: 1px solid rgba(28, 131, 225, 0.2);
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("⚡ EDGE Protocol Ultimate")
    st.markdown("**Volume Acceleration + Pattern Detection + Momentum = Maximum Edge**")
    
    # Sidebar with profiles (from Alternative)
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Profile selection
        profile_name = st.radio("Trading Profile", list(PROFILE_PRESETS.keys()), index=0)
        weights = PROFILE_PRESETS[profile_name]
        
        st.markdown("---")
        st.subheader("🎯 Filters")
        
        min_edge = st.slider("Min EDGE Score", 0, 100, 30, 5)
        show_smallcaps = st.checkbox("Include small/micro caps", value=False)
        show_super_edge_only = st.checkbox("Show SUPER EDGE Only", value=False)
        
        st.markdown("---")
        st.markdown("### 📊 Active Weights")
        st.write(f"Volume: {weights[0]*100:.0f}%")
        st.write(f"Momentum: {weights[1]*100:.0f}%")
        st.write(f"Risk/Reward: {weights[2]*100:.0f}%")
        st.write(f"Fundamentals: {weights[3]*100:.0f}%")
    
    # Load data
    df, diagnostics = load_data()
    
    if df.empty:
        st.error("❌ Unable to load data")
        with st.expander("🔧 Diagnostics", expanded=True):
            st.json(diagnostics)
        st.stop()
    
    # Show diagnostics (from Final)
    with st.expander("📊 Data Health Check", expanded=False):
        quality_score = diagnostics.get('data_quality_score', 0)
        
        if quality_score >= 80:
            st.success(f"✅ Data Quality: EXCELLENT ({quality_score:.1f}%)")
        elif quality_score >= 60:
            st.warning(f"⚠️ Data Quality: MODERATE ({quality_score:.1f}%)")
        else:
            st.error(f"❌ Data Quality: POOR ({quality_score:.1f}%)")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows Loaded", diagnostics.get('rows_loaded', 0))
            st.metric("Rows Valid", diagnostics.get('rows_valid', 0))
        
        with col2:
            vol_accel_status = "✅ Active" if df.get('has_volume_acceleration', pd.Series([False])).any() else "❌ Unavailable"
            st.metric("Volume Acceleration", vol_accel_status)
        
        with col3:
            if diagnostics.get('warnings'):
                st.metric("⚠️ Warnings", len(diagnostics['warnings']))
    
    # Apply filters
    if not show_smallcaps:
        df = df[~df['category'].str.contains('nano|micro', case=False, na=False)]
    
    # Calculate scores
    with st.spinner("🧮 Computing EDGE scores..."):
        df_scored = compute_edge_scores(df, weights)
    
    # Apply score filter
    df_filtered = df_scored[df_scored['edge_score'] >= min_edge].copy()
    
    if show_super_edge_only:
        df_filtered = df_filtered[df_filtered['signal_strength'] == 'SUPER_EDGE']
    
    # SUPER EDGE Alert (from Alternative)
    super_edge_count = (df_filtered['signal_strength'] == 'SUPER_EDGE').sum()
    if super_edge_count > 0:
        st.markdown(f"""
        <div class="super-edge-banner">
            <div class="super-edge-text">
                ⭐ {super_edge_count} SUPER EDGE SIGNAL{'S' if super_edge_count > 1 else ''} DETECTED! ⭐<br>
                Maximum conviction opportunities!
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Signals", len(df_filtered))
    with col2:
        st.metric("SUPER EDGE", (df_filtered['signal_strength'] == 'SUPER_EDGE').sum())
    with col3:
        st.metric("EXPLOSIVE", (df_filtered['signal_strength'] == 'EXPLOSIVE').sum())
    with col4:
        avg_edge = df_filtered['edge_score'].mean()
        st.metric("Avg EDGE", f"{avg_edge:.1f}")
    with col5:
        patterns_found = (df_filtered['top_pattern_score'] > 0).sum()
        st.metric("Patterns", patterns_found)
    
    # Main tabs
    tabs = st.tabs([
        "📊 Signals Dashboard",
        "🗺️ Volume Map",
        "🎯 Pattern Analysis",
        "🔥 Sector Heatmap",
        "🔍 Deep Dive",
        "📈 Analytics",
        "📚 Help"
    ])
    
    # Tab 1: Signals Dashboard
    with tabs[0]:
        st.header("📊 Trading Signals")
        
        # Quick filters
        col1, col2, col3 = st.columns(3)
        with col1:
            signal_filter = st.multiselect(
                "Signal Type",
                ['SUPER_EDGE', 'EXPLOSIVE', 'STRONG', 'MODERATE', 'WATCH'],
                default=['SUPER_EDGE', 'EXPLOSIVE', 'STRONG']
            )
        
        with col2:
            if 'sector' in df_filtered.columns:
                sectors = sorted(df_filtered['sector'].dropna().unique())
                sector_filter = st.multiselect("Sectors", sectors)
        
        with col3:
            search = st.text_input("Search Ticker", "")
        
        # Apply filters
        display_df = df_filtered[df_filtered['signal_strength'].isin(signal_filter)]
        
        if 'sector' in df_filtered.columns and sector_filter:
            display_df = display_df[display_df['sector'].isin(sector_filter)]
        
        if search:
            display_df = display_df[display_df['ticker'].str.contains(search.upper(), na=False)]
        
        # Sort by EDGE score
        display_df = display_df.sort_values('edge_score', ascending=False)
        
        # Display columns
        cols = ['ticker', 'company_name', 'sector', 'signal_strength', 'edge_score', 
                'signal_reason', 'volume_acceleration', 'top_pattern_name', 'price', 
                'rvol', 'position_size']
        
        available_cols = [col for col in cols if col in display_df.columns]
        
        # Style the dataframe
        def highlight_signal(row):
            if row['signal_strength'] == 'SUPER_EDGE':
                return ['background-color: gold'] * len(row)
            elif row['signal_strength'] == 'EXPLOSIVE':
                return ['background-color: #ffcccc'] * len(row)
            return [''] * len(row)
        
        st.dataframe(
            display_df[available_cols].style
            .apply(highlight_signal, axis=1)
            .format({
                'edge_score': '{:.1f}',
                'volume_acceleration': '{:.1f}%',
                'price': '₹{:.2f}',
                'rvol': '{:.1f}x',
                'position_size': '{:.1%}'
            }),
            use_container_width=True,
            height=600
        )
        
        # Export
        if not display_df.empty:
            csv = display_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Export Signals",
                csv,
                f"edge_signals_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                "text/csv",
                type="primary"
            )
    
    # Tab 2: Volume Map
    with tabs[1]:
        st.header("🗺️ Volume Acceleration Map")
        
        fig_volume = create_volume_acceleration_map(df_filtered)
        st.plotly_chart(fig_volume, use_container_width=True)
        
        # Volume insights
        if 'volume_acceleration' in df_filtered.columns:
            col1, col2, col3 = st.columns(3)
            with col1:
                inst_loading = (df_filtered['volume_classification'] == 'Institutional Loading').sum()
                st.metric("🏦 Institutional Loading", inst_loading)
            with col2:
                heavy_accum = (df_filtered['volume_classification'] == 'Heavy Accumulation').sum()
                st.metric("📊 Heavy Accumulation", heavy_accum)
            with col3:
                distribution = (df_filtered['volume_classification'].str.contains('Distribution', na=False)).sum()
                st.metric("📉 Distribution", distribution)
    
    # Tab 3: Pattern Analysis
    with tabs[2]:
        st.header("🎯 Pattern Discovery")
        
        pattern_df = df_filtered[df_filtered['top_pattern_score'] > 0]
        
        if not pattern_df.empty:
            # Pattern stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Patterns Found", len(pattern_df))
            with col2:
                high_conf = (pattern_df['pattern_confluence_score'] >= 85).sum()
                st.metric("High Confluence", high_conf)
            with col3:
                avg_pattern_score = pattern_df['top_pattern_score'].mean()
                st.metric("Avg Pattern Score", f"{avg_pattern_score:.0f}")
            
            # Pattern distribution
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
            
            high_conf_df = pattern_df[pattern_df['pattern_confluence_score'] >= 70].head(10)
            
            if not high_conf_df.empty:
                pattern_cols = ['ticker', 'company_name', 'edge_score', 'top_pattern_name',
                               'pattern_confluence_score', 'vp_divergence_score']
                
                st.dataframe(
                    high_conf_df[pattern_cols].style.format({
                        'edge_score': '{:.1f}',
                        'pattern_confluence_score': '{:.0f}',
                        'vp_divergence_score': '{:.1f}'
                    }),
                    use_container_width=True
                )
        else:
            st.info("No patterns detected in current selection")
    
    # Tab 4: Sector Heatmap
    with tabs[3]:
        st.header("🔥 Sector Analysis")
        
        fig_sector = create_sector_heatmap(df_filtered)
        st.plotly_chart(fig_sector, use_container_width=True)
    
    # Tab 5: Deep Dive
    with tabs[4]:
        st.header("🔍 Stock Deep Dive")
        
        if not df_filtered.empty:
            # Stock selector prioritizing high scores
            sorted_tickers = df_filtered.sort_values('edge_score', ascending=False)['ticker'].unique()
            
            selected_ticker = st.selectbox(
                "Select Stock",
                sorted_tickers,
                format_func=lambda x: f"⭐ {x}" if x in df_filtered[df_filtered['signal_strength'] == 'SUPER_EDGE']['ticker'].values else x
            )
            
            stock_data = df_filtered[df_filtered['ticker'] == selected_ticker].iloc[0]
            
            # Display SUPER EDGE banner if applicable
            if stock_data['signal_strength'] == 'SUPER_EDGE':
                st.markdown("""
                <div style="background: gold; padding: 10px; border-radius: 5px; text-align: center;">
                    <h2 style="margin: 0;">⭐ SUPER EDGE SIGNAL ⭐</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Stock info
            st.subheader(f"{stock_data['company_name']} ({stock_data['ticker']})")
            st.write(f"**Signal:** {stock_data['signal_reason']}")
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("EDGE Score", f"{stock_data['edge_score']:.1f}")
                st.metric("Classification", stock_data['signal_strength'])
            
            with col2:
                st.metric("Price", f"₹{stock_data.get('price', 0):.2f}")
                st.metric("RVOL", f"{stock_data.get('rvol', 1):.1f}x")
            
            with col3:
                st.metric("Vol Acceleration", f"{stock_data.get('volume_acceleration', 0):.1f}%")
                st.metric("7D Return", f"{stock_data.get('ret_7d', 0):.1f}%")
            
            with col4:
                st.metric("Position Size", f"{stock_data.get('position_size', 0):.1%}")
                st.metric("Pattern Score", f"{stock_data.get('top_pattern_score', 0):.0f}")
            
            # Radar chart
            fig_radar = create_stock_radar_chart(stock_data)
            st.plotly_chart(fig_radar, use_container_width=True)
            
            # Pattern analysis
            if stock_data.get('pattern_data'):
                st.subheader("📊 Pattern Analysis")
                
                pattern_info = stock_data['pattern_data']
                if 'patterns' in pattern_info:
                    for pattern in pattern_info['patterns'][:3]:
                        if pattern['score'] > 50:
                            with st.expander(f"{pattern['pattern']} (Score: {pattern['score']:.0f})"):
                                for signal in pattern.get('signals', []):
                                    st.write(f"• {signal}")
                                if pattern.get('target'):
                                    target_pct = (pattern['target'] / stock_data['price'] - 1) * 100
                                    st.success(f"Pattern Target: ₹{pattern['target']:.2f} (+{target_pct:.1f}%)")
        else:
            st.info("No stocks available for deep dive")
    
    # Tab 6: Analytics
    with tabs[5]:
        st.header("📈 Market Analytics")
        
        # EDGE distribution
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df_scored['edge_score'],
            nbinsx=30,
            marker_color='rgba(255, 102, 0, 0.7)',
            name='All Stocks'
        ))
        
        for threshold, color in [('SUPER_EDGE', 'gold'), ('EXPLOSIVE', 'red'), 
                                 ('STRONG', 'orange'), ('MODERATE', 'yellow')]:
            fig.add_vline(x=EDGE_THRESHOLDS[threshold], line_dash="dash", 
                         line_color=color, annotation_text=threshold)
        
        fig.update_layout(
            title="EDGE Score Distribution",
            xaxis_title="EDGE Score",
            yaxis_title="Count",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Stocks Analyzed", len(df_scored))
            st.metric("With Patterns", (df_scored['top_pattern_score'] > 0).sum())
        
        with col2:
            st.metric("Avg EDGE Score", f"{df_scored['edge_score'].mean():.1f}")
            st.metric("High Quality (>70)", (df_scored['edge_score'] > 70).sum())
        
        with col3:
            if 'volume_acceleration' in df_scored.columns:
                st.metric("Avg Vol Acceleration", f"{df_scored['volume_acceleration'].mean():.1f}%")
            st.metric("Profile", profile_name)
    
    # Tab 7: Help
    with tabs[6]:
        st.header("📚 Understanding EDGE Protocol")
        
        st.markdown("""
        ### 🎯 What is EDGE Protocol Ultimate?
        
        This is the definitive version combining the best features from all implementations:
        - **Bulletproof data parsing** that handles all edge cases
        - **6 advanced patterns** with confluence scoring
        - **SUPER EDGE detection** for maximum conviction trades
        - **Profile-based scoring** for different trading styles
        - **Comprehensive diagnostics** for data quality monitoring
        
        ### 🌟 SUPER EDGE Signals
        
        SUPER EDGE signals require 4+ conditions:
        1. RVOL > 2.0x (High relative volume)
        2. Volume Acceleration > 30% (Institutional loading)
        3. EPS Acceleration > 10% (Fundamental momentum)
        4. Price -30% to -15% from high (Sweet spot entry)
        5. Momentum alignment (1d < 3d < 7d returns, all positive)
        
        ### 📊 Trading Profiles
        
        - **Balanced**: Equal focus on all factors
        - **Swing**: Volume and momentum focused
        - **Positional**: Adds fundamental weight
        - **Momentum-only**: Pure technical momentum
        - **Breakout**: Volume breakout focus
        - **Long-Term**: Fundamental value focus
        
        ### 🎯 Pattern Types
        
        1. **Accumulation Under Resistance**: Big volume, flat price near highs
        2. **Coiled Spring**: Tight range with increasing volume
        3. **Absorption Pattern**: High RVOL with price stability
        4. **Failed Breakdown Reversal**: Bounce from lows with volume
        5. **Stealth Breakout**: Quiet strength above SMAs
        6. **Pre-Earnings Accumulation**: Volume spike before earnings
        
        ### 💡 How to Use
        
        1. Select your trading profile based on style
        2. Focus on SUPER EDGE and EXPLOSIVE signals
        3. Check pattern confluence for confirmation
        4. Use the Deep Dive tab for detailed analysis
        5. Monitor Volume Map for emerging opportunities
        
        ### ⚠️ Risk Disclaimer
        
        This tool is for educational purposes only. Not financial advice.
        Always do your own research and use proper risk management.
        """)
    
    # Footer
    st.markdown("---")
    st.caption(f"""
    **EDGE Protocol Ultimate** | 
    Data Quality: {diagnostics.get('data_quality_score', 0):.1f}% | 
    Volume Acceleration: {'✅ Active' if df.get('has_volume_acceleration', pd.Series([False])).any() else '❌ Unavailable'} |
    Profile: {profile_name} |
    Last Updated: {diagnostics.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M')}
    
    ⚡ The definitive edge in market intelligence ⚡
    """)

# ============================================================================
# MAIN ENTRY
# ============================================================================

if __name__ == "__main__":
    render_ui()
