import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import requests
import math
import warnings
import re
from functools import lru_cache
from scipy import stats
from typing import Dict, List, Tuple, Optional
import time

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================
SHEET_ID = "1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk"
GID_WATCHLIST = "2026492216"
SHEET_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID_WATCHLIST}"

PAGE_TITLE = "EDGE Protocol – Ultimate Trading Intelligence"

PROFILE_PRESETS = {
    "Balanced": (0.40, 0.25, 0.20, 0.15),
    "Swing": (0.50, 0.30, 0.20, 0.00),
    "Positional": (0.40, 0.25, 0.25, 0.10),
    "Momentum‑only": (0.60, 0.30, 0.10, 0.00),
    "Breakout": (0.45, 0.40, 0.15, 0.00),
    "Long‑Term": (0.25, 0.25, 0.15, 0.35),
}

EDGE_THRESHOLDS = {
    "SUPER_EDGE": 90,
    "EXPLOSIVE": 85,
    "STRONG": 70,
    "MODERATE": 50,
    "WATCH": 0
}

MIN_STOCKS_PER_SECTOR = 4

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
    rolling_std = price.rolling(20, min_periods=1).std()
    return rolling_std.fillna(rolling_std.mean()) * math.sqrt(2)

# ============================================================================
# DATA LOADING
# ============================================================================
@st.cache_data(ttl=300)
def load_sheet() -> pd.DataFrame:
    """Load and clean data from Google Sheets"""
    try:
        resp = requests.get(SHEET_URL, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(io.BytesIO(resp.content))
        
        # Standardize column names
        df.columns = (df.columns.str.strip()
                     .str.lower()
                     .str.replace("%", "pct")
                     .str.replace(" ", "_"))
        
        # Define numeric columns
        numeric_cols = [
            'market_cap', 'volume_1d', 'volume_7d', 'volume_30d', 'volume_90d', 'volume_180d',
            'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d',
            'vol_ratio_1d_180d', 'vol_ratio_7d_180d', 'vol_ratio_30d_180d', 'vol_ratio_90d_180d',
            'price', 'ret_1d', 'low_52w', 'high_52w', 'from_low_pct', 'from_high_pct',
            'sma_20d', 'sma_50d', 'sma_200d', 'ret_3d', 'ret_7d', 'ret_30d', 'ret_3m',
            'ret_6m', 'ret_1y', 'ret_3y', 'ret_5y', 'rvol', 'prev_close', 'pe',
            'eps_current', 'eps_last_qtr', 'eps_change_pct', 'year'
        ]
        
        # Clean and convert numeric columns
        for col in numeric_cols:
            if col in df.columns:
                if col == 'market_cap':
                    df[col] = df[col].astype(str).apply(parse_market_cap)
                else:
                    df[col] = pd.to_numeric(
                        df[col].astype(str).str.replace(r"[₹,$€£%,]", "", regex=True)
                        .replace(["nan", "", "-"], np.nan),
                        errors="coerce"
                    )
        
        # Fill critical columns
        df['price'] = df['price'].fillna(df.get('prev_close', 1)).fillna(1)
        df['volume_1d'] = df.get('volume_1d', pd.Series(0)).fillna(0).astype(int)
        df['rvol'] = df.get('rvol', pd.Series(1)).fillna(1)
        
        # Ensure string columns
        for col in ['ticker', 'company_name', 'sector', 'category']:
            if col in df.columns:
                df[col] = df[col].astype(str).fillna("Unknown")
        
        # Calculate derived columns
        df["atr_20"] = calc_atr20(df["price"])
        df["rs_volume_30d"] = df.get("volume_30d", 0) * df["price"]
        
        # Calculate volume acceleration
        df = calculate_volume_acceleration(df)
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

def parse_market_cap(val):
    """Parse market cap values with units"""
    if pd.isna(val) or not isinstance(val, str):
        return np.nan
    
    val_str = re.sub(r"[₹,$€£%,]", "", val.strip())
    
    multipliers = {
        'Cr': 1e7, 'cr': 1e7,
        'L': 1e5, 'l': 1e5,
        'K': 1e3, 'k': 1e3,
        'M': 1e6, 'm': 1e6,
        'B': 1e9, 'b': 1e9
    }
    
    for suffix, mult in multipliers.items():
        if suffix in val_str:
            try:
                return float(val_str.replace(suffix, '').strip()) * mult
            except:
                return np.nan
    
    try:
        return float(val_str)
    except:
        return np.nan

# ============================================================================
# VOLUME ACCELERATION
# ============================================================================
def calculate_volume_acceleration(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate volume acceleration metrics"""
    df = df.copy()
    
    # Calculate volume acceleration
    vol_30_90 = df.get('vol_ratio_30d_90d', 0)
    vol_30_180 = df.get('vol_ratio_30d_180d', 0)
    
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
# PATTERN DETECTION
# ============================================================================
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

def detect_pre_earnings_accumulation(row: pd.Series) -> dict:
    """Pattern 6: Unusual volume with EPS momentum"""
    score = 0
    signals = []
    
    eps_current = row.get('eps_current', 0)
    eps_last = row.get('eps_last_qtr', 0)
    eps_change = row.get('eps_change_pct', 0)
    vol_7d_90d = row.get('vol_ratio_7d_90d', 0)
    from_high = row.get('from_high_pct', 0)
    
    # EPS momentum
    if eps_last > 0 and eps_current > 0:
        eps_accel = safe_divide((eps_current - eps_last), eps_last) * 100
        if eps_accel > 10:
            score += 35
            signals.append(f"EPS accel: {eps_accel:.0f}%")
    
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
    
    # Earnings growth bonus
    if eps_change > 20:
        score = min(score * 1.2, 100)
        signals.append("Strong EPS growth")
    
    return {
        'pattern': 'Pre-Earnings Accumulation',
        'score': score,
        'signals': signals,
        'target': row.get('price', 0) * 1.12
    }

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
    vol_ratio = row.get('vol_ratio_30d_90d', 0)
    ret_30d = row.get('ret_30d', 0)
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
# SCORING FUNCTIONS
# ============================================================================
def score_vol_accel(row: pd.Series) -> float:
    """Enhanced Volume Acceleration scoring"""
    vol_30_90 = row.get("vol_ratio_30d_90d", 0)
    vol_30_180 = row.get("vol_ratio_30d_180d", 0)
    
    if pd.isna(vol_30_90) or pd.isna(vol_30_180):
        return 50  # Default neutral score
    
    delta = vol_30_90 - vol_30_180
    base_score = 50 + (delta * 2)  # More sensitive scaling
    base_score = np.clip(base_score, 0, 100)
    
    # RVOL bonus
    rvol = row.get("rvol", 1)
    if rvol > 2.0 and delta > 20:
        base_score = min(base_score * 1.5, 100)
    elif rvol > 1.5:
        base_score = min(base_score * 1.2, 100)
    
    return base_score

def score_momentum(row: pd.Series, df: pd.DataFrame) -> float:
    """Momentum scoring with consistency check"""
    ret_1d = row.get('ret_1d', 0)
    ret_3d = row.get('ret_3d', 0)
    ret_7d = row.get('ret_7d', 0)
    ret_30d = row.get('ret_30d', 0)
    
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

def score_risk_reward(row: pd.Series) -> float:
    """Risk/Reward scoring"""
    price = row.get("price", 1)
    high_52w = row.get("high_52w", price)
    low_52w = row.get("low_52w", price)
    
    upside = safe_divide(high_52w - price, price) * 100
    downside = safe_divide(price - low_52w, price) * 100
    
    # Risk/Reward ratio
    rr_ratio = safe_divide(upside, downside + 1)
    base_score = min(rr_ratio * 20, 100)
    
    # Quality bonus
    ret_3y = row.get("ret_3y", 0)
    ret_1y = row.get("ret_1y", 0)
    if ret_3y > 300 and ret_1y < 20:  # Quality stock on sale
        base_score = min(base_score * 1.4, 100)
    
    return base_score

def score_fundamentals(row: pd.Series, df: pd.DataFrame) -> float:
    """Fundamentals scoring"""
    scores = []
    
    # EPS change
    eps_change = row.get("eps_change_pct", 0)
    if not pd.isna(eps_change):
        eps_score = 50 + (eps_change * 2)  # Scale around 50
        scores.append(np.clip(eps_score, 0, 100))
    
    # PE ratio
    pe = row.get("pe", 0)
    if not pd.isna(pe) and pe > 0:
        if pe <= 25:
            pe_score = 100 - (pe * 2)
        elif pe <= 50:
            pe_score = 50 - ((pe - 25) * 1)
        else:
            pe_score = 25
        scores.append(max(pe_score, 0))
    
    # EPS acceleration
    eps_current = row.get("eps_current", 0)
    eps_last = row.get("eps_last_qtr", 0)
    if eps_last > 0 and eps_current > 0:
        eps_accel = safe_divide(eps_current - eps_last, eps_last) * 100
        if eps_accel > 10:
            scores.append(min(eps_accel * 5, 100))
    
    return np.mean(scores) if scores else 50

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
# MAIN SCORING ENGINE
# ============================================================================
def compute_scores(df: pd.DataFrame, weights: Tuple[float, float, float, float]) -> pd.DataFrame:
    """Complete scoring with pattern detection"""
    df = df.copy()
    
    # Calculate individual scores
    with st.spinner("Calculating component scores..."):
        df["vol_score"] = df.apply(score_vol_accel, axis=1)
        df["mom_score"] = df.apply(score_momentum, axis=1, df=df)
        df["rr_score"] = df.apply(score_risk_reward, axis=1)
        df["fund_score"] = df.apply(score_fundamentals, axis=1, df=df)
    
    # Calculate EDGE score
    block_cols = ["vol_score", "mom_score", "rr_score", "fund_score"]
    
    # Weighted average with adaptive weighting
    df["EDGE"] = 0
    for idx in df.index:
        scores = df.loc[idx, block_cols]
        valid_mask = ~scores.isna()
        
        if valid_mask.sum() == 0:
            continue
            
        valid_weights = np.array(weights)[valid_mask]
        valid_scores = scores[valid_mask]
        
        # Normalize weights
        norm_weights = valid_weights / valid_weights.sum()
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
    df['position_size_pct'] = df['tag'].map(position_map)
    
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
        high_potential = df[df['EDGE'] >= 30].copy()
        
        pattern_results = []
        for idx, row in high_potential.iterrows():
            pattern_data = detect_all_patterns(row)
            
            # Store pattern data
            df.loc[idx, 'pattern_analysis'] = pattern_data
            df.loc[idx, 'top_pattern_name'] = pattern_data['top_pattern']['pattern'] if pattern_data['top_pattern'] else ""
            df.loc[idx, 'top_pattern_score'] = pattern_data['top_pattern']['score'] if pattern_data['top_pattern'] else 0
            df.loc[idx, 'pattern_confluence_score'] = pattern_data['confluence_score']
            df.loc[idx, 'vp_divergence_score'] = pattern_data['vp_divergence']
        
        # Fill missing pattern data for low EDGE stocks
        pattern_cols = ['pattern_analysis', 'top_pattern_name', 'top_pattern_score', 
                       'pattern_confluence_score', 'vp_divergence_score']
        
        for col in pattern_cols:
            if col not in df.columns:
                df[col] = None
        
        df['top_pattern_name'] = df['top_pattern_name'].fillna("")
        df['top_pattern_score'] = df['top_pattern_score'].fillna(0)
        df['pattern_confluence_score'] = df['pattern_confluence_score'].fillna(0)
        df['vp_divergence_score'] = df['vp_divergence_score'].fillna(0)
    
    # Additional indicators
    df['eps_qoq_acceleration'] = 0
    mask = df['eps_last_qtr'] > 0
    df.loc[mask, 'eps_qoq_acceleration'] = (
        (df.loc[mask, 'eps_current'] - df.loc[mask, 'eps_last_qtr']) / 
        df.loc[mask, 'eps_last_qtr'] * 100
    )
    
    df['quality_consolidation'] = (
        (df.get('ret_3y', 0) > 300) & 
        (df.get('ret_1y', 0) < 20) & 
        (df.get('from_high_pct', 0) >= -40) & 
        (df.get('from_high_pct', 0) <= -15)
    )
    
    df['momentum_aligned'] = (
        (df.get('ret_1d', 0) > 0) & 
        (df.get('ret_3d', 0) > df.get('ret_1d', 0)) & 
        (df.get('ret_7d', 0) > df.get('ret_3d', 0)) & 
        (df.get('ret_30d', 0) > 0)
    )
    
    # Tier classifications
    df['eps_tier'] = df['eps_current'].apply(get_eps_tier)
    df['price_tier'] = df['price'].apply(get_price_tier)
    
    return df

# ============================================================================
# HELPER FUNCTIONS
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
# VISUALIZATION FUNCTIONS
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
# MAIN UI
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
    
    # Load data
    df = load_sheet()
    
    if df.empty:
        st.error("❌ No data available. Please check the data source.")
        return
    
    # Apply filters
    if not show_smallcaps:
        df = df[~df["category"].str.contains("nano|micro", case=False, na=False)]
    
    # Remove low liquidity stocks
    if "rs_volume_30d" in df.columns:
        df = df[(df["rs_volume_30d"] >= 1e7) | df["rs_volume_30d"].isna()]
    
    # Process data
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
            avg_edge = df_filtered["EDGE"].mean()
            st.metric("Avg EDGE", f"{avg_edge:.1f}")
        
        # Filters
        st.markdown("### 🔍 Filters")
        filter_cols = st.columns(4)
        
        with filter_cols[0]:
            unique_tags = df_filtered['tag'].unique().tolist()
            selected_tags = st.multiselect("Classification", unique_tags, default=unique_tags)
        
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
            st.info("Volume acceleration data not available.")
    
    # Tab 5: Sector Heatmap
    with tabs[4]:
        st.header("🔥 Sector Heatmap")
        
        # Aggregate by sector
        sector_agg = df_scored.groupby('sector').agg({
            'EDGE': 'mean',
            'ticker': 'count',
            'is_super_edge': 'sum'
        }).reset_index()
        
        sector_agg.columns = ['sector', 'avg_edge', 'count', 'super_edge_count']
        sector_agg = sector_agg[sector_agg['count'] >= 3]  # Min 3 stocks
        
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
            st.info("Insufficient data for sector analysis.")
    
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
            st.subheader(f"{stock_data['company_name']} ({stock_data['ticker']})")
            
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
                st.metric("1Y Return", f"{stock_data.get('ret_1y', 0)*100:.0f}%")
                st.metric("3Y Return", f"{stock_data.get('ret_3y', 0)*100:.0f}%")
                st.metric("From High", f"{stock_data.get('from_high_pct', 0)*100:.1f}%")
            
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
                    
                    # Show top 3 patterns
                    for pattern in patterns[:3]:
                        if pattern['score'] > 50:
                            with st.expander(f"{pattern['pattern']} (Score: {pattern['score']:.0f})"):
                                for signal in pattern.get('signals', []):
                                    st.write(f"• {signal}")
                                if pattern.get('target'):
                                    target_pct = (pattern['target'] / stock_data['price'] - 1) * 100
                                    st.success(f"Pattern Target: ₹{pattern['target']:.2f} (+{target_pct:.1f}%)")
            
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
        
        # Summary stats
        st.subheader("📊 Summary Statistics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Stocks Loaded", len(df))
            st.metric("After Filters", len(df_scored))
            st.metric("High EDGE (>70)", (df_scored['EDGE'] > 70).sum())
        
        with col2:
            st.metric("Avg EDGE Score", f"{df_scored['EDGE'].mean():.1f}")
            st.metric("Avg RVOL", f"{df_scored.get('rvol', pd.Series([1])).mean():.2f}")
            st.metric("Patterns Detected", (df_scored['top_pattern_score'] > 0).sum())
        
        with col3:
            st.metric("Data Timestamp", pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'))
            st.metric("Profile", profile_name)
            st.metric("Min EDGE Filter", min_edge)
        
        # Data quality check
        st.subheader("🔍 Data Quality Check")
        
        critical_cols = ['price', 'volume_1d', 'vol_ratio_30d_90d', 'vol_ratio_30d_180d']
        quality_data = []
        
        for col in critical_cols:
            if col in df_scored.columns:
                non_null = df_scored[col].notna().sum()
                pct = non_null / len(df_scored) * 100
                quality_data.append({
                    'Column': col,
                    'Non-null': non_null,
                    'Coverage %': f"{pct:.1f}%",
                    'Status': '✅' if pct > 90 else '⚠️' if pct > 70 else '❌'
                })
        
        quality_df = pd.DataFrame(quality_data)
        st.dataframe(quality_df, use_container_width=True, hide_index=True)
        
        # Sample data
        st.subheader("📄 Sample Data (Top 10 by EDGE)")
        
        sample_cols = ['ticker', 'company_name', 'EDGE', 'tag', 'price', 'rvol', 
                      'volume_acceleration', 'top_pattern_name', 'top_pattern_score']
        sample_cols = [col for col in sample_cols if col in df_scored.columns]
        
        st.dataframe(
            df_scored.nlargest(10, 'EDGE')[sample_cols],
            use_container_width=True
        )
        
        # Export full data
        st.subheader("💾 Export Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv_full = df_scored.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Full Dataset",
                csv_full,
                f"edge_protocol_full_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                "text/csv"
            )
        
        with col2:
            # High EDGE only
            high_edge_df = df_scored[df_scored['EDGE'] >= 70]
            if not high_edge_df.empty:
                csv_high = high_edge_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "🔥 Download High EDGE Only",
                    csv_high,
                    f"edge_high_signals_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                    "text/csv"
                )

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    render_ui()
