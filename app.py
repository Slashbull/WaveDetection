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
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore")

# Google Sheet Configuration
SHEET_ID = "1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk"
GID_WATCHLIST = "2026492216"
SHEET_URL = (
    f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID_WATCHLIST}"
)

# UI CONSTANTS
PAGE_TITLE = "EDGE Protocol – Volume‑Acceleration Intelligence (SUPER EDGE Enhanced)"

# Define weighting profiles for different trading styles
PROFILE_PRESETS = {
    "Balanced": (0.40, 0.25, 0.20, 0.15),
    "Swing": (0.50, 0.30, 0.20, 0.00),
    "Positional": (0.40, 0.25, 0.25, 0.10),
    "Momentum‑only": (0.60, 0.30, 0.10, 0.00),
    "Breakout": (0.45, 0.40, 0.15, 0.00),
    "Long‑Term": (0.25, 0.25, 0.15, 0.35),
}

# Enhanced EDGE Score thresholds - Added SUPER_EDGE
EDGE_THRESHOLDS = {
    "SUPER_EDGE": 90,  # NEW: Ultra-high conviction signals
    "EXPLOSIVE": 85,
    "STRONG": 70,
    "MODERATE": 50,
    "WATCH": 0
}

MIN_STOCKS_PER_SECTOR = 4
GLOBAL_BLOCK_COLS = ["vol_score", "mom_score", "rr_score", "fund_score"]

# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def winsorise_series(s: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    """Winsorises a pandas Series to cap outliers at specified quantiles."""
    if s.empty or not pd.api.types.is_numeric_dtype(s):
        return s
    lo, hi = s.quantile([lower_q, upper_q])
    return s.clip(lo, hi)

def calc_atr20(price: pd.Series) -> pd.Series:
    """Calculates a proxy for Average True Range (ATR) over 20 periods."""
    rolling_std = price.rolling(20).std()
    # Use bfill() instead of fillna(method="bfill") to avoid deprecation warning
    return rolling_std.bfill() * math.sqrt(2)

@lru_cache(maxsize=1)
def load_sheet() -> pd.DataFrame:
    """Loads data from the specified Google Sheet URL with all cleaning and conversions."""
    try:
        resp = requests.get(SHEET_URL, timeout=30)
        resp.raise_for_status()
        raw = pd.read_csv(io.BytesIO(resp.content))

        # Standardise column headers
        raw.columns = (
            raw.columns.str.strip()
            .str.lower()
            .str.replace("%", "pct")
            .str.replace(" ", "_")
        )

        df = raw.copy()

        # Define columns that are expected to be numeric
        numeric_cols = [
            'market_cap', 'volume_1d', 'volume_7d', 'volume_30d', 'volume_90d', 'volume_180d',
            'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d',
            'vol_ratio_1d_180d', 'vol_ratio_7d_180d', 'vol_ratio_30d_180d', 'vol_ratio_90d_180d',
            'price', 'ret_1d', 'low_52w', 'high_52w', 'from_low_pct', 'from_high_pct',
            'sma_20d', 'sma_50d', 'sma_200d', 'ret_3d', 'ret_7d', 'ret_30d', 'ret_3m',
            'ret_6m', 'ret_1y', 'ret_3y', 'ret_5y', 'rvol', 'prev_close', 'pe',
            'eps_current', 'eps_last_qtr', 'eps_change_pct', 'year'
        ]

        # Define columns that are percentages
        percentage_cols_to_normalize = [
            'ret_1d', 'from_low_pct', 'from_high_pct', 'ret_3d', 'ret_7d', 'ret_30d', 'ret_3m',
            'ret_6m', 'ret_1y', 'ret_3y', 'ret_5y', 'eps_change_pct',
            'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d',
            'vol_ratio_1d_180d', 'vol_ratio_7d_180d', 'vol_ratio_30d_180d', 'vol_ratio_90d_180d'
        ]

        # Helper function to parse market cap values
        def parse_market_cap_value(val):
            if pd.isna(val) or not isinstance(val, str):
                return np.nan
            val_str = val.strip()
            clean_val_str = re.sub(r"[₹,$€£%,]", "", val_str)
            
            multiplier = 1
            numeric_part = clean_val_str
            if 'Cr' in clean_val_str:
                numeric_part = clean_val_str.replace('Cr', '').strip()
                multiplier = 10**7
            elif 'L' in clean_val_str:
                numeric_part = clean_val_str.replace('L', '').strip()
                multiplier = 10**5
            elif 'K' in clean_val_str:
                numeric_part = clean_val_str.replace('K', '').strip()
                multiplier = 10**3
            elif 'M' in clean_val_str:
                numeric_part = clean_val_str.replace('M', '').strip()
                multiplier = 10**6
            elif 'B' in clean_val_str:
                numeric_part = clean_val_str.replace('B', '').strip()
                multiplier = 10**9
            
            try:
                return float(numeric_part) * multiplier
            except ValueError:
                return np.nan

        # Convert numeric columns
        for col in numeric_cols:
            if col in df.columns:
                s = df[col].astype(str)
                
                if col == 'market_cap':
                    df[col] = s.apply(parse_market_cap_value)
                else:
                    s = s.str.replace(r"[₹,$€£%,]", "", regex=True) 
                    s = s.replace({"nan": np.nan, "": np.nan, "-": np.nan})
                    df[col] = pd.to_numeric(s, errors="coerce")

        # Normalize percentage columns
        for col in percentage_cols_to_normalize:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                non_na_values = df[col].dropna()
                if not non_na_values.empty and non_na_values.abs().max() > 1 and non_na_values.abs().max() <= 1000:
                    df[col] = df[col] / 100.0

        # Winsorise numeric columns
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].apply(winsorise_series, axis=0)

        # Fill critical columns
        df['price'] = df['price'].fillna(df['prev_close']).fillna(1.0)
        df['prev_close'] = df['prev_close'].fillna(df['price']).fillna(1.0)
        df['volume_1d'] = df['volume_1d'].fillna(0).astype(int)
        df['volume_7d'] = df['volume_7d'].fillna(0)
        df['volume_30d'] = df['volume_30d'].fillna(0)
        df['volume_90d'] = df['volume_90d'].fillna(0)
        df['volume_180d'] = df['volume_180d'].fillna(0)
        df['rvol'] = df['rvol'].fillna(1.0)

        # Ensure string columns
        if 'sector' in df.columns:
            df['sector'] = df['sector'].astype(str).fillna("Unknown")
        if 'category' in df.columns:
            df['category'] = df['category'].astype(str).fillna("Unknown")

        # Derived columns
        df["atr_20"] = calc_atr20(df["price"])
        df["rs_volume_30d"] = df["volume_30d"] * df["price"]

        return df

    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.exception(e)
        return pd.DataFrame()

# ─────────────────────────────────────────────────────────────────────────────
# Sector statistics with fallback
# ─────────────────────────────────────────────────────────────────────────────

def sector_stats(df: pd.DataFrame, sector: str) -> Tuple[pd.Series, pd.Series, int]:
    """Calculates mean and standard deviation for a given sector."""
    sector_df = df[df["sector"] == sector]
    n = len(sector_df)

    if n < MIN_STOCKS_PER_SECTOR:
        sector_df = df
        n = len(sector_df)

    mean = sector_df.mean(numeric_only=True)
    std = sector_df.std(numeric_only=True).replace(0, 1e-6)
    return mean, std, n

# ─────────────────────────────────────────────────────────────────────────────
# ENHANCED Scoring blocks with new features
# ─────────────────────────────────────────────────────────────────────────────

def score_vol_accel(row: pd.Series) -> float:
    """Enhanced Volume Acceleration scoring with RVOL integration."""
    if pd.isna(row.get("vol_ratio_30d_90d")) or pd.isna(row.get("vol_ratio_30d_180d")):
        return np.nan

    delta = row["vol_ratio_30d_90d"] - row["vol_ratio_30d_180d"]
    pct = stats.norm.cdf(delta / 0.2) * 100

    # Pattern bonus for consolidation
    if (delta >= 0.20 and not pd.isna(row.get("from_high_pct")) and row["from_high_pct"] <= -0.10):
        pct = min(pct + 5, 100)

    # NEW: RVOL BONUS - This is HUGE!
    if not pd.isna(row.get("rvol")):
        if row["rvol"] > 2.0 and delta > 0.20:
            # INSTITUTIONAL URGENCY - Double the score impact!
            pct = min(pct * 1.5, 100)
        elif row["rvol"] > 1.5:
            pct = min(pct * 1.2, 100)

    return pct

def score_momentum(row: pd.Series, df: pd.DataFrame) -> float:
    """Enhanced momentum scoring with consistency check."""
    ret_cols = [c for c in df.columns if c.startswith("ret_") and c.endswith("d")]
    if not ret_cols:
        return np.nan

    mean, std, n = sector_stats(df, row["sector"])
    valid_ret_cols = [col for col in ret_cols if col in row and col in mean and col in std]
    if not valid_ret_cols:
        return np.nan

    valid_std = std[valid_ret_cols].replace(0, 1e-6)
    z_scores = (row[valid_ret_cols] - mean[valid_ret_cols]) / valid_std
    raw_momentum_score = z_scores.mean()
    base_score = np.clip(stats.norm.cdf(raw_momentum_score) * 100, 0, 100)

    # NEW: MOMENTUM CONSISTENCY CHECK
    if all(col in row for col in ['ret_1d', 'ret_3d', 'ret_7d', 'ret_30d']):
        if (not pd.isna(row['ret_1d']) and not pd.isna(row['ret_3d']) and 
            not pd.isna(row['ret_7d']) and not pd.isna(row['ret_30d'])):
            if (row['ret_1d'] > 0 and 
                row['ret_3d'] > row['ret_1d'] and 
                row['ret_7d'] > row['ret_3d'] and 
                row['ret_30d'] > 0):
                # PERFECT MOMENTUM ALIGNMENT - Big bonus!
                base_score = min(base_score * 1.3, 100)

    return base_score

def score_risk_reward(row: pd.Series) -> float:
    """Enhanced risk/reward scoring with quality filter."""
    required = ["price", "low_52w", "high_52w", "atr_20"]
    if any(pd.isna(row.get(c)) for c in required):
        return np.nan

    upside = row["high_52w"] - row["price"]
    downside = row["price"] - row["low_52w"]
    atr = row["atr_20"] if pd.notna(row["atr_20"]) and row["atr_20"] > 0 else 1
    rr_metric = (upside - downside) / atr
    base_score = np.clip(stats.norm.cdf(rr_metric / 4) * 100, 0, 100)

    # NEW: QUALITY FILTER USING 3Y/5Y RETURNS
    if not pd.isna(row.get("ret_3y")) and not pd.isna(row.get("ret_1y")):
        if row["ret_3y"] > 3.0 and row["ret_1y"] < 0.20:  # 300% in 3y, <20% in 1y
            # PROVEN WINNER taking a breather!
            base_score = min(base_score * 1.4, 100)

    return base_score

def score_fundamentals(row: pd.Series, df: pd.DataFrame) -> float:
    """Enhanced fundamentals scoring with EPS acceleration."""
    if pd.isna(row.get("eps_change_pct")) and pd.isna(row.get("pe")):
        return np.nan

    scores = []

    # Original EPS change score
    if not pd.isna(row.get("eps_change_pct")):
        eps_score = np.clip(row["eps_change_pct"], -0.50, 1.00)
        eps_score = (eps_score + 0.50) / 0.015
        scores.append(eps_score)

    # Original PE score
    if not pd.isna(row.get("pe")) and row["pe"] > 0:
        if row["pe"] <= 100:
            pe_score = 100 - (row["pe"] / 100 * 100)
        else:
            pe_score = 0
        scores.append(pe_score)

    # NEW: EPS MOMENTUM ACCELERATION
    if not pd.isna(row.get("eps_current")) and not pd.isna(row.get("eps_last_qtr")):
        if row["eps_last_qtr"] > 0:
            eps_qoq_acceleration = (row["eps_current"] - row["eps_last_qtr"]) / row["eps_last_qtr"]
            if eps_qoq_acceleration > 0.10:  # 10% QoQ acceleration
                accel_score = min(eps_qoq_acceleration * 200, 100)  # Scale to 0-100
                scores.append(accel_score)

    if not scores:
        return np.nan
    return np.mean(scores)

# ─────────────────────────────────────────────────────────────────────────────
# EXPLOSIVE PATTERN DETECTION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def detect_volume_price_divergence(row: pd.Series) -> float:
    """Calculate volume-price divergence score."""
    if pd.isna(row.get('vol_ratio_30d_90d')) or pd.isna(row.get('ret_30d')):
        return 0
    
    # Prevent division by zero
    price_move = abs(row['ret_30d']) + 0.01
    # Higher score when volume increases but price doesn't move much
    divergence = row['vol_ratio_30d_90d'] / price_move
    
    # Normalize to 0-100 scale
    return min(divergence * 2, 100)

def detect_accumulation_under_resistance(row: pd.Series) -> dict:
    """Pattern 1: Volume explodes but price stays flat near resistance."""
    score = 0
    signals = []
    
    # Check volume explosion (>50% increase)
    if not pd.isna(row.get('vol_ratio_30d_90d')) and row['vol_ratio_30d_90d'] > 0.50:
        score += 40
        signals.append(f"Volume up {row['vol_ratio_30d_90d']:.0%}")
    
    # Check price is flat (moved less than 5%)
    if not pd.isna(row.get('ret_30d')) and abs(row['ret_30d']) < 0.05:
        score += 30
        signals.append(f"Price flat ({row['ret_30d']:.1%})")
    
    # Check if near 52w high (potential resistance)
    if not pd.isna(row.get('from_high_pct')) and -0.10 <= row['from_high_pct'] <= 0:
        score += 30
        signals.append("Near 52w high resistance")
        
    # Bonus for high RVOL
    if not pd.isna(row.get('rvol')) and row['rvol'] > 2.0:
        score = min(score * 1.2, 100)
        signals.append(f"RVOL: {row['rvol']:.1f}x")
    
    return {
        'pattern': 'Accumulation Under Resistance',
        'score': score,
        'signals': signals,
        'target': row.get('high_52w', row.get('price', 0)) * 1.05  # 5% above resistance
    }

def detect_coiled_spring(row: pd.Series) -> dict:
    """Pattern 2: Volume up significantly but price in tight range."""
    score = 0
    signals = []
    
    # Volume increase check
    if not pd.isna(row.get('vol_ratio_30d_90d')) and row['vol_ratio_30d_90d'] > 0.30:
        score += 35
        signals.append(f"Volume +{row['vol_ratio_30d_90d']:.0%}")
    
    # Price in tight range (compare multiple timeframes)
    tight_range = True
    for col in ['ret_7d', 'ret_30d']:
        if not pd.isna(row.get(col)) and abs(row[col]) > 0.05:
            tight_range = False
            break
    
    if tight_range:
        score += 35
        signals.append("Tight consolidation")
    
    # Check if price above key SMAs (bullish bias)
    sma_score = 0
    for sma in ['sma_20d', 'sma_50d', 'sma_200d']:
        if not pd.isna(row.get(sma)) and not pd.isna(row.get('price')):
            if row['price'] > row[sma]:
                sma_score += 10
    score += sma_score
    if sma_score == 30:
        signals.append("Above all SMAs")
    
    # Bonus for decreasing ATR (tighter coil)
    if not pd.isna(row.get('atr_20')) and not pd.isna(row.get('price')):
        atr_pct = row['atr_20'] / row['price']
        if atr_pct < 0.02:  # Less than 2% ATR
            score = min(score * 1.1, 100)
            signals.append("Very tight ATR")
    
    return {
        'pattern': 'Coiled Spring',
        'score': score,
        'signals': signals,
        'target': row.get('price', 0) * 1.08  # 8% breakout target
    }

def detect_absorption_pattern(row: pd.Series) -> dict:
    """Pattern 3: High RVOL sustained over multiple periods."""
    score = 0
    signals = []
    
    # Check sustained high RVOL
    if not pd.isna(row.get('rvol')) and row['rvol'] > 1.5:
        score += 30
        signals.append(f"RVOL: {row['rvol']:.1f}x")
        
        # Extra points for very high RVOL
        if row['rvol'] > 2.5:
            score += 20
            signals.append("Extreme volume!")
    
    # Check if volume ratios are consistently positive
    vol_consistency = 0
    for col in ['vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d']:
        if not pd.isna(row.get(col)) and row[col] > 0:
            vol_consistency += 1
    
    if vol_consistency == 3:
        score += 30
        signals.append("Consistent volume across timeframes")
    
    # Check for price stability (absorption characteristic)
    if not pd.isna(row.get('ret_7d')) and abs(row['ret_7d']) < 0.03:
        score += 20
        signals.append("Price absorbed at levels")
    
    return {
        'pattern': 'Absorption Pattern',
        'score': score,
        'signals': signals,
        'target': row.get('price', 0) * 1.10  # 10% target after absorption
    }

def detect_failed_breakdown_reversal(row: pd.Series) -> dict:
    """Pattern 4: Near 52w low but volume accelerating."""
    score = 0
    signals = []
    
    # Check if near 52w low
    if not pd.isna(row.get('from_low_pct')) and row['from_low_pct'] < 0.10:
        score += 40
        signals.append(f"Near 52w low (+{row['from_low_pct']:.0%})")
    
    # Check volume acceleration
    if not pd.isna(row.get('volume_acceleration')) and row['volume_acceleration'] > 20:
        score += 40
        signals.append(f"Volume accel: {row['volume_acceleration']:.0f}%")
    
    # Check for positive momentum reversal
    if not pd.isna(row.get('ret_7d')) and row['ret_7d'] > 0:
        score += 20
        signals.append("Momentum turning positive")
    
    # Bonus for quality stocks in reversal
    if not pd.isna(row.get('ret_3y')) and row['ret_3y'] > 2.0:  # 200% in 3 years
        score = min(score * 1.2, 100)
        signals.append("Quality stock reversal")
    
    return {
        'pattern': 'Failed Breakdown Reversal',
        'score': score,
        'signals': signals,
        'target': row.get('sma_50d', row.get('price', 0) * 1.15)  # Target 50 SMA or 15% up
    }

def detect_stealth_breakout(row: pd.Series) -> dict:
    """Pattern 5: Quiet accumulation with gradual volume build."""
    score = 0
    signals = []
    
    # Check if above all SMAs (stealth strength)
    sma_count = 0
    for sma in ['sma_20d', 'sma_50d', 'sma_200d']:
        if not pd.isna(row.get(sma)) and not pd.isna(row.get('price')):
            if row['price'] > row[sma]:
                sma_count += 1
    
    if sma_count == 3:
        score += 40
        signals.append("Above all SMAs")
    
    # Check gradual volume build (positive but not explosive)
    if not pd.isna(row.get('vol_ratio_30d_90d')):
        if 0.10 <= row['vol_ratio_30d_90d'] <= 0.30:
            score += 30
            signals.append("Gradual volume increase")
    
    # Check steady price appreciation
    if not pd.isna(row.get('ret_30d')) and 0.05 <= row['ret_30d'] <= 0.15:
        score += 30
        signals.append("Steady price climb")
    
    # Bonus for low volatility breakout
    if not pd.isna(row.get('atr_20')) and not pd.isna(row.get('price')):
        if row['atr_20'] / row['price'] < 0.03:
            score = min(score * 1.1, 100)
            signals.append("Low volatility breakout")
    
    return {
        'pattern': 'Stealth Breakout',
        'score': score,
        'signals': signals,
        'target': row.get('high_52w', row.get('price', 0) * 1.20)  # Target 52w high or 20% up
    }

def detect_pre_earnings_accumulation(row: pd.Series) -> dict:
    """Pattern 6: Unusual volume with EPS momentum."""
    score = 0
    signals = []
    
    # Check EPS momentum
    if not pd.isna(row.get('eps_qoq_acceleration')) and row['eps_qoq_acceleration'] > 10:
        score += 35
        signals.append(f"EPS accel: {row['eps_qoq_acceleration']:.0f}%")
    
    # Check unusual volume in recent periods
    if not pd.isna(row.get('vol_ratio_7d_90d')) and row['vol_ratio_7d_90d'] > 0.50:
        score += 35
        signals.append("Unusual recent volume")
    
    # Check if price is in accumulation phase
    if not pd.isna(row.get('from_high_pct')) and -0.20 <= row['from_high_pct'] <= -0.05:
        score += 30
        signals.append("Accumulation zone")
    
    # Bonus for improving fundamentals
    if not pd.isna(row.get('eps_change_pct')) and row['eps_change_pct'] > 0.20:
        score = min(score * 1.2, 100)
        signals.append("Strong earnings growth")
    
    return {
        'pattern': 'Pre-Earnings Accumulation',
        'score': score,
        'signals': signals,
        'target': row.get('price', 0) * 1.12  # 12% earnings surprise target
    }

def calculate_pattern_confluence(patterns: list) -> dict:
    """Calculate confluence score when multiple patterns align."""
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
    
    # Calculate average target from all strong patterns
    if high_score_patterns:
        avg_target = np.mean([p['target'] for p in high_score_patterns])
    else:
        avg_target = 0
    
    return {
        'score': confluence_score,
        'pattern_count': len(high_score_patterns),
        'signals': confluence_signals,
        'target': avg_target,
        'patterns': [p['pattern'] for p in high_score_patterns]
    }

def detect_all_patterns(row: pd.Series) -> dict:
    """Detect all explosive patterns for a stock."""
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
    confluence = calculate_pattern_confluence(patterns)
    
    # Volume-Price Divergence
    vp_divergence = detect_volume_price_divergence(row)
    
    return {
        'patterns': patterns,
        'confluence': confluence,
        'vp_divergence': vp_divergence,
        'top_pattern': patterns[0] if patterns else None
    }

# ─────────────────────────────────────────────────────────────────────────────
# SUPER EDGE Detection Function
# ─────────────────────────────────────────────────────────────────────────────

def detect_super_edge(row: pd.Series) -> bool:
    """Detect SUPER EDGE conditions - the absolute best setups."""
    conditions_met = 0

    # 1. RVOL > 2.0 (unusual activity TODAY)
    if not pd.isna(row.get("rvol")) and row["rvol"] > 2.0:
        conditions_met += 1

    # 2. Volume Acceleration > 30%
    if not pd.isna(row.get("volume_acceleration")) and row["volume_acceleration"] > 30:
        conditions_met += 1

    # 3. EPS QoQ acceleration > 10%
    if (not pd.isna(row.get("eps_current")) and not pd.isna(row.get("eps_last_qtr")) 
        and row.get("eps_last_qtr", 0) > 0):
        eps_qoq = (row["eps_current"] - row["eps_last_qtr"]) / row["eps_last_qtr"]
        if eps_qoq > 0.10:
            conditions_met += 1

    # 4. From High between -15% to -30% (perfect entry zone)
    if not pd.isna(row.get("from_high_pct")):
        if -0.30 <= row["from_high_pct"] <= -0.15:
            conditions_met += 1

    # 5. Momentum consistency
    if all(not pd.isna(row.get(col)) for col in ['ret_1d', 'ret_3d', 'ret_7d', 'ret_30d']):
        if (row['ret_1d'] > 0 and row['ret_3d'] > row['ret_1d'] and 
            row['ret_7d'] > row['ret_3d'] and row['ret_30d'] > 0):
            conditions_met += 1

    # Need at least 4 out of 5 conditions for SUPER EDGE
    return conditions_met >= 4

# ─────────────────────────────────────────────────────────────────────────────
# Enhanced Edge score computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_scores(df: pd.DataFrame, weights: Tuple[float, float, float, float]) -> pd.DataFrame:
    """Enhanced EDGE score computation with SUPER EDGE detection."""
    df = df.copy()

    # Calculate individual component scores
    df["vol_score"] = df.apply(score_vol_accel, axis=1)
    df["mom_score"] = df.apply(score_momentum, axis=1, df=df)
    df["rr_score"] = df.apply(score_risk_reward, axis=1)
    df["fund_score"] = df.apply(score_fundamentals, axis=1, df=df)

    block_cols = ["vol_score", "mom_score", "rr_score", "fund_score"]

    # Adaptive weighting
    w_array = np.array(weights)
    out_scores = []
    for idx, row in df.iterrows():
        active_mask = ~row[block_cols].isna()
        if not active_mask.any():
            out_scores.append(np.nan)
            continue

        active_weights = w_array[active_mask]
        norm_w = active_weights / active_weights.sum()
        edge = (row[block_cols][active_mask] * norm_w).sum()
        out_scores.append(edge)

    df["EDGE"] = out_scores

    # Detect SUPER EDGE conditions
    df["is_super_edge"] = df.apply(detect_super_edge, axis=1)

    # Boost EDGE score for SUPER EDGE stocks
    df.loc[df["is_super_edge"], "EDGE"] = df.loc[df["is_super_edge"], "EDGE"] * 1.1
    df["EDGE"] = df["EDGE"].clip(upper=100)

    # Enhanced classification with SUPER EDGE
    conditions = [
        df["is_super_edge"] & (df["EDGE"] >= EDGE_THRESHOLDS["SUPER_EDGE"]),
        df["EDGE"] >= EDGE_THRESHOLDS["EXPLOSIVE"],
        df["EDGE"] >= EDGE_THRESHOLDS["STRONG"],
        df["EDGE"] >= EDGE_THRESHOLDS["MODERATE"],
    ]
    choices = ["SUPER_EDGE", "EXPLOSIVE", "STRONG", "MODERATE"]
    df["tag"] = np.select(conditions, choices, default="WATCH")

    # Enhanced position sizing for SUPER EDGE
    df['position_size_pct'] = df['tag'].apply(
        lambda x: (
            0.15 if x == "SUPER_EDGE" else  # 15% for SUPER EDGE!
            0.10 if x == "EXPLOSIVE" else
            0.05 if x == "STRONG" else
            0.02 if x == "MODERATE" else
            0.00
        )
    )

    # Calculate stops and targets
    df['dynamic_stop'] = df['price'] * 0.95
    df['target1'] = df['price'] * 1.05
    df['target2'] = df['price'] * 1.10

    # Adjust targets for SUPER EDGE stocks
    df.loc[df["tag"] == "SUPER_EDGE", 'target1'] = df.loc[df["tag"] == "SUPER_EDGE", 'price'] * 1.10
    df.loc[df["tag"] == "SUPER_EDGE", 'target2'] = df.loc[df["tag"] == "SUPER_EDGE", 'price'] * 1.20

    # Ensure stops and targets are within bounds
    df['dynamic_stop'] = np.maximum(df['dynamic_stop'], df['low_52w'].fillna(-np.inf))
    df['target1'] = np.minimum(df['target1'], df['high_52w'].fillna(np.inf))
    df['target2'] = np.minimum(df['target2'], df['high_52w'].fillna(np.inf))

    # Add special indicators
    df['eps_qoq_acceleration'] = np.where(
        df['eps_last_qtr'] > 0,
        (df['eps_current'] - df['eps_last_qtr']) / df['eps_last_qtr'] * 100,
        0
    )
    
    df['quality_consolidation'] = (
        (df['ret_3y'] > 3.0) & 
        (df['ret_1y'] < 0.20) & 
        (df['from_high_pct'] >= -0.40) & 
        (df['from_high_pct'] <= -0.15)
    )

    df['momentum_aligned'] = (
        (df['ret_1d'] > 0) & 
        (df['ret_3d'] > df['ret_1d']) & 
        (df['ret_7d'] > df['ret_3d']) & 
        (df['ret_30d'] > 0)
    )

    return df

# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def get_eps_tier(eps: float) -> str:
    """Categorizes EPS into predefined tiers."""
    if pd.isna(eps):
        return ""
    if eps < 0.05:
        return "5↓"
    elif 0.05 <= eps < 0.15:
        return "5↑"
    elif 0.15 <= eps < 0.35:
        return "15↑"
    elif 0.35 <= eps < 0.55:
        return "35↑"
    elif 0.55 <= eps < 0.75:
        return "55↑"
    elif 0.75 <= eps < 0.95:
        return "75↑"
    elif eps >= 0.95:
        return "95↑"
    return ""

def get_price_tier(price: float) -> str:
    """Categorizes Price into predefined tiers."""
    if pd.isna(price):
        return ""
    if price >= 5000:
        return "5K↑"
    elif 2000 <= price < 5000:
        return "2K↑"
    elif 1000 <= price < 2000:
        return "1K↑"
    elif 500 <= price < 1000:
        return "500↑"
    elif 200 <= price < 500:
        return "200↑"
    elif 100 <= price < 200:
        return "100↑"
    elif price < 100:
        return "100↓"
    return ""

def calculate_volume_acceleration_and_classify(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates volume acceleration metrics and classifies accumulation/distribution."""
    df = df.copy()

    df['avg_vol_30d'] = df['volume_30d'] / 30.0
    df['avg_vol_90d'] = df['volume_90d'] / 90.0
    df['avg_vol_180d'] = df['volume_180d'] / 180.0

    df['vol_ratio_30d_90d_calc'] = np.where(df['avg_vol_90d'] != 0,
                                            (df['avg_vol_30d'] / df['avg_vol_90d'] - 1) * 100, 0)
    df['vol_ratio_30d_180d_calc'] = np.where(df['avg_vol_180d'] != 0,
                                             (df['avg_vol_30d'] / df['avg_vol_180d'] - 1) * 100, 0)
    df['vol_ratio_90d_180d_calc'] = np.where(df['avg_vol_180d'] != 0,
                                             (df['avg_vol_90d'] / df['avg_vol_180d'] - 1) * 100, 0)

    df['volume_acceleration'] = df['vol_ratio_30d_90d_calc'] - df['vol_ratio_30d_180d_calc']

    def classify_volume(row):
        ratio_30_90 = row['vol_ratio_30d_90d_calc']
        ratio_30_180 = row['vol_ratio_30d_180d_calc']
        acceleration = row['volume_acceleration']

        if acceleration > 20 and ratio_30_90 > 5 and ratio_30_180 > 5:
            return "Institutional Loading"
        elif acceleration > 5 and ratio_30_90 > 0 and ratio_30_180 > 0:
            return "Heavy Accumulation"
        elif ratio_30_90 > 0 and ratio_30_180 > 0:
            return "Accumulation"
        elif ratio_30_90 < 0 and ratio_30_180 < 0 and acceleration < -5:
            return "Exodus"
        elif ratio_30_90 < 0 and ratio_30_180 < 0:
            return "Distribution"
        else:
            return "Neutral"

    df['volume_classification'] = df.apply(classify_volume, axis=1)
    return df

def plot_stock_radar_chart(df_row: pd.Series):
    """Enhanced radar chart for stock EDGE components."""
    categories = ['Volume Acceleration', 'Momentum Divergence', 'Risk/Reward', 'Fundamentals']
    scores = [
        df_row.get('vol_score', 0),
        df_row.get('mom_score', 0),
        df_row.get('rr_score', 0),
        df_row.get('fund_score', 0)
    ]
    scores = [0 if pd.isna(s) else s for s in scores]

    fig = go.Figure()

    # Different color for SUPER EDGE stocks
    line_color = 'gold' if df_row.get('tag') == 'SUPER_EDGE' else 'darkblue'
    
    fig.add_trace(go.Scatterpolar(
            r=scores,
            theta=categories,
            fill='toself',
            name=df_row['company_name'],
            line_color=line_color,
            line_width=3
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=False,
        title=f"EDGE Component Breakdown for {df_row['company_name']} ({df_row['ticker']})" + 
              (" ⭐ SUPER EDGE ⭐" if df_row.get('tag') == 'SUPER_EDGE' else ""),
        font_size=16
    )
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Enhanced Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────

def render_ui():
    """Enhanced Streamlit UI with SUPER EDGE highlights."""
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    
    # Enhanced header with SUPER EDGE branding
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
    </style>
    """, unsafe_allow_html=True)
    
    st.title(PAGE_TITLE)
    st.markdown("Your unfair advantage: **Volume acceleration + RVOL + EPS Momentum + Quality Filters = SUPER EDGE**")

    # Sidebar controls
    with st.sidebar:
        st.header("Settings")
        profile_name = st.radio("Profile", list(PROFILE_PRESETS.keys()), index=0)
        weights = PROFILE_PRESETS[profile_name]
        min_edge = st.slider("Min EDGE Score for Display", 0, 100, 50, 1)
        show_smallcaps = st.checkbox("Include small/micro caps", value=False)
        
        # NEW: Super Edge Only filter
        show_super_edge_only = st.checkbox("Show SUPER EDGE Only", value=False, 
                                          help="Filter to show only the highest conviction SUPER EDGE signals")

    # Load and preprocess data
    df = load_sheet()

    if df.empty:
        st.error("No data available to process.")
        return

    # String conversions
    if 'ticker' in df.columns:
        df['ticker'] = df['ticker'].astype(str)
    if 'sector' in df.columns:
        df['sector'] = df['sector'].astype(str).fillna("Unknown")
    if 'category' in df.columns:
        df['category'] = df['category'].astype(str).fillna("Unknown")

    # Filters
    if not show_smallcaps and "category" in df.columns:
        df = df[~df["category"].str.contains("nano|micro", case=False, na=False)]

    if "rs_volume_30d" in df.columns:
        df = df[df["rs_volume_30d"].notna() & (df["rs_volume_30d"] >= 1e7)]

    if df.empty:
        st.info("No stocks remain after initial filtering criteria.")
        return

    # Process data
    df_processed = calculate_volume_acceleration_and_classify(df.copy())
    df_processed['eps_tier'] = df_processed['eps_current'].apply(get_eps_tier)
    df_processed['price_tier'] = df_processed['price'].apply(get_price_tier)
    
    # Show progress for compute_scores
    with st.spinner("Computing EDGE scores and detecting patterns..."):
        df_scored = compute_scores(df_processed, weights)
    df_filtered_by_min_edge = df_scored[df_scored["EDGE"].notna() & (df_scored["EDGE"] >= min_edge)].copy()

    # SUPER EDGE Alert
    super_edge_count = (df_filtered_by_min_edge["tag"] == "SUPER_EDGE").sum()
    if super_edge_count > 0:
        st.markdown(f"""
        <div class="super-edge-banner">
            <div class="super-edge-text">
                ⭐ {super_edge_count} SUPER EDGE SIGNAL{'S' if super_edge_count > 1 else ''} DETECTED! ⭐<br>
                These are the absolute BEST opportunities combining ALL edge factors!
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Apply Super Edge filter if selected
    if show_super_edge_only:
        df_filtered_by_min_edge = df_filtered_by_min_edge[df_filtered_by_min_edge["tag"] == "SUPER_EDGE"]

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Daily EDGE Signals", 
        "⭐ SUPER EDGE Analysis",
        "🎯 Explosive Patterns",
        "📈 Volume Acceleration", 
        "🔥 Sector Heatmap", 
        "🔍 Stock Deep Dive", 
        "⚙️ Raw Data & Logs"
    ])

    with tab1:
        st.header("Daily EDGE Signals")
        
        # Filters
        current_filtered_df = df_filtered_by_min_edge.copy()
        
        filter_cols_1 = st.columns(4)
        with filter_cols_1[0]:
            all_edge_class_options = ["SUPER_EDGE", "EXPLOSIVE", "STRONG", "MODERATE", "WATCH"]
            available_edge_classes = current_filtered_df['tag'].dropna().unique().tolist()
            display_edge_options = [opt for opt in all_edge_class_options if opt in available_edge_classes]
            
            selected_edge_class_display = st.multiselect(
                "Filter by EDGE Classification:",
                options=display_edge_options,
                default=display_edge_options
            )
            if selected_edge_class_display:
                current_filtered_df = current_filtered_df[current_filtered_df["tag"].isin(selected_edge_class_display)]

        with filter_cols_1[1]:
            unique_sectors = current_filtered_df['sector'].dropna().unique().tolist()
            unique_sectors.sort()
            selected_sectors = st.multiselect("Filter by Sector:", options=unique_sectors, default=unique_sectors)
            if selected_sectors:
                current_filtered_df = current_filtered_df[current_filtered_df["sector"].isin(selected_sectors)]

        with filter_cols_1[2]:
            unique_categories = current_filtered_df['category'].dropna().unique().tolist()
            unique_categories.sort()
            selected_categories = st.multiselect("Filter by Category:", options=unique_categories, default=unique_categories)
            if selected_categories:
                current_filtered_df = current_filtered_df[current_filtered_df["category"].isin(selected_categories)]

        with filter_cols_1[3]:
            unique_volume_classifications = current_filtered_df['volume_classification'].dropna().unique().tolist()
            unique_volume_classifications.sort()
            selected_volume_classifications = st.multiselect("Filter by Volume Classification:", 
                                                           options=unique_volume_classifications, 
                                                           default=unique_volume_classifications)
            if selected_volume_classifications:
                current_filtered_df = current_filtered_df[current_filtered_df["volume_classification"].isin(selected_volume_classifications)]

        filter_cols_2 = st.columns(3)
        with filter_cols_2[0]:
            unique_eps_tiers = current_filtered_df['eps_tier'].dropna().unique().tolist()
            eps_tier_order = ["5↓", "5↑", "15↑", "35↑", "55↑", "75↑", "95↑", ""]
            sorted_eps_tiers = [tier for tier in eps_tier_order if tier in unique_eps_tiers]
            selected_eps_tiers = st.multiselect("Filter by EPS Tier:", options=sorted_eps_tiers, default=sorted_eps_tiers)
            if selected_eps_tiers:
                current_filtered_df = current_filtered_df[current_filtered_df["eps_tier"].isin(selected_eps_tiers)]

        with filter_cols_2[1]:
            unique_price_tiers = current_filtered_df['price_tier'].dropna().unique().tolist()
            price_tier_order = ["100↓", "100↑", "200↑", "500↑", "1K↑", "2K↑", "5K↑", ""]
            sorted_price_tiers = [tier for tier in price_tier_order if tier in unique_price_tiers]
            selected_price_tiers = st.multiselect("Filter by Price Tier:", options=sorted_price_tiers, default=sorted_price_tiers)
            if selected_price_tiers:
                current_filtered_df = current_filtered_df[current_filtered_df["price_tier"].isin(selected_price_tiers)]

        with filter_cols_2[2]:
            if not current_filtered_df.empty and 'pe' in current_filtered_df.columns and current_filtered_df['pe'].notna().any():
                min_pe, max_pe = float(current_filtered_df['pe'].min()), float(current_filtered_df['pe'].max())
                selected_pe_range = st.slider(
                    "Filter by PE Ratio:",
                    min_value=min_pe,
                    max_value=max_pe,
                    value=(min_pe, max_pe),
                    step=0.1,
                    format="%.1f"
                )
                current_filtered_df = current_filtered_df[
                    (current_filtered_df["pe"] >= selected_pe_range[0]) &
                    (current_filtered_df["pe"] <= selected_pe_range[1])
                ]
        
        display_df = current_filtered_df.sort_values("EDGE", ascending=False)

        if not display_df.empty:
            # Highlight SUPER EDGE rows
            def highlight_super_edge(row):
                if row['tag'] == 'SUPER_EDGE':
                    return ['background-color: gold'] * len(row)
                return [''] * len(row)
            
            st.dataframe(
                display_df[[
                    "ticker", "company_name", "sector", "category", "tag", "EDGE",
                    "vol_score", "mom_score", "rr_score", "fund_score",
                    "price", "price_tier", "eps_current", "eps_tier", "pe", "rvol",
                    "position_size_pct", "dynamic_stop", "target1", "target2",
                    "volume_acceleration", "volume_classification",
                    "eps_qoq_acceleration", "quality_consolidation", "momentum_aligned"
                ]].style.apply(highlight_super_edge, axis=1).background_gradient(
                    cmap='RdYlGn', subset=['EDGE']
                ).format({
                    "EDGE": "{:.2f}",
                    "vol_score": "{:.2f}", "mom_score": "{:.2f}", 
                    "rr_score": "{:.2f}", "fund_score": "{:.2f}",
                    "price": "₹{:.2f}",
                    "eps_current": "{:.2f}",
                    "pe": "{:.2f}",
                    "rvol": "{:.2f}",
                    "position_size_pct": "{:.2%}",
                    "dynamic_stop": "₹{:.2f}", "target1": "₹{:.2f}", "target2": "₹{:.2f}",
                    "volume_acceleration": "{:.2f}%",
                    "eps_qoq_acceleration": "{:.2f}%"
                }),
                use_container_width=True
            )

            csv = display_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Export Filtered Signals to CSV",
                data=csv,
                file_name=f"edge_signals_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
            )
        else:
            st.info("No stocks match the selected filters.")

    with tab2:
        st.header("⭐ SUPER EDGE Analysis")
        st.markdown("""
        SUPER EDGE stocks meet multiple high-conviction criteria:
        - ✅ RVOL > 2.0 (Unusual activity TODAY)
        - ✅ Volume Acceleration > 30% (Aggressive institutional loading)
        - ✅ EPS QoQ Acceleration > 10% (Fundamental momentum)
        - ✅ Perfect consolidation zone (-15% to -30% from high)
        - ✅ Momentum alignment (all timeframes positive and accelerating)
        """)
        
        super_edge_df = df_scored[df_scored["tag"] == "SUPER_EDGE"].sort_values("EDGE", ascending=False)
        
        if not super_edge_df.empty:
            st.success(f"🎯 {len(super_edge_df)} SUPER EDGE opportunities found!")
            
            for idx, row in super_edge_df.head(5).iterrows():
                with st.expander(f"⭐ {row['ticker']} - {row['company_name']} (EDGE: {row['EDGE']:.2f})"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Price", f"₹{row['price']:.2f}")
                        st.metric("RVOL", f"{row['rvol']:.2f}x")
                        st.metric("Position Size", f"{row['position_size_pct']:.1%}")
                    
                    with col2:
                        st.metric("Volume Accel", f"{row['volume_acceleration']:.1f}%")
                        st.metric("EPS QoQ Accel", f"{row['eps_qoq_acceleration']:.1f}%")
                        st.metric("From High", f"{row['from_high_pct']:.1f}%")
                    
                    with col3:
                        st.metric("Stop Loss", f"₹{row['dynamic_stop']:.2f}")
                        st.metric("Target 1", f"₹{row['target1']:.2f}")
                        st.metric("Target 2", f"₹{row['target2']:.2f}")
                    
                    st.info(f"""
                    **Why SUPER EDGE:**
                    - Volume Classification: {row['volume_classification']}
                    - Quality Stock: {'Yes' if row['quality_consolidation'] else 'No'} 
                    - Momentum Aligned: {'Yes' if row['momentum_aligned'] else 'No'}
                    - 3Y Return: {row['ret_3y']:.0f}% | 1Y Return: {row['ret_1y']:.0f}%
                    """)
        else:
            st.info("No SUPER EDGE signals detected today. Check EXPLOSIVE category for next best opportunities.")

    with tab3:
        st.header("🎯 Explosive Patterns Discovery")
        st.markdown("""
        **Pattern-based trading catches moves 2-5 days BEFORE they explode!**
        
        These 6 patterns identify stocks about to make significant moves based on volume-price dynamics:
        """)
        
        # Pattern explanations
        with st.expander("📚 Pattern Guide"):
            st.markdown("""
            1. **Accumulation Under Resistance** - Volume explodes but price stays flat near highs
            2. **Coiled Spring** - Tight consolidation with increasing volume pressure
            3. **Absorption Pattern** - Big players absorbing supply at current levels
            4. **Failed Breakdown Reversal** - Smart money buying near 52w lows
            5. **Stealth Breakout** - Quiet accumulation breaking above all SMAs
            6. **Pre-Earnings Accumulation** - Unusual volume before earnings announcements
            
            **🔥 Pattern Confluence**: When 3+ patterns align = ULTRA HIGH CONVICTION!
            """)
        
        # Filter for high-scoring patterns
        pattern_min_score = st.slider("Minimum Pattern Score", 0, 100, 70, 5, 
                                     help="Show only patterns with score above this threshold")
        
        # Find stocks with high-scoring patterns
        pattern_stocks = []
        
        # Check if pattern columns exist before trying to access them
        if 'top_pattern_score' in df_scored.columns:
            for idx, row in df_scored.iterrows():
                if row['top_pattern_score'] >= pattern_min_score:
                    pattern_stocks.append({
                        'ticker': row['ticker'],
                        'company_name': row['company_name'],
                        'price': row['price'],
                        'edge_score': row['EDGE'],
                        'pattern': row['top_pattern_name'],
                        'pattern_score': row['top_pattern_score'],
                        'confluence_score': row['pattern_confluence_score'],
                        'vp_divergence': row['vp_divergence_score'],
                        'pattern_data': row['pattern_analysis']
                    })
            
            pattern_df = pd.DataFrame(pattern_stocks).sort_values('confluence_score', ascending=False)
        else:
            st.warning("⚠️ Pattern analysis is still processing or unavailable. Please check the data quality.")
            pattern_df = pd.DataFrame()  # Empty dataframe
        
        if not pattern_df.empty:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Pattern Signals", len(pattern_df))
            with col2:
                confluence_high = len(pattern_df[pattern_df['confluence_score'] >= 85])
                st.metric("High Confluence (3+ patterns)", confluence_high)
            with col3:
                avg_pattern_score = pattern_df['pattern_score'].mean()
                st.metric("Avg Pattern Score", f"{avg_pattern_score:.0f}")
            with col4:
                unique_patterns = pattern_df['pattern'].nunique()
                st.metric("Pattern Types Active", unique_patterns)
            
            # Pattern distribution
            st.subheader("Pattern Distribution")
            pattern_counts = pattern_df['pattern'].value_counts()
            
            fig_patterns = go.Figure(data=[go.Bar(
                x=pattern_counts.index,
                y=pattern_counts.values,
                text=pattern_counts.values,
                textposition='auto',
                marker_color=['#ff4444', '#ff8844', '#ffcc44', '#88cc44', '#44ff44', '#4444ff']
            )])
            fig_patterns.update_layout(
                title="Active Pattern Types",
                xaxis_title="Pattern",
                yaxis_title="Count",
                height=400
            )
            st.plotly_chart(fig_patterns, use_container_width=True)
            
            # Confluence Leaders
            st.subheader("🔥 High Confluence Signals (Multiple Patterns Aligned)")
            
            high_confluence = pattern_df[pattern_df['confluence_score'] >= 70].head(10)
            if not high_confluence.empty:
                for _, stock in high_confluence.iterrows():
                    # Safely get pattern data
                    pattern_data = stock.get('pattern_data', {})
                    patterns = pattern_data.get('patterns', [])
                    confluence = pattern_data.get('confluence', {})
                    
                    with st.expander(f"💎 {stock['ticker']} - {stock['company_name']} (Confluence: {stock['confluence_score']:.0f})"):
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.write(f"**Price**: ₹{stock['price']:.2f} | **EDGE Score**: {stock['edge_score']:.1f}")
                            st.write(f"**Volume-Price Divergence**: {stock['vp_divergence']:.1f}")
                            
                            # Show aligned patterns
                            st.write("**Aligned Patterns:**")
                            shown_patterns = 0
                            for pattern in patterns:
                                if pattern.get('score', 0) >= 50 and shown_patterns < 3:
                                    st.write(f"- {pattern['pattern']}: {pattern.get('score', 0):.0f}/100")
                                    for signal in pattern.get('signals', [])[:2]:
                                        st.write(f"  • {signal}")
                                    shown_patterns += 1
                            
                            # Confluence target
                            if confluence.get('target', 0) > 0:
                                target_pct = (confluence['target'] / stock['price'] - 1) * 100
                                st.success(f"**Pattern Target**: ₹{confluence['target']:.2f} (+{target_pct:.1f}%)")
                        
                        with col2:
                            # Pattern strength meter
                            fig_gauge = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=stock['confluence_score'],
                                domain={'x': [0, 1], 'y': [0, 1]},
                                gauge={'axis': {'range': [None, 100]},
                                      'bar': {'color': "darkgreen" if stock['confluence_score'] >= 85 else "orange"},
                                      'steps': [
                                          {'range': [0, 50], 'color': "lightgray"},
                                          {'range': [50, 70], 'color': "yellow"},
                                          {'range': [70, 85], 'color': "orange"},
                                          {'range': [85, 100], 'color': "red"}],
                                      'threshold': {'line': {'color': "red", 'width': 4},
                                                   'thickness': 0.75, 'value': 90}}))
                            fig_gauge.update_layout(height=200, margin=dict(l=20, r=20, t=20, b=20))
                            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # All Pattern Signals Table
            st.subheader("All Pattern Signals")
            
            display_pattern_df = pattern_df[[
                'ticker', 'company_name', 'price', 'edge_score', 
                'pattern', 'pattern_score', 'confluence_score', 'vp_divergence'
            ]]
            
            # Color code by confluence score
            def color_confluence(val):
                if val >= 85:
                    return 'background-color: #ffcccc'  # Light red
                elif val >= 70:
                    return 'background-color: #ffffcc'  # Light yellow
                return ''
            
            st.dataframe(
                display_pattern_df.style.applymap(
                    color_confluence, subset=['confluence_score']
                ).format({
                    'price': '₹{:.2f}',
                    'edge_score': '{:.1f}',
                    'pattern_score': '{:.0f}',
                    'confluence_score': '{:.0f}',
                    'vp_divergence': '{:.1f}'
                }),
                use_container_width=True
            )
            
            # Export pattern signals
            csv_patterns = display_pattern_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Export Pattern Signals",
                csv_patterns,
                f"explosive_patterns_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                "text/csv"
            )
            
        else:
            st.info(f"No patterns found with score >= {pattern_min_score}. Try lowering the threshold.")
        
        # Pattern combination insights
        st.subheader("🎯 Today's Best Pattern Combinations")
        
        # Find stocks with specific powerful combinations
        power_combos = []
        for idx, row in df_scored.iterrows():
            if 'pattern_analysis' in row and isinstance(row.get('pattern_analysis'), dict):
                patterns = row['pattern_analysis'].get('patterns', [])
                pattern_names = [p['pattern'] for p in patterns if p.get('score', 0) >= 70]
                
                # Check for power combinations
                if ('Accumulation Under Resistance' in pattern_names and 
                    'Coiled Spring' in pattern_names):
                    power_combos.append({
                        'ticker': row['ticker'],
                        'combo': 'Resistance Breakout Imminent',
                        'signal': 'Volume + Coil at resistance'
                    })
                
                if ('Failed Breakdown Reversal' in pattern_names and 
                    'Pre-Earnings Accumulation' in pattern_names):
                    power_combos.append({
                        'ticker': row['ticker'],
                        'combo': 'Earnings Reversal Play',
                        'signal': 'Bottom + Earnings catalyst'
                    })
        
        if power_combos:
            st.success("🚀 Power Combinations Detected!")
            for combo in power_combos[:5]:
                st.write(f"**{combo['ticker']}**: {combo['combo']} - {combo['signal']}")

    with tab4:
        st.header("Volume Acceleration Insights")
        st.markdown("Enhanced visualization with RVOL overlay to spot INSTITUTIONAL URGENCY")

        if "volume_acceleration" in df_scored.columns and "from_high_pct" in df_scored.columns and not df_scored.empty:
            order = ["SUPER_EDGE", "EXPLOSIVE", "STRONG", "MODERATE", "WATCH"]
            df_scored['tag'] = pd.Categorical(df_scored['tag'], categories=order, ordered=True)
            df_scored_plot = df_scored.sort_values('tag')

            # Create size based on RVOL for visual impact
            df_scored_plot['marker_size'] = df_scored_plot['rvol'] * 10

            fig2 = px.scatter(
                df_scored_plot,
                x="from_high_pct",
                y="volume_acceleration",
                color="tag",
                size="marker_size",  # Size by RVOL
                hover_data=["ticker", "company_name", "sector", "EDGE", "rvol", 
                           "vol_score", "mom_score", "volume_classification"],
                title="Volume Acceleration vs. Distance from 52-Week High (Size = RVOL)",
                labels={
                    "from_high_pct": "% From 52-Week High",
                    "volume_acceleration": "Volume Acceleration %"
                },
                color_discrete_map={
                    "SUPER_EDGE": "#FFD700",  # Gold
                    "EXPLOSIVE": "#FF4B4B",   # Red
                    "STRONG": "#FFA500",      # Orange
                    "MODERATE": "#FFD700",    # Yellow
                    "WATCH": "#1F77B4"        # Blue
                }
            )
            
            # Add zones
            fig2.add_vline(x=-0.15, line_dash="dash", line_color="gold", 
                          annotation_text="SUPER EDGE Zone Start")
            fig2.add_vline(x=-0.30, line_dash="dash", line_color="gold", 
                          annotation_text="SUPER EDGE Zone End")
            fig2.add_hline(y=30, line_dash="dash", line_color="red", 
                          annotation_text="Vol Accel > 30% (SUPER)")

            # Highlight SUPER EDGE zone
            fig2.add_vrect(x0=-0.30, x1=-0.15, fillcolor="gold", opacity=0.1,
                          annotation_text="SUPER EDGE ZONE", annotation_position="top")

            st.plotly_chart(fig2, use_container_width=True)
            
            # Additional insights
            high_rvol_stocks = df_scored_plot[df_scored_plot['rvol'] > 2.0]
            if len(high_rvol_stocks) > 0:
                st.warning(f"🔥 {len(high_rvol_stocks)} stocks showing HIGH RVOL (>2.0) - Institutional urgency!")
        else:
            st.info("Volume acceleration data not available.")

    with tab4:
        st.header("Sector Heatmap (Average EDGE Score)")
        st.markdown("Sectors with concentration of SUPER EDGE signals are highlighted")

        agg = df_scored.groupby("sector").agg(
            edge_mean=("EDGE", "mean"),
            n=("EDGE", "size"),
            super_edge_count=("is_super_edge", "sum")
        ).reset_index()
        
        agg.dropna(subset=['edge_mean'], inplace=True)

        if not agg.empty:
            agg['edge_mean'] = pd.to_numeric(agg['edge_mean'], errors='coerce')
            agg['n'] = pd.to_numeric(agg['n'], errors='coerce')
            agg.dropna(subset=['edge_mean', 'n'], inplace=True)

            if not agg.empty:
                # Add super edge indicator
                agg['has_super_edge'] = agg['super_edge_count'] > 0
                agg["opacity"] = np.where(agg["n"] < MIN_STOCKS_PER_SECTOR, 0.4, 1.0)

                fig = px.treemap(agg, path=["sector"], values="n", color="edge_mean",
                                hover_data={"super_edge_count": True},
                                range_color=(0, 100),
                                color_continuous_scale='RdYlGn',
                                title="Sector EDGE Heatmap (with SUPER EDGE counts)"
                                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show sectors with SUPER EDGE signals
                super_sectors = agg[agg['super_edge_count'] > 0].sort_values('super_edge_count', ascending=False)
                if len(super_sectors) > 0:
                    st.success("🌟 Sectors with SUPER EDGE signals:")
                    for _, row in super_sectors.iterrows():
                        st.write(f"- **{row['sector']}**: {int(row['super_edge_count'])} SUPER EDGE signal(s)")

    with tab5:
        st.header("Stock Deep Dive (Enhanced)")
        
        available_stocks = df_scored[df_scored['EDGE'].notnull()]['ticker'].dropna().astype(str).tolist()
        
        if available_stocks:
            # Prioritize SUPER EDGE stocks in dropdown
            super_edge_stocks = df_scored[df_scored['tag'] == 'SUPER_EDGE']['ticker'].tolist()
            other_stocks = [s for s in available_stocks if s not in super_edge_stocks]
            sorted_stocks = super_edge_stocks + other_stocks
            
            if 'selected_ticker' not in st.session_state or st.session_state.selected_ticker not in sorted_stocks:
                st.session_state.selected_ticker = sorted_stocks[0]

            selected_ticker = st.selectbox(
                "Select Ticker (⭐ = SUPER EDGE):", 
                sorted_stocks, 
                format_func=lambda x: f"⭐ {x}" if x in super_edge_stocks else x,
                key='selected_ticker'
            )
            
            selected_stock_row_df = df_scored[df_scored['ticker'] == selected_ticker]
            if not selected_stock_row_df.empty:
                selected_stock_row = selected_stock_row_df.iloc[0]
                
                # Show SUPER EDGE banner if applicable
                if selected_stock_row['tag'] == 'SUPER_EDGE':
                    st.markdown("""
                    <div style="background-color: gold; padding: 10px; border-radius: 5px; text-align: center;">
                        <h2>⭐ THIS IS A SUPER EDGE SIGNAL ⭐</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                plot_stock_radar_chart(selected_stock_row)

                st.subheader(f"Enhanced Metrics for {selected_stock_row['company_name']} ({selected_stock_row['ticker']})")
                
                # Enhanced metrics display
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current Price", f"₹{selected_stock_row.get('price', 0):.2f}")
                    st.metric("EDGE Score", f"{selected_stock_row.get('EDGE', 0):.2f}")
                    st.metric("Classification", selected_stock_row.get('tag', 'N/A'))
                    st.metric("RVOL", f"{selected_stock_row.get('rvol', 0):.2f}x",
                             delta="HIGH" if selected_stock_row.get('rvol', 0) > 2.0 else None)
                
                with col2:
                    st.metric("Volume Accel", f"{selected_stock_row.get('volume_acceleration', 0):.2f}%")
                    st.metric("Volume Class", selected_stock_row.get('volume_classification', 'N/A'))
                    st.metric("EPS QoQ Accel", f"{selected_stock_row.get('eps_qoq_acceleration', 0):.1f}%")
                
                with col3:
                    st.metric("3Y Return", f"{selected_stock_row.get('ret_3y', 0):.0f}%")
                    st.metric("1Y Return", f"{selected_stock_row.get('ret_1y', 0):.0f}%")
                    st.metric("Quality Stock", "Yes" if selected_stock_row.get('quality_consolidation', False) else "No")
                
                with col4:
                    st.metric("Stop Loss", f"₹{selected_stock_row.get('dynamic_stop', 0):.2f}")
                    st.metric("Target 1", f"₹{selected_stock_row.get('target1', 0):.2f}")
                    st.metric("Target 2", f"₹{selected_stock_row.get('target2', 0):.2f}")
                    st.metric("Position Size", f"{selected_stock_row.get('position_size_pct', 0):.1%}")

                # Special indicators
                st.markdown("---")
                st.subheader("Special Indicators")
                
                indicators = []
                if selected_stock_row.get('rvol', 0) > 2.0:
                    indicators.append("🔥 **HIGH RVOL**: Unusual activity detected")
                if selected_stock_row.get('quality_consolidation', False):
                    indicators.append("💎 **QUALITY CONSOLIDATION**: Proven winner taking a breather")
                if selected_stock_row.get('momentum_aligned', False):
                    indicators.append("📈 **MOMENTUM ALIGNED**: All timeframes in sync")
                if selected_stock_row.get('eps_qoq_acceleration', 0) > 10:
                    indicators.append("💰 **EPS ACCELERATING**: Fundamental momentum building")
                
                if indicators:
                    for indicator in indicators:
                        st.write(indicator)
                else:
                    st.info("No special indicators triggered")

                st.markdown("---")
                st.subheader("All Data")
                st.dataframe(
                    selected_stock_row.to_frame().T,
                    use_container_width=True
                )

    with tab6:
        st.header("Raw Data and Enhanced Metrics")
        
        st.subheader("Data Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Stocks", len(df))
            st.metric("After Filters", len(df_scored))
        with col2:
            st.metric("SUPER EDGE", (df_scored['tag'] == 'SUPER_EDGE').sum())
            st.metric("EXPLOSIVE", (df_scored['tag'] == 'EXPLOSIVE').sum())
        with col3:
            st.metric("Avg RVOL", f"{df_scored['rvol'].mean():.2f}")
            st.metric("High RVOL (>2)", (df_scored['rvol'] > 2.0).sum())
        
        st.subheader("Enhanced EDGE Thresholds")
        st.json(EDGE_THRESHOLDS)
        
        st.subheader("SUPER EDGE Criteria")
        st.markdown("""
        A stock achieves **SUPER EDGE** status when it meets at least 4 of these 5 criteria:
        1. **RVOL > 2.0** - Unusual trading activity today
        2. **Volume Acceleration > 30%** - Aggressive institutional accumulation
        3. **EPS QoQ Acceleration > 10%** - Accelerating fundamentals
        4. **From High: -15% to -30%** - Perfect consolidation zone
        5. **Momentum Aligned** - All timeframes (1d, 3d, 7d, 30d) positive and accelerating
        """)
        
        st.subheader("Component Weights")
        st.write(f"""
        Current profile: **{profile_name}**
        - Volume Acceleration: {weights[0]*100:.0f}%
        - Momentum Divergence: {weights[1]*100:.0f}%
        - Risk/Reward: {weights[2]*100:.0f}%
        - Fundamentals: {weights[3]*100:.0f}%
        """)
        
        st.subheader("Raw Data Sample")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Export enhanced data
        csv_full = df_scored.to_csv(index=False).encode('utf-8')
        st.download_button("Download Full Enhanced Data", csv_full, "edge_protocol_enhanced.csv", "text/csv")
        
        st.write(f"Data processed: {pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")

if __name__ == "__main__":
    render_ui()
