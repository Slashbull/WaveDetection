"""
Wave Detection System 2.0 - Professional Stock Analysis Platform
Author: AI Assistant
Version: 2.0.0
Last Updated: December 2024

Features:
- Traffic Light System (GREEN/YELLOW/RED)
- Market Regime Detection
- Sector-Aware Analysis
- Portfolio Risk Management
- Performance Tracking
- Data Quality Validation
- Exit Signal Detection
- Professional Reports

MAJOR IMPROVEMENTS:
- Handles sectors with 2-100+ stocks intelligently
- Market context before stock signals
- Liquidity-based filtering
- Relative strength vs sector/market
- Position sizing recommendations
- Historical performance tracking
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
import logging
from io import BytesIO
import json
import os
from typing import Dict, List, Tuple, Optional, Any
import requests

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Wave Detection System 2.0",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional appearance
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .green-signal {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
    }
    .yellow-signal {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
    }
    .red-signal {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
    }
    .market-health {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        text-align: center;
    }
    .liquidity-warning {
        background-color: #ff6b6b;
        color: white;
        padding: 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.8rem;
        margin-top: 0.25rem;
    }
    .sector-badge {
        background-color: #e9ecef;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.8rem;
        display: inline-block;
        margin: 0.25rem;
    }
    .performance-tracker {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# CONSTANTS AND CONFIGURATION
# ============================================

# Market regime thresholds
MARKET_REGIME_THRESHOLDS = {
    'BULL': 0.6,      # 60%+ stocks above SMA50
    'BEAR': 0.4,      # <40% stocks above SMA50
    'NEUTRAL': 0.5    # 40-60% range
}

# Liquidity tiers
LIQUIDITY_TIERS = {
    'A': 1_000_000,   # Institutional grade
    'B': 100_000,     # Tradeable
    'C': 50_000,      # Risky
    'D': 0            # Avoid
}

# Position sizing rules
POSITION_SIZE_RULES = {
    'GREEN_A': 0.02,   # 2% for best signals with high liquidity
    'GREEN_B': 0.015,  # 1.5% for good signals
    'YELLOW_A': 0.01,  # 1% for watch signals
    'YELLOW_B': 0.005, # 0.5% for lower quality
    'DEFAULT': 0.005   # 0.5% minimum
}

# Portfolio constraints
PORTFOLIO_CONSTRAINTS = {
    'MAX_PER_SECTOR': 0.25,          # 25% max in one sector
    'MAX_PER_CATEGORY': 0.30,        # 30% max in one market cap
    'MAX_CORRELATED_POSITIONS': 3,    # Max similar stocks
    'MIN_STOCKS_FOR_SECTOR_AVG': 3   # Min stocks to calculate sector average
}

# Data validation limits
DATA_VALIDATION_LIMITS = {
    'MAX_RVOL': 100,
    'MAX_DAILY_RETURN': 20,  # ±20% circuit limits
    'MIN_PE': -500,
    'MAX_PE': 1000,
    'MIN_PRICE': 0.01
}

# ============================================
# UTILITY FUNCTIONS
# ============================================

def safe_float(value, default=0.0):
    """Safely convert value to float"""
    try:
        if pd.isna(value):
            return default
        if isinstance(value, str):
            value = value.replace('₹', '').replace(',', '').replace('%', '').strip()
            if value == '' or value == '-' or value == 'N/A':
                return default
        return float(value)
    except:
        return default

def safe_divide(numerator, denominator, default=0.0):
    """Safely divide two numbers"""
    try:
        num = safe_float(numerator, 0)
        den = safe_float(denominator, 0)
        if den == 0:
            return default
        return num / den
    except:
        return default

def get_current_month():
    """Get current month name"""
    return datetime.now().strftime('%B')

def get_liquidity_tier(volume_30d):
    """Determine liquidity tier based on 30-day average volume"""
    volume = safe_float(volume_30d, 0)
    for tier, threshold in LIQUIDITY_TIERS.items():
        if volume >= threshold:
            return tier
    return 'D'

# ============================================
# DATA LOADING AND CLEANING
# ============================================

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_google_sheets_data(sheet_url: str, gid: str) -> pd.DataFrame:
    """Load data from Google Sheets with error handling"""
    try:
        csv_url = f"{sheet_url.split('/edit')[0]}/export?format=csv&gid={gid}"
        df = pd.read_csv(csv_url)
        logger.info(f"Successfully loaded {len(df)} rows from Google Sheets")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        st.error(f"Failed to load data: {str(e)}")
        return pd.DataFrame()

def validate_and_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Comprehensive data cleaning and validation"""
    try:
        df = df.copy()
        initial_count = len(df)
        
        # Define column types
        currency_cols = ['price', 'low_52w', 'high_52w', 'sma_20d', 'sma_50d', 'sma_200d', 'prev_close']
        percentage_cols = ['ret_1d', 'ret_3d', 'ret_7d', 'ret_30d', 'ret_3m', 'ret_6m', 'ret_1y', 'ret_3y', 'ret_5y',
                          'from_low_pct', 'from_high_pct', 'eps_change_pct',
                          'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d',
                          'vol_ratio_1d_180d', 'vol_ratio_7d_180d', 'vol_ratio_30d_180d', 'vol_ratio_90d_180d']
        volume_cols = ['volume_1d', 'volume_7d', 'volume_30d', 'volume_90d', 'volume_180d']
        numeric_cols = ['rvol', 'pe', 'eps_current', 'eps_last_qtr', 'year']
        
        # Clean all numeric columns
        all_numeric_cols = currency_cols + percentage_cols + volume_cols + numeric_cols
        
        for col in all_numeric_cols:
            if col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.replace('₹', '', regex=False)
                    df[col] = df[col].str.replace('%', '', regex=False)
                    df[col] = df[col].str.replace(',', '', regex=False)
                    df[col] = df[col].str.strip()
                    df[col] = df[col].replace(['', '-', 'N/A', 'n/a', '#N/A'], np.nan)
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Clean market cap
        if 'market_cap' in df.columns:
            df['market_cap_clean'] = df['market_cap'].astype(str).str.replace('₹', '', regex=False)
            df['market_cap_clean'] = df['market_cap_clean'].str.replace(' Cr', '', regex=False)
            df['market_cap_clean'] = df['market_cap_clean'].str.replace(',', '', regex=False)
            df['market_cap_clean'] = pd.to_numeric(df['market_cap_clean'], errors='coerce')
        
        # Ensure categorical columns are strings
        categorical_cols = ['ticker', 'company_name', 'category', 'sector']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace('nan', '')
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['ticker'], keep='first')
        
        # Apply data validation limits
        if 'rvol' in df.columns:
            df = df[df['rvol'] < DATA_VALIDATION_LIMITS['MAX_RVOL']]
        
        if 'ret_1d' in df.columns:
            df = df[df['ret_1d'].between(-DATA_VALIDATION_LIMITS['MAX_DAILY_RETURN'], 
                                         DATA_VALIDATION_LIMITS['MAX_DAILY_RETURN'])]
        
        if 'pe' in df.columns:
            df = df[(df['pe'].isna()) | 
                   df['pe'].between(DATA_VALIDATION_LIMITS['MIN_PE'], 
                                   DATA_VALIDATION_LIMITS['MAX_PE'])]
        
        if 'price' in df.columns:
            df = df[df['price'] > DATA_VALIDATION_LIMITS['MIN_PRICE']]
        
        # Fill NaN values appropriately
        for col in percentage_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        for col in volume_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        if 'rvol' in df.columns:
            df['rvol'] = df['rvol'].fillna(0)
        
        # Add liquidity tier
        if 'volume_30d' in df.columns:
            df['liquidity_tier'] = df['volume_30d'].apply(get_liquidity_tier)
        
        # Calculate sector statistics
        df = calculate_sector_statistics(df)
        
        # Calculate relative strength
        df = calculate_relative_strength(df)
        
        final_count = len(df)
        logger.info(f"Data cleaned: {initial_count} → {final_count} rows")
        
        return df
        
    except Exception as e:
        logger.error(f"Error cleaning data: {str(e)}")
        st.error(f"Data cleaning failed: {str(e)}")
        return df

# ============================================
# MARKET ANALYSIS FUNCTIONS
# ============================================

def calculate_market_health(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate overall market health metrics"""
    try:
        total_stocks = len(df)
        if total_stocks == 0:
            return {
                'regime': 'UNKNOWN',
                'regime_color': 'gray',
                'total_stocks': 0,
                'market_score': 0
            }
        
        # Stocks above key SMAs
        above_sma20 = len(df[df['price'] > df['sma_20d']])
        above_sma50 = len(df[df['price'] > df['sma_50d']])
        above_sma200 = len(df[df['price'] > df['sma_200d']])
        
        # Market breadth
        advancing = len(df[df['ret_1d'] > 0])
        declining = len(df[df['ret_1d'] < 0])
        
        # New highs/lows
        new_highs = len(df[df['from_high_pct'] > -5])
        new_lows = len(df[df['from_low_pct'] < 5])
        
        # Average metrics
        avg_ret_1d = df['ret_1d'].mean() if not df.empty else 0
        avg_ret_30d = df['ret_30d'].mean() if not df.empty else 0
        avg_rvol = df['rvol'].mean() if not df.empty else 1
        
        # Market regime
        sma50_ratio = above_sma50 / total_stocks
        if sma50_ratio >= MARKET_REGIME_THRESHOLDS['BULL']:
            regime = 'BULL'
            regime_color = 'green'
        elif sma50_ratio <= MARKET_REGIME_THRESHOLDS['BEAR']:
            regime = 'BEAR'
            regime_color = 'red'
        else:
            regime = 'NEUTRAL'
            regime_color = 'orange'
        
        # Volume analysis
        high_volume_stocks = len(df[df['rvol'] > 2])
        volume_acceleration_stocks = len(df[df.apply(calculate_volume_acceleration, axis=1) > 0.1])
        
        # Calculate advance/decline ratio safely
        if declining > 0:
            advance_decline_ratio = advancing / declining
        elif advancing > 0:
            advance_decline_ratio = float(advancing)  # All advancing
        else:
            advance_decline_ratio = 1.0  # Neutral
        
        # Calculate high/low ratio safely
        if new_lows > 0:
            high_low_ratio = new_highs / new_lows
        elif new_highs > 0:
            high_low_ratio = float(new_highs)  # All highs
        else:
            high_low_ratio = 1.0  # Neutral
        
        return {
            'regime': regime,
            'regime_color': regime_color,
            'total_stocks': total_stocks,
            'above_sma20_pct': (above_sma20 / total_stocks * 100),
            'above_sma50_pct': (above_sma50 / total_stocks * 100),
            'above_sma200_pct': (above_sma200 / total_stocks * 100),
            'advance_decline_ratio': advance_decline_ratio,
            'new_highs': new_highs,
            'new_lows': new_lows,
            'high_low_ratio': high_low_ratio,
            'avg_return_1d': avg_ret_1d,
            'avg_return_30d': avg_ret_30d,
            'avg_rvol': avg_rvol,
            'high_volume_stocks': high_volume_stocks,
            'volume_acceleration_stocks': volume_acceleration_stocks,
            'market_score': calculate_market_score(sma50_ratio, advancing/total_stocks)
        }
        
    except Exception as e:
        logger.error(f"Error calculating market health: {str(e)}")
        return {
            'regime': 'ERROR',
            'regime_color': 'gray',
            'total_stocks': 0,
            'market_score': 0
        }

def calculate_market_score(sma50_ratio: float, advance_ratio: float) -> int:
    """Calculate overall market score (0-100)"""
    sma_score = sma50_ratio * 50
    breadth_score = advance_ratio * 50
    return int(sma_score + breadth_score)

def calculate_sector_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate sector-level statistics"""
    try:
        # Filter out empty sectors
        df_clean = df[df['sector'].notna() & (df['sector'] != '') & (df['sector'] != 'nan')]
        
        if df_clean.empty:
            return df
        
        # Group by sector
        sector_stats = df_clean.groupby('sector').agg({
            'ticker': 'count',
            'ret_1d': 'mean',
            'ret_7d': 'mean',
            'ret_30d': 'mean',
            'rvol': 'mean',
            'volume_30d': 'mean'
        }).rename(columns={'ticker': 'stock_count'})
        
        # Calculate sector scores
        sector_stats['sector_score'] = (
            sector_stats['ret_30d'] * 0.4 +
            sector_stats['ret_7d'] * 0.3 +
            sector_stats['ret_1d'] * 0.2 +
            (sector_stats['rvol'] - 1) * 10 * 0.1
        )
        
        # Merge back to main dataframe
        df = df.merge(
            sector_stats[['stock_count', 'ret_30d', 'sector_score']].rename(columns={
                'ret_30d': 'sector_avg_ret_30d',
                'stock_count': 'sector_stock_count'
            }),
            left_on='sector',
            right_index=True,
            how='left'
        )
        
        # Fill NaN values for stocks with no sector
        df['sector_stock_count'] = df['sector_stock_count'].fillna(0)
        df['sector_avg_ret_30d'] = df['sector_avg_ret_30d'].fillna(0)
        df['sector_score'] = df['sector_score'].fillna(0)
        
        return df
        
    except Exception as e:
        logger.error(f"Error calculating sector statistics: {str(e)}")
        return df

def calculate_relative_strength(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate relative strength vs sector and market"""
    try:
        # Market averages
        market_avg_ret_30d = df['ret_30d'].mean()
        market_avg_ret_7d = df['ret_7d'].mean()
        
        # Stock vs Sector
        if 'sector_avg_ret_30d' in df.columns:
            df['rs_vs_sector'] = df['ret_30d'] - df['sector_avg_ret_30d']
        
        # Stock vs Market
        df['rs_vs_market'] = df['ret_30d'] - market_avg_ret_30d
        
        # Combined RS score
        df['relative_strength_score'] = (
            df.get('rs_vs_sector', 0) * 0.6 +
            df.get('rs_vs_market', 0) * 0.4
        )
        
        return df
        
    except Exception as e:
        logger.error(f"Error calculating relative strength: {str(e)}")
        return df

# ============================================
# WAVE DETECTION ALGORITHMS
# ============================================

def calculate_volume_acceleration(row: pd.Series) -> float:
    """Calculate Volume Acceleration"""
    try:
        vol_30d = safe_float(row.get('volume_30d', 0))
        vol_90d = safe_float(row.get('volume_90d', 0))
        vol_180d = safe_float(row.get('volume_180d', 0))
        
        if vol_90d > 0 and vol_180d > 0:
            ratio_30_90 = vol_30d / vol_90d
            ratio_30_180 = vol_30d / vol_180d
            acceleration = ratio_30_90 - ratio_30_180
            return acceleration
        return 0
    except:
        return 0

def calculate_traffic_light_signal(row: pd.Series, market_health: Dict) -> Dict[str, Any]:
    """Calculate traffic light signal with all factors"""
    try:
        # Volume Story (40% weight)
        volume_score = 0
        rvol = safe_float(row.get('rvol', 0))
        vol_ratio_90d_180d = safe_float(row.get('vol_ratio_90d_180d', 0))
        vol_acceleration = calculate_volume_acceleration(row)
        
        if rvol > 2:
            volume_score += 30
        elif rvol > 1.5:
            volume_score += 20
        elif rvol > 1:
            volume_score += 10
            
        if vol_ratio_90d_180d > 1:
            volume_score += 30
        elif vol_ratio_90d_180d > 0.9:
            volume_score += 20
            
        if vol_acceleration > 0.1:
            volume_score += 40
        elif vol_acceleration > 0:
            volume_score += 20
        
        # Position Story (30% weight)
        position_score = 0
        from_low = safe_float(row.get('from_low_pct', 0))
        from_high = safe_float(row.get('from_high_pct', 0))
        
        if from_low < 40:
            position_score += 50
        elif from_low < 60:
            position_score += 30
        elif from_low < 80:
            position_score += 10
            
        if from_high < -20:
            position_score += 50
        elif from_high < -10:
            position_score += 30
        
        # Momentum Story (30% weight)
        momentum_score = 0
        ret_7d = safe_float(row.get('ret_7d', 0))
        ret_30d = safe_float(row.get('ret_30d', 0))
        price = safe_float(row.get('price', 0))
        sma_20d = safe_float(row.get('sma_20d', 0))
        
        if ret_7d > 0:
            momentum_score += 30
        if ret_30d > 0 and ret_7d > ret_30d / 4:
            momentum_score += 40
        if price > sma_20d and sma_20d > 0:
            momentum_score += 30
        
        # Calculate total score
        total_score = (
            volume_score * 0.4 +
            position_score * 0.3 +
            momentum_score * 0.3
        )
        
        # Market regime adjustment
        if market_health.get('regime') == 'BEAR':
            total_score *= 0.8  # More conservative in bear market
        elif market_health.get('regime') == 'BULL':
            total_score *= 1.1  # More aggressive in bull market
        
        # Liquidity adjustment
        liquidity_tier = row.get('liquidity_tier', 'D')
        if pd.isna(liquidity_tier):
            liquidity_tier = 'D'
        
        if liquidity_tier == 'D':
            total_score *= 0.5  # Heavily penalize illiquid stocks
        elif liquidity_tier == 'C':
            total_score *= 0.8
        
        # Relative strength bonus
        rs_score = safe_float(row.get('relative_strength_score', 0))
        if rs_score > 5:
            total_score += 10
        elif rs_score > 0:
            total_score += 5
        
        # Determine signal
        if total_score >= 80:
            signal = 'GREEN'
            action = 'BUY NOW'
            confidence = min(95, total_score)
        elif total_score >= 60:
            signal = 'YELLOW'
            action = 'WATCH CLOSELY'
            confidence = total_score
        else:
            signal = 'RED'
            action = 'AVOID/EXIT'
            confidence = 100 - total_score
        
        # Calculate targets and stops
        if signal == 'GREEN':
            target_pct = 20 if from_low < 40 else 15
            stop_pct = 5
        elif signal == 'YELLOW':
            target_pct = 15 if from_low < 50 else 10
            stop_pct = 7
        else:
            target_pct = 0
            stop_pct = 3
        
        target_price = price * (1 + target_pct / 100)
        stop_price = price * (1 - stop_pct / 100)
        
        # Position size recommendation
        if signal == 'GREEN' and liquidity_tier in ['A', 'B']:
            position_size = POSITION_SIZE_RULES.get(f'{signal}_{liquidity_tier}', 0.01)
        else:
            position_size = POSITION_SIZE_RULES.get('DEFAULT', 0.005)
        
        return {
            'signal': signal,
            'action': action,
            'score': total_score,
            'confidence': confidence,
            'target_price': target_price,
            'target_pct': target_pct,
            'stop_price': stop_price,
            'stop_pct': stop_pct,
            'position_size': position_size,
            'volume_score': volume_score,
            'position_score': position_score,
            'momentum_score': momentum_score,
            'entry_price': price
        }
        
    except Exception as e:
        logger.error(f"Error calculating signal: {str(e)}")
        return {
            'signal': 'RED',
            'action': 'ERROR',
            'score': 0,
            'confidence': 0
        }

def check_exit_signals(row: pd.Series) -> Dict[str, Any]:
    """Check for exit signals"""
    try:
        exit_reasons = []
        exit_urgency = 'NORMAL'
        
        # Volume exhaustion
        rvol = safe_float(row.get('rvol', 0))
        if rvol < 0.5:
            exit_reasons.append('Volume dried up')
            
        # Momentum failure
        ret_1d = safe_float(row.get('ret_1d', 0))
        ret_3d = safe_float(row.get('ret_3d', 0))
        if ret_1d < -3:
            exit_reasons.append('Sharp daily decline')
            exit_urgency = 'HIGH'
        if ret_3d < -5:
            exit_reasons.append('3-day momentum broken')
            exit_urgency = 'HIGH'
            
        # Position exhaustion
        from_high = safe_float(row.get('from_high_pct', 0))
        if from_high > -5 and ret_1d < 0:
            exit_reasons.append('Rejected at highs')
            
        # Technical breakdown
        price = safe_float(row.get('price', 0))
        sma_20d = safe_float(row.get('sma_20d', 0))
        if price < sma_20d and sma_20d > 0:
            exit_reasons.append('Below SMA20')
        
        return {
            'has_exit_signal': len(exit_reasons) > 0,
            'exit_reasons': exit_reasons,
            'exit_urgency': exit_urgency
        }
        
    except Exception as e:
        logger.error(f"Error checking exit signals: {str(e)}")
        return {'has_exit_signal': False, 'exit_reasons': []}

# ============================================
# PORTFOLIO MANAGEMENT
# ============================================

def check_portfolio_constraints(df: pd.DataFrame, selected_stocks: List[str]) -> Dict[str, Any]:
    """Check portfolio concentration and constraints"""
    try:
        selected_df = df[df['ticker'].isin(selected_stocks)]
        
        warnings = []
        
        # Sector concentration
        sector_counts = selected_df['sector'].value_counts()
        total_selected = len(selected_stocks)
        
        for sector, count in sector_counts.items():
            concentration = count / total_selected if total_selected > 0 else 0
            if concentration > PORTFOLIO_CONSTRAINTS['MAX_PER_SECTOR']:
                warnings.append(f"High concentration in {sector}: {concentration:.1%}")
        
        # Category concentration
        category_counts = selected_df['category'].value_counts()
        for category, count in category_counts.items():
            concentration = count / total_selected if total_selected > 0 else 0
            if concentration > PORTFOLIO_CONSTRAINTS['MAX_PER_CATEGORY']:
                warnings.append(f"High concentration in {category}: {concentration:.1%}")
        
        # Correlation check (simplified - same sector = correlated)
        if sector_counts.max() > PORTFOLIO_CONSTRAINTS['MAX_CORRELATED_POSITIONS']:
            warnings.append(f"Too many correlated positions in same sector")
        
        return {
            'warnings': warnings,
            'sector_distribution': sector_counts.to_dict(),
            'category_distribution': category_counts.to_dict(),
            'is_diversified': len(warnings) == 0
        }
        
    except Exception as e:
        logger.error(f"Error checking portfolio constraints: {str(e)}")
        return {'warnings': [], 'is_diversified': True}

# ============================================
# PERFORMANCE TRACKING
# ============================================

def load_performance_history():
    """Load historical performance tracking data"""
    try:
        if os.path.exists('performance_history.json'):
            with open('performance_history.json', 'r') as f:
                return json.load(f)
        return {}
    except:
        return {}

def save_performance_history(data):
    """Save performance tracking data"""
    try:
        with open('performance_history.json', 'w') as f:
            json.dump(data, f)
    except Exception as e:
        logger.error(f"Error saving performance history: {str(e)}")

def track_signal_performance(signal_data: Dict):
    """Track performance of signals"""
    history = load_performance_history()
    
    date_key = datetime.now().strftime('%Y-%m-%d')
    if date_key not in history:
        history[date_key] = []
    
    history[date_key].append(signal_data)
    save_performance_history(history)

# ============================================
# REPORT GENERATION
# ============================================

def generate_professional_excel_report(df: pd.DataFrame, analysis_results: Dict) -> BytesIO:
    """Generate comprehensive Excel report with all improvements"""
    output = BytesIO()
    
    try:
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            workbook = writer.book
            
            # Define formats
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'fg_color': '#1e3c72',
                'font_color': 'white',
                'border': 1
            })
            
            green_format = workbook.add_format({'bg_color': '#d4edda', 'border': 1})
            yellow_format = workbook.add_format({'bg_color': '#fff3cd', 'border': 1})
            red_format = workbook.add_format({'bg_color': '#f8d7da', 'border': 1})
            
            number_format = workbook.add_format({'num_format': '#,##0.00'})
            percent_format = workbook.add_format({'num_format': '0.00%'})
            currency_format = workbook.add_format({'num_format': '₹#,##0.00'})
            
            # Sheet 1: Executive Summary
            summary_data = {
                'Metric': [
                    'Analysis Date',
                    'Market Regime',
                    'Market Score',
                    'Total Stocks Analyzed',
                    'Green Signals (BUY)',
                    'Yellow Signals (WATCH)',
                    'Red Signals (AVOID)',
                    'Avg Market Return (30d)',
                    'Top Performing Sector',
                    'Weakest Sector'
                ],
                'Value': [
                    datetime.now().strftime('%Y-%m-%d %H:%M'),
                    analysis_results['market_health']['regime'],
                    f"{analysis_results['market_health']['market_score']}/100",
                    len(df),
                    len(analysis_results['green_signals']),
                    len(analysis_results['yellow_signals']),
                    len(analysis_results['red_signals']),
                    f"{analysis_results['market_health']['avg_return_30d']:.2f}%",
                    analysis_results['top_sector'],
                    analysis_results['bottom_sector']
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Executive Summary', index=False)
            
            # Sheet 2: Action Items (Most Important)
            action_items = []
            
            # Add GREEN signals
            if not analysis_results['green_signals'].empty:
                for _, row in analysis_results['green_signals'].iterrows():
                    action_items.append({
                        'Action': 'BUY',
                        'Ticker': row['ticker'],
                        'Company': row['company_name'][:50] if len(row['company_name']) > 50 else row['company_name'],
                        'Entry Price': row['signal_details']['entry_price'],
                        'Target': row['signal_details']['target_price'],
                        'Stop Loss': row['signal_details']['stop_price'],
                        'Position Size': f"{row['signal_details']['position_size']*100:.1f}%",
                        'Score': row['signal_details']['score'],
                        'Liquidity': row.get('liquidity_tier', 'N/A')
                    })
            
            # Add exit signals
            if not analysis_results.get('exit_signals', pd.DataFrame()).empty:
                for _, row in analysis_results['exit_signals'].iterrows():
                    action_items.append({
                        'Action': 'SELL',
                        'Ticker': row['ticker'],
                        'Company': row['company_name'][:50] if len(row['company_name']) > 50 else row['company_name'],
                        'Current Price': row.get('price', 0),
                        'Reason': ', '.join(row['exit_details']['exit_reasons'][:2]),  # Limit reasons
                        'Urgency': row['exit_details'].get('exit_urgency', 'NORMAL')
                    })
            
            if action_items:
                action_df = pd.DataFrame(action_items)
                action_df.to_excel(writer, sheet_name='ACTION ITEMS', index=False)
                
                # Format the action sheet
                worksheet = writer.sheets['ACTION ITEMS']
                for idx, row in action_df.iterrows():
                    if row.get('Action') == 'BUY':
                        worksheet.set_row(idx + 1, None, green_format)
                    elif row.get('Action') == 'SELL':
                        worksheet.set_row(idx + 1, None, red_format)
            
            # Sheet 3: Market Overview
            market_data = {
                'Indicator': [
                    'Stocks Above SMA20',
                    'Stocks Above SMA50',
                    'Stocks Above SMA200',
                    'Advance/Decline Ratio',
                    'New Highs',
                    'New Lows',
                    'High Volume Stocks',
                    'Volume Acceleration Stocks'
                ],
                'Value': [
                    f"{analysis_results['market_health']['above_sma20_pct']:.1f}%",
                    f"{analysis_results['market_health']['above_sma50_pct']:.1f}%",
                    f"{analysis_results['market_health']['above_sma200_pct']:.1f}%",
                    f"{analysis_results['market_health']['advance_decline_ratio']:.2f}",
                    analysis_results['market_health']['new_highs'],
                    analysis_results['market_health']['new_lows'],
                    analysis_results['market_health']['high_volume_stocks'],
                    analysis_results['market_health']['volume_acceleration_stocks']
                ]
            }
            market_df = pd.DataFrame(market_data)
            market_df.to_excel(writer, sheet_name='Market Overview', index=False)
            
            # Sheet 4: Sector Analysis
            if not df.empty:
                df_sectors = df[df['sector'].notna() & (df['sector'] != '') & (df['sector'] != 'nan')]
                if not df_sectors.empty:
                    sector_analysis = df_sectors.groupby('sector').agg({
                        'ticker': 'count',
                        'ret_30d': 'mean',
                        'ret_7d': 'mean',
                        'rvol': 'mean',
                        'volume_30d': 'mean'
                    }).round(2)
                    sector_analysis.columns = ['Stock Count', 'Avg 30d Return', 'Avg 7d Return', 
                                              'Avg RVol', 'Avg Volume']
                    sector_analysis = sector_analysis.sort_values('Avg 30d Return', ascending=False)
                    sector_analysis.to_excel(writer, sheet_name='Sector Analysis')
                else:
                    # Create empty sheet with message
                    empty_df = pd.DataFrame({'Message': ['No sector data available']})
                    empty_df.to_excel(writer, sheet_name='Sector Analysis', index=False)
            
            # Sheet 5: Green Signals Detail
            if 'green_signals' in analysis_results and not analysis_results['green_signals'].empty:
                # Get available columns
                available_cols = analysis_results['green_signals'].columns.tolist()
                
                # Select columns that exist
                detail_cols = []
                for col in ['ticker', 'company_name', 'sector', 'price', 'ret_7d', 'ret_30d',
                           'rvol', 'from_low_pct', 'liquidity_tier']:
                    if col in available_cols:
                        detail_cols.append(col)
                
                if detail_cols:
                    green_detail = analysis_results['green_signals'][detail_cols].copy()
                    
                    # Add signal details safely
                    if 'signal_details' in available_cols:
                        for col in ['score', 'target_price', 'stop_price', 'position_size']:
                            try:
                                green_detail[col] = analysis_results['green_signals'].apply(
                                    lambda x: x['signal_details'].get(col, 0) if isinstance(x['signal_details'], dict) else 0, 
                                    axis=1
                                )
                            except:
                                green_detail[col] = 0
                    
                    green_detail.to_excel(writer, sheet_name='Green Signals', index=False)
                else:
                    empty_df = pd.DataFrame({'Message': ['No data available for green signals']})
                    empty_df.to_excel(writer, sheet_name='Green Signals', index=False)
            else:
                # Create empty sheet with message
                empty_df = pd.DataFrame({'Message': ['No green signals found']})
                empty_df.to_excel(writer, sheet_name='Green Signals', index=False)
            
            # Sheet 6: Portfolio Warnings
            if 'portfolio_warnings' in analysis_results:
                warnings_df = pd.DataFrame({
                    'Warning Type': analysis_results['portfolio_warnings']
                })
                warnings_df.to_excel(writer, sheet_name='Portfolio Warnings', index=False)
            
            # Auto-fit columns for all sheets
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                worksheet.set_zoom(90)
                
                # Get the dimensions of the dataframe
                if sheet_name == 'Executive Summary':
                    max_col = 2
                else:
                    max_col = 10
                    
                for i in range(max_col):
                    worksheet.set_column(i, i, 15)
        
        output.seek(0)
        return output
        
    except Exception as e:
        logger.error(f"Error generating Excel report: {str(e)}")
        output.seek(0)
        return output

# ============================================
# VISUALIZATION FUNCTIONS
# ============================================

def create_market_health_gauge(market_score: int) -> go.Figure:
    """Create market health gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=market_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Market Health Score"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 40], 'color': "lightgray"},
                {'range': [40, 60], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def create_sector_performance_chart(df: pd.DataFrame) -> go.Figure:
    """Create sector performance comparison chart"""
    try:
        # Filter out empty sectors
        df_clean = df[df['sector'].notna() & (df['sector'] != '') & (df['sector'] != 'nan')]
        
        if df_clean.empty:
            # Return empty chart
            fig = go.Figure()
            fig.add_annotation(
                text="No sector data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(height=600)
            return fig
        
        sector_perf = df_clean.groupby('sector').agg({
            'ret_30d': 'mean',
            'ticker': 'count'
        }).round(2)
        
        # Filter sectors with at least minimum stocks
        sector_perf = sector_perf[sector_perf['ticker'] >= PORTFOLIO_CONSTRAINTS['MIN_STOCKS_FOR_SECTOR_AVG']]
        
        if sector_perf.empty:
            # Return empty chart
            fig = go.Figure()
            fig.add_annotation(
                text="No sectors with sufficient stocks for analysis",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(height=600)
            return fig
        
        sector_perf = sector_perf.sort_values('ret_30d', ascending=True).tail(15)
        
        # Create color scale
        colors = ['red' if x < 0 else 'green' for x in sector_perf['ret_30d']]
        
        fig = go.Figure(go.Bar(
            x=sector_perf['ret_30d'],
            y=sector_perf.index,
            orientation='h',
            marker_color=colors,
            text=[f"{x:.1f}% ({sector_perf.loc[idx, 'ticker']} stocks)" 
                  for idx, x in zip(sector_perf.index, sector_perf['ret_30d'])],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Top 15 Sectors by 30-Day Performance",
            xaxis_title="Average 30-Day Return (%)",
            yaxis_title="Sector",
            height=600,
            margin=dict(l=200)
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating sector performance chart: {str(e)}")
        # Return empty chart on error
        fig = go.Figure()
        fig.add_annotation(
            text="Error creating sector chart",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(height=600)
        return fig

def create_signal_distribution_chart(signal_counts: Dict) -> go.Figure:
    """Create signal distribution pie chart"""
    labels = ['Green (BUY)', 'Yellow (WATCH)', 'Red (AVOID)']
    values = [signal_counts.get('green', 0), 
              signal_counts.get('yellow', 0), 
              signal_counts.get('red', 0)]
    colors = ['#28a745', '#ffc107', '#dc3545']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker_colors=colors,
        hole=0.3
    )])
    
    fig.update_layout(
        title="Signal Distribution",
        height=400
    )
    
    return fig

def create_liquidity_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Create liquidity tier distribution chart"""
    try:
        if 'liquidity_tier' not in df.columns or df.empty:
            # Return empty chart
            fig = go.Figure()
            fig.add_annotation(
                text="No liquidity data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(height=400)
            return fig
        
        liquidity_counts = df['liquidity_tier'].value_counts().sort_index()
        
        if liquidity_counts.empty:
            # Return empty chart
            fig = go.Figure()
            fig.add_annotation(
                text="No liquidity data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(height=400)
            return fig
        
        colors = {
            'A': '#28a745',  # Green
            'B': '#20c997',  # Teal
            'C': '#ffc107',  # Yellow
            'D': '#dc3545'   # Red
        }
        
        fig = go.Figure(data=[go.Bar(
            x=liquidity_counts.index,
            y=liquidity_counts.values,
            marker_color=[colors.get(x, '#6c757d') for x in liquidity_counts.index],
            text=liquidity_counts.values,
            textposition='auto'
        )])
        
        fig.update_layout(
            title="Stock Distribution by Liquidity Tier",
            xaxis_title="Liquidity Tier (A=Best, D=Worst)",
            yaxis_title="Number of Stocks",
            height=400
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating liquidity distribution chart: {str(e)}")
        # Return empty chart on error
        fig = go.Figure()
        fig.add_annotation(
            text="Error creating liquidity chart",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(height=400)
        return fig

# ============================================
# MAIN APPLICATION
# ============================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🚦 Wave Detection System 2.0</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #6c757d;">Professional Stock Analysis with Traffic Light Signals</p>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📊 Data Source")
        sheet_url = st.text_input(
            "Google Sheets URL",
            value="https://docs.google.com/spreadsheets/d/1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk/edit?usp=sharing",
            help="Enter your Google Sheets URL"
        )
        gid = st.text_input("Sheet GID", value="2026492216", help="Sheet identifier")
        
        if st.button("🔄 Refresh Data", type="primary"):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        
        # Load data
        df = load_google_sheets_data(sheet_url, gid)
        
        if not df.empty:
            df = validate_and_clean_data(df)
            
            # Calculate market health
            market_health = calculate_market_health(df)
            
            # Ensure market_health has required keys
            if not market_health or 'regime' not in market_health:
                market_health = {
                    'regime': 'UNKNOWN',
                    'regime_color': 'gray',
                    'total_stocks': len(df),
                    'market_score': 50,
                    'above_sma20_pct': 0,
                    'above_sma50_pct': 0,
                    'above_sma200_pct': 0,
                    'advance_decline_ratio': 1,
                    'new_highs': 0,
                    'new_lows': 0,
                    'high_low_ratio': 1,
                    'avg_return_1d': 0,
                    'avg_return_30d': 0,
                    'avg_rvol': 1,
                    'high_volume_stocks': 0,
                    'volume_acceleration_stocks': 0
                }
            
            # Filters
            st.markdown("## 🔍 Filters")
            
            # Market regime filter
            apply_regime_filter = st.checkbox(
                "Apply Market Regime Filter",
                value=True,
                help="Adjust signals based on market conditions"
            )
            
            # Liquidity filter
            min_liquidity = st.selectbox(
                "Minimum Liquidity Tier",
                options=['A', 'B', 'C', 'D'],
                index=1,  # Default to B
                help="A=Institutional, B=Tradeable, C=Risky, D=Avoid"
            )
            
            # Category filter
            categories = ['All'] + sorted(df['category'].dropna().unique().tolist())
            selected_categories = st.multiselect("Category", categories, default=['All'])
            
            # Sector filter - with stock count
            sector_options = []
            sector_counts = df['sector'].value_counts()
            for sector, count in sector_counts.items():
                if pd.notna(sector) and sector != '' and sector != 'nan':
                    sector_options.append(f"{sector} ({count} stocks)")
            
            selected_sectors_with_count = st.multiselect(
                "Sector (with stock count)",
                ['All'] + sector_options,
                default=['All']
            )
            
            # Extract sector names without counts
            selected_sectors = []
            if 'All' not in selected_sectors_with_count:
                for sector_with_count in selected_sectors_with_count:
                    # Handle the case where sector name might contain parentheses
                    if ' (' in sector_with_count and sector_with_count.endswith(')'):
                        sector_name = sector_with_count.rsplit(' (', 1)[0]
                        selected_sectors.append(sector_name)
                    else:
                        # Fallback in case format is different
                        selected_sectors.append(sector_with_count)
            
            # Relative strength filter
            min_rs_score = st.slider(
                "Minimum Relative Strength Score",
                min_value=-20.0,
                max_value=20.0,
                value=0.0,
                step=1.0,
                help="Stock performance vs sector/market"
            )
            
            # Settings
            st.markdown("---")
            st.markdown("## ⚙️ Settings")
            
            position_limit = st.number_input(
                "Max Positions",
                min_value=1,
                max_value=50,
                value=10,
                help="Maximum number of positions to show"
            )
            
            show_warnings = st.checkbox(
                "Show Portfolio Warnings",
                value=True,
                help="Alert for concentration risks"
            )
    
    # Main content
    if not df.empty:
        # Apply filters
        filtered_df = df.copy()
        
        # Liquidity filter
        if 'liquidity_tier' in filtered_df.columns:
            liquidity_tiers = ['A', 'B', 'C', 'D']
            selected_tiers = liquidity_tiers[liquidity_tiers.index(min_liquidity):]
            filtered_df = filtered_df[filtered_df['liquidity_tier'].isin(selected_tiers)]
        
        # Category filter
        if selected_categories and 'All' not in selected_categories:
            filtered_df = filtered_df[filtered_df['category'].isin(selected_categories)]
        
        # Sector filter
        if selected_sectors and 'All' not in selected_sectors_with_count:
            filtered_df = filtered_df[filtered_df['sector'].isin(selected_sectors)]
        
        # Relative strength filter
        if 'relative_strength_score' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['relative_strength_score'] >= min_rs_score]
        
        # Calculate signals for all stocks
        filtered_df['signal_details'] = filtered_df.apply(
            lambda x: calculate_traffic_light_signal(x, market_health), axis=1
        )
        
        # Extract signal type and score for easier filtering
        filtered_df['signal'] = filtered_df['signal_details'].apply(lambda x: x['signal'])
        filtered_df['signal_score'] = filtered_df['signal_details'].apply(lambda x: x.get('score', 0))
        
        # Separate by signal type
        green_signals = filtered_df[filtered_df['signal'] == 'GREEN'].nlargest(
            position_limit, 
            'signal_score'
        )
        yellow_signals = filtered_df[filtered_df['signal'] == 'YELLOW'].nlargest(
            position_limit, 
            'signal_score'
        )
        red_signals = filtered_df[filtered_df['signal'] == 'RED']
        
        # Check for exit signals in current positions (simulated)
        # In real implementation, this would load from portfolio
        
        # Tab content continues...
        
        # Market Health Dashboard
        st.markdown("## 🌍 Market Health Dashboard")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: {market_health['regime_color']};">{market_health['regime']} MARKET</h3>
                <p>Market Score: {market_health['market_score']}/100</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            fig_gauge = create_market_health_gauge(market_health['market_score'])
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Market Stats</h4>
                <p>Above SMA50: {market_health['above_sma50_pct']:.1f}%</p>
                <p>A/D Ratio: {market_health['advance_decline_ratio']:.2f}</p>
                <p>New H/L: {market_health['high_low_ratio']:.1f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Signal Summary
        signal_counts = {
            'green': len(green_signals),
            'yellow': len(yellow_signals),
            'red': len(red_signals)
        }
        
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🟢 Green Signals", signal_counts['green'])
        with col2:
            st.metric("🟡 Yellow Signals", signal_counts['yellow'])
        with col3:
            st.metric("🔴 Red Signals", signal_counts['red'])
        with col4:
            st.metric("📊 Total Analyzed", len(filtered_df))
        
        # Tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🟢 Green Lights (BUY)", "🟡 Yellow Lights (WATCH)", 
            "🔴 Exit Signals", "📊 Market Analysis", 
            "📈 Reports", "🔧 System Info"
        ])
        
        with tab1:
            st.markdown("## 🟢 GREEN LIGHTS - Buy These Now!")
            
            if not green_signals.empty:
                # Check portfolio constraints if showing warnings
                if show_warnings:
                    selected_tickers = green_signals['ticker'].tolist()
                    portfolio_check = check_portfolio_constraints(df, selected_tickers)
                    
                    if portfolio_check['warnings']:
                        st.warning("⚠️ Portfolio Concentration Warnings:")
                        for warning in portfolio_check['warnings']:
                            st.write(f"• {warning}")
                
                # Display green signals
                for idx, (_, row) in enumerate(green_signals.iterrows()):
                    signal_details = row['signal_details']
                    
                    # Determine card color based on liquidity
                    liquidity_tier = row.get('liquidity_tier', 'D')
                    if pd.isna(liquidity_tier):
                        liquidity_tier = 'D'
                    
                    if liquidity_tier == 'A':
                        card_class = "green-signal"
                    elif liquidity_tier == 'B':
                        card_class = "green-signal"
                    else:
                        card_class = "yellow-signal"  # Lower liquidity gets yellow background
                    
                    st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
                    
                    col1, col2, col3, col4, col5 = st.columns([2.5, 1.5, 1.5, 1.5, 1])
                    
                    with col1:
                        st.markdown(f"### {row['ticker']}")
                        st.markdown(f"{row['company_name'][:60]}...")
                        sector_count = row.get('sector_stock_count', 0)
                        st.markdown(f'<span class="sector-badge">{row["sector"]} ({int(sector_count)} stocks)</span>', 
                                  unsafe_allow_html=True)
                        if liquidity_tier not in ['A', 'B']:
                            st.markdown(f'<span class="liquidity-warning">⚠️ Low Liquidity (Tier {liquidity_tier})</span>', 
                                      unsafe_allow_html=True)
                    
                    with col2:
                        st.metric("Entry Price", f"₹{signal_details['entry_price']:.2f}")
                        st.metric("Score", f"{signal_details['score']:.0f}/100")
                    
                    with col3:
                        st.metric("Target", f"₹{signal_details['target_price']:.2f}")
                        st.metric("Upside", f"+{signal_details['target_pct']}%")
                    
                    with col4:
                        st.metric("Stop Loss", f"₹{signal_details['stop_price']:.2f}")
                        st.metric("Risk", f"-{signal_details['stop_pct']}%")
                    
                    with col5:
                        st.metric("Position", f"{signal_details['position_size']*100:.1f}%")
                        risk_reward = signal_details['target_pct'] / signal_details['stop_pct']
                        st.metric("R:R", f"{risk_reward:.1f}")
                    
                    # Additional details
                    with st.expander("📊 View Details"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.write("**Performance Metrics:**")
                            st.write(f"• 7-day Return: {row['ret_7d']:.1f}%")
                            st.write(f"• 30-day Return: {row['ret_30d']:.1f}%")
                            st.write(f"• From 52w Low: {row['from_low_pct']:.0f}%")
                        
                        with col2:
                            st.write("**Volume Metrics:**")
                            st.write(f"• Relative Volume: {row['rvol']:.1f}x")
                            vol_accel = calculate_volume_acceleration(row)
                            st.write(f"• Volume Acceleration: {vol_accel:.3f}")
                            st.write(f"• 90d/180d Ratio: {row['vol_ratio_90d_180d']:.2f}")
                        
                        with col3:
                            st.write("**Signal Breakdown:**")
                            st.write(f"• Volume Score: {signal_details['volume_score']:.0f}/100")
                            st.write(f"• Position Score: {signal_details['position_score']:.0f}/100")
                            st.write(f"• Momentum Score: {signal_details['momentum_score']:.0f}/100")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown("")
                
            else:
                st.info("No GREEN signals found with current filters. Try adjusting your criteria.")
        
        with tab2:
            st.markdown("## 🟡 YELLOW LIGHTS - Watch These Closely")
            
            if not yellow_signals.empty:
                for _, row in yellow_signals.iterrows():
                    signal_details = row['signal_details']
                    
                    st.markdown('<div class="yellow-signal">', unsafe_allow_html=True)
                    
                    col1, col2, col3, col4 = st.columns([3, 1.5, 1.5, 2])
                    
                    with col1:
                        st.markdown(f"### {row['ticker']}")
                        st.markdown(f"{row['company_name'][:60]}...")
                        sector_count = row.get('sector_stock_count', 0)
                        st.markdown(f"**Sector:** {row['sector']} ({int(sector_count)} stocks)")
                    
                    with col2:
                        st.metric("Current Price", f"₹{row['price']:.2f}")
                        st.metric("Score", f"{signal_details['score']:.0f}/100")
                    
                    with col3:
                        st.metric("From Low", f"{row['from_low_pct']:.0f}%")
                        st.metric("RVol", f"{row['rvol']:.1f}x")
                    
                    with col4:
                        st.markdown("**Watch For:**")
                        if row['rvol'] < 2:
                            st.write("• Volume spike (rvol > 2)")
                        if row['ret_7d'] <= 0:
                            st.write("• Positive momentum")
                        if row['price'] < row['sma_20d']:
                            st.write("• Break above SMA20")
                        st.write(f"• Entry: ₹{row['price'] * 1.02:.2f}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown("")
            else:
                st.info("No YELLOW signals found with current filters.")
        
        with tab3:
            st.markdown("## 🔴 EXIT SIGNALS - Sell or Avoid These")
            
            # Check exit signals for all stocks
            all_exit_checks = filtered_df.copy()
            all_exit_checks['exit_details'] = all_exit_checks.apply(check_exit_signals, axis=1)
            exit_signals = all_exit_checks[all_exit_checks['exit_details'].apply(lambda x: x.get('has_exit_signal', False))]
            
            if not exit_signals.empty:
                # Separate by urgency
                high_urgency = exit_signals[exit_signals['exit_details'].apply(lambda x: x['exit_urgency'] == 'HIGH')]
                normal_urgency = exit_signals[exit_signals['exit_details'].apply(lambda x: x['exit_urgency'] == 'NORMAL')]
                
                if not high_urgency.empty:
                    st.markdown("### 🚨 HIGH URGENCY EXITS")
                    for _, row in high_urgency.iterrows():
                        st.markdown('<div class="red-signal">', unsafe_allow_html=True)
                        
                        col1, col2, col3 = st.columns([3, 2, 2])
                        
                        with col1:
                            st.markdown(f"### {row['ticker']} - SELL NOW")
                            st.markdown(f"{row['company_name'][:60]}...")
                        
                        with col2:
                            st.metric("Current Price", f"₹{row['price']:.2f}")
                            st.metric("Today's Return", f"{row['ret_1d']:.1f}%")
                        
                        with col3:
                            st.markdown("**Exit Reasons:**")
                            for reason in row['exit_details']['exit_reasons']:
                                st.write(f"• {reason}")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                
                if not normal_urgency.empty:
                    st.markdown("### ⚠️ Normal Priority Exits")
                    for _, row in normal_urgency.head(10).iterrows():
                        col1, col2, col3 = st.columns([3, 2, 3])
                        
                        with col1:
                            st.write(f"**{row['ticker']}** - {row['company_name'][:40]}...")
                        
                        with col2:
                            st.write(f"₹{row['price']:.2f} ({row['ret_1d']:.1f}%)")
                        
                        with col3:
                            reasons = ", ".join(row['exit_details']['exit_reasons'])
                            st.write(f"Exit: {reasons}")
            else:
                st.success("No immediate exit signals detected!")
        
        with tab4:
            st.markdown("## 📊 Market Analysis")
            
            # Market metrics
            col1, col2 = st.columns(2)
            
            with col1:
                # Sector performance
                st.plotly_chart(create_sector_performance_chart(filtered_df), use_container_width=True)
            
            with col2:
                # Signal distribution
                st.plotly_chart(create_signal_distribution_chart(signal_counts), use_container_width=True)
            
            # Liquidity analysis
            st.plotly_chart(create_liquidity_distribution_chart(filtered_df), use_container_width=True)
            
            # Market breadth indicators
            st.markdown("### 📈 Market Breadth Indicators")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                ad_ratio = market_health['advance_decline_ratio']
                color = "green" if ad_ratio > 1.5 else "red" if ad_ratio < 0.67 else "orange"
                st.markdown(f"""
                <div class="metric-card">
                    <h4>A/D Ratio</h4>
                    <h2 style="color: {color};">{ad_ratio:.2f}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                hl_ratio = market_health['high_low_ratio']
                color = "green" if hl_ratio > 2 else "red" if hl_ratio < 0.5 else "orange"
                st.markdown(f"""
                <div class="metric-card">
                    <h4>High/Low Ratio</h4>
                    <h2 style="color: {color};">{hl_ratio:.1f}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                vol_stocks = market_health['volume_acceleration_stocks']
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Volume Accel</h4>
                    <h2>{vol_stocks}</h2>
                    <p>stocks</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                avg_rvol = market_health['avg_rvol']
                color = "green" if avg_rvol > 1.2 else "red" if avg_rvol < 0.8 else "orange"
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Avg RVol</h4>
                    <h2 style="color: {color};">{avg_rvol:.2f}x</h2>
                </div>
                """, unsafe_allow_html=True)
        
        with tab5:
            st.markdown("## 📈 Download Professional Reports")
            
            # Prepare analysis results
            if not filtered_df.empty:
                sector_perfs = filtered_df[filtered_df['sector'].notna()].groupby('sector')['ret_30d'].mean()
                top_sector = sector_perfs.idxmax() if not sector_perfs.empty else "N/A"
                bottom_sector = sector_perfs.idxmin() if not sector_perfs.empty else "N/A"
            else:
                top_sector = "N/A"
                bottom_sector = "N/A"
            
            analysis_results = {
                'market_health': market_health,
                'green_signals': green_signals,
                'yellow_signals': yellow_signals,
                'red_signals': red_signals,
                'exit_signals': exit_signals if 'exit_signals' in locals() else pd.DataFrame(),
                'top_sector': top_sector,
                'bottom_sector': bottom_sector,
                'portfolio_warnings': []
            }
            
            # Add portfolio warnings if enabled
            if show_warnings and not green_signals.empty:
                portfolio_check = check_portfolio_constraints(df, green_signals['ticker'].tolist())
                analysis_results['portfolio_warnings'] = portfolio_check['warnings']
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📊 Generate Excel Report", type="primary"):
                    with st.spinner("Generating comprehensive report..."):
                        excel_file = generate_professional_excel_report(filtered_df, analysis_results)
                        
                        st.download_button(
                            label="📥 Download Excel Report",
                            data=excel_file,
                            file_name=f"wave_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
            
            with col2:
                # Quick text summary
                if st.button("📝 Generate Text Summary", type="secondary"):
                    summary = f"""
WAVE DETECTION SYSTEM - DAILY SUMMARY
{datetime.now().strftime('%Y-%m-%d %H:%M')}

MARKET STATUS: {market_health['regime']} (Score: {market_health['market_score']}/100)

🟢 BUY NOW ({len(green_signals)} stocks):
"""
                    for _, row in green_signals.head(5).iterrows():
                        signal_details = row['signal_details']
                        summary += f"\n• {row['ticker']}: Entry ₹{signal_details['entry_price']:.0f}, Target ₹{signal_details['target_price']:.0f} (+{signal_details['target_pct']}%)"
                    
                    summary += f"\n\n🟡 WATCH LIST ({len(yellow_signals)} stocks):"
                    for _, row in yellow_signals.head(5).iterrows():
                        summary += f"\n• {row['ticker']}: Watch for volume > 2x or break above ₹{row['price']*1.02:.0f}"
                    
                    if 'exit_signals' in locals() and not exit_signals.empty:
                        summary += f"\n\n🔴 EXIT SIGNALS ({len(exit_signals)} stocks):"
                        for _, row in exit_signals.head(3).iterrows():
                            reasons = ", ".join(row['exit_details']['exit_reasons'][:2])
                            summary += f"\n• {row['ticker']}: {reasons}"
                    
                    summary += f"\n\nTOP SECTOR: {top_sector}"
                    summary += f"\nWEAK SECTOR: {bottom_sector}"
                    
                    if analysis_results['portfolio_warnings']:
                        summary += "\n\n⚠️ WARNINGS:"
                        for warning in analysis_results['portfolio_warnings'][:3]:
                            summary += f"\n• {warning}"
                    
                    st.text_area("Summary", summary, height=400)
        
        with tab6:
            st.markdown("## 🔧 System Information")
            
            # Data quality metrics
            st.markdown("### 📊 Data Quality")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Stocks", len(df))
                st.metric("After Filters", len(filtered_df))
            
            with col2:
                st.metric("Data Freshness", datetime.now().strftime('%H:%M'))
                total_cells = len(df) * len(df.columns)
                if total_cells > 0:
                    missing_pct = (df.isnull().sum().sum() / total_cells * 100)
                    st.metric("Data Completeness", f"{100-missing_pct:.1f}%")
                else:
                    st.metric("Data Completeness", "N/A")
            
            with col3:
                st.metric("Sectors Covered", df['sector'].nunique())
                st.metric("Categories", df['category'].nunique())
            
            with col4:
                if 'liquidity_tier' in df.columns:
                    high_quality = len(df[df['liquidity_tier'].isin(['A', 'B'])])
                    st.metric("High Liquidity Stocks", high_quality)
                    if len(df) > 0:
                        st.metric("Liquidity %", f"{high_quality/len(df)*100:.1f}%")
                    else:
                        st.metric("Liquidity %", "0%")
                else:
                    st.metric("High Liquidity Stocks", "N/A")
                    st.metric("Liquidity %", "N/A")
            
            # System configuration
            st.markdown("### ⚙️ Configuration")
            
            config_data = {
                'Parameter': [
                    'Market Regime Filter',
                    'Min Liquidity Tier',
                    'Position Limit',
                    'Max per Sector',
                    'Max per Category',
                    'Data Cache Time'
                ],
                'Value': [
                    'Enabled' if apply_regime_filter else 'Disabled',
                    min_liquidity,
                    position_limit,
                    f"{PORTFOLIO_CONSTRAINTS['MAX_PER_SECTOR']*100:.0f}%",
                    f"{PORTFOLIO_CONSTRAINTS['MAX_PER_CATEGORY']*100:.0f}%",
                    '5 minutes'
                ]
            }
            
            config_df = pd.DataFrame(config_data)
            st.dataframe(config_df, hide_index=True)
            
            # Performance tracking (if available)
            st.markdown("### 📈 Historical Performance")
            st.info("Performance tracking will be available after running the system for a few days.")
            
            # Version info
            st.markdown("---")
            st.markdown("""
            <div style="text-align: center; color: #6c757d; font-size: 0.9rem;">
                Wave Detection System v2.0.0<br>
                Professional Stock Analysis Platform<br>
                Last Updated: December 2024
            </div>
            """, unsafe_allow_html=True)
    
    else:
        st.error("No data loaded. Please check your Google Sheets URL and GID.")

if __name__ == "__main__":
    main()
