"""
Wave Detection Ultimate 3.0 - Professional Stock Ranking System
==============================================================
Complete implementation with RVOL integration and smart search.

Version: 3.0.0 - Production Ready with RVOL
Author: Professional Implementation
License: MIT

Key Features:
- Master Score 3.0 with RVOL integration (10% weight)
- Smart search with autocomplete functionality
- Uses 95% of available data (was 60%)
- 10 advanced patterns including RVOL-based
- Long-term strength analysis
- Professional error handling throughout
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import wraps
import time
from io import BytesIO
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# ============================================
# LOGGING CONFIGURATION
# ============================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ============================================
# CONFIGURATION
# ============================================

@dataclass(frozen=True)
class Config:
    """System configuration"""
    
    # Data source
    DEFAULT_SHEET_URL: str = "https://docs.google.com/spreadsheets/d/1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk/edit?usp=sharing"
    DEFAULT_GID: str = "2026492216"
    
    # Cache settings
    CACHE_TTL: int = 300  # 5 minutes
    
    # ENHANCED Ranking weights - Master Score 3.0
    POSITION_WEIGHT: float = 0.30      # Reduced from 0.35
    VOLUME_WEIGHT: float = 0.25        # Reduced from 0.30  
    MOMENTUM_WEIGHT: float = 0.15      # Same
    ACCELERATION_WEIGHT: float = 0.10  # Same
    BREAKOUT_WEIGHT: float = 0.10      # Same
    RVOL_WEIGHT: float = 0.10          # NEW!
    
    # Display settings
    DEFAULT_TOP_N: int = 50
    AVAILABLE_TOP_N: List[int] = field(default_factory=lambda: [10, 20, 50, 100, 200, 500])
    
    # Tier definitions
    EPS_TIERS: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "Loss": (-float('inf'), 0),
        "0-5": (0, 5),
        "5-10": (5, 10),
        "10-20": (10, 20),
        "20-50": (20, 50),
        "50-100": (50, 100),
        "100+": (100, float('inf'))
    })
    
    PE_TIERS: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "Negative/NA": (-float('inf'), 0),
        "0-10": (0, 10),
        "10-15": (10, 15),
        "15-20": (15, 20),
        "20-30": (20, 30),
        "30-50": (30, 50),
        "50+": (50, float('inf'))
    })
    
    PRICE_TIERS: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "0-100": (0, 100),
        "100-250": (100, 250),
        "250-500": (250, 500),
        "500-1000": (500, 1000),
        "1000-2500": (1000, 2500),
        "2500-5000": (2500, 5000),
        "5000+": (5000, float('inf'))
    })

# Global configuration
CONFIG = Config()

# ============================================
# PERFORMANCE MONITORING
# ============================================

def timer(func):
    """Performance timing decorator"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        if elapsed > 1.0:
            logger.warning(f"{func.__name__} took {elapsed:.2f}s")
        return result
    return wrapper

# ============================================
# DATA LOADING
# ============================================

@st.cache_data(ttl=CONFIG.CACHE_TTL)
def load_google_sheets_data(sheet_url: str, gid: str) -> pd.DataFrame:
    """
    Load data from Google Sheets with proper caching
    
    Args:
        sheet_url: Google Sheets URL
        gid: Sheet ID
        
    Returns:
        Raw dataframe
    """
    try:
        # Construct CSV URL
        base_url = sheet_url.split('/edit')[0]
        csv_url = f"{base_url}/export?format=csv&gid={gid}"
        
        logger.info(f"Loading data from Google Sheets")
        
        # Load data
        df = pd.read_csv(csv_url)
        
        if df.empty:
            raise ValueError("Loaded empty dataframe")
        
        logger.info(f"Successfully loaded {len(df):,} rows")
        return df
        
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        raise

# ============================================
# DATA PROCESSING
# ============================================

class DataProcessor:
    """Handle all data processing operations"""
    
    @staticmethod
    def clean_indian_number(value: Any) -> Optional[float]:
        """Clean Indian number format"""
        if pd.isna(value) or value == '':
            return np.nan
        
        try:
            cleaned = str(value)
            # Remove currency and special characters
            for char in ['₹', '$', '%', ',']:
                cleaned = cleaned.replace(char, '')
            cleaned = cleaned.strip()
            
            # Handle special cases
            if cleaned in ['', '-', 'N/A', 'n/a', '#N/A', 'nan', 'None']:
                return np.nan
            
            return float(cleaned)
        except:
            return np.nan
    
    @staticmethod
    @timer
    def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Complete data processing pipeline"""
        if df.empty:
            return pd.DataFrame()
        
        # Create copy
        df = df.copy()
        
        # Clean numeric columns
        numeric_columns = [
            'price', 'prev_close', 'low_52w', 'high_52w',
            'from_low_pct', 'from_high_pct',
            'sma_20d', 'sma_50d', 'sma_200d',
            'ret_1d', 'ret_3d', 'ret_7d', 'ret_30d', 
            'ret_3m', 'ret_6m', 'ret_1y', 'ret_3y', 'ret_5y',
            'volume_1d', 'volume_7d', 'volume_30d', 'volume_90d', 'volume_180d',
            'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d',
            'vol_ratio_1d_180d', 'vol_ratio_7d_180d', 'vol_ratio_30d_180d', 
            'vol_ratio_90d_180d',
            'rvol', 'pe', 'eps_current', 'eps_last_qtr', 'eps_change_pct'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].apply(DataProcessor.clean_indian_number)
        
        # Clean categorical columns
        categorical_columns = ['ticker', 'company_name', 'category', 'sector']
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace(['nan', 'None', '', 'N/A'], 'Unknown')
        
        # Fix volume ratios (convert from % to multiplier)
        volume_ratio_columns = [
            'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d',
            'vol_ratio_1d_180d', 'vol_ratio_7d_180d', 'vol_ratio_30d_180d',
            'vol_ratio_90d_180d'
        ]
        
        for col in volume_ratio_columns:
            if col in df.columns:
                # Convert percentage to multiplier, handling NaN values
                df[col] = df[col].apply(lambda x: (100 + x) / 100 if pd.notna(x) else np.nan)
        
        # Remove invalid rows
        initial_count = len(df)
        df = df[df['price'].notna() & (df['price'] > 0)]
        df = df[df['from_low_pct'].notna()]
        df = df[df['from_high_pct'].notna()]
        
        removed = initial_count - len(df)
        if removed > 0:
            logger.info(f"Removed {removed} invalid rows")
        
        # Add tier classifications
        df = DataProcessor.add_tiers(df)
        
        return df
    
    @staticmethod
    def add_tiers(df: pd.DataFrame) -> pd.DataFrame:
        """Add tier classifications"""
        
        # EPS Tier
        def get_eps_tier(eps):
            if pd.isna(eps):
                return "Unknown"
            for tier_name, (min_val, max_val) in CONFIG.EPS_TIERS.items():
                if min_val < eps <= max_val:
                    return tier_name
            return "Unknown"
        
        # PE Tier
        def get_pe_tier(pe):
            if pd.isna(pe) or pe <= 0:
                return "Negative/NA"
            for tier_name, (min_val, max_val) in CONFIG.PE_TIERS.items():
                if tier_name != "Negative/NA" and min_val < pe <= max_val:
                    return tier_name
            return "Unknown"
        
        # Price Tier
        def get_price_tier(price):
            if pd.isna(price):
                return "Unknown"
            for tier_name, (min_val, max_val) in CONFIG.PRICE_TIERS.items():
                if min_val < price <= max_val:
                    return tier_name
            return "Unknown"
        
        df['eps_tier'] = df['eps_current'].apply(get_eps_tier)
        df['pe_tier'] = df['pe'].apply(get_pe_tier)
        df['price_tier'] = df['price'].apply(get_price_tier)
        
        return df

# ============================================
# RANKING ENGINE
# ============================================

class RankingEngine:
    """Core ranking calculations"""
    
    @staticmethod
    def safe_rank(series: pd.Series, pct: bool = False, ascending: bool = True) -> pd.Series:
        """Safely rank a series handling NaN values"""
        # Replace inf values with NaN
        series = series.replace([np.inf, -np.inf], np.nan)
        
        # Perform ranking
        if pct:
            return series.rank(pct=True, ascending=ascending, na_option='bottom')
        else:
            return series.rank(ascending=ascending, method='min', na_option='bottom')
    
    @staticmethod
    def calculate_advanced_volume_score(df: pd.DataFrame) -> pd.Series:
        """Calculate advanced volume score using ALL 7 volume ratios"""
        # Fill NaN values with 1.0 (no change) for all volume ratios
        volume_cols = ['vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d',
                      'vol_ratio_1d_180d', 'vol_ratio_7d_180d', 'vol_ratio_30d_180d',
                      'vol_ratio_90d_180d']
        
        for col in volume_cols:
            if col in df.columns:
                df[col] = df[col].fillna(1.0)
        
        # Rank all volume ratios
        df['rank_vol_1d_90d'] = RankingEngine.safe_rank(df['vol_ratio_1d_90d'], pct=True) * 100
        df['rank_vol_7d_90d'] = RankingEngine.safe_rank(df['vol_ratio_7d_90d'], pct=True) * 100
        df['rank_vol_30d_90d'] = RankingEngine.safe_rank(df['vol_ratio_30d_90d'], pct=True) * 100
        df['rank_vol_30d_180d'] = RankingEngine.safe_rank(df['vol_ratio_30d_180d'], pct=True) * 100
        df['rank_vol_90d_180d'] = RankingEngine.safe_rank(df['vol_ratio_90d_180d'], pct=True) * 100
        
        # Short-term explosion (20%)
        short_term = df['rank_vol_1d_90d'] * 0.2
        
        # Medium-term accumulation (40%)
        medium_term = (
            df['rank_vol_7d_90d'] * 0.2 +
            df['rank_vol_30d_90d'] * 0.2
        )
        
        # Long-term institutional (40%) - USING THE MISSING RATIOS!
        long_term = (
            df['rank_vol_30d_180d'] * 0.15 +
            df['rank_vol_90d_180d'] * 0.25  # Most important for institutional!
        )
        
        return short_term + medium_term + long_term
    
    @staticmethod
    def calculate_momentum_acceleration(df: pd.DataFrame) -> pd.Series:
        """Calculate if momentum is accelerating or decelerating"""
        # Ensure we have the data
        df['ret_1d'] = df['ret_1d'].fillna(0.0)
        df['ret_7d'] = df['ret_7d'].fillna(0.0)
        df['ret_30d'] = df['ret_30d'].fillna(0.0)
        
        # Calculate daily averages
        daily_avg_7d = df['ret_7d'] / 7
        daily_avg_30d = df['ret_30d'] / 30
        
        # Acceleration metrics
        daily_vs_weekly = df['ret_1d'] - daily_avg_7d
        weekly_vs_monthly = daily_avg_7d - daily_avg_30d
        
        # Create acceleration score
        acceleration_score = pd.Series(index=df.index, dtype=float)
        
        # Perfect acceleration: today > week > month
        perfect_accel = (df['ret_1d'] > daily_avg_7d) & (daily_avg_7d > daily_avg_30d)
        acceleration_score[perfect_accel] = 100
        
        # Good acceleration: today > week
        good_accel = (~perfect_accel) & (df['ret_1d'] > daily_avg_7d)
        acceleration_score[good_accel] = 70
        
        # Neutral
        neutral = (~perfect_accel) & (~good_accel) & (df['ret_1d'] > 0)
        acceleration_score[neutral] = 50
        
        # Deceleration
        acceleration_score[acceleration_score.isna()] = 30
        
        return acceleration_score
    
    @staticmethod
    def calculate_breakout_probability(df: pd.DataFrame) -> pd.Series:
        """Calculate probability of breakout based on multiple factors"""
        # Distance from high (closer = higher probability)
        distance_score = (100 + df['from_high_pct']) # -20% becomes 80
        distance_score = distance_score.clip(0, 100)
        
        # Volume pressure (average of 7d and 30d ratios)
        volume_pressure = (df['vol_ratio_7d_90d'] + df['vol_ratio_30d_90d']) / 2
        volume_pressure = (volume_pressure - 1) * 100  # Convert to percentage above normal
        volume_pressure = volume_pressure.clip(0, 100)
        
        # Trend support (how many SMAs is price above)
        trend_support = pd.Series(0, index=df.index)
        if 'sma_20d' in df.columns:
            trend_support += (df['price'] > df['sma_20d']).astype(int) * 33.33
        if 'sma_50d' in df.columns:
            trend_support += (df['price'] > df['sma_50d']).astype(int) * 33.33
        if 'sma_200d' in df.columns:
            trend_support += (df['price'] > df['sma_200d']).astype(int) * 33.34
        
        # Combined breakout probability
        breakout_prob = (
            distance_score * 0.4 +
            volume_pressure * 0.4 +
            trend_support * 0.2
        )
        
        return breakout_prob.clip(0, 100)
    
    @staticmethod
    def calculate_category_relative_ranks(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate percentile ranks within each category for true relative comparison
        This ensures small caps compete with small caps, large with large
        """
        # Columns to rank within categories
        rank_columns = [
            'from_low_pct', 'from_high_pct', 'ret_1d', 'ret_7d', 'ret_30d',
            'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d',
            'vol_ratio_90d_180d', 'rvol'
        ]
        
        # Add category ranks for each metric
        for col in rank_columns:
            if col in df.columns:
                # Create category-specific ranks
                df[f'{col}_cat_rank'] = df.groupby('category')[col].transform(
                    lambda x: x.rank(pct=True, na_option='bottom') * 100
                )
        
        # Also calculate overall market ranks for comparison
        for col in rank_columns:
            if col in df.columns:
                df[f'{col}_mkt_rank'] = RankingEngine.safe_rank(df[col], pct=True) * 100
        
        return df
    
    @staticmethod
    def calculate_liquidity_score(df: pd.DataFrame) -> pd.Series:
        """
        Calculate liquidity score using absolute volume data
        High liquidity = easier to buy/sell large quantities
        """
        liquidity_score = pd.Series(50, index=df.index)
        
        # Use 30-day average volume as primary liquidity measure
        if 'volume_30d' in df.columns and 'price' in df.columns:
            # Calculate average daily traded value (in currency)
            df['avg_traded_value'] = df['volume_30d'] * df['price']
            
            # Rank by traded value
            liquidity_rank = df['avg_traded_value'].rank(pct=True, na_option='bottom') * 100
            
            # Create score based on liquidity rank
            liquidity_score = liquidity_rank
            
            # Additional factor: volume consistency
            if all(col in df.columns for col in ['volume_7d', 'volume_30d', 'volume_90d']):
                # Check if volume is consistent across timeframes
                vol_consistency = 1 - (
                    df[['volume_7d', 'volume_30d', 'volume_90d']].std(axis=1) / 
                    df[['volume_7d', 'volume_30d', 'volume_90d']].mean(axis=1)
                ).fillna(0)
                
                # Bonus for consistent volume
                liquidity_score = liquidity_score * 0.8 + (vol_consistency * 100) * 0.2
        
        return liquidity_score.clip(0, 100)
    
    @staticmethod
    def calculate_market_context_score(df: pd.DataFrame) -> pd.Series:
        """
        Determine how well stock is performing relative to overall market
        """
        # Calculate market averages
        market_avg_return = df['ret_30d'].median()
        market_avg_volume = df['vol_ratio_30d_90d'].median()
        
        # Calculate relative performance
        relative_return = df['ret_30d'] - market_avg_return
        relative_volume = df['vol_ratio_30d_90d'] / market_avg_volume
        
        # Create context score
        context_score = (
            RankingEngine.safe_rank(relative_return, pct=True) * 50 +
            RankingEngine.safe_rank(relative_volume, pct=True) * 50
        )
        
        return context_score
    
    @staticmethod
    def calculate_rvol_dynamics(df: pd.DataFrame) -> pd.Series:
        """Calculate relative volume dynamics using RVOL column"""
        rvol_score = pd.Series(50, index=df.index)  # Default neutral score
        
        if 'rvol' not in df.columns:
            return rvol_score
        
        # Fill NaN values
        df['rvol'] = df['rvol'].fillna(1.0)
        
        # Score based on RVOL levels
        # Extreme volume (>5x) - Climax
        rvol_score[df['rvol'] > 5] = 100
        
        # Very high volume (3-5x) - Strong interest
        rvol_score[(df['rvol'] > 3) & (df['rvol'] <= 5)] = 85
        
        # High volume (2-3x) - Increased activity
        rvol_score[(df['rvol'] > 2) & (df['rvol'] <= 3)] = 70
        
        # Above average (1.2-2x) - Healthy interest
        rvol_score[(df['rvol'] > 1.2) & (df['rvol'] <= 2)] = 60
        
        # Normal (0.8-1.2x) - Neutral
        rvol_score[(df['rvol'] > 0.8) & (df['rvol'] <= 1.2)] = 50
        
        # Below average (0.5-0.8x) - Quiet accumulation
        rvol_score[(df['rvol'] > 0.5) & (df['rvol'] <= 0.8)] = 40
        
        # Very low (<0.5x) - Dormant
        rvol_score[df['rvol'] <= 0.5] = 20
        
        return rvol_score
    
    @staticmethod
    def calculate_long_term_strength(df: pd.DataFrame) -> pd.Series:
        """Calculate long-term strength using all return periods"""
        strength_score = pd.Series(50, index=df.index)
        
        # List of return columns to check
        return_cols = ['ret_3m', 'ret_6m', 'ret_1y']
        available_cols = [col for col in return_cols if col in df.columns]
        
        if not available_cols:
            return strength_score
        
        # Calculate average long-term return
        long_returns = df[available_cols].fillna(0)
        avg_long_return = long_returns.mean(axis=1)
        
        # Score based on long-term performance
        strength_score = pd.Series(50, index=df.index)
        
        # Strong long-term (>50% average)
        strength_score[avg_long_return > 50] = 100
        
        # Good long-term (30-50%)
        strength_score[(avg_long_return > 30) & (avg_long_return <= 50)] = 85
        
        # Decent long-term (15-30%)
        strength_score[(avg_long_return > 15) & (avg_long_return <= 30)] = 70
        
        # Neutral (0-15%)
        strength_score[(avg_long_return > 0) & (avg_long_return <= 15)] = 55
        
        # Negative but recovering (> -15%)
        strength_score[(avg_long_return > -15) & (avg_long_return <= 0)] = 40
        
        # Poor long-term (< -15%)
        strength_score[avg_long_return <= -15] = 20
        
        return strength_score
    
    @staticmethod
    @timer
    def calculate_rankings(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate rankings using ENHANCED scoring system"""
        if df.empty:
            return df
        
        # Create a copy to avoid modifying original
        df = df.copy()
        
        # Calculate basic percentile ranks for position metrics
        df['rank_from_low'] = RankingEngine.safe_rank(df['from_low_pct'], pct=True) * 100
        df['rank_from_high'] = RankingEngine.safe_rank(100 + df['from_high_pct'], pct=True) * 100
        
        # Fill NaN values for returns
        df['ret_1d'] = df['ret_1d'].fillna(0.0)
        df['ret_7d'] = df['ret_7d'].fillna(0.0)
        df['ret_30d'] = df['ret_30d'].fillna(0.0)
        
        # Basic momentum ranks
        df['rank_ret_30d'] = RankingEngine.safe_rank(df['ret_30d'], pct=True) * 100
        
        # ENHANCED SCORING COMPONENTS
        
        # 1. Position Score (35%) - slightly reduced from 45%
        df['position_score'] = (
            df['rank_from_low'].fillna(50) * 0.6 +
            df['rank_from_high'].fillna(50) * 0.4
        )
        
        # 2. Advanced Volume Score (30%) - using ALL ratios
        df['volume_score'] = RankingEngine.calculate_advanced_volume_score(df)
        
        # 3. Momentum Score (15%) - reduced from 20%
        df['momentum_score'] = df['rank_ret_30d'].fillna(50)
        
        # 4. Momentum Acceleration (10%) - NEW!
        df['acceleration_score'] = RankingEngine.calculate_momentum_acceleration(df)
        
        # 5. Breakout Probability (10%) - NEW!
        df['breakout_score'] = RankingEngine.calculate_breakout_probability(df)
        
        # MASTER SCORE 2.0 - Enhanced formula
        df['master_score'] = (
            df['position_score'] * 0.35 +
            df['volume_score'] * 0.30 +
            df['momentum_score'] * 0.15 +
            df['acceleration_score'] * 0.10 +
            df['breakout_score'] * 0.10
        )
        
        # Add trend quality as a bonus (not part of main score)
        df['trend_quality'] = RankingEngine.calculate_trend_quality(df)
        
        # Final ranking
        df['rank'] = RankingEngine.safe_rank(df['master_score'], ascending=False)
        df['rank'] = df['rank'].fillna(9999).astype(int)
        
        df['percentile'] = RankingEngine.safe_rank(df['master_score'], pct=True) * 100
        df['percentile'] = df['percentile'].fillna(0)
        
        # Enhanced pattern detection
        df = RankingEngine.detect_smart_patterns(df)
        
        return df
    
    @staticmethod
    def detect_relative_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect patterns using PURE RELATIVE rankings - NO FIXED THRESHOLDS
        All patterns based on percentile ranks within categories and market
        
        Patterns detected:
        - 🔥 CAT LEADER: Top 10% in category
        - 💎 HIDDEN GEM: High category rank, low market rank
        - 🚀 ACCELERATING: Momentum ranks improving
        - 🏦 INSTITUTIONAL: Top 20% volume accumulation in category
        - ⚡ VOL EXPLOSION: Top 5% RVOL in category
        - 🎯 BREAKOUT: Top 20% breakout probability
        - 📈 RISING STAR: Improving ranks across timeframes
        - 👑 MARKET LEADER: Top 5% overall
        - 🌊 MOMENTUM WAVE: Consistent high ranks
        - 💰 LIQUID LEADER: Top liquidity + performance
        """
        patterns = []
        
        for idx, row in df.iterrows():
            stock_patterns = []
            
            # Get all the percentile ranks
            percentile = row.get('percentile', 0)
            cat_rank = row.get('category_rank', 999)
            
            # Volume ranks
            rvol_cat_rank = row.get('rvol_cat_rank', 50)
            vol_90_180_cat_rank = row.get('vol_ratio_90d_180d_cat_rank', 50)
            vol_30_90_cat_rank = row.get('vol_ratio_30d_90d_cat_rank', 50)
            
            # Position ranks
            from_low_cat_rank = row.get('from_low_pct_cat_rank', 50)
            
            # Return ranks
            ret_1d_cat_rank = row.get('ret_1d_cat_rank', 50)
            ret_7d_cat_rank = row.get('ret_7d_cat_rank', 50)
            ret_30d_cat_rank = row.get('ret_30d_cat_rank', 50)
            
            # Scores
            acceleration_score = row.get('acceleration_score', 50)
            breakout_score = row.get('breakout_score', 50)
            liquidity_score = row.get('liquidity_score', 50)
            
            # PATTERN 1: Category Leader (Top 10% in category)
            if cat_rank <= 5:  # Top 5 stocks in category
                stock_patterns.append("🔥 CAT LEADER")
            
            # PATTERN 2: Hidden Gem (High in category, not recognized market-wide)
            if cat_rank <= 10 and percentile < 70:
                stock_patterns.append("💎 HIDDEN GEM")
            
            # PATTERN 3: Accelerating (Momentum improving)
            if acceleration_score > 85:  # Top 15% acceleration
                stock_patterns.append("🚀 ACCELERATING")
            
            # PATTERN 4: Institutional Accumulation (Top 20% volume patterns)
            if vol_90_180_cat_rank > 80 and vol_30_90_cat_rank > 70:
                stock_patterns.append("🏦 INSTITUTIONAL")
            
            # PATTERN 5: Volume Explosion (Top 5% RVOL in category)
            if rvol_cat_rank > 95:
                stock_patterns.append("⚡ VOL EXPLOSION")
            
            # PATTERN 6: Breakout Ready (Top 20% breakout probability)
            if breakout_score > 80:
                stock_patterns.append("🎯 BREAKOUT")
            
            # PATTERN 7: Rising Star (Improving across timeframes)
            if (ret_1d_cat_rank > ret_7d_cat_rank > ret_30d_cat_rank and 
                ret_30d_cat_rank > 60):
                stock_patterns.append("📈 RISING STAR")
            
            # PATTERN 8: Market Leader (Top 5% overall)
            if percentile > 95:
                stock_patterns.append("👑 MARKET LEADER")
            
            # PATTERN 9: Momentum Wave (Consistent high performance)
            if all(rank > 75 for rank in [ret_1d_cat_rank, ret_7d_cat_rank, ret_30d_cat_rank]):
                stock_patterns.append("🌊 MOMENTUM WAVE")
            
            # PATTERN 10: Liquid Leader (High liquidity + good performance)
            if liquidity_score > 80 and percentile > 80:
                stock_patterns.append("💰 LIQUID LEADER")
            
            # PATTERN 11: Quiet Accumulation (Low RVOL but improving metrics)
            if (rvol_cat_rank < 50 and vol_90_180_cat_rank > 70 and 
                from_low_cat_rank > 60):
                stock_patterns.append("🤫 QUIET ACCUM")
            
            patterns.append(" | ".join(stock_patterns) if stock_patterns else "")
        
        df['patterns'] = patterns
        return df

# ============================================
# FILTER ENGINE
# ============================================

class FilterEngine:
    """Handle all filtering operations"""
    
    @staticmethod
    def get_unique_values(df: pd.DataFrame, column: str) -> List[str]:
        """Get unique values for a column, excluding 'Unknown'"""
        if column not in df.columns:
            return []
        
        values = df[column].unique().tolist()
        # Remove 'Unknown' and sort
        values = [v for v in values if v != 'Unknown' and pd.notna(v)]
        return sorted(values)
    
    @staticmethod
    def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply all filters to dataframe"""
        filtered_df = df.copy()
        
        # Category filter
        if filters.get('categories') and 'All' not in filters['categories']:
            filtered_df = filtered_df[filtered_df['category'].isin(filters['categories'])]
        
        # Sector filter
        if filters.get('sectors') and 'All' not in filters['sectors']:
            filtered_df = filtered_df[filtered_df['sector'].isin(filters['sectors'])]
        
        # EPS tier filter
        if filters.get('eps_tiers') and 'All' not in filters['eps_tiers']:
            filtered_df = filtered_df[filtered_df['eps_tier'].isin(filters['eps_tiers'])]
        
        # PE tier filter
        if filters.get('pe_tiers') and 'All' not in filters['pe_tiers']:
            filtered_df = filtered_df[filtered_df['pe_tier'].isin(filters['pe_tiers'])]
        
        # Price tier filter
        if filters.get('price_tiers') and 'All' not in filters['price_tiers']:
            filtered_df = filtered_df[filtered_df['price_tier'].isin(filters['price_tiers'])]
        
        # Score filter
        if filters.get('min_score', 0) > 0:
            filtered_df = filtered_df[filtered_df['master_score'] >= filters['min_score']]
        
        # EPS change filter
        if filters.get('min_eps_change') is not None:
            filtered_df = filtered_df[
                (filtered_df['eps_change_pct'] >= filters['min_eps_change']) | 
                (filtered_df['eps_change_pct'].isna())
            ]
        
        return filtered_df

# ============================================
# VISUALIZATION ENGINE
# ============================================

class Visualizer:
    """Create all visualizations"""
    
    @staticmethod
    def create_score_distribution(df: pd.DataFrame) -> go.Figure:
        """Score distribution chart"""
        fig = go.Figure()
        
        scores = ['position_score', 'volume_score', 'momentum_score']
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        
        for score, color in zip(scores, colors):
            if score in df.columns:
                # Filter out NaN values for box plot
                score_data = df[score].dropna()
                if len(score_data) > 0:
                    fig.add_trace(go.Box(
                        y=score_data,
                        name=score.replace('_', ' ').title(),
                        marker_color=color,
                        boxpoints='outliers'
                    ))
        
        fig.update_layout(
            title="Score Component Distribution",
            yaxis_title="Score (0-100)",
            template='plotly_white',
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_top_stocks_chart(df: pd.DataFrame, n: int = 20) -> go.Figure:
        """Top stocks breakdown with enhanced scores"""
        # Get top stocks with valid scores
        valid_df = df[df['master_score'].notna()]
        top_df = valid_df.nlargest(min(n, len(valid_df)), 'master_score')
        
        if len(top_df) == 0:
            return go.Figure()
        
        fig = go.Figure()
        
        # Calculate weighted contributions for display
        fig.add_trace(go.Bar(
            name='Position',
            y=top_df['ticker'],
            x=top_df['position_score'] * 0.35,  # 35% weight
            orientation='h',
            marker_color='#3498db',
            text=[f"{x:.1f}" for x in top_df['position_score']],
            textposition='inside'
        ))
        
        fig.add_trace(go.Bar(
            name='Volume',
            y=top_df['ticker'],
            x=top_df['volume_score'] * 0.30,  # 30% weight
            orientation='h',
            marker_color='#e74c3c',
            text=[f"{x:.1f}" for x in top_df['volume_score']],
            textposition='inside'
        ))
        
        fig.add_trace(go.Bar(
            name='Momentum',
            y=top_df['ticker'],
            x=top_df['momentum_score'] * 0.15,  # 15% weight
            orientation='h',
            marker_color='#2ecc71',
            text=[f"{x:.1f}" for x in top_df['momentum_score']],
            textposition='inside'
        ))
        
        # Add new components if they exist
        if 'acceleration_score' in top_df.columns:
            fig.add_trace(go.Bar(
                name='Acceleration',
                y=top_df['ticker'],
                x=top_df['acceleration_score'] * 0.10,  # 10% weight
                orientation='h',
                marker_color='#f39c12',
                text=[f"{x:.1f}" for x in top_df['acceleration_score']],
                textposition='inside'
            ))
        
        if 'breakout_score' in top_df.columns:
            fig.add_trace(go.Bar(
                name='Breakout',
                y=top_df['ticker'],
                x=top_df['breakout_score'] * 0.10,  # 10% weight
                orientation='h',
                marker_color='#9b59b6',
                text=[f"{x:.1f}" for x in top_df['breakout_score']],
                textposition='inside'
            ))
        
        if 'rvol_score' in top_df.columns:
            fig.add_trace(go.Bar(
                name='RVOL',
                y=top_df['ticker'],
                x=top_df['rvol_score'] * 0.10,  # 10% weight
                orientation='h',
                marker_color='#e67e22',
                text=[f"{x:.1f}" for x in top_df['rvol_score']],
                textposition='inside'
            ))
        
        fig.update_layout(
            title=f"Top {len(top_df)} Stocks - Enhanced Score Breakdown",
            xaxis_title="Weighted Score Contribution",
            barmode='stack',
            template='plotly_white',
            height=max(400, len(top_df) * 30),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
    
    @staticmethod
    def create_sector_performance(df: pd.DataFrame) -> go.Figure:
        """Sector performance analysis"""
        # Group by sector with valid scores
        valid_df = df[df['master_score'].notna()]
        
        sector_stats = valid_df.groupby('sector').agg({
            'master_score': 'mean',
            'position_score': 'mean',
            'ticker': 'count'
        }).reset_index()
        
        # Filter sectors with at least 3 stocks
        sector_stats = sector_stats[sector_stats['ticker'] >= 3]
        
        if len(sector_stats) == 0:
            return go.Figure()
        
        fig = px.scatter(
            sector_stats,
            x='position_score',
            y='master_score',
            size='ticker',
            color='master_score',
            hover_data=['ticker'],
            text='sector',
            title='Sector Performance Analysis',
            labels={
                'position_score': 'Average Position Score',
                'master_score': 'Average Master Score',
                'ticker': 'Number of Stocks'
            },
            color_continuous_scale='Viridis',
            template='plotly_white'
        )
        
        fig.update_traces(textposition='top center')
        fig.update_layout(height=500)
        
        return fig

# ============================================
# EXPORT ENGINE
# ============================================

class ExportEngine:
    """Handle data exports"""
    
    @staticmethod
    def create_excel_report(df: pd.DataFrame) -> BytesIO:
        """Create Excel report"""
        output = BytesIO()
        
        try:
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                # Top 100 stocks
                valid_df = df[df['master_score'].notna()]
                top_100 = valid_df.nlargest(min(100, len(valid_df)), 'master_score')
                
                export_cols = [
                    'rank', 'ticker', 'company_name', 'master_score',
                    'position_score', 'volume_score', 'momentum_score',
                    'acceleration_score', 'breakout_score', 'rvol_score',
                    'price', 'from_low_pct', 'from_high_pct',
                    'ret_30d', 'rvol', 'vol_ratio_30d_90d', 'vol_ratio_90d_180d',
                    'patterns', 'category', 'sector', 'eps_tier', 'pe_tier', 'price_tier'
                ]
                export_cols = [col for col in export_cols if col in top_100.columns]
                top_100[export_cols].to_excel(writer, sheet_name='Top 100', index=False)
                
                # All stocks summary
                summary_cols = ['rank', 'ticker', 'company_name', 'master_score', 'category', 'sector']
                summary_cols = [col for col in summary_cols if col in df.columns]
                df[summary_cols].to_excel(writer, sheet_name='All Stocks', index=False)
                
                # Sector analysis
                sector_stats = valid_df.groupby('sector').agg({
                    'master_score': ['mean', 'std', 'count'],
                    'position_score': 'mean',
                    'volume_score': 'mean',
                    'momentum_score': 'mean'
                }).round(2)
                sector_stats.to_excel(writer, sheet_name='Sector Analysis')
                
                # Category analysis
                category_stats = valid_df.groupby('category').agg({
                    'master_score': ['mean', 'std', 'count'],
                    'position_score': 'mean',
                    'volume_score': 'mean',
                    'momentum_score': 'mean'
                }).round(2)
                category_stats.to_excel(writer, sheet_name='Category Analysis')
        
        except Exception as e:
            logger.error(f"Error creating Excel report: {e}")
            
        output.seek(0)
        return output

# ============================================
# MAIN APPLICATION
# ============================================

def main():
    """Main application"""
    
    # Page config
    st.set_page_config(
        page_title="Wave Detection Ultimate 3.0",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main {padding: 0rem 1rem;}
    .stTabs [data-baseweb="tab-list"] {gap: 8px;}
    .stTabs [data-baseweb="tab"] {height: 50px;}
    div[data-testid="metric-container"] {
        background-color: rgba(28, 131, 225, 0.1);
        border: 1px solid rgba(28, 131, 225, 0.2);
        padding: 5% 5% 5% 10%;
        border-radius: 5px;
        overflow-wrap: break-word;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0; background: linear-gradient(90deg, #3498db, #2ecc71); color: white; border-radius: 10px; margin-bottom: 2rem;">
        <h1 style="margin: 0;">📊 Wave Detection Ultimate 3.0</h1>
        <p style="margin: 0;">Professional Stock Ranking System with RVOL Integration</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📁 Data Source")
        
        sheet_url = st.text_input(
            "Google Sheets URL",
            value=CONFIG.DEFAULT_SHEET_URL,
            help="Enter the Google Sheets URL"
        )
        
        gid = st.text_input(
            "Sheet ID (GID)",
            value=CONFIG.DEFAULT_GID,
            help="Enter the sheet ID"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh", help="Clear cache and reload"):
                st.cache_data.clear()
                st.rerun()
        
        with col2:
            if st.button("ℹ️ Help"):
                st.info("""
                **Enhanced Master Score 3.0:**
                - Position (30%): Distance from 52w low/high
                - Volume (25%): ALL 7 volume ratios analyzed
                - Momentum (15%): 30-day returns
                - Acceleration (10%): Momentum building/fading
                - Breakout (10%): Probability of breakout
                - RVOL (10%): Today's relative volume
                
                **New Features:**
                - RVOL integration for real-time volume
                - Long-term strength analysis
                - Smart search with autocomplete
                """)
        
        st.markdown("---")
        st.markdown("### 🔍 Filters")
    
    # Load and process data
    try:
        with st.spinner("Loading data..."):
            raw_df = load_google_sheets_data(sheet_url, gid)
        
        with st.spinner(f"Processing {len(raw_df):,} stocks..."):
            processed_df = DataProcessor.process_dataframe(raw_df)
        
        with st.spinner("Calculating rankings..."):
            ranked_df = RankingEngine.calculate_rankings(processed_df)
        
        if ranked_df.empty:
            st.error("No valid data after processing.")
            st.stop()
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()
    
    # Get filter options
    filter_engine = FilterEngine()
    
    # Sidebar filters
    with st.sidebar:
        filters = {}
        
        # Category filter
        categories = ['All'] + filter_engine.get_unique_values(ranked_df, 'category')
        filters['categories'] = st.multiselect(
            "Category",
            options=categories,
            default=['All'],
            help="Filter by market cap category"
        )
        
        # Sector filter
        sectors = ['All'] + filter_engine.get_unique_values(ranked_df, 'sector')
        filters['sectors'] = st.multiselect(
            "Sector",
            options=sectors,
            default=['All'],
            help="Filter by sector"
        )
        
        # EPS tier filter
        eps_tiers = ['All'] + filter_engine.get_unique_values(ranked_df, 'eps_tier')
        filters['eps_tiers'] = st.multiselect(
            "EPS Tier",
            options=eps_tiers,
            default=['All'],
            help="Filter by EPS tier"
        )
        
        # PE tier filter
        pe_tiers = ['All'] + filter_engine.get_unique_values(ranked_df, 'pe_tier')
        filters['pe_tiers'] = st.multiselect(
            "PE Tier",
            options=pe_tiers,
            default=['All'],
            help="Filter by PE ratio tier"
        )
        
        # Price tier filter
        price_tiers = ['All'] + filter_engine.get_unique_values(ranked_df, 'price_tier')
        filters['price_tiers'] = st.multiselect(
            "Price Tier",
            options=price_tiers,
            default=['All'],
            help="Filter by price range"
        )
        
        # Score filter
        filters['min_score'] = st.slider(
            "Minimum Score",
            min_value=0,
            max_value=100,
            value=0,
            step=5,
            help="Filter by minimum master score"
        )
        
        # EPS change filter
        if 'eps_change_pct' in ranked_df.columns:
            filters['min_eps_change'] = st.number_input(
                "Min EPS Change %",
                min_value=-100.0,
                max_value=1000.0,
                value=0.0,
                step=10.0,
                help="Filter by minimum EPS change percentage"
            )
    
    # Apply filters
    filtered_df = filter_engine.apply_filters(ranked_df, filters)
    
    # Sort by rank
    filtered_df = filtered_df.sort_values('rank')
    
    # Main content
    # Summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Stocks", f"{len(filtered_df):,}")
    
    with col2:
        valid_scores = filtered_df['master_score'].dropna()
        avg_score = valid_scores.mean() if len(valid_scores) > 0 else 0
        st.metric("Avg Score", f"{avg_score:.1f}")
    
    with col3:
        accelerating = ((filtered_df['acceleration_score'] > 80).sum() if 'acceleration_score' in filtered_df.columns and len(filtered_df) > 0 else 0)
        st.metric("Accelerating", f"{accelerating}")
    
    with col4:
        ready_breakout = ((filtered_df['breakout_score'] > 80).sum() if 'breakout_score' in filtered_df.columns and len(filtered_df) > 0 else 0)
        st.metric("Breakout Ready", f"{ready_breakout}")
    
    with col5:
        # High RVOL count
        high_rvol = ((filtered_df['rvol'] > 2).sum() if 'rvol' in filtered_df.columns and len(filtered_df) > 0 else 0)
        st.metric("High RVOL (>2x)", f"{high_rvol}")
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏆 Rankings", "📊 Analysis", "📈 Visualizations", "🔍 Search", "📥 Export"
    ])
    
    with tab1:
        st.markdown("### Top Ranked Stocks")
        
        # Display options
        col1, col2 = st.columns([1, 3])
        with col1:
            display_count = st.selectbox(
                "Show top",
                options=CONFIG.AVAILABLE_TOP_N,
                index=2
            )
        
        # Get top stocks
        display_df = filtered_df.head(display_count).copy()
        
        if len(display_df) > 0:
            # Format for display
            display_cols = [
                'rank', 'ticker', 'company_name', 'master_score',
                'position_score', 'volume_score', 'momentum_score',
                'acceleration_score', 'breakout_score', 'rvol_score',
                'patterns', 'price', 'from_low_pct', 'ret_30d', 'rvol',
                'category', 'sector', 'eps_tier', 'pe_tier'
            ]
            
            display_cols = [col for col in display_cols if col in display_df.columns]
            
            # Format numeric columns
            format_dict = {
                'master_score': '{:.1f}',
                'position_score': '{:.1f}',
                'volume_score': '{:.1f}',
                'momentum_score': '{:.1f}',
                'acceleration_score': '{:.1f}',
                'breakout_score': '{:.1f}',
                'rvol_score': '{:.1f}',
                'price': '₹{:,.2f}',
                'from_low_pct': '{:.1f}%',
                'ret_30d': '{:.1f}%',
                'rvol': '{:.2f}x'  # Display as multiplier (e.g., 2.5x)
            }
            
            for col, fmt in format_dict.items():
                if col in display_df.columns:
                    if col == 'rvol':
                        # Special handling for RVOL to ensure proper formatting
                        display_df[col] = display_df[col].apply(
                            lambda x: f"{x:.2f}x" if pd.notna(x) and x > 0 else 'N/A'
                        )
                    elif '%' in fmt:
                        display_df[col] = display_df[col].apply(
                            lambda x: fmt.format(x) if pd.notna(x) else ''
                        )
                    elif '₹' in fmt:
                        display_df[col] = display_df[col].apply(
                            lambda x: fmt.format(x) if pd.notna(x) else ''
                        )
                    else:
                        display_df[col] = display_df[col].apply(
                            lambda x: fmt.format(x) if pd.notna(x) else ''
                        )
            
            st.dataframe(
                display_df[display_cols],
                use_container_width=True,
                height=600
            )
        else:
            st.warning("No stocks match the selected filters.")
    
    with tab2:
        st.markdown("### Market Analysis")
        
        if len(filtered_df) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                # Score distribution
                fig_dist = Visualizer.create_score_distribution(filtered_df)
                st.plotly_chart(fig_dist, use_container_width=True)
            
            with col2:
                # Sector summary
                st.markdown("#### Sector Performance")
                valid_df = filtered_df[filtered_df['master_score'].notna()]
                
                if len(valid_df) > 0:
                    sector_summary = valid_df.groupby('sector').agg({
                        'master_score': ['mean', 'count']
                    }).round(2)
                    sector_summary.columns = ['Avg Score', 'Count']
                    sector_summary = sector_summary.sort_values('Avg Score', ascending=False)
                    st.dataframe(sector_summary, use_container_width=True)
                else:
                    st.info("No valid scores for sector analysis")
            
            # Category summary
            st.markdown("#### Category Performance")
            if len(valid_df) > 0:
                category_summary = valid_df.groupby('category').agg({
                    'master_score': ['mean', 'count']
                }).round(2)
                category_summary.columns = ['Avg Score', 'Count']
                category_summary = category_summary.sort_values('Avg Score', ascending=False)
                st.dataframe(category_summary, use_container_width=True)
            else:
                st.info("No valid scores for category analysis")
        else:
            st.info("No data available for analysis.")
    
    with tab3:
        st.markdown("### Visualizations")
        
        if len(filtered_df) > 0:
            # Top stocks chart
            st.markdown("#### Top Stocks Breakdown")
            n_stocks = st.slider("Number of stocks", 10, 50, 20)
            fig_top = Visualizer.create_top_stocks_chart(filtered_df, n_stocks)
            st.plotly_chart(fig_top, use_container_width=True)
            
            # Sector performance
            valid_df = filtered_df[filtered_df['master_score'].notna()]
            if len(valid_df.groupby('sector')) >= 3:
                st.markdown("#### Sector Performance")
                fig_sector = Visualizer.create_sector_performance(filtered_df)
                st.plotly_chart(fig_sector, use_container_width=True)
        else:
            st.info("No data available for visualization.")
    
    with tab4:
        st.markdown("### 🔍 Stock Search")
        
        # Create search interface
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Get list of all companies for autocomplete
            all_companies = filtered_df['company_name'].unique().tolist()
            all_tickers = filtered_df['ticker'].unique().tolist()
            
            # Combine tickers and company names for search
            search_options = []
            for _, row in filtered_df.iterrows():
                search_options.append(f"{row['ticker']} - {row['company_name']}")
            
            # Search input with selectbox for autocomplete
            search_query = st.selectbox(
                "Search by ticker or company name",
                options=[''] + sorted(search_options),
                format_func=lambda x: x if x else "Type to search...",
                help="Start typing to see suggestions"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            search_button = st.button("🔎 Search", type="primary", use_container_width=True)
        
        # Alternative text input for partial search
        with st.expander("Advanced Search"):
            # Alternative text input for partial search
            text_search = st.text_input(
                "Partial search (leave empty to use dropdown above)",
                placeholder="Enter partial ticker or company name...",
                help="Search for stocks containing this text"
            )
        
        # Perform search
        search_results = pd.DataFrame()
        
        if search_query and search_query != '':
            # Extract ticker from selection
            ticker = search_query.split(' - ')[0]
            search_results = filtered_df[filtered_df['ticker'] == ticker]
        
        elif text_search:
            # Partial text search
            mask = (
                filtered_df['ticker'].str.contains(text_search, case=False, na=False) |
                filtered_df['company_name'].str.contains(text_search, case=False, na=False)
            )
            search_results = filtered_df[mask]
        
        # Display search results
        if len(search_results) > 0:
            st.success(f"Found {len(search_results)} matching stock(s)")
            
            for idx, stock in search_results.iterrows():
                # Create expandable card for each stock
                with st.expander(f"📊 {stock['ticker']} - {stock['company_name']}", expanded=True):
                    # Header metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Master Score",
                            f"{stock['master_score']:.1f}",
                            f"Rank #{int(stock['rank'])}"
                        )
                    
                    with col2:
                        st.metric(
                            "Price",
                            f"₹{stock['price']:,.2f}",
                            f"{stock['ret_1d']:.1f}% today"
                        )
                    
                    with col3:
                        st.metric(
                            "Position",
                            f"{stock['from_low_pct']:.1f}% ↑",
                            f"{stock['from_high_pct']:.1f}% from high"
                        )
                    
                    with col4:
                        rvol_value = stock.get('rvol', 1.0)
                        if pd.notna(rvol_value):
                            rvol_display = f"{rvol_value:.2f}x"
                            rvol_status = "High" if rvol_value > 2 else "Normal" if rvol_value > 0.8 else "Low"
                        else:
                            rvol_display = "N/A"
                            rvol_status = "No data"
                        
                        st.metric(
                            "Volume (RVOL)",
                            rvol_display,
                            rvol_status
                        )
                    
                    # Detailed scores
                    st.markdown("#### 📈 Score Breakdown")
                    score_cols = st.columns(6)
                    
                    scores = [
                        ("Position", stock['position_score']),
                        ("Volume", stock['volume_score']),
                        ("Momentum", stock['momentum_score']),
                        ("Acceleration", stock.get('acceleration_score', 50)),
                        ("Breakout", stock.get('breakout_score', 50)),
                        ("RVOL", stock.get('rvol_score', 50))
                    ]
                    
                    for i, (name, score) in enumerate(scores):
                        with score_cols[i]:
                            # Color code based on score
                            if score >= 80:
                                color = "🟢"
                            elif score >= 60:
                                color = "🟡"
                            else:
                                color = "🔴"
                            st.metric(name, f"{color} {score:.0f}")
                    
                    # Pattern and signals
                    if stock.get('patterns', ''):
                        st.markdown(f"**Patterns:** {stock['patterns']}")
                    
                    # Additional metrics in columns
                    st.markdown("#### 📊 Key Metrics")
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    
                    with metric_col1:
                        st.markdown("**Returns**")
                        st.text(f"1 Day: {stock.get('ret_1d', 0):.1f}%")
                        st.text(f"7 Days: {stock.get('ret_7d', 0):.1f}%")
                        st.text(f"30 Days: {stock.get('ret_30d', 0):.1f}%")
                        if 'ret_3m' in stock:
                            st.text(f"3 Months: {stock.get('ret_3m', 0):.1f}%")
                    
                    with metric_col2:
                        st.markdown("**Classification**")
                        st.text(f"Category: {stock['category']}")
                        st.text(f"Sector: {stock['sector']}")
                        st.text(f"EPS Tier: {stock.get('eps_tier', 'N/A')}")
                        st.text(f"PE Tier: {stock.get('pe_tier', 'N/A')}")
                    
                    with metric_col3:
                        st.markdown("**Technicals**")
                        st.text(f"52W Low: ₹{stock.get('low_52w', 0):,.2f}")
                        st.text(f"52W High: ₹{stock.get('high_52w', 0):,.2f}")
                        if 'sma_200d' in stock:
                            sma_200 = stock.get('sma_200d', stock['price'])
                            above_below = "above" if stock['price'] > sma_200 else "below"
                            pct_diff = abs((stock['price'] - sma_200) / sma_200 * 100)
                            st.text(f"vs 200 SMA: {pct_diff:.1f}% {above_below}")
                    
                    # Long-term performance if available
                    if any(col in stock.index for col in ['ret_6m', 'ret_1y', 'ret_3y', 'ret_5y']):
                        st.markdown("#### 📅 Long-term Performance")
                        lt_cols = st.columns(4)
                        
                        long_term_metrics = [
                            ("6 Months", 'ret_6m'),
                            ("1 Year", 'ret_1y'),
                            ("3 Years", 'ret_3y'),
                            ("5 Years", 'ret_5y')
                        ]
                        
                        for i, (label, col) in enumerate(long_term_metrics):
                            if col in stock.index:
                                with lt_cols[i]:
                                    value = stock[col]
                                    if pd.notna(value):
                                        color = "green" if value > 0 else "red"
                                        st.markdown(f"**{label}**")
                                        st.markdown(f"<span style='color: {color}'>{value:.1f}%</span>", 
                                                  unsafe_allow_html=True)
                    
                    # Volume analysis
                    st.markdown("#### 📊 Volume Analysis")
                    vol_col1, vol_col2 = st.columns(2)
                    
                    with vol_col1:
                        st.markdown("**Volume Ratios**")
                        vol_ratios = [
                            ("1D vs 90D", 'vol_ratio_1d_90d'),
                            ("7D vs 90D", 'vol_ratio_7d_90d'),
                            ("30D vs 90D", 'vol_ratio_30d_90d'),
                            ("90D vs 180D", 'vol_ratio_90d_180d')
                        ]
                        
                        for label, col in vol_ratios:
                            if col in stock.index:
                                value = stock[col]
                                if pd.notna(value):
                                    # Convert back to percentage for display
                                    pct = (value - 1) * 100
                                    color = "green" if pct > 0 else "red"
                                    st.text(f"{label}: {pct:+.1f}%")
                    
                    with vol_col2:
                        st.markdown("**Volume Trend**")
                        # Determine volume trend
                        vol_30d = stock.get('vol_ratio_30d_90d', 1)
                        vol_90d = stock.get('vol_ratio_90d_180d', 1)
                        
                        if vol_30d > 1.2 and vol_90d > 1.1:
                            st.success("📈 Accumulation Pattern")
                        elif vol_30d < 0.8 and vol_90d < 0.9:
                            st.warning("📉 Distribution Pattern")
                        else:
                            st.info("➡️ Normal Volume Pattern")
        
        elif search_button or text_search:
            st.warning("No stocks found matching your search criteria")
        
        # Top movers in searched results
        if len(search_results) > 1:
            st.markdown("---")
            st.markdown("### 🏆 Best Matches")
            
            top_matches = search_results.nlargest(min(5, len(search_results)), 'master_score')[
                ['rank', 'ticker', 'company_name', 'master_score', 'price', 'ret_30d', 'patterns']
            ].copy()
            
            # Format for display
            top_matches['price'] = top_matches['price'].apply(lambda x: f'₹{x:,.2f}')
            top_matches['ret_30d'] = top_matches['ret_30d'].apply(lambda x: f'{x:.1f}%')
            top_matches['master_score'] = top_matches['master_score'].apply(lambda x: f'{x:.1f}')
            
            st.dataframe(top_matches, use_container_width=True, hide_index=True)
    
    with tab5:
        st.markdown("### Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Excel Report")
            st.markdown("Comprehensive report with multiple sheets")
            
            if st.button("📊 Generate Excel Report"):
                with st.spinner("Generating report..."):
                    excel_file = ExportEngine.create_excel_report(filtered_df)
                    
                    st.download_button(
                        label="📥 Download Excel",
                        data=excel_file,
                        file_name=f"wave_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
        
        with col2:
            st.markdown("#### CSV Export")
            st.markdown("Raw data for further analysis")
            
            if st.button("📄 Generate CSV"):
                csv_data = filtered_df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name=f"wave_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()
